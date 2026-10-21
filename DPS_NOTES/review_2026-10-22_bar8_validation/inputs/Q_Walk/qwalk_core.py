import numpy as np
import pandas as pd
import awkward as ak
import uproot
from dataclasses import dataclass

CLOCK_PS = 6250.0
COARSE_MODULO = 1024
DAC_STEP_UA = 0.3125


def decode_thresholds(step2):
    step2 = np.asarray(step2, dtype=float)
    vth1 = np.floor(step2 / 10_000).astype(int) - 1
    vth2 = np.floor((step2 - 10_000 * (vth1 + 1)) / 100).astype(int) - 1
    return vth1, vth2, (vth2 - vth1) * DAC_STEP_UA


def reconstruct_time2(time1, t1coarse, t2coarse, calib_t1, calib_t2):
    # The reference CSV used uint16 pandas columns, which overflowed when multiplied by 6250.
    # We must reproduce this exact arithmetic to match the notebook input.
    t1c = np.asarray(t1coarse, dtype=np.uint16) % np.uint16(COARSE_MODULO)
    t2c = np.asarray(t2coarse, dtype=np.uint16) % np.uint16(COARSE_MODULO)
    delta_coarse = (t2c - t1c) % np.uint16(COARSE_MODULO)
    delta_coarse_ps = (delta_coarse * np.uint16(6250))
    return np.asarray(time1, dtype=float) + delta_coarse_ps.astype(float) - calib_t1 + calib_t2


def reconstruct_time2_safe(time1, t1coarse, t2coarse, calib_t1, calib_t2):
    t1c = np.asarray(t1coarse, dtype=np.int64) % COARSE_MODULO
    t2c = np.asarray(t2coarse, dtype=np.int64) % COARSE_MODULO
    delta_coarse = (t2c - t1c) % COARSE_MODULO
    return (
        np.asarray(time1, dtype=float)
        + delta_coarse.astype(float) * CLOCK_PS
        - np.asarray(calib_t1, dtype=float)
        + np.asarray(calib_t2, dtype=float)
    )


def calculate_slope(time1, time2, delta_i):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1000.0 * np.asarray(delta_i, dtype=float) / (np.asarray(time2, dtype=float) - np.asarray(time1, dtype=float))


def walk_empirical_137(q):
    q = np.asarray(q, dtype=float)
    return 0.1218 + 198.5 / q - 15230.0 / q**2


def walk_slope_model_137(q):
    q = np.asarray(q, dtype=float)
    return -0.7 + 20.0 / (11.3429 + 0.0133429 * q)


def build_calibration_table(channel, tac, fine, time):
    table = {}
    for ch, tac_id, fine_code, value in zip(channel, tac, fine, np.asarray(time) % CLOCK_PS):
        key = (int(ch), int(tac_id), int(fine_code))
        total, count = table.get(key, (0.0, 0))
        table[key] = (total + float(value), count + 1)
    return {key: total / count for key, (total, count) in table.items()}


def lookup_calibration(table, channel, tac, fine):
    keys = zip(channel, tac, fine)
    values = np.empty(len(channel), dtype=float)
    missing = np.zeros(len(channel), dtype=bool)
    for index, key in enumerate(keys):
        value = table.get(tuple(map(int, key)))
        missing[index] = value is None
        values[index] = 0.0 if value is None else value
    return values, missing


def read_root_hits(root_path, time2_mode="notebook"):
    if time2_mode == "notebook":
        reconstruct = reconstruct_time2
    elif time2_mode == "safe":
        reconstruct = reconstruct_time2_safe
    else:
        raise ValueError(f"Unsupported time2_mode: {time2_mode!r}")

    branches = [
        "step1", "step2", "tot", "t1coarse", "t1fine", 
        "t2coarse", "t2fine", "time", "channelID", "energy", "tacID"
    ]
    with uproot.open(root_path) as f:
        tree = f["data"]
        df = ak.to_dataframe(tree.arrays(branches, library="ak")).reset_index()

    calib_table = build_calibration_table(
        df["channelID"].to_numpy(),
        df["tacID"].to_numpy(),
        df["t1fine"].to_numpy(),
        df["time"].to_numpy(),
    )

    calib_t1, missing_t1 = lookup_calibration(
        calib_table,
        df["channelID"].to_numpy(),
        df["tacID"].to_numpy(),
        df["t1fine"].to_numpy(),
    )
    calib_t2, missing_t2 = lookup_calibration(
        calib_table,
        df["channelID"].to_numpy(),
        df["tacID"].to_numpy(),
        df["t2fine"].to_numpy(),
    )

    df["time2"] = reconstruct(
        df["time"].to_numpy(),
        df["t1coarse"].to_numpy(),
        df["t2coarse"].to_numpy(),
        calib_t1,
        calib_t2,
    )
    df["missing_calib_t1"] = missing_t1
    df["missing_calib_t2"] = missing_t2

    return df


NOTEBOOK_CHANNELS = (133, 137, 150, 154)
NOTEBOOK_ANCHOR_CHANNELS = (133, 137)
UPSTREAM_CHANNELS = tuple(range(128, 160))
FIXED_DOWNSTREAM_CHANNELS = (234, 243)
FIT_WINDOWS = {
    137: {"q_min": 100.0, "q_max": 650.0, "slope_min": 10.0, "slope_max": 24.0},
    150: {"q_min": 100.0, "q_max": 800.0, "slope_min": 10.0, "slope_max": 24.0},
}


def select_notebook_hits(hits):
    event_sizes = hits.groupby("entry")["subentry"].transform("size")
    anchor_entries = hits.loc[hits["channelID"].isin(NOTEBOOK_ANCHOR_CHANNELS), "entry"].unique()
    has_anchor = hits["entry"].isin(anchor_entries)
    return hits.loc[(event_sizes == 4) & has_anchor & hits["channelID"].isin(NOTEBOOK_CHANNELS)].copy()


def select_upstream_hits_with_fixed_downstream_bar(
    hits,
    upstream_channels=UPSTREAM_CHANNELS,
    downstream_channels=FIXED_DOWNSTREAM_CHANNELS,
):
    """Select upstream hits from exact 2-upstream + 2-fixed-downstream events.

    The threshold-scan slope does not require an external time reference, but
    keeping the fixed downstream-bar coincidence restricts the calibration to
    the same particle-selection topology used by the timing analysis.
    """
    upstream_channels = tuple(map(int, upstream_channels))
    downstream_channels = tuple(map(int, downstream_channels))
    if len(downstream_channels) != 2 or len(set(downstream_channels)) != 2:
        raise ValueError("Exactly two distinct fixed downstream channels are required")

    is_upstream = hits["channelID"].isin(upstream_channels)
    is_downstream = hits["channelID"].isin(downstream_channels)
    event_size = hits.groupby("entry")["subentry"].transform("size")
    upstream_count = is_upstream.groupby(hits["entry"]).transform("sum")
    allowed_count = (is_upstream | is_downstream).groupby(hits["entry"]).transform("sum")
    unique_channel_count = hits.groupby("entry")["channelID"].transform("nunique")
    downstream_unique = hits.loc[is_downstream].groupby("entry")["channelID"].nunique()
    has_fixed_downstream = hits["entry"].map(downstream_unique).eq(len(downstream_channels))

    exact_topology = (
        (event_size == 4)
        & (upstream_count == 2)
        & (allowed_count == 4)
        & (unique_channel_count == 4)
        & has_fixed_downstream
    )
    return hits.loc[exact_topology & is_upstream].copy()


def select_notebook_fit_hits(hits, channel):
    """Apply the strict Q/slope window used by the Mathematica notebook."""
    window = FIT_WINDOWS[int(channel)]
    mask = (
        (hits["energy"] > window["q_min"])
        & (hits["energy"] < window["q_max"])
        & (hits["slope_ua_per_ns"] > window["slope_min"])
        & (hits["slope_ua_per_ns"] < window["slope_max"])
    )
    return hits.loc[mask].copy()


@dataclass
class SlopeFit:
    fit_mask: np.ndarray
    linear_coefficients: np.ndarray
    quadratic_coefficients: np.ndarray
    local_trend: pd.DataFrame


def fit_slope_vs_q(q, slope, bandwidth=20.0):
    q = np.asarray(q, dtype=float)
    slope = np.asarray(slope, dtype=float)
    
    # Polynomial order 1 -> polyfit returns [b, a]. We want intercept-first: [a, b].
    lin_coefs = np.polyfit(q, slope, 1)[::-1]
    
    # Quadratic fit: slope = a + b*q + c*q^2 -> [c, b, a]. We want [a, b, c].
    quad_coefs = np.polyfit(q, slope, 2)[::-1]
    
    # Local trend using gaussian weighted local linear regression
    grid_q = np.linspace(q.min(), q.max(), 100)
    trend_slope = np.zeros_like(grid_q)
    for i, g in enumerate(grid_q):
        w = np.exp(-0.5 * ((q - g) / bandwidth)**2)
        X = np.vstack([np.ones_like(q), q - g]).T
        try:
            weighted_x = X * w[:, None]
            beta = np.linalg.solve(X.T @ weighted_x, X.T @ (w * slope))
            trend_slope[i] = beta[0]
        except np.linalg.LinAlgError:
            trend_slope[i] = np.nan
            
    local_trend = pd.DataFrame({"q": grid_q, "slope": trend_slope})
    
    # fit_mask is just True for all here since filtering is done before
    fit_mask = np.ones_like(q, dtype=bool)
    
    return SlopeFit(fit_mask, lin_coefs, quad_coefs, local_trend)
