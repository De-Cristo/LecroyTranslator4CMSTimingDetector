import numpy as np


def compute_zero_line(a, zero_line_override=None):
    if zero_line_override is not None:
        return float(zero_line_override)
    return 0.5 * (np.nanmax(a) + np.nanmin(a))


def normalize_trace(values):
    if np.nanstd(values) == 0:
        return None
    return (values - np.mean(values)) / np.std(values)


def _interpolate_crossing(t_rel, a, j, zero_line, crossing_type=None):
    a0 = a[j]
    a1 = a[j + 1]
    t0 = float(t_rel[j])
    t1 = float(t_rel[j + 1])
    denom = a1 - a0
    if denom == 0:
        frac = 0.5
    else:
        frac = float((zero_line - a0) / denom)
    frac = max(0.0, min(1.0, frac))
    t_edge = t0 + frac * (t1 - t0)
    edge = (t_edge, int(j), float(zero_line))
    if crossing_type is None:
        return edge
    return edge + (crossing_type,)


def detect_zero_crossing(t_rel, a, polarity="rising", zero_line_override=None):
    zero_line = compute_zero_line(a, zero_line_override=zero_line_override)
    v = a - zero_line
    if polarity == "rising":
        idx = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    elif polarity == "falling":
        idx = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
    else:
        idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
        idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
        idx = np.sort(np.concatenate((idx_r, idx_f)))

    if len(idx) == 0:
        return np.nan, -1, zero_line
    return _interpolate_crossing(t_rel, a, int(idx[0]), zero_line)


def detect_zero_crossings(t_rel, a, polarity="rising", zero_line_override=None):
    zero_line = compute_zero_line(a, zero_line_override=zero_line_override)
    v = a - zero_line
    if polarity == "rising":
        idxs = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    elif polarity == "falling":
        idxs = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
    else:
        idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
        idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
        idxs = np.sort(np.concatenate((idx_r, idx_f)))

    return [_interpolate_crossing(t_rel, a, int(j), zero_line) for j in idxs]


def detect_zero_crossings_typed(t_rel, a, zero_line_override=None):
    zero_line = compute_zero_line(a, zero_line_override=zero_line_override)
    v = a - zero_line
    idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]

    out = []
    for j in idx_r:
        out.append(_interpolate_crossing(t_rel, a, int(j), zero_line, "rising"))
    for j in idx_f:
        out.append(_interpolate_crossing(t_rel, a, int(j), zero_line, "falling"))
    out.sort(key=lambda x: x[0])
    return out


def build_template_from_edges(
    waves,
    meta=None,
    polarity="rising",
    pre_ns=1.0,
    post_ns=5.0,
    max_cycles=200,
    zero_line_override=None,
):
    snippets = []
    dt_ns = None
    for i, evt in enumerate(sorted(waves.keys())):
        if i >= max_cycles:
            break
        tns, a = waves[evt]
        t_rel = tns - tns[0]
        if dt_ns is None:
            if len(t_rel) > 1:
                dt_ns = float(t_rel[1] - t_rel[0])
            else:
                continue

        if polarity == "both":
            edges = detect_zero_crossings_typed(
                t_rel, a, zero_line_override=zero_line_override
            )
        else:
            edges = detect_zero_crossings(
                t_rel, a, polarity=polarity, zero_line_override=zero_line_override
            )
        if len(edges) == 0:
            continue

        t_edge = edges[0][0]
        start = t_edge - pre_ns
        stop = t_edge + post_ns
        n_points = int(round((post_ns + pre_ns) / dt_ns)) + 1
        ts = np.linspace(start, stop, n_points)
        try:
            snippet = np.interp(ts, t_rel, a)
        except Exception:
            continue
        snippet_norm = normalize_trace(snippet)
        if snippet_norm is None:
            continue
        snippets.append(snippet_norm)

    print(
        f"[info] build_template_from_edges: collected snippets={len(snippets)} dt_ns={dt_ns}"
    )
    if len(snippets) == 0:
        return None, None, None, None

    template = np.mean(np.vstack(snippets), axis=0)
    t_axis = np.linspace(-pre_ns, post_ns, len(template))
    return template, t_axis, dt_ns, None


def cross_correlate_align(snippet, template, dt_ns):
    s_norm = normalize_trace(snippet)
    t_norm = normalize_trace(template)
    if s_norm is None or t_norm is None:
        return np.nan, np.nan, np.nan, np.nan

    corr = np.correlate(s_norm, t_norm, mode="full")
    i0 = int(np.argmax(corr))
    n_points = len(s_norm)
    lag = i0 - (n_points - 1)
    if 1 <= i0 < len(corr) - 1:
        y0, y1, y2 = corr[i0 - 1], corr[i0], corr[i0 + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            peak_offset = 0.5 * (y0 - y2) / denom
        else:
            peak_offset = 0.0
    else:
        peak_offset = 0.0

    lag_refined = lag + peak_offset
    shift_ns = -lag_refined * dt_ns
    peak = corr[i0]
    peak_norm = peak / float(n_points) if n_points > 0 else np.nan
    return float(shift_ns), float(lag_refined), float(peak), float(peak_norm)


def collect_interedge_diffs(df, time_column):
    diffs = []
    for _, g_evt in df.groupby("eventNo"):
        times = g_evt[time_column].dropna().to_numpy()
        if len(times) <= 1:
            continue
        delta = np.diff(np.sort(times))
        delta = delta[~np.isnan(delta)]
        delta = delta[delta > 0]
        if len(delta) > 0:
            diffs.extend(delta.tolist())
    return np.asarray(diffs, dtype=float)


def filter_range(values, xmin, xmax):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return values[(values >= xmin) & (values <= xmax)]


def fit_edge_times(edge_times, drop_last=0):
    precise_times = np.asarray(edge_times, dtype=float)
    precise_times = precise_times[~np.isnan(precise_times)]
    if len(precise_times) == 0:
        return None

    precise_times = np.sort(precise_times)
    edge_indices_full = np.arange(len(precise_times))
    precise_times_used = precise_times.copy()
    edge_indices_used = edge_indices_full.copy()

    try:
        n_drop = int(drop_last) if drop_last is not None else 0
    except Exception:
        n_drop = 0

    if n_drop > 0 and len(precise_times_used) >= 2:
        if n_drop >= len(precise_times_used):
            if len(precise_times_used) > 1:
                n_drop = len(precise_times_used) - 1
            else:
                n_drop = 0
        if n_drop > 0:
            precise_times_used = precise_times_used[:-n_drop]
            edge_indices_used = edge_indices_used[:-n_drop]

    n_edges_total = len(precise_times)
    n_edges_used = len(precise_times_used)
    fit_vals = np.full(n_edges_used, np.nan, dtype=float)
    slope = np.nan
    intercept = np.nan
    sigma_single = np.nan
    sigma_t0 = np.nan
    sigma_t_ave = np.nan
    t_ave_ps = np.nan

    if n_edges_used >= 2:
        slope, intercept = np.polyfit(edge_indices_used, precise_times_used, 1)
        fit_vals = intercept + slope * edge_indices_used
        residuals = precise_times_used - fit_vals
        sigma_single = float(np.std(residuals, ddof=1))
    elif n_edges_used == 1:
        intercept = float(precise_times_used[0])

    if n_edges_used >= 2 and not np.isnan(sigma_single):
        mean_x = np.mean(edge_indices_used)
        sum_sq_diff_x = np.sum((edge_indices_used - mean_x) ** 2)
        sigma_t0 = sigma_single * np.sqrt(
            1.0 / n_edges_used + (mean_x ** 2) / sum_sq_diff_x
        )
        sigma_t_ave = sigma_single * np.sqrt(1.0 / n_edges_used)
        t_ave_ps = (intercept + slope * mean_x) * 1000.0

    return {
        "precise_times_all": precise_times,
        "precise_times_used": precise_times_used,
        "edge_indices_used": edge_indices_used,
        "fit_vals": fit_vals,
        "n_edges_total": int(n_edges_total),
        "n_edges_used": int(n_edges_used),
        "t0_ns": float(intercept),
        "tclk_ns": float(slope) if not np.isnan(slope) else np.nan,
        "sigma_single_edge_ns": sigma_single,
        "sigma_t0_ns": sigma_t0,
        "sigma_t_ave_ns": sigma_t_ave,
        "t_ave_ps": t_ave_ps,
    }
