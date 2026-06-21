# Current Timing Method (Bar-Level Analysis)

This note summarizes the timing method implemented by `bar_analysis_main.py` and its supporting modules (`bar_processing.py`, `bar_plotting.py`, `bar_helpers.py`, `channel_mapping.py`). It reflects the current behavior of the code, not a proposed change.

## 1. Inputs, channels, and event selection
- **Input**: ROOT files with a data TTree found by `find_data_tree()` (prefers `data*` keys, highest cycle).
- **Channels**:
  - L/R channel indices for each LYSO bar are defined in `channel_mapping.py` and offset by module base (`UP_MODULE_BASE=128`, `DOWN_MODULE_BASE=224`).
  - Trigger channel is fixed at `TRIGGER_CHANNEL=192`.
- **Event presence**:
  - Events must contain both selected bar channels (`ch_L`, `ch_R`).
  - Optional `--strict-bar-only` rejects events containing unexpected channels (except trigger and optionally the other module or down-bar channels).
  - In fast mode, `--require-trigger` (and MCP mode) additionally require channel 192 to be present.

## 2. Two-pass workflow and energy cuts
**Pass 1 (discovery)**
- Extracts raw energy spectra for each channel.
- Fits a Landau-like Moyal distribution to the *peak region* of each energy histogram.
- Derives automatic channel-wise energy cuts:
  - `cut_lo = MPV - 1 * scale`
  - `cut_hi = MPV + 3 * scale`

**Pass 2 (final)**
- Re-runs extraction with:
  - Optional global `--energy-min/--energy-max` on the average energy.
  - Auto-derived per-channel Landau cuts for L/R (and down-bar if enabled).
- Produces final timing/phase distributions and fits.

## 3. Primary timing observables
**Raw time difference**
- `t_diff = T_L - T_R` (from the `time` branch).
- Used for the main bar timing distribution and Gaussian fit.

**Bar average time**
- `t_bar = (T_L + T_R) / 2`.
- Used for MCP comparisons and cross-module (up vs down) timing.

## 4. Phase (`phi`) computation from `t1coarse`
If `t1coarse` exists:
- Per-channel phase:
  - `phi = (t - t1coarse * 6250) % 6250` (ps)
- Phase difference within a bar:
  - `phi_diff = (phi_L - phi_R + 3125) % 6250 - 3125` (wrap to `[-3125, 3125]` ps)
- Trigger phase (if channel 192 present):
  - `phi_192 = (t_192 - t1coarse_192 * 6250) % 6250`

## 5. MCP-referenced timing and phase (optional `--mcp`)
An MCP tree provides per-event reference values (matched by MCP event index):
- `phi_peak`, `phi_trigger`, `peak_time`, `trigger_time`, optional `peak_amp` cuts.
- Optional internal MCP timing cut:
  - enable with `--mcp-internal-dt-cut`
  - compute `delta_t_mcp = peak_time - trigger_time`
  - keep events inside a robust window around the median
  - default half-window = `3.0 * robust_width`, configured by `--mcp-internal-dt-nmad`
  - `robust_width` uses MAD, with IQR fallback if MAD is zero

**Phase vs MCP**
- Bar average phase (wrap-aware):
  - `d = wrap(phi_L - phi_R)`
  - `phi_avg* = wrap(phi_R + d/2)`
- MCP-referenced phase:
  - `phi_vs_mcp = wrap(phi_avg* - phi_peak)`
- Per-channel MCP-referenced phase:
  - `phi_L - phi_peak` and `phi_R - phi_peak` (same wrapping)

**Raw time and phase differences vs MCP**
- `raw_time_diff = (t_bar - t_192) - (peak_time - trigger_time)`
- `raw_phi_diff = (phi_bar - phi_192) - (phi_peak - phi_trigger)` with wrapping

**Average time vs MCP**
- `t_avg_vs_mcp = t_bar - peak_time`

## 6. Dual-module (up vs down) timing
When `--down-lyso-bar` is provided:
- Down bar times and phases are computed identically.
- Cross-module observables:
  - `t_avg_diff = (t_avg_up - t_avg_down)`
  - `phi_avg_diff = wrap(phi_avg_up - phi_avg_down)` using wrap-aware bar averages
  - `phi_L_up - phi_avg_down` for cross-module energy calibration
  - `phi_R_up - phi_avg_down` for cross-module energy calibration

## 7. Time-walk correction (energy-dependent calibration)
In MCP mode, per-channel phase vs energy is used to correct time-walk:
1. Build a **profile** of `phi` vs energy and fit a quadratic:
   - `phi(E) = a*E^2 + b*E + c`
2. Apply per-channel correction:
   - `phi_corr = phi_raw - (a*E^2 + b*E + c)`
3. Compute corrected bar average using synchronized per-channel data.

## 7b. Cross-module calibration (up vs down)
When `--down-lyso-bar` is enabled, a second time-walk calibration is performed using the **down-bar average as reference**:
1. Fit `phi_L_up - phi_avg_down` vs `E_L_up` with a quadratic.
2. Fit `phi_R_up - phi_avg_down` vs `E_R_up` with a quadratic.
3. Apply both corrections and form a calibrated up–down average:
   `phi_avg_diff_calib = 0.5 * ( (phi_L_up - phi_avg_down - f_L(E_L)) + (phi_R_up - phi_avg_down - f_R(E_R)) )`
4. The width of `phi_avg_diff_calib` is used in the **calibrated absolute resolution** calculation.

## 8. Fitting and resolution extraction
**Gaussian fits**
- Timing/phase distributions (`t_diff`, `phi_diff`, MCP-referenced histograms) are fit with a two-pass Gaussian procedure centered around the peak.
- The fitted sigma is used as the measured timing resolution proxy.

**Absolute resolution (three-point method)**
When all three sigmas are available:
- `sigma(up - down)`
- `sigma(up - MCP)`
- `sigma(down - MCP)`

Intrinsic variances:
- `var_up   = 0.5 * (v1 + v2 - v3)`
- `var_down = 0.5 * (v1 + v3 - v2)`
- `var_mcp  = 0.5 * (v2 + v3 - v1)`

(where `v1 = sigma(up-down)^2`, `v2 = sigma(up-MCP)^2`, `v3 = sigma(down-MCP)^2`).

The same calculation is repeated for **per-channel calibrated** MCP sigmas when available.

## 8b. Trigger-subtracted absolute resolution
An additional end-of-run calculation can be printed using the measured trigger term:
- `sigma(Delta_trig)` is taken from `phi_192 - phi_trigger`
- For each bar-MCP width, a trigger-subtracted variance is formed:
  - `var(up-MCP)_sub = var(up-MCP) - var(Delta_trig)`
  - `var(down-MCP)_sub = var(down-MCP) - var(Delta_trig)`
- Those trigger-subtracted pair variances are then used in the same three-point intrinsic-resolution formulas.
- This calculation assumes the trigger term can be subtracted in quadrature from the bar-MCP widths.

## 9. Key outputs
- `t_diff_*`, `phi_diff_*`, MCP-referenced histograms, and energy spectra.
- Optional calibrated histograms showing improved MCP-referenced resolution.
- Absolute timing resolution printout when sufficient sigmas are available.
