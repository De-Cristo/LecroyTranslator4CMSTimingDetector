# MCP Timing Method — Detailed Scientific Summary

This document provides a highly detailed description of the MCP timing reconstruction pipeline implemented in `sandbox_test/MCP_Timing/`. This pipeline is designed to extract precise, per-event MCP peak arrival times and absolute clock phase ($t_0$) from LeCroy oscilloscope waveforms. The calibrated results serve as the high-precision timing reference for the bar-level LYSO analysis.

---

## 1. Hardware Context & Data Acquisition

The framework expects waveforms acquired from an oscilloscope (typically two segments per event):

| Channel | Content | Typical Filename Pattern |
|---------|---------|--------------------------|
| **C1** | MCP photomultiplier signal (negative inverted pulse) | `raw_C1_*_data.csv` |
| **C2** | 160 MHz reference clock (sinusoidal) | `raw_C2_*_data.csv` |

### Waveform and Metadata Structure
Each CSV waveform file must contain minimum columns: `Segment`, `Time_s`, `Voltage_V`. 
A companion `*_meta.csv` provides crucial per-segment absolute timing records:
*   **`trigger_time`**: The global event timestamp (seconds).
*   **`trigger_offset`**: The digitiser delay for the specific channel/segment (seconds).

Internal processing immediately converts all times from seconds to nanoseconds (ns) for fitting stability. Final deliverables use picoseconds (ps) for maximal arithmetic precision.

---

## 2. MCP Peak Reconstruction — `MCP_wave_reco.py`

This module is responsible for loading the MCP signal, correcting for digitiser delays, and fitting the primary pulse with sub-nanosecond precision.

### 2.1 Waveform Alignment
1. `load_wave_csv` extracts the waveform and its corresponding metadata arrays.
2. The raw scope timestamp `Time_s` is converted to ns (`g['_time_ns']`).
3. The absolute scope time is calculated by applying the digitiser correction:
   $t_{abs} = 10^9 \times \text{trigger\_offset} + \text{Time\_s\_ns}$
4. To maximize numerical stability during the fit, all coordinates are shifted relative to the absolute trigger reference:
   $t_{rel} = t_{abs} - t_{trigger\_abs}$

### 2.2 Sub-Sample Peak Fitting Algorithm
The `fit_largest_peak` function executes a robust two-pass Gaussian fit targeted at the largest absolute peak in the waveform block.

*   **Step 1: Extrema Detection**
    *   Uses `scipy.signal.find_peaks` on both $+V$ and $-V$ to locate signal minima and maxima, granting immunity to signal polarity inversion.
    *   The prominence threshold adapts dynamically to 5% of the full waveform peak-to-peak amplitude (with a strict $1\text{ mV}$ floor).
    *   Selects the single candidate with the largest absolute amplitude.

*   **Step 2: Sub-Sample Precision Refinement (Parabolic)**
    *   Initial baseline is estimated from the median of the first 10 waveform samples.
    *   The raw discrete peak position $t[i]$ is refined using a 3-point parabolic interpolation through $(t_{i-1}, y_{i-1}), (t_i, y_i), (t_{i+1}, y_{i+1})$.
    *   The fractional offset shift $\delta$ is calculated as:
      $\delta = 0.5 \times \frac{y_{i-1} - y_{i+1}}{y_{i-1} - 2y_i + y_{i+1}}$
    *   Refined position: $t_{\text{refined}} = t_i + \delta \cdot \Delta t$.

*   **Step 3: Constrained Gaussian Fit**
    *   A symmetric fitting window is drawn around the peak: $\pm 2.5 \times \text{estimated\_FWHM}$ (minimum 20 samples).
    *   Coordinates are re-centered around $t_{\text{refined}}$ to prevent floating-point catastrophic cancellation.
    *   A Levenberg-Marquardt solver fits a 4-parameter Gaussian model $G(x) = \text{baseline} + \text{amp} \times \exp(-\frac{1}{2}(\frac{x-\mu}{\sigma})^2)$.
    *   Yields `peak_time_ns` (relative to the fit window), `peak_amp`, `peak_sigma_ns` (width), and `baseline`.

---

## 3. Clock Edge Detection & Phase Extraction — `clock_study.py`

The core of the timing calibration relies on reconstructing the 160 MHz clock phase. Instead of tracking a single edge, it performs a multi-edge linear fit over the entire event window to aggressively reduce per-edge jitter.

### 3.1 Initial Zero-Crossing Detection
*   A target **Zero Line** is established at $\frac{\max(V) + \min(V)}{2}$ or a fixed user override.
*   Pairs of threshold-spanning samples are found. Sub-sample crossing times are computed via linear interpolation:
    $f = \frac{V_{\text{target}} - V_j}{V_{j+1} - V_j}$ ; $t_{edge} = t_j + f \cdot (t_{j+1} - t_j)$
*   A typed detector tags crossings as `rising` or `falling`.

### 3.2 Template-Based Precision Inter-Edge Timing (Primary Method)
Standard thresholding is susceptible to localized noise. The template method leverages whole-period waveform shapes. 

*   **Dual Template Construction (`build_template_from_edges`)**
    1. Collects $\le 200$ valid zero-crossing edge snippets (configurable window: typically $[-1\text{ ns}, +5\text{ ns}]$ relative to the crossing).
    2. Snippets are interpolated onto a uniform high-density time grid.
    3. Snippets are $Z$-score normalized: $s_{\text{norm}} = \frac{s - \mu}{\sigma}$.
    4. An average normalized template is built. If `--polarity both` is passed, the script concurrently builds strictly independent `rising` and `falling` templates.

*   **Cross-Correlation Alignment (`cross_correlate_align`)**
    1. For every edge candidate, the local snippet is extracted, normalized, and mapped to its corresponding template type (`rising` or `falling`).
    2. The full cross-correlation array $R$ is computed: $R = s_{\text{norm}} \star T_{\text{norm}}$.
    3. The integer lag is determined by $\operatorname{argmax}(R)$. 
    4. **Quadratic continuous alignment**: The precise peak of the correlation function is resolved analytically using a 3-point parabolic fit around the maximum lag bin.
    5. The final precise edge time is $t_{\text{precise}} = t_{\text{zero\_cross}} + \operatorname{shift}_{ns}$.
    6. Quality filters drop edges below a correlation threshold (`--template-min-corr`) or violating spacing logic (`--min-edge-spacing-ns`).

### 3.3 The Multi-Edge Linear Fit
For events featuring $N \ge 2$ successfully template-aligned edges, a linear model extracts the global event parameters:
$t_j = t_0 + j \times T_{clk}$ for $j \in [0, N-1]$

*   Fitted via least-squares (`np.polyfit`):
    *   **Slope ($T_{clk}$)**: The measured clock period for that specific event (expected $\approx 6.25 \text{ ns}$).
    *   **Intercept ($t_0$)**: The clock phase (ns) assigned to the leading zero-edge index.
    *   **Absolute $t_0$**: Reconstructed by re-adding the leading waveform absolute timestamp.
*   **Jitter Metric**: $\sigma_{t_0} = \frac{\sigma_{\text{residuals}}}{\sqrt{N_{\text{edges}}}}$. This standard error serves as an excellent diagnostic of phase lock health. Events exceeding `--high-jitter-threshold-ps` automatically yield diagnostic debug plots.

---

## 4. Signal Integration — `combine_mcp_clock.py`

This unifying script imports the functional objects from the aforementioned scripts, avoiding disk-IO bottlenecks. 

### 4.1 Fusion Workflow
1. Dispatches the clock logic to `clock_study.process_group()`. Extracts the exact event-by-event dictionary of $t_{0\_\text{abs\_ns}}$ and sorted absolute rising edges.
2. Iterates over MCP components. Fits the MCP peaks. 
3. Recovers the physical absolute peak time reversing the relative shifts: `peak_time_abs_ns = peak_time_raw_ns + float(trigger_offset)`.
4. **Edge proximity lookup**: Iterates the sorted clock edges to identify the exact last physically real rising clock edge that occurred *before or concurrently* with the MCP centroid.
5. Emits the unified CSV, strictly enforcing **picosecond (ps)** scaling mapping for all downstream timing variables.

### 4.2 Key output dimensions (All scaled to `ps`)
| Column | Final Output Definition |
|--------|-----------|
| `peak_time_ps` | The global absolute scope arrival time of the MCP centroid |
| `t0_abs_ps` | The absolute time of the $0^\text{th}$ clock edge extracted from the multi-edge linear fit |
| `prev_rising_edge_abs_ps` | The explicit absolute timestamp of the $160\text{ MHz}$ edge immediately leading the MCP hit |
| `peak_amp` | The true peak Gaussian envelope amplitude (V) |
| `peak_sigma_ps` | Envelope functional width (ps). Used for dynamic shape cuts. |
| `trigger_offset_ps` | The captured channel-delay relative to trigger timing |

---

## 5. Automation Interface — `process_mcp_clock.sh`

The top-level shell wrapper iterates over all detector chunks in an input directory:
- Locates combinations of `raw_C1_*_data.csv` (MCP).
- Automatically pairs with corresponding `raw_C2_*_data.csv` matrices by substitution semantics.
- Executes `combine_mcp_clock.py` enforcing base parameters:
  - `--clock-polarity rising`
  - `--clock-min-edge-spacing-ns 3` (ignores ringing)
  - `--clock-drop-last-edge 2` (guards against array wrap-around signal distortion)

---

## 6. Purpose within Bar-Level LYSO Analysis

The generated output precisely characterizes the reference timing plane per event:
- `peak_time_ps`: Directly supplies `peak_time` array upstream. Subtracted from `t_bar` to extract purely the detector timing resolution $\sigma_t$.
- `t0_abs_ps`: Subtracts from `peak_time` to identify the phase fraction (`phi_peak`) within the continuous 160 MHz clock cycles.
- Correcting intrinsic time-walk distributions to extract limit-case LYSO matrix capabilities.
