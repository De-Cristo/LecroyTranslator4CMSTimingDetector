# Engineering Noise Cancelling for Timing Residuals

**Context:** Analysis of the `ch192_vs_trig_resid_fitonly_seg1.png` plot showing the residuals between channel 192 (trigger) and the MCP trigger time.

Based on the residual plot, the noise consists of two distinct components:

1. **Low-Frequency Baseline Drift (Wander):** The predominant feature is a slow, macroscopic oscillation (the cubic/S-shape traced by the red polynomial line). This drift occurs over a very long timescale ($\sim 6 \times 10^{11}$ ps, or about 10 minutes) and is typically caused by thermal fluctuations, clock synchronization drifts, or $1/f$ (flicker) electronic noise.
2. **High-Frequency Jitter (White Noise):** The scattered blue points around the red baseline represent event-by-event random noise (intrinsic timing resolution, statistical fluctuations, thermal electronic noise). 

To apply an "engineering noise cancelling" method as a parallel alternative to the current static polynomial or binned-median fit, you need a dynamic filter that tracks and cancels out the slow baseline drift while being robust against the high-frequency jitter. 

Here are the most suitable digital filtering approaches for this specific shape:

## 1. Savitzky-Golay Filter (Trend Tracking)
* **Why:** The Savitzky-Golay filter performs local polynomial regression over a sliding window. It is widely used in physics and engineering to smooth noisy data and track wandering baselines without introducing the severe phase delays (lag) or distortion that standard moving averages cause.
* **How to use it:** Sort the events chronologically and pass the residuals through a SavGol filter. The filter's continuous output becomes your dynamic baseline correction, which you subtract from the original signal event-by-event.
* **Implementation:** `scipy.signal.savgol_filter`

## 2. Moving Median Filter (Robust Low-Pass)
* **Why:** A standard moving average is highly susceptible to outliers (which are present as tightly scattered points in the plot). A **rolling median filter** acts as a robust low-pass filter. It closely mirrors the "binned median" approach already in use, but operates in a continuous, unbinned, sliding-window fashion.
* **How to use it:** Use a sliding window (e.g., 50–200 events) and calculate the median. This will cleanly trace the underlying slow drift dynamically while ignoring outlier jitter.
* **Implementation:** `pandas.Series.rolling.median()` or `scipy.ndimage.median_filter`

## 3. High-Pass Digital Filter (e.g., Butterworth)
* **Why:** If the time series of residuals is treated as a continuous signal, the slow drift is effectively very low-frequency noise (near DC). A standard digital High-Pass Filter (like a 1st or 2nd order Butterworth) applied to the chronological sequence of residuals will suppress the slow macroscopic wandering and forcefully zero-center the entire dataset.
* **How to use it:** Apply the high-pass filter sequentially over the time series of the residuals.
* **Implementation:** `scipy.signal.butter` combined with `scipy.signal.filtfilt` (for zero-phase filtering)

## 4. 1D Kalman Filter (Optimal State Tracking)
* **Why:** From a control systems perspective, the "true" baseline is a hidden state that drifts over time (analogous to a random walk), and the scattered points are noisy measurements of that state. A Kalman filter estimates this wandering baseline optimally and dynamically, updating event-by-event.
* **How to use it:** Define a 1D state model where the baseline velocity is near-zero but allowed to drift (process noise), and the measurement noise corresponds to the observed high-frequency jitter spread (e.g., $\sim 60$ ps).
* **Implementation:** Standard Kalman filter iterations or libraries like `pykalman`

---

### Recommendation
The **Savitzky-Golay filter** or a **Rolling Median filter** are the most direct drop-in replacements for the current polynomial fit. They are easy to implement and uniquely excel at dynamically capturing the smooth, snake-like baseline drift without requiring you to guess a global polynomial degree mapping the entire segment upfront.
