# Conditional statistical errors for the historical quadratic table

The saved binned fits reproduce the four historical coefficients on short
Note 2 slide 3. Their CSV files did not contain errors. This calculation adds
conditional statistical errors without changing the estimator or coefficients.
It does not substitute the newer MCP calibrations used in other comparisons.

## Calculation

For each saved bin, use its mean inverse charge x = mean(1/Q), mean timing y
in ns, and timing SEM. The original script fits y = a + b*x + c*x^2 with
unweighted `numpy.polyfit`; the SEM values were plotted, not used as fit weights.
The existing +26 ns display offset is already in y and a, and is not added again.

Let X have columns (1, x, x^2), and let J be its least-squares pseudoinverse.
The fitted coefficient vector is J*y. Assuming independent bin means and fixed
x coordinates, the propagated covariance is

```
V = J diag(SEM^2) J^T
sigma_a, sigma_b, sigma_c = sqrt(diag(V))
```

The implementation centers and scales x for numerical stability, then
transforms coefficients and covariance back to (a, b, c). Every coefficient
is checked against both `numpy.polyfit` and the original coefficient CSV.
No residual chi-square scaling is applied. This is propagation of the saved
statistical errors, not the homoscedastic residual covariance returned by
`polyfit(..., cov=True)`.

## Interpretation

These are conditional one-standard-deviation statistical errors, with binning,
selection, coordinates and calibration held fixed. The calculation excludes
charge-coordinate uncertainty, shared clock/calibration correlations,
selection/binning changes and uncertainty in the functional form.
Consequently these are not total calibration uncertainties.

The coefficients are strongly correlated. Their separate errors must not be
added in quadrature to construct a curve band; use v(Q)^T V v(Q), with
v(Q) = (1, 1/Q, 1/Q^2). Large marginal coefficient errors do not by themselves
imply large uncertainty in the curve within its fitted range.

## Artifacts

- `coefficients.csv`: original full-precision coefficients and recovered errors.
- `covariance.json`: covariance and correlation matrices, units and input hashes.
- `table.tex`: rounded slide table; the c column is expressed in units of
  10^4 ns ADC^2 to keep the table readable.
- `verification.json`: numerical checks against independent fits and propagation.

Rebuild with `prepare_note2_quadratic_errors.py` in the short-slide directory.
Its inputs are the existing frozen profile and coefficient snapshots bundled
with Note 1; no ROOT input or event reprocessing is required.
