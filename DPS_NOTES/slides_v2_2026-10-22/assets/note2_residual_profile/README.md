# Numerical residual illustration

This replaces the historical raster on short Note 2 slide 10. It is a new
profile-level calculation, not an exact reproduction of original Figure 5 or
a reprocessing of individual events. Regenerate with
`python DPS_NOTES/slides_v2_2026-10-22/prepare_note2_residual_profile.py`.
All numerical inputs are already bundled in the notes repository.

## Definition

For each original inverse-charge bin, define `Q_eff = 1 / invE_mean`.
This is the reciprocal of the mean inverse charge, not the mean charge.
Use the saved mean slope `Sbar(Q) = c0 + c1*Q` and fixed AWC = 20 uA
(original supplied detector Note 2, Table 2, bar 8 left SiPM). Compute

`r = delta_t_mean - AWC / Sbar(Q_eff) - b`.

The constant `b` is the inverse-variance weighted mean of the first two terms,
using weights `1 / delta_t_sem**2`. Input timing values are already in ns,
including the old +26 ns display offset; do not apply that offset again.
Output residuals and SEM are multiplied by 1000 for ps.

Keep a bin when its representative Q_eff is in the saved slope-profile range
[184.92645, 656.44073] ADC. This retains 43 of 48 bins, spanning approximately
185.55 to 400.69 ADC. Full-bin containment is not asserted. Five lower-charge
bins are excluded; no residual clipping or extrapolation is performed.

Evaluating a nonlinear correction at a representative charge is not the same
as averaging that correction over all events in a bin. These inputs do not
provide that event-level calculation; finite-bin effects are not estimated.

## Errors and diagnostic line

Displayed bars are exactly the saved input profile-mean SEM in ps. They are
conditional bars with AWC, slope coefficients and fitted centering treated as
fixed, not the total uncertainty of post-fit residuals. They omit slope-fit and
AWC uncertainties, shared timing/calibration systematics, charge-coordinate and
finite-bin effects, and cross-acquisition transfer effects. If the fitted
offset uncertainty were propagated, the centered residuals would be correlated;
these bars do not include that covariance.

The inverse-variance diagnostic line is not subtracted and does not change AWC.
Its covariance is the fixed-input-SEM regression covariance, without chi-square
rescaling. The same unknown constant is accounted for once in its 41 degrees
of freedom (43 bins minus intercept and trend). Residual RMS is 6.08 ps over
bin means, not a single-hit timing resolution; five central points exceed the
visual +/-10 ps guide. That guide is not a confidence band.

## Historical comparison

`historical_marker_audit.csv` contains digitized centers solely to compare
the old illustration with this calculation. No digitized values enter the new
residuals, centering or diagnostic fit. On the same 43 bins, removing an
arbitrary constant between the illustrations leaves a 1.61 ps RMS difference,
larger than the approximately 0.058 ps/pixel raster scale. Thus the shapes are
close but not identical. The saved slope coefficient also differs from the
rounded coefficient in the original Table 2. The original figure and its
unverified error-bar definition remain archived unchanged.

`residual_profile.csv` retains all 48 original input rows, the selection reason
and numerical outputs for selected bins. Excluded residual fields are empty.
`provenance.json` records exact parameters, domains, source hashes and results.
This illustration tests neither a reference-free AWC prediction nor transfer
to another operating configuration.
