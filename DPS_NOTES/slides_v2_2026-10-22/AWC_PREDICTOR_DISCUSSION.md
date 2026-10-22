# Internal discussion: what justifies the second AWC candidate?

This is an exploratory model, not an established CMS result or a
first-principles determination of AWC. This candidate is deferred and its page
has been removed from the current short deck. The discussion is retained for
future consideration; only the empirical AWC trend remains as backup.

## Exact algebra versus the additional hypothesis

Start with the walk model and the measured mean-slope parametrization:

\[
W_i(Q)=\frac{A_i}{S_i(Q)},\qquad S_i(Q)=c_{0,i}+c_{1,i}Q.
\]

Holding a channel's AWC fixed, differentiation gives exactly

\[
-\frac{dW_i}{d\ln Q}=\frac{A_i c_{1,i}Q}{S_i(Q)^2}.
\]

The left-hand side is the change in walk for a small fractional change in
charge, with a minus sign. The new, unproven hypothesis is that this response
is approximately common across channels at a prescribed typical charge:

\[
-\left.\frac{dW_i}{d\ln Q}\right|_{Q_{*,i}}\simeq K
\quad\Longrightarrow\quad
A_i^{\rm pred}=K\frac{S_i(Q_{*,i})^2}{c_{1,i}Q_{*,i}}.
\]

Here K has units of ns. Q* is the median of each channel's selected charge
sample, computed once, not the event charge. AWC remains a channel constant.

## Why consider it, and what is not justified?

Similar scintillators and front-end shaping motivate testing whether channels
have similar timing response to a relative pulse-amplitude change. The feature
also survives a purely multiplicative ADC gain change: Q* scales with gain,
c1 inversely with gain, and their product is unchanged. This makes it a
reasonable candidate to test, not a derived universal relation.

Neither dual-threshold timing nor identical nominal detector construction
requires a common K. Pulse shapes, effective thresholds, noise, nonlinear
response and the selected charge population can change it. The median-Q*
choice is a practical convention, not a uniquely physical reference point.

K is fitted to MCP-derived AWC targets on other bars. Applying a learned K to
a new channel uses internal observables, but the current study does not remove
the reference from model development. The saved ten-channel, six-bar held-out
test gives 3.22 uA RMS error and 14.1% RMS fractional error. This small selected
sample, shared slope inputs in targets and predictors, and incomplete
uncertainty propagation limit the conclusion. Independent configurations and
corrected-event timing are still required before presenting a validated method.

## Suggested reply to the coauthor

The criticism is fair: the second model is a physics-motivated empirical
hypothesis, not a consequence of the dual-threshold principle. The algebra
rewrites AWC in terms of the local change of walk with fractional charge. We
then assume that change is approximately shared across channels. That extra
assumption, and the choice of typical charge, need discussion and independent
tests. The encouraging held-out-bar result is a reason to investigate it, not
yet a public performance claim. This candidate has now been removed from the
short deck; the empirical trend remains internal backup after the references.

## Provenance

- Formula and fit implementation: `Q_Walk/results/2026-09-24_measured_q_overlap/try_physical_candidate.py`.
- Displayed saved test: `Q_Walk/results/2026-10-22_clock_quality/physical_candidate_quick_test/`.
- No new fit or event processing was performed for this reorganization.
