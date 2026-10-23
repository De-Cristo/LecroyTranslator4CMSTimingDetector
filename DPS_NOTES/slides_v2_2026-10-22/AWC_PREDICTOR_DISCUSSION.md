# Internal discussion: what justifies the second AWC candidate?

This is an exploratory model, not an established CMS result or a
first-principles determination of AWC. At the author's request on 24 October,
this candidate is restored as main-body slide 8, following the empirical trend
on slide 7. Both are explicitly exploratory approaches, not established
reference-free calibrations.

## Exact algebra versus the additional hypothesis

Start with the walk model and the measured mean-slope parametrization:

\[
W(Q)=\frac{\mathrm{AWC}}{\overline S(Q)},\qquad \overline S(Q)=c_0+c_1Q.
\]

Holding a channel's AWC fixed, differentiation gives exactly

\[
-\frac{dW}{d\ln Q}=\frac{\mathrm{AWC}\,c_1Q}{\overline S(Q)^2}.
\]

The left-hand side is the change in walk for a small fractional change in
charge, with a minus sign. The new, unproven hypothesis is that this response
is approximately common across channels at a prescribed typical charge:

\[
-\left.\frac{dW}{d\ln Q}\right|_{Q_*}\simeq K
\quad\Longrightarrow\quad
\mathrm{AWC}_{\rm pred}=K\frac{\overline S(Q_*)^2}{c_1Q_*}.
\]

Here K has units of ns. Q* is the median of each channel's selected charge
sample, computed once, not the event charge. AWC remains a channel constant.

## Required assumptions on slide 8

1. The local inverse-slope approximation describes the walk, with AWC constant per SiPM and configuration.
2. The linear mean-slope fit is reliable near the selected-charge median Q*.
3. These medians sample comparable pulse regions across the SiPMs.
4. Similar pulse shapes and operating conditions give a common fractional-charge timing response K that transfers between SiPMs and bars.

The first two support the derivative expression; the last two are additional
physical hypotheses needed to use one shared K for prediction. They are not
established by the algebra or by nominally identical detector construction.

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
yet a validated performance claim. Both candidate slides are now restored
before the comparison results, with their assumptions and limitations retained.

## Provenance

- Formula and fit implementation: `Q_Walk/results/2026-09-24_measured_q_overlap/try_physical_candidate.py`.
- Displayed saved test: `Q_Walk/results/2026-10-22_clock_quality/physical_candidate_quick_test/`.
- No new fit or event processing was performed for this reorganization.
