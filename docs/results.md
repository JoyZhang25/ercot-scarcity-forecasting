# Locked-test results

The model family was frozen using 2024. Gradient boosting won validation PR-AUC
(0.084 versus a 0.017 event rate), so it became the primary model before the 2025
test was examined.

On 8,759 test hours in 2025, 235 exceeded $100/MWh (2.68%). The selected model's
raw PR-AUC was 0.095 and ROC-AUC was 0.816. The highest predicted-risk decile
contained a 10.62% realized spike rate—4.0 times the unconditional rate. These
numbers support a tail-risk ranking claim, not a point forecast or certain alarm.

The selected model also ranks first descriptively on the locked test; that fact does
not alter the validation-governed selection rule.

The economic diagnostic is deliberately less flattering. The selected model's
top-risk decile had a mean RT−DA spread of −$6.59/MWh. The top-decile-minus-market
spread estimate was −$4.14/MWh with a 95% daily block-bootstrap interval of
[−$7.67, −$0.42]. Operational tail predictability therefore did **not** become a
virtual-load alpha claim; the prespecified direction was wrong in this test year.

That distinction is the main research lesson: a model can identify stressful hours
while the day-ahead auction prices the same risk more aggressively than the average
real-time settlement. Reversing the trade after seeing this result would be post hoc.

## Prospective alpha audit

A separate model predicts `RT−DA` directly and trades at most one virtual-supply
hour per day under a rule frozen before acquiring 2026 outcomes.  After a $2/MWh
hurdle, the 2026 lockbox earns +$13.76/MWh across 111 positions, but the 95% daily
block-bootstrap interval is [−$21.15, +$53.00], the HAC t-statistic is 1.11, and
the five best days account for 62.4% of positive P&L.  Removing those five days
changes the remaining mean to −$7.01/MWh.

The selector beats an equal-turnover random date/hour null (one-sided p=0.024),
so the point estimate is not devoid of information.  It does not beat a random
hour conditional on the same selected dates at conventional levels (p=0.124), and
it fails the preregistered uncertainty gate.  The result is an alpha candidate,
not evidence of a stable deployable alpha.
