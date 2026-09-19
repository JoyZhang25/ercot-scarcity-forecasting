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
