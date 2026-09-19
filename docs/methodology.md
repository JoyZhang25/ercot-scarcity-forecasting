# Methodology

## Estimand

The project estimates the incremental out-of-sample information in dated financial headlines for predicting a future S&P 500 close-to-close return. It does **not** estimate a causal effect of news and it does not claim a deployable trading edge.

Let \(\mathcal F_t\) contain headlines dated on or before trading day \(t\) and market prices through the close of \(t\). Because the dataset does not contain reliable intraday publication times, the default signal is executed with a one-session delay. The binary target is

\[
Y_t = \mathbf 1\!\left\{\log(P_{t+2}/P_{t+1}) > 0\right\}.
\]

The delay ensures that all day-\(t\) headlines precede the assumed close-\(t+1\) execution time.

## Fixed research protocol

- Training outcomes end on 2019-12-31.
- Validation outcomes end on 2021-12-31.
- Test features begin after 2021-12-31.
- Vocabulary construction, scaling, and regularization selection use no test observations.
- The test set is evaluated once after choosing regularization on the validation set.

## Ablations

1. **Prior:** constant probability equal to the train-plus-validation up frequency.
2. **Market:** logistic regression on lagged return, five-day momentum, twenty-day realized volatility, and headline intensity.
3. **Text:** TF-IDF unigrams/bigrams with logistic regression.
4. **Combined:** the market and text feature blocks fit jointly.

The linear models are intentional. A complex neural network is not persuasive unless it first beats correctly specified, regularized baselines on a locked test period.

## Evaluation

Primary forecast metrics are log loss, Brier score, ROC-AUC, balanced accuracy, and macro-F1. The project reports a moving-block bootstrap confidence interval for the market-baseline log-loss improvement and a Newey-West/HAC test of the paired loss differential.

An illustrative long/short strategy is included as a diagnostic. It applies a configurable transaction cost to changes in position. It is not a backtest claim because the dataset lacks venue-specific execution prices, intraday timestamps, and a full transaction-cost model.

## Why the original HMM labels were removed

The course project fit a Gaussian HMM and used smoothed posterior states as labels. That design mixes an unsupervised state model with the supervised prediction target, makes the economic meaning of the label unclear, and can leak future observations when smoothing or full-sample fitting is used. The reconstruction predicts an observable future return directly. A filtered, expanding-window regime model can be added later as a feature or conditional evaluation, but never as a retrospectively smoothed target.
