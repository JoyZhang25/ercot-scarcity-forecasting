# Reconstruction audit

## Purpose

This document records why the independent reconstruction does not simply copy the 2025 course pipeline. The point is not to retroactively criticize exploratory classroom work. It is to separate useful ideas from design choices that cannot support a forecasting claim under a professional evidence standard.

The original project explored whether financial news and tweets could predict S&P 500 direction using sentence embeddings, principal components, Gaussian hidden Markov models, clustering, logistic regression, random forests, SMOTE, and LSTMs. The breadth made sense for a course project. A portfolio study, however, needs one estimand, one information clock, protected model selection, and a locked test set.

## Findings from the original artifacts

### 1. The prediction target was not stable across the report

Different sections referred to close-to-close and open-to-open returns, and the precise time at which a forecast became tradable was not defined. Without a publication-time cutoff, a headline dated on day \(t\) cannot safely be assumed available before the day-\(t\) close.

**Reconstruction:** all headlines dated \(t\) are treated as complete only after that close. The signal is entered at the close of \(t+1\), and the target is the close-\(t+1\) to close-\(t+2\) log return.

### 2. Retrospective HMM states were used as supervised labels

The stock-label notebook standardizes the full price sample, fits a Gaussian HMM to that full sample, and obtains state probabilities from rolling windows. Those states are useful for retrospective regime description, but they are not an observable economic target. Full-sample scaling, fitting, or smoothing also allows future observations to influence earlier state assignments. The report described this construction as supervised even though HMM state inference is unsupervised.

**Reconstruction:** the target is an observable future return sign. No smoothed latent state is used as ground truth. A regime model would be allowed only as an expanding-window, filtered feature or a pre-specified subgroup diagnostic.

### 3. Several evaluations used random row-level splits

The tweet-classification notebook uses stratified random train/test splitting. With repeated observations from the same date and security, random row splitting can place closely related observations on both sides of the boundary and does not emulate deployment into the future.

**Reconstruction:** every split is chronological and defined using outcome dates. Training outcomes end in 2019, regularization is selected on 2020–2021, and test features begin only after 2021. Boundary observations whose outcomes cross a split are purged.

### 4. The random-forest stack mixed in-sample and out-of-sample features

The first-stage random forest predicts next-market movement for individual tweets. Its predictions are then described as sentiment, aggregated by day, and passed to a second random forest. Training-date aggregate features come from first-stage in-sample predictions, while test-date features come from first-stage out-of-sample predictions. This changes the feature-generating process across the split. Correct stacking would require out-of-fold predictions for every second-stage training observation.

There is also a semantic problem: a classifier trained on market direction estimates predicted market direction, not linguistic sentiment.

**Reconstruction:** the benchmark uses one daily row, makes the text and market information sets explicit, and fits them jointly in a single regularized pipeline. The study calls model outputs return probabilities, not sentiment scores.

### 5. Model breadth substituted for a benchmark hierarchy

The course project compared many algorithms but did not establish whether text improved over a historical-prior forecast or a market-only model under the same target clock. Accuracy screenshots did not quantify paired uncertainty, and reported performance could not be tied to one immutable experiment specification.

**Reconstruction:** four nested models answer a single ablation question: prior, market-only, text-only, and market-plus-text. Log loss is the primary probability score. Moving-block bootstrap intervals and HAC tests operate on paired daily loss differences. Configuration, input hashes, package versions, predictions, and tables are written to the run artifacts.

## What was retained

The reconstruction preserves the original economic question and the useful intuition that text must be evaluated jointly with market-history controls. It also retains the focus on dimensionality control, but implements it through train-only TF–IDF and regularized linear models rather than full-sample PCA or an unnecessarily flexible neural network.

## What the locked test changed

The more disciplined protocol does not reproduce the optimistic classroom narrative. On 542 test dates from January 2022 through March 2024, neither text alone nor text added to market features improves log loss in a statistically resolved way. The combined specification is significantly worse than the historical prior and indistinguishable from the market-only model.

This is the intended behavior of a locked test: it is allowed to invalidate the hypothesis. The portfolio contribution is the redesign, the leakage controls, the reproducible evidence, and the calibrated conclusion—not a manufactured claim of market predictability.

## Attribution boundary

The research question originated in a Georgia Tech group project by Aryan Gupta, Mengwei Sun, Eddy Wang, Huopu Zhang, and Jingyi Zhang. The original report credited Jingyi Zhang with preprocessing methods and machine-learning algorithms/models. This repository is an independent reimplementation and methodological audit; it does not claim sole authorship of the group submission.
