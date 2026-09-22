# Methodology and Design Decisions

## 1. Problem framing
Binary classification: does a record resemble the diabetic (1) or non-diabetic (0) class in the PIMA dataset?
The data are cross-sectional (features and label recorded together), from one population (women ≥21 of Pima heritage),
so the model is a *risk-classification prototype*. It cannot diagnose or forecast onset, and it should not be
assumed to generalise to other populations.

## 2. Data quality decisions
| Issue | Decision | Reason |
|---|---|---|
| Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI | Treated as missing (NaN) | Physiologically impossible; they encode "not measured". Pregnancies = 0 is valid and untouched. |
| Missing values | Median imputation inside the Pipeline | Robust to skew; learned on training folds only. |
| Outliers | Reported (IQR), **not removed** | Extreme insulin/BMI/age values can be genuine and clinically informative; deleting them shrinks an already small dataset. Tree ensembles are insensitive to them; SVM/LR use scaling. |
| Duplicates | Counted and reported | Duplicates across the split would leak information. |
| Scaling | StandardScaler for LR, SVM, MLP only | Distance/gradient-based models need it; trees do not. |
| Class imbalance (~65/35) | Stratified splits; `class_weight` tuned; threshold tuned; PR-AUC reported | Mild imbalance: accuracy alone is misleading. |

An ablation (`results/preprocessing_ablation.csv`) checks with CV whether the zero→NaN→impute step helps.

## 3. Leakage prevention
Only stateless steps (zero→NaN) and steps fitted inside `sklearn.pipeline.Pipeline` are used. During
`GridSearchCV`/`cross_val_score` the imputer medians and scaler statistics are recomputed on each training fold.
`tests/test_pipeline.py` asserts that imputer statistics equal the training-set medians and differ from full-data medians.
The decision threshold is chosen from out-of-fold *training* predictions, never from the test set.

## 4. Training protocol
1. Stratified 80/20 split (`random_state=42`), matching the original presentation.
2. Stratified 5-fold CV on the 80% part: tune hyper-parameters (refit on ROC-AUC; accuracy, recall, F1 also recorded).
3. Model choice uses **CV only**: pick the best CV AUC, then apply the one-standard-error rule, choosing the simplest model whose CV AUC is within one SD of the best (LR < DT < SVM < RF/MLP < boosting < ensemble).
4. Single final evaluation on the held-out 20%, with bootstrap 95% CIs.

## 5. Models
Baselines from the presentation: Logistic Regression, Decision Tree (depth/leaf regularised), Random Forest (tuned).
Extensions: SVM, Gradient Boosting, HistGradientBoosting, XGBoost (if installed), CatBoost (opt-in: it adds a heavy
dependency and long training time for typically marginal gains on 768 rows), small MLP (≤ 2 hidden layers, early stopping),
soft-voting ensemble of the three best tuned models. Grids are intentionally small to limit over-fitting the CV folds.

## 6. Evaluation
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, ROC/PR curves. For a screening-style
tool, recall (sensitivity) matters; set `THRESHOLD_BETA = 2` in `src/config.py` to weight recall more heavily.

## 7. Explainability
Permutation importance (test AUC drop), partial dependence for the top features, standardised logistic-regression
odds ratios, optional SHAP summary. Expect glucose, BMI, age and pedigree function to dominate. Verify this in *your*
results rather than assuming it. Importance shows what the model uses, not causal risk factors.

## 8. Limitations (include in your report)
* 768 samples, one population, cross-sectional labels, no external validation.
* Insulin (~49%) and skin thickness (~30%) have heavy missingness, so imputed values carry little information.
* Small test set (154 rows) gives wide CIs; model rankings within ~2 points are not statistically meaningful.
* Probabilities are not clinically calibrated.
* Not a diagnostic device; no regulatory validation.

## 9. Likely viva questions
1. Why did you treat zeros as missing, and why not drop those rows? (~50% of rows would be lost.)
2. How did you prevent data leakage? (Pipeline + per-fold fitting.)
3. Why ROC-AUC for tuning rather than accuracy? (Threshold-independent, robust to imbalance.)
4. Why might Logistic Regression match boosting here? (Small n, roughly additive signal.)
5. Why is the test set used only once? (Otherwise the estimate is optimistically biased.)
6. What does permutation importance not tell you? (Causality; importance is shared among correlated features.)
7. Why can't this be used clinically? (Population, sample size, no external validation, cross-sectional design.)

## 10. Comparison with the reference Kaggle notebook
See README ("Reference notebook and the leakage finding"). `src/leakage_demo.py` reproduces the notebook's preprocessing
(outcome-conditional median imputation and global scaling before the split) and its two headline models on the identical
split, next to the leak-free pipeline. The gap between the two rows is the optimism caused by leakage, not a real
performance difference. Also note that CV for tuning here is 5-fold on the training set only, and the final model is chosen
without looking at the test set.
