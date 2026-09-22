# Diabetes Risk Prediction Using Machine Learning: Comparative Analysis and Explainable Predictive Modeling

B.Tech (CSE) minor project. Extends the presentation *"Predicting Diabetes Using Machine Learning"* (PIMA Indians
Diabetes dataset, 768 records, Logistic Regression / Decision Tree / Random Forest, 80/20 split).

> **Scope disclaimer.** This is a machine-learning **risk-classification prototype** trained on a small, cross-sectional
> dataset. It is **not** a clinical diagnostic system and does **not** predict future diabetes onset.

## What is new compared with the presentation

| Area | Original presentation | This project |
|---|---|---|
| Data quality | basic cleaning | validation report; impossible zeros → missing; outliers **reported, not deleted** |
| Leakage | not addressed | imputer + scaler inside `Pipeline`, refitted per CV fold (unit-tested) |
| Split | 80/20 | 80/20 **stratified**; test set used once |
| Model selection | single split | 5-fold stratified CV + Grid/RandomizedSearchCV |
| Models | LR, DT, RF | + Extra Trees, AdaBoost, kNN (from the reference notebook), SVM, Gradient Boosting, HistGradientBoosting, XGBoost (opt.), CatBoost (opt.), small MLP, soft-voting ensemble |
| Metrics | Acc / Prec / Rec / F1 | + ROC-AUC, PR-AUC, Brier score, confusion matrix, bootstrap 95% CIs |
| Threshold | fixed 0.5 | tuned on out-of-fold **train** predictions |
| Explainability | none | permutation importance, partial dependence, LR odds ratios, SHAP (opt.) |
| Selection rule | highest test accuracy | highest CV AUC with **one-standard-error rule** favouring simpler models |
| Deliverable | slides | reproducible code + Streamlit demo + methodology document |

## Reference notebook (Kaggle) and the leakage finding

The Kaggle notebook `diabetes-prediction-pima-indian-dataset.ipynb` was used as a reference. Its ideas (Extra Trees, AdaBoost,
kNN, RandomizedSearchCV, ROC analysis) are integrated, but its protocol is **not** copied, because:

1. Missing values are filled with the median *per Outcome class* on the **whole** dataset before the split, so each test row's
   label leaks into its features.
2. The scaler is fitted on the whole dataset before the split.
3. The final Gradient Boosting model (max_depth=20) has 100% train accuracy, i.e. it is overfitted.
4. Its first ROC curves are drawn from hard 0/1 predictions instead of probabilities.

`python -m src.train` automatically re-runs the notebook's protocol and this project's leak-free pipeline on the same split
(`results/leakage_demonstration.csv`). Expect the leak-free scores to be noticeably lower; that is the honest estimate, and the
comparison itself is a strong result for your report.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the PIMA CSV at `data/diabetes.csv` (structure in `data/README.md`).

## Run

```bash
python -m src.train                 # full pipeline (a few minutes)
python -m src.train --fast          # quick smoke test
python -m src.train --with-catboost # include CatBoost
streamlit run app/app.py            # demo UI
python -m tests.test_pipeline       # leakage / preprocessing tests
```

Outputs: `results/model_comparison.csv`, `results/final_metrics.json`, `results/data_quality_report.json`,
`results/preprocessing_ablation.csv`, `results/figures/*.png`, `models/final_model.joblib`.

## Project layout

```
data/           put diabetes.csv here
src/            config, data_utils, preprocessing, models, eda, explain, train
app/app.py      Streamlit demo with disclaimers
tests/          pipeline tests (+ synthetic generator used ONLY for code tests)
docs/           METHODOLOGY.md (design decisions, limitations, viva questions)
```

## Notes for your report
* All numbers in your report must come from **your own run** of `src.train` on the real CSV; none are pre-filled here.
* Report CV results and test results separately, and quote the bootstrap confidence intervals: the test set has only 154 rows,
  so differences of 1-3 accuracy points between models are within noise.
* If the MLP or XGBoost does not beat Logistic Regression, say so; that is a legitimate finding on small tabular data.
