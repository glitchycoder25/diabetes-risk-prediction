import React from "react";

export default function About() {
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>How this was built</h3>
      <p>
        Pipeline: physiologically impossible zero values (Glucose, BloodPressure, SkinThickness, Insulin,
        BMI) are treated as missing → median imputation → scaling (for scale-sensitive models) → classifier,
        all inside one leakage-safe scikit-learn <code>Pipeline</code> so every statistic is learned on
        training folds only.
      </p>
      <p>
        80/20 stratified train/test split, 5-fold stratified cross-validation for tuning, model selection by
        highest CV ROC-AUC subject to a one-standard-error rule that prefers the simplest model within one
        SD of the best score. The test set is evaluated exactly once.
      </p>
      <p>
        Models compared: Logistic Regression, Decision Tree, Random Forest (the original presentation's
        baselines), plus Extra Trees, AdaBoost, k-Nearest Neighbours, SVM, Gradient Boosting,
        HistGradientBoosting, XGBoost (if installed), a small MLP, and a soft-voting ensemble of the top
        three tuned models.
      </p>
      <h4>Limitations</h4>
      <ul className="muted">
        <li>768 records, one population (women 21+ of Pima heritage), cross-sectional labels.</li>
        <li>Insulin and skin-fold thickness have heavy missingness (~45% and ~30% of rows).</li>
        <li>Small test set (~154 rows) → wide confidence intervals; small score gaps are not meaningful.</li>
        <li>Predicted probabilities are not clinically calibrated.</li>
        <li>Not a diagnostic device; no external or clinical validation was performed.</li>
      </ul>
      <h4>Stack</h4>
      <p className="muted">
        Model training: Python, scikit-learn (+ optional XGBoost). Backend: FastAPI, serving the model
        already trained by <code>python -m src.train</code>. Frontend: React (Vite) + Recharts for the
        interactive dashboard.
      </p>
    </div>
  );
}
