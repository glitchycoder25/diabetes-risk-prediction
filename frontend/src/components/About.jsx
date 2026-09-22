import React from "react";

export default function About() {
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>How this was built</h3>
      <p>Pipeline: physiologically impossible zero values (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are treated as missing then median-imputed, then scaled where needed, then classified, all inside one leakage-safe scikit-learn Pipeline.</p>
      <p>80/20 stratified train/test split, 5-fold stratified cross-validation for tuning, model selection by highest CV ROC-AUC using a one-standard-error rule. The test set is evaluated exactly once.</p>
      <p>Models compared: Logistic Regression, Decision Tree, Random Forest, plus Extra Trees, AdaBoost, k-Nearest Neighbours, SVM, Gradient Boosting, HistGradientBoosting, XGBoost, a small MLP, and a soft-voting ensemble.</p>
      <h4>Limitations</h4>
      <ul className="muted">
        <li>768 records, one population, cross-sectional labels.</li>
        <li>Insulin and skin-fold thickness have heavy missingness.</li>
        <li>Small test set means wide confidence intervals.</li>
        <li>Predicted probabilities are not clinically calibrated.</li>
        <li>Not a diagnostic device; no external validation performed.</li>
      </ul>
    </div>
  );
}
