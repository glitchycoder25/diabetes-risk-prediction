"""Streamlit demo.  Run from the project root:  streamlit run app/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURES, MODEL_DIR, FIG_DIR, RESULTS_DIR

st.set_page_config(page_title="Diabetes Risk Prediction (ML prototype)", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Prediction - ML Prototype")
st.warning(
    "**Educational prototype - not a medical device.** It classifies whether a record *resembles* the diabetic class of the "
    "PIMA Indians dataset (a small, cross-sectional dataset of adult women of Pima heritage). It does **not** diagnose "
    "diabetes and does **not** predict future onset. Consult a healthcare professional for any medical decision.")

model_path = MODEL_DIR / "final_model.joblib"
if not model_path.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()
bundle = joblib.load(model_path)
model, thr = bundle["model"], bundle["threshold"]

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance & explainability", "About"])

with tab_pred:
    st.caption(f"Model: **{bundle['name']}** - decision threshold: {thr:.2f}")
    c1, c2 = st.columns(2)
    v = {}
    with c1:
        v["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 2)
        v["Glucose"] = st.number_input("Plasma glucose (mg/dL)", 40, 300, 110)
        v["BloodPressure"] = st.number_input("Diastolic blood pressure (mm Hg)", 30, 150, 70)
        v["BMI"] = st.number_input("BMI (kg/m²)", 12.0, 70.0, 28.0, step=0.1)
    with c2:
        v["Age"] = st.number_input("Age (years)", 18, 100, 30)
        v["DiabetesPedigreeFunction"] = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.4, step=0.01)
        skin_na = st.checkbox("Skin-fold thickness not available")
        v["SkinThickness"] = 0 if skin_na else st.number_input("Triceps skin-fold thickness (mm)", 5, 100, 25)
        ins_na = st.checkbox("Insulin not available")
        v["Insulin"] = 0 if ins_na else st.number_input("2-h serum insulin (µU/mL)", 10, 900, 80)
    st.caption("Unavailable values are sent as 0 and imputed by the pipeline exactly as during training.")

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([[v[f] for f in FEATURES]], columns=FEATURES)
        p = float(model.predict_proba(row)[0, 1])
        label = "Resembles the **diabetic** class" if p >= thr else "Resembles the **non-diabetic** class"
        st.subheader(label)
        st.progress(min(max(p, 0.0), 1.0), text=f"Model-estimated probability: {p:.1%}")
        st.info("This probability comes from a small research dataset and is not calibrated for clinical use.")

with tab_perf:
    cmp_path = RESULTS_DIR / "model_comparison.csv"
    if cmp_path.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(cmp_path).drop(columns=["best_params"]), use_container_width=True)
    for f, cap in [("model_comparison_cv_auc.png", "Cross-validated AUC"), ("roc_pr_curves.png", "ROC / PR curves (test)"),
                   ("confusion_matrix_final.png", "Confusion matrix"), ("explain_permutation_importance.png", "Permutation importance"),
                   ("explain_partial_dependence.png", "Partial dependence"), ("explain_shap_summary.png", "SHAP summary")]:
        if (FIG_DIR / f).exists():
            st.image(str(FIG_DIR / f), caption=cap)

with tab_about:
    st.markdown("""
**Pipeline:** zero→missing for impossible values → median imputation → scaling (where needed) → classifier, all inside one
leakage-safe scikit-learn `Pipeline`. 80/20 stratified split, 5-fold stratified CV for tuning and selection, test set used once.

**Limitations:** 768 records, one population (women ≥21 of Pima heritage), cross-sectional labels, heavy missingness in
insulin/skin-fold, no external validation. Results will not generalise to other populations without re-validation.
""")
