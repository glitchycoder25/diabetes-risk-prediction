"""Re-runs the preprocessing protocol of the reference Kaggle notebook and compares it with the
leakage-free pipeline used in this project, on the SAME stratified 80/20 split (random_state=42).

Why: the reference notebook (a) fills missing values with the median *of each Outcome class*,
computed on the WHOLE dataset before splitting, so the target label of every test row leaks into its
features, and (b) fits the StandardScaler on the whole dataset. Its final Gradient Boosting model also
uses max_depth=20 and reaches 100% train accuracy. High notebook scores are therefore optimistic.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.preprocessing import build_pipeline


def _notebook_preprocess(df):
    d = df.copy()
    d[C.ZERO_IS_MISSING] = d[C.ZERO_IS_MISSING].replace(0, np.nan)
    for col in C.FEATURES:                                    # LEAK: uses Outcome, uses all rows
        med = d.groupby(C.TARGET)[col].median()
        for o in (0, 1):
            d.loc[(d[C.TARGET] == o) & d[col].isnull(), col] = med[o]
    X = pd.DataFrame(StandardScaler().fit_transform(d[C.FEATURES]), columns=C.FEATURES)  # LEAK: global scaler
    return X, d[C.TARGET]


def _models():
    return {"Random Forest (n=100)": lambda: RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_STATE),
            "Gradient Boosting (notebook 'best')": lambda: GradientBoostingClassifier(
                subsample=0.6, n_estimators=500, max_depth=20, learning_rate=0.01, random_state=C.RANDOM_STATE)}


def _score(m, Xtr, ytr, Xte, yte, protocol, name):
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return {"protocol": protocol, "model": name, "train_accuracy": accuracy_score(ytr, m.predict(Xtr)),
            "test_accuracy": accuracy_score(yte, m.predict(Xte)), "test_f1": f1_score(yte, m.predict(Xte)),
            "test_roc_auc": roc_auc_score(yte, p)}


def run_leakage_demo(df):
    rows = []
    Xn, yn = _notebook_preprocess(df)
    Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=C.TEST_SIZE, stratify=yn, random_state=C.RANDOM_STATE)
    for name, mk in _models().items():
        rows.append(_score(mk(), Xtr, ytr, Xte, yte, "reference notebook (leaky)", name))
    X, y = df[C.FEATURES], df[C.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    for name, mk in _models().items():
        rows.append(_score(build_pipeline(mk(), scale=False), Xtr, ytr, Xte, yte, "this project (leak-free pipeline)", name))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "leakage_demonstration.csv", index=False)
    return out
