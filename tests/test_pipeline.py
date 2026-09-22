"""Run with:  python -m pytest -q   (or python -m tests.test_pipeline)"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.config import FEATURES, TARGET, ZERO_IS_MISSING
from src.preprocessing import ZeroToNaN, build_pipeline
from tests.make_synthetic_data import make_synthetic


def test_zero_to_nan_only_touches_invalid_columns():
    df = make_synthetic()
    df.loc[0, ["Pregnancies", "Glucose"]] = 0
    out = ZeroToNaN().fit_transform(df[FEATURES].values)
    assert np.isnan(out[0, FEATURES.index("Glucose")])
    assert out[0, FEATURES.index("Pregnancies")] == 0          # zero pregnancies is valid


def test_no_leakage_imputer_uses_train_medians_only():
    df = make_synthetic()
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True).fit(Xtr, ytr)
    learned = pipe.named_steps["imputer"].statistics_
    train_med = Xtr.replace(0, np.nan).median()[FEATURES].values
    full_med = X.replace(0, np.nan).median()[FEATURES].values
    assert np.allclose(learned, train_med)            # fitted on the training rows only
    assert not np.allclose(train_med, full_med)       # ...and genuinely different from full-data medians
    assert pipe.predict_proba(Xte).shape == (len(Xte), 2)


if __name__ == "__main__":
    test_zero_to_nan_only_touches_invalid_columns(); test_no_leakage_imputer_uses_train_medians_only(); print("ok")
