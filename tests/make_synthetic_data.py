"""SYNTHETIC data used ONLY by automated tests to check the code runs. Never used for real results."""
import numpy as np
import pandas as pd
from src.config import FEATURES, TARGET


def make_synthetic(n=768, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Pregnancies": rng.poisson(3.8, n), "Glucose": rng.normal(120, 30, n).clip(50, 200),
        "BloodPressure": rng.normal(70, 12, n).clip(30, 120), "SkinThickness": rng.normal(25, 10, n).clip(5, 60),
        "Insulin": rng.gamma(2, 50, n).clip(15, 600), "BMI": rng.normal(32, 6.5, n).clip(18, 60),
        "DiabetesPedigreeFunction": rng.gamma(2, 0.25, n).clip(0.08, 2.4), "Age": rng.gamma(6, 5.5, n).clip(21, 80).round()})
    z = 0.035 * (df.Glucose - 120) + 0.06 * (df.BMI - 32) + 0.02 * (df.Age - 33) + 0.8 * df.DiabetesPedigreeFunction - 0.9
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    for c, frac in [("Glucose", .01), ("BloodPressure", .05), ("SkinThickness", .3), ("Insulin", .45), ("BMI", .01)]:
        df.loc[rng.random(n) < frac, c] = 0
    return df[FEATURES + [TARGET]]


if __name__ == "__main__":
    import sys
    make_synthetic().to_csv(sys.argv[1], index=False)
