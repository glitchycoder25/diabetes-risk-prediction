"""Loading, validation and quality reporting. Nothing here modifies the data."""
import json
import numpy as np
import pandas as pd
from src.config import FEATURES, TARGET, ZERO_IS_MISSING, PLAUSIBLE_RANGE, RESULTS_DIR


def load_data(path):
    path = str(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the PIMA Indians Diabetes CSV there "
            "(see data/README.md for the expected structure).") from e
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")
    return df[FEATURES + [TARGET]]


def validate_data(df, save=True):
    """Shape, dtypes, duplicates, NaNs, hidden zeros, implausible values, IQR outliers, class balance."""
    rep = {"shape": list(df.shape),
           "dtypes": {c: str(t) for c, t in df.dtypes.items()},
           "non_numeric_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
           "duplicate_rows": int(df.duplicated().sum()),
           "explicit_missing": {c: int(v) for c, v in df.isna().sum().items()}}
    rep["invalid_target_values"] = sorted(float(v) for v in set(df[TARGET].dropna().unique()) - {0, 1})
    rep["negative_values"] = {c: int((df[c] < 0).sum()) for c in FEATURES}
    rep["hidden_zeros_as_missing"] = {c: int((df[c] == 0).sum()) for c in ZERO_IS_MISSING}
    rep["hidden_zero_pct"] = {c: round(100 * v / len(df), 2) for c, v in rep["hidden_zeros_as_missing"].items()}
    clean = df.copy()
    clean[ZERO_IS_MISSING] = clean[ZERO_IS_MISSING].replace(0, np.nan)
    out_of_range, iqr_outliers = {}, {}
    for c in FEATURES:
        lo, hi = PLAUSIBLE_RANGE[c]
        s = clean[c].dropna()
        out_of_range[c] = int(((s < lo) | (s > hi)).sum())
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        iqr_outliers[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    rep["outside_plausible_range"] = out_of_range
    rep["iqr_outliers_reported_not_removed"] = iqr_outliers
    counts = df[TARGET].value_counts().sort_index()
    rep["class_counts"] = {int(k): int(v) for k, v in counts.items()}
    rep["class_pct"] = {int(k): round(100 * v / len(df), 2) for k, v in counts.items()}
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "data_quality_report.json", "w") as f:
            json.dump(rep, f, indent=2)
    return rep
