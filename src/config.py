"""Central configuration so every experiment is reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20          # same 80/20 split as the original presentation
CV_FOLDS = 5

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"

# Zero is physiologically impossible for these -> it encodes "not measured".
# Pregnancies = 0 is valid, so it is NOT in this list.
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Rough plausibility limits used only for the data-quality REPORT (nothing is deleted).
PLAUSIBLE_RANGE = {
    "Pregnancies": (0, 20), "Glucose": (40, 300), "BloodPressure": (30, 150),
    "SkinThickness": (5, 100), "Insulin": (10, 900), "BMI": (12, 70),
    "DiabetesPedigreeFunction": (0.05, 2.5), "Age": (18, 100),
}

# Decision-threshold tuning on out-of-fold TRAIN predictions.
# beta=1 -> F1. Use 2.0 to favour recall (screening-style: missing a positive costs more).
THRESHOLD_BETA = 1.0
