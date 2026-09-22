"""Model zoo: original 3 baselines + stronger tabular models + a small MLP.

Grids are deliberately small: the dataset has ~600 training rows, so a huge search would
mostly overfit the cross-validation folds.
"""
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE as RS


@dataclass
class Spec:
    estimator: object
    scale: bool
    grid: dict
    search: str = "grid"      # "grid" or "random"
    n_iter: int = 20
    complexity: int = 1       # used by the one-standard-error selection rule
    origin: str = "extension"  # "baseline" = from the original presentation


def get_model_specs(with_catboost=False):
    specs = {
        "Logistic Regression": Spec(
            LogisticRegression(max_iter=5000, random_state=RS), True,
            {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["liblinear", "lbfgs"],
             "clf__class_weight": [None, "balanced"]},
            complexity=1, origin="baseline"),
        "Decision Tree": Spec(
            DecisionTreeClassifier(random_state=RS), False,
            {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_split": [2, 10, 20],
             "clf__min_samples_leaf": [1, 5, 10], "clf__class_weight": [None, "balanced"]},
            complexity=2, origin="baseline"),
        "Random Forest": Spec(
            RandomForestClassifier(random_state=RS, n_jobs=-1), False,
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 4, 6, 8, 12],
             "clf__min_samples_split": [2, 5, 10], "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", "log2", 0.5],
             "clf__class_weight": [None, "balanced_subsample"]},
            search="random", n_iter=25, complexity=4, origin="baseline"),
        "SVM (RBF/Linear)": Spec(
            SVC(probability=True, random_state=RS), True,
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"],
             "clf__gamma": ["scale", 0.01, 0.1], "clf__class_weight": [None, "balanced"]},
            complexity=3),
        "Gradient Boosting": Spec(
            GradientBoostingClassifier(random_state=RS), False,
            {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__max_depth": [2, 3, 4], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__min_samples_leaf": [1, 5, 10]},
            search="random", n_iter=20, complexity=5),
        "HistGradientBoosting": Spec(
            HistGradientBoostingClassifier(random_state=RS), False,
            {"clf__learning_rate": [0.02, 0.05, 0.1], "clf__max_depth": [2, 3, 4, None],
             "clf__max_leaf_nodes": [7, 15, 31], "clf__min_samples_leaf": [10, 20, 30],
             "clf__l2_regularization": [0.0, 0.1, 1.0], "clf__max_iter": [100, 200, 300]},
            search="random", n_iter=20, complexity=5),
        # --- models taken from the reference Kaggle notebook (kept, but tuned with small grids) ---
        "Extra Trees": Spec(
            ExtraTreesClassifier(random_state=RS, n_jobs=-1), False,
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 4, 6, 8, 12],
             "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2", 0.5]},
            search="random", n_iter=20, complexity=4, origin="notebook"),
        "AdaBoost": Spec(
            AdaBoostClassifier(random_state=RS), False,
            {"clf__n_estimators": [50, 100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]},
            complexity=5, origin="notebook"),
        "k-Nearest Neighbours": Spec(
            KNeighborsClassifier(), True,
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21], "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},
            complexity=3, origin="notebook"),
        "MLP (small NN)": Spec(
            MLPClassifier(early_stopping=True, max_iter=1000, random_state=RS), True,
            {"clf__hidden_layer_sizes": [(8,), (16,), (16, 8)],
             "clf__alpha": [1e-4, 1e-3, 1e-2], "clf__learning_rate_init": [1e-3, 1e-2]},
            complexity=4),
    }
    try:
        from xgboost import XGBClassifier
        specs["XGBoost"] = Spec(
            XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1, verbosity=0), False,
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.01, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0], "clf__reg_lambda": [1, 5, 10]},
            search="random", n_iter=20, complexity=5)
    except ImportError:
        print("[info] xgboost not installed -> skipping XGBoost (pip install xgboost)")
    if with_catboost:
        try:
            from catboost import CatBoostClassifier
            specs["CatBoost"] = Spec(
                CatBoostClassifier(verbose=0, random_seed=RS, thread_count=2), False,
                {"clf__iterations": [200, 400], "clf__depth": [3, 4, 6],
                 "clf__learning_rate": [0.03, 0.05, 0.1], "clf__l2_leaf_reg": [3, 5, 10]},
                search="random", n_iter=10, complexity=5)
        except ImportError:
            print("[info] catboost not installed -> skipping CatBoost")
    return specs
