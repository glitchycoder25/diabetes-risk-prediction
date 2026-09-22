"""Model explainability: permutation importance, partial dependence, LR odds ratios, optional SHAP."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from src.config import FEATURES, FIG_DIR, RESULTS_DIR, RANDOM_STATE


def permutation_report(model, X_test, y_test, name):
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=30,
                               random_state=RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame({"feature": FEATURES, "importance_mean": r.importances_mean,
                       "importance_std": r.importances_std}).sort_values("importance_mean")
    df.sort_values("importance_mean", ascending=False).to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df.feature, df.importance_mean, xerr=df.importance_std, color="#3E7CB1")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled (test set)")
    ax.set_title(f"Permutation importance - {name}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "explain_permutation_importance.png", dpi=150); plt.close(fig)
    return df.sort_values("importance_mean", ascending=False)


def partial_dependence_plot(model, X_train, top_features):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(model, X_train, top_features[:4], ax=ax, n_cols=2)
        fig.tight_layout(); fig.savefig(FIG_DIR / "explain_partial_dependence.png", dpi=150); plt.close(fig)
    except Exception as e:  # never let a plot kill the pipeline
        print(f"[warn] partial dependence skipped: {e}")


def logistic_odds_ratios(lr_pipeline):
    """Odds ratio per +1 SD of each feature (features are standardised inside the pipeline)."""
    coef = lr_pipeline.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": FEATURES, "coefficient": coef, "odds_ratio_per_SD": np.exp(coef)})
    df = df.sort_values("odds_ratio_per_SD", ascending=False)
    df.to_csv(RESULTS_DIR / "logistic_odds_ratios.csv", index=False)
    return df


def shap_summary(model, X_train, X_test):
    """Optional: only runs if `shap` is installed."""
    try:
        import shap
    except ImportError:
        print("[info] shap not installed -> skipping SHAP (pip install shap)")
        return
    try:
        f = lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1]
        bg = shap.sample(X_train, 100, random_state=RANDOM_STATE)
        sv = shap.Explainer(f, bg)(X_test)
        shap.summary_plot(sv.values, X_test, feature_names=FEATURES, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / "explain_shap_summary.png", dpi=150); plt.close()
    except Exception as e:
        print(f"[warn] SHAP skipped: {e}")
