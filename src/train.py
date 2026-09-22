"""End-to-end training + evaluation.

Usage:
    python -m src.train                      # full run on data/diabetes.csv
    python -m src.train --fast               # tiny searches, for a quick smoke test
    python -m src.train --with-catboost      # also evaluate CatBoost
    python -m src.train --data path/to.csv

The 20% test set is touched ONLY for final reporting. Model selection uses CV on the 80% train part.
"""
import argparse
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             ConfusionMatrixDisplay, f1_score, fbeta_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import (GridSearchCV, ParameterGrid, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)

from src import config as C
from src.data_utils import load_data, validate_data
from src.eda import run_eda
from src.leakage_demo import run_leakage_demo
from src.explain import (logistic_odds_ratios, partial_dependence_plot,
                         permutation_report, shap_summary)
from src.models import get_model_specs
from src.preprocessing import build_pipeline

warnings.filterwarnings("ignore")
SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "recall": "recall", "f1": "f1"}


def test_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba), "brier": brier_score_loss(y, proba)}


def make_search(spec, pipe, cv, fast):
    grid = spec.grid
    size = len(ParameterGrid(grid))
    n_iter = min(4, size) if fast else spec.n_iter
    kw = dict(scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)
    if (spec.search == "grid" and not fast) or n_iter >= size:
        return GridSearchCV(pipe, grid, **kw)
    return RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=C.RANDOM_STATE, **kw)


def preprocessing_ablation(X_train, y_train, specs, cv):
    """Does zero->NaN + median imputation actually help? (train-set CV only)"""
    rows = []
    for name in ["Logistic Regression", "Random Forest"]:
        spec = specs[name]
        for label, tz in [("raw zeros kept", False), ("zeros -> NaN -> median impute", True)]:
            est = clone(spec.estimator)
            s = cross_val_score(build_pipeline(est, spec.scale, treat_zeros=tz), X_train, y_train,
                                scoring="roc_auc", cv=cv)
            rows.append({"model": name, "preprocessing": label, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "preprocessing_ablation.csv", index=False)
    return df


def bootstrap_ci(y, proba, thr, n=1000):
    rng = np.random.default_rng(C.RANDOM_STATE)
    y = np.asarray(y); out = {k: [] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = test_metrics(y[idx], proba[idx], thr)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def plot_curves(models_proba, y_test, final_name, final_pred, final_proba):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for n, p in models_proba.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, label=n)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall curves (test set)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "roc_pr_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, display_labels=["Non-diabetic", "Diabetic"], ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {final_name}")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "confusion_matrix_final.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_PATH))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--with-catboost", action="store_true")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-leakage-demo", action="store_true")
    a = ap.parse_args()
    for d in (C.RESULTS_DIR, C.FIG_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1. data ----------
    df = load_data(a.data)
    rep = validate_data(df)
    print(f"Loaded {rep['shape']}, duplicates={rep['duplicate_rows']}, class counts={rep['class_counts']}")
    print("Hidden zeros:", rep["hidden_zeros_as_missing"])
    if not a.skip_eda:
        run_eda(df)
    if not a.skip_leakage_demo:
        print("\nReference-notebook protocol vs leak-free pipeline (same split, same models):")
        print(run_leakage_demo(df).round(4).to_string(index=False))
    X, y = df[C.FEATURES], df[C.TARGET]

    # ---------- 2. split (same 80/20 as the presentation, but stratified) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    print(f"Train={len(X_train)}  Test={len(X_test)} (test set untouched until final evaluation)")

    specs = get_model_specs(a.with_catboost)
    ab = preprocessing_ablation(X_train, y_train, specs, cv)
    print("\nPreprocessing ablation (CV AUC on train):\n", ab.round(4).to_string(index=False))

    # ---------- 3. tuning by CV on the training set only ----------
    tuned, rows = {}, []
    for name, spec in specs.items():
        t0 = time.time()
        pipe = build_pipeline(spec.estimator, spec.scale)
        base_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv).mean()
        search = make_search(spec, pipe, cv, a.fast).fit(X_train, y_train)
        i = search.best_index_; r = search.cv_results_
        tuned[name] = search.best_estimator_
        rows.append({"model": name, "origin": spec.origin, "cv_auc_untuned": base_auc,
                     "cv_auc_mean": r["mean_test_roc_auc"][i], "cv_auc_std": r["std_test_roc_auc"][i],
                     "cv_accuracy": r["mean_test_accuracy"][i], "cv_recall": r["mean_test_recall"][i],
                     "cv_f1": r["mean_test_f1"][i], "best_params": json.dumps(
                         {k.replace("clf__", ""): (v if isinstance(v, (int, float, str, type(None))) else str(v))
                          for k, v in search.best_params_.items()}),
                     "complexity": spec.complexity})
        print(f"[{name:22s}] CV AUC {base_auc:.4f} -> {r['mean_test_roc_auc'][i]:.4f} ({time.time()-t0:.0f}s)")

    # soft-voting ensemble of the 3 best tuned models (by CV AUC)
    top3 = sorted(rows, key=lambda d: -d["cv_auc_mean"])[:3]
    ens = VotingClassifier([(d["model"], tuned[d["model"]]) for d in top3], voting="soft")
    s = cross_val_score(ens, X_train, y_train, scoring="roc_auc", cv=cv)
    ens.fit(X_train, y_train)
    tuned["Soft-Voting Ensemble"] = ens
    rows.append({"model": "Soft-Voting Ensemble", "origin": "extension", "cv_auc_untuned": np.nan,
                 "cv_auc_mean": s.mean(), "cv_auc_std": s.std(), "cv_accuracy": np.nan, "cv_recall": np.nan,
                 "cv_f1": np.nan, "best_params": "top-3: " + ", ".join(d["model"] for d in top3), "complexity": 6})

    # ---------- 4. model selection: 1-SE rule on CV AUC (NO test data) ----------
    res = pd.DataFrame(rows)
    best = res.loc[res.cv_auc_mean.idxmax()]
    eligible = res[res.cv_auc_mean >= best.cv_auc_mean - best.cv_auc_std]
    final_name = eligible.sort_values(["complexity", "cv_auc_mean"], ascending=[True, False]).iloc[0]["model"]
    print(f"\nBest CV AUC: {best.model} ({best.cv_auc_mean:.4f}); "
          f"selected by 1-SE rule (simplest within one SD): {final_name}")

    # ---------- 5. final evaluation on the untouched test set ----------
    probas, trows = {}, []
    for name, m in tuned.items():
        p = m.predict_proba(X_test)[:, 1]; probas[name] = p
        trows.append({"model": name, **test_metrics(y_test, p, 0.5)})
    res = res.merge(pd.DataFrame(trows), on="model").sort_values("cv_auc_mean", ascending=False)
    res.insert(0, "selected", res.model == final_name)
    res.to_csv(C.RESULTS_DIR / "model_comparison.csv", index=False)
    show = ["selected", "model", "origin", "cv_auc_mean", "cv_auc_std", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\n", res[show].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    o = res.sort_values("cv_auc_mean")
    ax.barh(o.model, o.cv_auc_mean, xerr=o.cv_auc_std, color=["#D1495B" if s else "#3E7CB1" for s in o.selected])
    ax.set_xlim(max(0.5, o.cv_auc_mean.min() - 0.05), 1); ax.set_xlabel("5-fold CV ROC-AUC (train set)")
    ax.set_title("Model comparison (red = selected)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "model_comparison_cv_auc.png", dpi=150); plt.close(fig)

    # ---------- 6. decision threshold from OUT-OF-FOLD TRAIN predictions ----------
    final = tuned[final_name]
    oof = cross_val_predict(clone(final), X_train, y_train, cv=StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=7),
                            method="predict_proba")[:, 1]
    grid = np.arange(0.10, 0.91, 0.01)
    scores = [fbeta_score(y_train, oof >= t, beta=C.THRESHOLD_BETA) for t in grid]
    thr = float(grid[int(np.argmax(scores))])
    p_final = probas[final_name]
    m05, mthr = test_metrics(y_test, p_final, 0.5), test_metrics(y_test, p_final, thr)
    ci = bootstrap_ci(y_test, p_final, thr)
    print(f"\nFinal model: {final_name}\n  threshold 0.50 : {({k: round(v, 3) for k, v in m05.items()})}"
          f"\n  threshold {thr:.2f} : {({k: round(v, 3) for k, v in mthr.items()})}")
    print("  95% bootstrap CI (test set, tuned thr):", {k: [round(x, 3) for x in v] for k, v in ci.items()})
    plot_curves(probas, y_test, final_name, (p_final >= thr).astype(int), p_final)

    # ---------- 7. explainability ----------
    imp = permutation_report(final, X_test, y_test, final_name)
    print("\nTop features (permutation importance):", list(imp.feature[:4]))
    partial_dependence_plot(final, X_train, list(imp.feature[:4]))
    logistic_odds_ratios(tuned["Logistic Regression"])
    shap_summary(final, X_train, X_test)

    # ---------- 8. persist ----------
    joblib.dump({"model": final, "threshold": thr, "features": C.FEATURES, "name": final_name,
                 "train_medians": X_train.replace(0, np.nan).median().to_dict()}, C.MODEL_DIR / "final_model.joblib")
    with open(C.RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({"model": final_name, "threshold": thr, "test_at_0.5": m05, "test_at_tuned_threshold": mthr,
                   "bootstrap_95ci": ci, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)
    print("\nSaved model to models/final_model.joblib and results to results/")


if __name__ == "__main__":
    main()
