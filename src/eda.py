"""Exploratory plots saved to results/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import FEATURES, TARGET, ZERO_IS_MISSING, FIG_DIR


def run_eda(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    clean[ZERO_IS_MISSING] = clean[ZERO_IS_MISSING].replace(0, np.nan)

    # 1. class balance
    fig, ax = plt.subplots(figsize=(4, 3.5))
    df[TARGET].value_counts().sort_index().plot.bar(ax=ax, color=["#4C9F70", "#D1495B"])
    ax.set_xticklabels(["Non-diabetic (0)", "Diabetic (1)"], rotation=0)
    ax.set_title("Target distribution"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_target_distribution.png", dpi=150); plt.close(fig)

    # 2. hidden zeros
    zeros = (df[ZERO_IS_MISSING] == 0).sum().sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    zeros.plot.barh(ax=ax, color="#EDAE49")
    ax.set_title("Zeros treated as 'not measured'"); ax.set_xlabel("Rows with value 0")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_hidden_zeros.png", dpi=150); plt.close(fig)

    # 3. distributions by class
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.histplot(data=clean, x=c, hue=TARGET, kde=True, stat="density",
                     common_norm=False, ax=ax, palette=["#4C9F70", "#D1495B"])
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_distributions_by_class.png", dpi=130); plt.close(fig)

    # 4. boxplots (outliers inspected visually, not removed)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=clean, x=TARGET, y=c, ax=ax, palette=["#4C9F70", "#D1495B"], hue=TARGET, legend=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_boxplots.png", dpi=130); plt.close(fig)

    # 5. correlations (Spearman: robust to skew/outliers)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(clean.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation (zeros treated as missing)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_correlation.png", dpi=150); plt.close(fig)
