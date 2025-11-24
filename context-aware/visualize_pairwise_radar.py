# visualize_pairwise_radar.py
# Generates two-model radar charts for TAS components and PEPPER, plus correlation between TAS and PEPPER across all models.

import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SUMMARY_CSV

# Models to compare directly
PAIR_MODELS = ["qwen:7b", "llama3:instruct"]

# Metrics for TAS (no inversion; add 0.5)
TAS_METRICS = ["context_retention", "cross_coherence", "topic_interference"]
TAS_LABELS = {
    "context_retention": "Context Retention",
    "cross_coherence": "Cross Coherence",
    "topic_interference": "Topic Interference",
}

# Metrics for PEPPER (shift +0.5, then min-max normalize to [0, 1])
PEPPER_METRICS = ["proactiveness", "personalization", "coherence"]
PEPPER_LABELS = {
    "proactiveness": "Proactiveness",
    "personalization": "Personalization",
    "coherence": "Coherence",
}


def min_max_normalize(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Normalize columns to [0, 1]; if constant, set to 0.5 and clamp."""
    for col in columns:
        if col not in df.columns:
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_max == col_min:
            df[col] = 0.5
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        df[col] = df[col].clip(0, 1)
    return df


def load_summaries() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary CSV at '{SUMMARY_CSV}'")
    df = pd.read_csv(SUMMARY_CSV)
    # Apply +0.5 shift for TAS and PEPPER raw scores before any normalization/plotting
    if "context_adaptation_score" in df.columns:
        df["topic_adaptation_score"] = df["context_adaptation_score"] + 0.5
    for col in TAS_METRICS + PEPPER_METRICS:
        if col in df.columns:
            df[col] = df[col] + 0.5
    # Normalize PEPPER metrics (after shift)
    df = min_max_normalize(df, PEPPER_METRICS)
    return df


def plot_radar_two_models(
    df: pd.DataFrame,
    metrics: List[str],
    labels: Optional[dict],
    title: str,
    filename: str,
) -> None:
    """Plot a radar chart for exactly two models with outside legend and readable fonts."""
    subset = df[df["model"].isin(PAIR_MODELS)]
    if subset.empty:
        print(f"Skipping {title}: no data for selected models.")
        return

    labels_order = [labels.get(m, m) if labels else m for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = ["#1f77b4", "#ff7f0e"]

    for idx, (_, row) in enumerate(subset.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=3, label=row["model"])
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_order, fontsize=24)
    ax.tick_params(axis="y", labelsize=18)

    plt.title(title, fontsize=28, pad=18)
    legend = plt.legend(
        title="Model",
        fontsize=18,
        title_fontsize=20,
        loc="upper left",
        bbox_to_anchor=(1.25, 1),
        borderaxespad=0,
    )
    if legend:
        legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def compute_alignment_correlation(df: pd.DataFrame) -> float:
    """Compute Pearson correlation between TAS and PEPPER mean across all models."""
    cols_needed = {"topic_adaptation_score"} | set(f"{m}" for m in PEPPER_METRICS)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for correlation: {missing}")
    pepper_mean = df[PEPPER_METRICS].mean(axis=1)
    return float(df["topic_adaptation_score"].corr(pepper_mean))


def main() -> None:
    df = load_summaries()

    # TAS radar for two models
    tas_path = "figures/radars/topic_adaptation_score_pair.png"
    plot_radar_two_models(df, TAS_METRICS, TAS_LABELS, "Topic Adaptation Score", tas_path)

    # PEPPER radar for two models
    pepper_path = "figures/radars/pepper_pair.png"
    plot_radar_two_models(df, PEPPER_METRICS, PEPPER_LABELS, "PEPPER", pepper_path)

    # Correlation between TAS and PEPPER across all models
    try:
        corr = compute_alignment_correlation(df)
        print(f"Pearson correlation between TAS and PEPPER (all models): {corr:.3f}")
    except KeyError as e:
        print(f"Correlation skipped: {e}")


if __name__ == "__main__":
    main()
