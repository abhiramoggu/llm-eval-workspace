# visualize_radar_custom.py
# Generates custom radar charts for CAS components and LLM-as-judge scores

import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG_DIR, SUMMARY_CSV


CAS_METRICS = ["context_retention", "cross_coherence", "topic_interference"]
JUDGE_METRICS = ["proactiveness", "personalization", "coherence"]
CORE_MODELS = ["gemma:2b", "qwen:4b", "llama3:instruct"]
RADAR_DIR = os.path.join(FIG_DIR, "radars")


def load_model_metrics() -> pd.DataFrame:
    """Load aggregated metric CSV produced by evaluation pipeline."""
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary CSV at '{SUMMARY_CSV}'.")
    return pd.read_csv(SUMMARY_CSV)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_labels(metrics: Iterable[str], label_map: Optional[Dict[str, str]] = None) -> List[str]:
    if not label_map:
        return [m.replace("_", " ").title() for m in metrics]
    return [label_map.get(m, m) for m in metrics]


def plot_radar(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    filename: str,
    label_map: Optional[Dict[str, str]] = None,
    legend_title: Optional[str] = None,
) -> None:
    """Generic radar plotting helper with large fonts per requirements."""
    if df.empty:
        print(f"Skipping '{title}' because no data is available.")
        return

    labels = format_labels(metrics, label_map)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Determine radial ticks that fit the data range.
    max_value = df[metrics].max().max()
    tick_values = np.linspace(0, max_value if max_value > 0 else 1, 5)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([f"{tick:.2f}" for tick in tick_values], fontsize=45)
    ax.set_ylim(0, tick_values[-1] if tick_values[-1] > 0 else 1)

    for _, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3, label=row["model"])
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=45)
    ax.tick_params(axis="y", labelsize=45)

    plt.title(title, fontsize=50, pad=25)
    legend = plt.legend(
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
        fontsize=22,
    )
    if legend_title and legend:
        legend.get_title().set_fontsize(24)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def prepare_cas_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return CAS metrics with required adjustments (add 0.5; no inversion)."""
    missing = [m for m in CAS_METRICS if m not in df.columns]
    if missing:
        raise KeyError(f"CAS metrics missing from data: {missing}")

    cas_df = df[["model"] + CAS_METRICS].copy()
    for metric in CAS_METRICS:
        cas_df[metric] = cas_df[metric] + 0.5
    return cas_df


def prepare_judge_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract LLM-as-judge metrics."""
    missing = [m for m in JUDGE_METRICS if m not in df.columns]
    if missing:
        raise KeyError(f"Judge metrics missing from data: {missing}")
    return df[["model"] + JUDGE_METRICS].copy()


def main() -> None:
    ensure_output_dir(RADAR_DIR)
    metrics_df = load_model_metrics()

    # --- Topic Adaptation Score (CAS components) ---
    cas_df = prepare_cas_data(metrics_df)
    cas_labels = {
        "context_retention": "Context Retention",
        "cross_coherence": "Cross Coherence",
        "topic_interference": "Topic Interference",
    }
    cas_core = cas_df[cas_df["model"].isin(CORE_MODELS)]
    plot_radar(
        cas_core,
        CAS_METRICS,
        title="Topic Adaptation Score",
        filename=os.path.join(RADAR_DIR, "topic_adaptation_score_core.png"),
        label_map=cas_labels,
        legend_title="Model",
    )
    plot_radar(
        cas_df,
        CAS_METRICS,
        title="Topic Adaptation Score",
        filename=os.path.join(RADAR_DIR, "topic_adaptation_score_all.png"),
        label_map=cas_labels,
        legend_title="Model",
    )

    # --- PEPPER (LLM-as-judge) ---
    judge_df = prepare_judge_data(metrics_df)
    judge_labels = {
        "proactiveness": "Proactiveness",
        "personalization": "Personalization",
        "coherence": "Coherence",
    }
    judge_core = judge_df[judge_df["model"].isin(CORE_MODELS)]
    plot_radar(
        judge_core,
        JUDGE_METRICS,
        title="PEPPER",
        filename=os.path.join(RADAR_DIR, "pepper_core.png"),
        label_map=judge_labels,
        legend_title="Model",
    )
    plot_radar(
        judge_df,
        JUDGE_METRICS,
        title="PEPPER",
        filename=os.path.join(RADAR_DIR, "pepper_all_models.png"),
        label_map=judge_labels,
        legend_title="Model",
    )


if __name__ == "__main__":
    main()
