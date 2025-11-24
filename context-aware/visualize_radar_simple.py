# visualize_radar_simple.py
# Lightweight radar charts for TAS and PEPPER with readable fonts and concise legends.

import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG_DIR, SUMMARY_CSV


RADAR_DIR = os.path.join(FIG_DIR, "radars_new")

PEPPER_METRICS = ["personalization", "coherence", "proactiveness"]
TAS_METRICS = ["cross_coherence", "context_retention", "topic_interference"]

SUBSET_MODELS = ["mistral:7b", "llama3:instruct"]

MODEL_COLORS = {
    "llama3:instruct": "#1f77b4",  # blue
    "mistral:7b": "#ff7f0e",       # orange
}

PEPPER_LABELS = {
    "personalization": "Personalization",
    "coherence": "Coherence",
    "proactiveness": "Proactiveness",
}

TAS_LABELS = {
    "cross_coherence": "Cross Coherence",
    "context_retention": "Context Retention",
    "topic_interference": "Topic Interference",
}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_summary() -> pd.DataFrame:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Summary CSV not found at '{SUMMARY_CSV}'")
    return pd.read_csv(SUMMARY_CSV)


def metric_labels(metrics: Iterable[str], label_map: Dict[str, str]) -> List[str]:
    return [label_map.get(m, m) for m in metrics]


def format_legend_label(model: str, values: List[float]) -> str:
    joined = ", ".join(f"{v:.3f}" for v in values)
    return f"{model} ({joined})"


def get_color(model: str, fallback_idx: int) -> str:
    palette = plt.get_cmap("tab10").colors
    return MODEL_COLORS.get(model, palette[fallback_idx % len(palette)])


def plot_radar(df: pd.DataFrame, metrics: List[str], label_map: Dict[str, str], title: str, filename: str) -> None:
    if df.empty:
        print(f"Skipping {title}: no data available.")
        return

    labels = metric_labels(metrics, label_map)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    handles: List = []
    legend_labels: List[str] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        legend_label = format_legend_label(row["model"], row[metrics].tolist())
        color = get_color(row["model"], idx)
        (line,) = ax.plot(angles, values, linewidth=3, label=legend_label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
        handles.append(line)
        legend_labels.append(legend_label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=32)
    ax.set_yticklabels([])  # Hide tick numbers
    ax.tick_params(axis="y", labelsize=32)

    plt.title(title, fontsize=36, pad=18, fontweight="bold")
    # Save main plot without legend
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")

    # Save legend separately
    if handles:
        legend_fig = plt.figure(figsize=(6, 4))
        legend = legend_fig.legend(
            handles,
            legend_labels,
            loc="center",
            fontsize=18,
            title="Model (values)",
            title_fontsize=20,
            framealpha=0.9,
        )
        for text in legend.get_texts():
            text.set_multialignment("left")
        legend_fig.tight_layout()
        legend_path = filename.replace(".png", "_legend.png")
        legend_fig.savefig(legend_path, bbox_inches="tight")
        plt.close(legend_fig)
        print(f"Saved legend: {legend_path}")


def save_model_color_legend(path: str) -> None:
    """Save a standalone legend showing model-color mapping."""
    handles = []
    labels = []
    for idx, (model, color) in enumerate(MODEL_COLORS.items()):
        line = plt.Line2D([0], [0], color=color, linewidth=3)
        handles.append(line)
        labels.append(model)
    fig = plt.figure(figsize=(4, 2))
    legend = fig.legend(handles, labels, loc="center", fontsize=14, title="Model Colors", title_fontsize=16, framealpha=0.9)
    legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model color legend: {path}")


def main() -> None:
    ensure_output_dir(RADAR_DIR)
    df = load_summary()

    pepper_available = [m for m in PEPPER_METRICS if m in df.columns]
    tas_available = [m for m in TAS_METRICS if m in df.columns]

    # All-model plots
    if len(pepper_available) == len(PEPPER_METRICS):
        plot_radar(
            df,
            PEPPER_METRICS,
            PEPPER_LABELS,
            "PEPPER (LLM-as-Judge)",
            os.path.join(RADAR_DIR, "pepper_radar_simple.png"),
        )
    else:
        print(f"Skipping PEPPER radar; missing metrics: {set(PEPPER_METRICS) - set(pepper_available)}")

    if len(tas_available) == len(TAS_METRICS):
        plot_radar(
            df,
            TAS_METRICS,
            TAS_LABELS,
            "Topic Adaptation Score (TAS)",
            os.path.join(RADAR_DIR, "tas_radar_simple.png"),
        )
    else:
        print(f"Skipping TAS radar; missing metrics: {set(TAS_METRICS) - set(tas_available)}")

    # Subset (mistral:7b vs llama3:instruct) plots
    subset_df = df[df["model"].isin(SUBSET_MODELS)]
    if len(pepper_available) == len(PEPPER_METRICS):
        plot_radar(
            subset_df,
            PEPPER_METRICS,
            PEPPER_LABELS,
            "PEPPER (mistral:7b vs llama3:instruct)",
            os.path.join(RADAR_DIR, "pepper_radar_simple_subset.png"),
        )
    if len(tas_available) == len(TAS_METRICS):
        plot_radar(
            subset_df,
            TAS_METRICS,
            TAS_LABELS,
            "TAS (mistral:7b vs llama3:instruct)",
            os.path.join(RADAR_DIR, "tas_radar_simple_subset.png"),
        )

    # Model color legend
    save_model_color_legend(os.path.join(RADAR_DIR, "model_color_legend.png"))


if __name__ == "__main__":
    main()
