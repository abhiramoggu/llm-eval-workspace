# visualize_distributions_custom.py
# Builds four large-format distribution plots for coherence and Topic Adaptation Score.

import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIG_DIR, RESULTS_FILE


CORE_MODELS = ["llama3:instruct", "qwen:7b"]
OUTPUT_DIR = os.path.join(FIG_DIR, "distributions")
PEPPER_COLS = ["proactiveness", "personalization", "coherence"]


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def min_max_normalize(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Normalize columns to [0, 1]; if constant, set to 0.5."""
    for col in columns:
        if col not in df.columns:
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_max == col_min:
            df[f"{col}_norm"] = 0.5
        else:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        # Clamp to [0, 1] to avoid any bleed-over from numerical noise.
        df[f"{col}_norm"] = df[f"{col}_norm"].clip(0, 1)
    return df


def load_metrics() -> pd.DataFrame:
    """Load per-conversation scores from results.jsonl to enable KDE distributions."""
    import json

    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Expected results file at '{RESULTS_FILE}'")

    records = [json.loads(line) for line in open(RESULTS_FILE)]
    df = pd.DataFrame(records)

    # Flatten judge metrics if present.
    if "judge" in df.columns:
        judge_df = pd.json_normalize(df["judge"])
        df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)

    if "context_adaptation_score" in df.columns:
        # Apply +0.5 shift as requested for the topic adaptation score.
        df["topic_adaptation_score"] = df["context_adaptation_score"] + 0.5

    # Apply +0.5 shift for PEPPER metrics before normalization.
    for col in PEPPER_COLS:
        if col in df.columns:
            df[col] = df[col] + 0.5

    # Normalize PEPPER metrics into 0-1 (including the shift) and clamp.
    df = min_max_normalize(df, PEPPER_COLS)
    return df


def plot_distribution(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    models: Optional[Iterable[str]] = None,
) -> None:
    subset = df.copy()
    if models is not None:
        subset = subset[subset["model"].isin(models)]
    if subset.empty or metric not in subset.columns:
        print(f"Skipping {title}: no data available.")
        return

    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    unique_models = sorted(subset["model"].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_models))

    for model_name in unique_models:
        scores = subset[subset["model"] == model_name][metric].dropna()
        if scores.empty:
            continue
        if scores.nunique() > 1:
            sns.kdeplot(
                scores,
                label=model_name,
                fill=True,
                alpha=0.25,
                linewidth=3,
                bw_adjust=0.8,
                color=palette[unique_models.index(model_name)],
                ax=ax,
            )
        else:
            # With no variance, fall back to a single marker line.
            ax.axvline(
                scores.iloc[0],
                linestyle="--",
                linewidth=3,
                color=palette[unique_models.index(model_name)],
                label=f"{model_name} (single value)",
            )

        # Add per-model mean line
        mean_val = scores.mean()
        ax.axvline(
            mean_val,
            color=palette[unique_models.index(model_name)],
            linewidth=3,
            linestyle=":",
            label=f"{model_name} mean = {mean_val:.2f}",
        )

    plt.title(title, fontsize=50, pad=20)
    plt.xlabel("Score", fontsize=40)
    plt.ylabel("Density", fontsize=40)
    plt.tick_params(axis="both", which="major", labelsize=35)
    plt.legend(
        title="Model",
        fontsize=24,
        title_fontsize=26,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Keep axes within [0, 1] for visual clarity across all plotted metrics.
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    df = load_metrics()

    # Core-model distributions (using normalized PEPPER coherence).
    plot_distribution(
        df,
        metric="coherence_norm",
        title="PEPPER - Normalized Coherence Distribution (Core Models)",
        filename=os.path.join(OUTPUT_DIR, "coherence_core_distribution.png"),
        models=CORE_MODELS,
    )
    plot_distribution(
        df,
        metric="topic_adaptation_score",
        title="Topic Adaptation Score Distribution (Core Models)",
        filename=os.path.join(OUTPUT_DIR, "topic_adaptation_score_core_distribution.png"),
        models=CORE_MODELS,
    )

    # All-model distributions
    plot_distribution(
        df,
        metric="coherence_norm",
        title="PEPPER - Normalized Coherence Distribution (All Models)",
        filename=os.path.join(OUTPUT_DIR, "coherence_all_models_distribution.png"),
    )
    plot_distribution(
        df,
        metric="topic_adaptation_score",
        title="Topic Adaptation Score Distribution (All Models)",
        filename=os.path.join(OUTPUT_DIR, "topic_adaptation_score_all_models_distribution.png"),
    )


if __name__ == "__main__":
    main()
