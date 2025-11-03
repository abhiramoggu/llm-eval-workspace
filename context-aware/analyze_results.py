# analyze_results.py
"""
Visualization of model-wise metrics and correlations from results.jsonl.
Saves figures to config.FIG_DIR. Handles missing libraries and sparse data gracefully.
"""

import json
import os

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required to analyze results. Install it with `pip install pandas` and rerun."
    ) from exc

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to analyze results. Install it with `pip install matplotlib` and rerun."
    ) from exc

try:
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "seaborn is required to analyze results. Install it with `pip install seaborn` and rerun."
    ) from exc

from config import RESULTS_FILE, FIG_DIR


def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run `python batch_run.py` first.")

    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if not records:
        raise SystemExit(f"{path} is empty. Run `python batch_run.py` to generate results.")

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("Results file contains no usable rows. Rerun your simulations.")
    if "model" not in df.columns:
        raise SystemExit("`model` column missing from results. Check batch_run output.")

    if "judge" in df.columns and df["judge"].notna().any():
        judge_df = pd.json_normalize(df["judge"].fillna({}))
        df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)
    else:
        for col in ["clarity", "politeness", "recovery", "context_memory", "engagement"]:
            if col not in df.columns:
                df[col] = None

    return df


def maybe_plot(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)

    backend = matplotlib.get_backend().lower()
    interactive = backend in {"tkagg", "qt5agg", "macosx", "gtk3agg"}
    if interactive:
        plt.show()
    else:
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def ensure_numeric(df: pd.DataFrame, cols):
    return [col for col in cols if col in df.columns and df[col].notna().any()]


def main():
    df = load_results(RESULTS_FILE)

    quant_cols = [
        "topic_recovery_rate",
        "avg_recovery_delay",
        "topic_interference",
        "cross_coherence",
        "context_retention",
        "context_adaptation_score",
    ]
    judge_cols = ["clarity", "politeness", "recovery", "context_memory", "engagement"]

    available_quant = ensure_numeric(df, quant_cols)
    available_judge = ensure_numeric(df, judge_cols)

    # 1) Grouped bar for quantitative metrics
    if available_quant:
        qmean = df.groupby("model")[available_quant].mean()
        fig, ax = plt.subplots(figsize=(11, 6))
        qmean.plot(kind="bar", ax=ax)
        ax.set_title("Model-wise Quantitative Metrics")
        ax.set_ylabel("Score (0-1 or raw delay)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        maybe_plot(fig, "quant_metrics_grouped_bar.png")
    else:
        print("Skipping quantitative grouped bar chart; no numeric metrics available.")

    # 2) Box plot for recovery delays (skip if all NaN)
    if "avg_recovery_delay" in available_quant:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="model", y="avg_recovery_delay", data=df, ax=ax)
        ax.set_title("Distribution of Average Recovery Delay (per run)")
        ax.set_ylabel("System turns")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        maybe_plot(fig, "recovery_delay_boxplot.png")
    else:
        print("Skipping recovery delay box plot; metric missing or empty.")

    # 3) Heatmap of correlations
    corr_cols = available_quant + available_judge
    if corr_cols:
        corr = df[corr_cols].astype(float).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Metric Correlation Heatmap")
        fig.tight_layout()
        maybe_plot(fig, "metric_correlation_heatmap.png")
    else:
        print("Skipping correlation heatmap; no numeric metrics available.")

    # 4) Horizontal bar for CAS
    if "context_adaptation_score" in available_quant:
        cas = df.groupby("model")["context_adaptation_score"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        cas.plot(kind="barh", ax=ax)
        ax.set_title("Context Adaptation Score (CAS) by Model")
        ax.set_xlabel("CAS (0..1)")
        fig.tight_layout()
        maybe_plot(fig, "cas_by_model.png")
    else:
        print("Skipping CAS bar chart; metric missing or empty.")


if __name__ == "__main__":
    main()
