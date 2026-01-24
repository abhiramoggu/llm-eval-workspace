# analyze_results.py
# Visualizes quantitative & LLM-judge evaluation results across models

import os
import shutil
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from config import RESULTS_FILE, FIG_DIR

# === Directory Setup ===
SUB_DIRS = ["distributions", "bar_charts", "radars", "correlations", "over_time", "box_plots", "stacked_bars"]

def setup_figure_directories(base_dir: str, sub_dirs: List[str]):
    """Clears the base figure directory and creates it along with specified subdirectories."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    for sub in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
    print(f"--- Cleared and prepared figure directories in '{base_dir}' ---")

# === Load Results ===
setup_figure_directories(FIG_DIR, SUB_DIRS)

records = [json.loads(line) for line in open(RESULTS_FILE)]
df = pd.DataFrame(records)

# Flatten LLM-judge nested dict
judge_df = pd.json_normalize(df["judge"])
df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)

# Ensure detail column is always a dict for downstream access
if "detail" in df.columns:
    df["detail"] = df["detail"].apply(lambda x: x if isinstance(x, dict) else {})
else:
    df["detail"] = [{} for _ in range(len(df))]

# === Color Palette for Metrics ===
metric_colors = {
    "topic_recovery_rate": "#1f77b4",
    "topic_interference": "#ff7f0e",
    "cross_coherence": "#2ca02c",
    "context_retention": "#d62728",
    "context_adaptation_score": "#9467bd",
    "avg_recovery_delay": "#8c564b",
    "proactiveness": "#e377c2",
    "coherence": "#7f7f7f",
    "personalization": "#bcbd22",
}


# === 1. Individual Bar Charts for Each Metric ===
all_metrics = [
    "topic_recovery_rate", "topic_interference",
    "cross_coherence", "context_retention", "context_adaptation_score",
    "avg_recovery_delay", "proactiveness", "coherence",
    "personalization"
]
# Filter out metrics that might not be in the dataframe to prevent KeyErrors
metrics_present = [m for m in all_metrics if m in df.columns]

df_summary = df.groupby("model")[metrics_present].mean().round(3)

for metric in metrics_present:
    plt.figure(figsize=(10, 6))

    # Plot a Kernel Density Estimate for each model on the same axes
    for model_name in df['model'].unique():
        model_scores = df[df['model'] == model_name][metric].dropna()
        if model_scores.empty:
            continue
        # Check if there is variance in the data
        if model_scores.nunique() > 1:
            sns.kdeplot(model_scores, label=model_name, fill=True, alpha=0.2)
        else:
            # If no variance, plot a vertical line at the single value
            plt.axvline(model_scores.iloc[0], linestyle='--', label=f'{model_name} (single value)')

    plt.title(f'Distribution of {metric.replace("_", " ").title()} per Model')
    plt.xlabel("Score")
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "distributions", f"metric_{metric}_distribution.png")
    print(f"Saved plot: {save_path}")
    plt.savefig(save_path)
    plt.close()

# === 1.5 Bar Charts for Average Metrics ===
for metric in metrics_present:
    plt.figure(figsize=(8, 5))

    # Use the assigned color for the metric, defaulting to a standard blue
    color = metric_colors.get(metric, "#1f77b4")

    sns.barplot(x=df_summary.index, y=df_summary[metric], color=color)

    # Add value labels on top of each bar
    for index, value in enumerate(df_summary[metric]):
        plt.text(index, value, f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.title(f'Average {metric.replace("_", " ").title()} per Model')
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "bar_charts", f"metric_{metric}_bar.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# === 2. Radar Chart (LLM-Judge metrics) ===
def plot_radar(df_mean, metrics, title, filename):
    import numpy as np
    labels = np.array(metrics)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in df_mean.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["model"], linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

judge_metrics = ["proactiveness", "coherence", "personalization"]
judge_metrics = [m for m in judge_metrics if m in df.columns]
if judge_metrics:
    judge_mean = df.groupby("model")[judge_metrics].mean().reset_index()
    radar_path = os.path.join(FIG_DIR, "radars", "llm_judge_radar.png")
    plot_radar(judge_mean, judge_metrics, "LLM-Judge Scores Radar Chart", radar_path)

# === 3. Box Plot: Recovery Turns Distribution ===
if "avg_recovery_delay" in df.columns:
    delay_df = df[["model", "avg_recovery_delay"]].dropna()
    if not delay_df.empty and delay_df["model"].nunique() > 0:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="model", y="avg_recovery_delay", data=delay_df)
        plt.title("Distribution of Recovery Delay (Turns) Across Models")
        plt.ylabel("Turns to Recover")
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, "box_plots", "recovery_delay_boxplot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")
    else:
        print("Skipping recovery delay boxplot: no valid avg_recovery_delay values.")

# === 4. Correlation Heatmap ===
all_metrics = [m for m in (metrics_present + judge_metrics) if m in df.columns]
if len(all_metrics) > 1:
    corr = df[all_metrics].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "correlations", "metrics_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# === 5. Stacked Bar: Combined Recovery Metrics ===
recovery_metrics = ["topic_recovery_rate", "topic_interference"]
if all(m in df_summary.columns for m in recovery_metrics):
    stacked = df_summary[recovery_metrics]
    stacked.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Stacked Recovery & Interference per Model")
    plt.ylabel("Score")
    plt.tight_layout() 
    save_path = os.path.join(FIG_DIR, "stacked_bars", "stacked_recovery_bar.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# === 6. Radar Chart for Context Adaptation Score (CAS) Components ===
cas_metrics = [
    "topic_recovery_rate",
    "cross_coherence",
    "context_retention",
    "norm_inv_recovery_delay",
    "norm_inv_topic_interference"
]

df_cas = df.copy()

# Normalize avg_recovery_delay (lower is better) -> (higher is better)
# We assume a practical range of 1 to 6 turns for recovery.
if "avg_recovery_delay" in df_cas.columns:
    min_delay, max_delay = 1, 6
    delay_norm = (df_cas["avg_recovery_delay"] - min_delay) / (max_delay - min_delay)
    df_cas["norm_inv_recovery_delay"] = 1 - delay_norm.clip(0, 1)

# Normalize topic_interference (lower is better) -> (higher is better)
if "topic_interference" in df_cas.columns:
    min_inter, max_inter = 0, 1
    inter_norm = (df_cas["topic_interference"] - min_inter) / (max_inter - min_inter)
    df_cas["norm_inv_topic_interference"] = 1 - inter_norm.clip(0, 1)

# Filter for metrics that actually exist in the dataframe
cas_metrics_present = [m for m in cas_metrics if m in df_cas.columns]

if len(cas_metrics_present) > 2: # A radar chart needs at least 3 axes
    cas_mean = df_cas.groupby("model")[cas_metrics_present].mean().reset_index()
    
    # Prettier labels for the chart
    cas_labels = {
        "topic_recovery_rate": "Recovery Rate",
        "cross_coherence": "Coherence",
        "context_retention": "Retention",
        "norm_inv_recovery_delay": "Recovery Speed",
        "norm_inv_topic_interference": "Focus"
    }
    chart_labels = [cas_labels.get(m, m) for m in cas_metrics_present]

    cas_radar_path = os.path.join(FIG_DIR, "radars", "cas_components_radar.png")
    plot_radar(cas_mean, cas_metrics_present, "Context Adaptation Score Components", cas_radar_path)

# === 7. Per-Model Conversation Deviation Step Graph ===
def plot_combined_deviation_graph(df: pd.DataFrame):
    """
    Plots the average deviation for all models on a single graph using pre-calculated
    alignment scores from the results file.
    """
    from collections import defaultdict
    import numpy as np

    plt.figure(figsize=(12, 4))

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # Collect all alignment scores per turn index
        turn_alignments = defaultdict(list)
        for _, row in model_df.iterrows():
            if 'detail' in row and 'turn_alignments' in row['detail']:
                for item in row['detail']['turn_alignments']:
                    turn_alignments[item['turn_idx']].append(item['alignment'])

        if not turn_alignments:
            continue

        # Calculate mean alignment for each turn
        turns = sorted(turn_alignments.keys())
        mean_alignments = [np.mean(turn_alignments[t]) for t in turns]
        
        plt.step(turns, mean_alignments, where='post', label=model)

    plt.axhline(y=0.65, color='r', linestyle='--', label='Alignment Threshold (0.65)')
    plt.title("Average Conversation Deviation Across Models")
    plt.xlabel("Conversation Turn Index")
    plt.ylabel("Topic Alignment Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "over_time", "combined_avg_deviation_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# Generate a single, combined deviation graph for all models
plot_combined_deviation_graph(df)

# === 8. Distribution of Number of Topics per System Response ===
all_topic_counts_data = []
for index, row in df.iterrows():
    model = row['model']
    # Ensure 'detail' and 'system_topic_counts' exist
    counts = row['detail'].get('system_topic_counts', []) if isinstance(row['detail'], dict) else []
    for count in counts:
        all_topic_counts_data.append({'model': model, 'topic_count': count})

if all_topic_counts_data:
    df_topic_counts = pd.DataFrame(all_topic_counts_data)
    
    plt.figure(figsize=(10, 6))
    for model_name in df_topic_counts['model'].unique():
        model_data = df_topic_counts[df_topic_counts['model'] == model_name]['topic_count']
        if model_data.nunique() > 1: # Check for variance
            sns.kdeplot(model_data, label=model_name, fill=True, alpha=0.2)
        else:
            plt.axvline(model_data.iloc[0], linestyle='--', label=f'{model_name} (single value)')

    plt.title('Distribution of Number of Topics per System Response')
    plt.xlabel('Number of Topics')
    plt.ylabel('Density')
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "distributions", "system_topic_counts_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
else:
    print("No system topic count data available for plotting.")


# ============================================================
# Additional stats + visualizations for TAS (grounded evaluation)
# ============================================================

def _safe_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y, nan_policy='omit')
        return float(r), float(p)
    except Exception:
        # fallback: rank then pearson
        import numpy as np
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            return float('nan'), float('nan')
        rx = x.argsort().argsort()
        ry = y.argsort().argsort()
        r = np.corrcoef(rx, ry)[0, 1]
        return float(r), float('nan')


def bootstrap_spearman(x, y, n_boot=1000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 5:
        return (float('nan'), float('nan'), float('nan'))
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        r, _ = _safe_spearman(x[idx], y[idx])
        rs.append(r)
    rs = np.asarray(rs, dtype=float)
    return (float(np.nanmedian(rs)), float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)))


def paired_model_tests(df: pd.DataFrame, metric: str, out_path: str):
    """Run paired tests across models assuming shared session_id."""
    import numpy as np
    from itertools import combinations
    rows = []
    if "session_id" not in df.columns:
        return
    pivot = df.pivot_table(index="session_id", columns="model", values=metric, aggfunc="mean")
    models = [c for c in pivot.columns if pivot[c].notna().sum() > 3]
    if len(models) < 2:
        return

    for a, b in combinations(models, 2):
        x = pivot[a].values
        y = pivot[b].values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 5:
            continue
        # Wilcoxon signed-rank is robust; fall back to paired t-test if scipy unavailable
        p = None
        stat = None
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(x, y)
        except Exception:
            try:
                from scipy.stats import ttest_rel
                stat, p = ttest_rel(x, y)
            except Exception:
                stat, p = float('nan'), float('nan')
        rows.append({"metric": metric, "model_a": a, "model_b": b, "n": len(x), "stat": float(stat), "p": float(p)})

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved paired tests: {out_path}")


def plot_tas_vs_judge(df: pd.DataFrame, out_path: str):
    if "context_adaptation_score" not in df.columns:
        return
    judge_col = None
    for c in ["overall", "pepper_score", "coherence"]:
        if c in df.columns:
            judge_col = c
            break
    if judge_col is None:
        return
    plt.figure(figsize=(7, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.scatter(sub["context_adaptation_score"], sub[judge_col], label=model, alpha=0.6)
    r, p = _safe_spearman(df["context_adaptation_score"], df[judge_col])
    med, lo, hi = bootstrap_spearman(df["context_adaptation_score"], df[judge_col])
    plt.title(f"TAS vs Judge ({judge_col})\nSpearman r={r:.3f}, p={p:.3g} | boot95%=[{lo:.3f},{hi:.3f}]")
    plt.xlabel("TAS (context_adaptation_score)")
    plt.ylabel(f"Judge ({judge_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_timeline_example(df: pd.DataFrame, out_path: str):
    """Plot CC/CR/I/TAS over turns for one example conversation per model."""
    import numpy as np
    plt.figure(figsize=(12, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        row = sub.iloc[0]
        detail = row.get("detail", {}) or {}
        turns = detail.get("turn_level", [])
        if not turns:
            continue
        x = [t["turn"] for t in turns]
        y = [t["tas"] for t in turns]
        plt.plot(x, y, label=f"{model} TAS")
        # mark shift points
        for sp in detail.get("shift_points", []):
            plt.axvline(sp, linestyle="--", alpha=0.25)
    plt.title("Example TAS timeline (vertical lines = user shift points)")
    plt.xlabel("Turn")
    plt.ylabel("TAS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


# Run the extra analyses if possible
try:
    scatter_path = os.path.join(FIG_DIR, "correlations", "tas_vs_judge_scatter.png")
    plot_tas_vs_judge(df, scatter_path)

    timeline_path = os.path.join(FIG_DIR, "over_time", "tas_timeline_example.png")
    plot_timeline_example(df, timeline_path)

    paired_out = os.path.join(FIG_DIR, "correlations", "paired_tests_tas.csv")
    paired_model_tests(df, "context_adaptation_score", paired_out)

    paired_out2 = os.path.join(FIG_DIR, "correlations", "paired_tests_recovery.csv")
    paired_model_tests(df, "topic_recovery_rate", paired_out2)

except Exception as e:
    print("Extra TAS analysis skipped due to error:", e)


# ============================================================
# Concept delta aggregation (missing/hallucinated concepts)
# ============================================================

def _extract_turn_concepts(row: pd.Series, per_turn_key: str, detail_key: str) -> List[List[str]]:
    per_turn = row.get(per_turn_key)
    if isinstance(per_turn, list):
        return per_turn
    detail = row.get("detail", {}) if isinstance(row, dict) or isinstance(row, pd.Series) else {}
    if isinstance(detail, dict):
        turns = detail.get("turn_level", [])
        if isinstance(turns, list):
            return [t.get(detail_key, []) for t in turns if isinstance(t, dict)]
    return []


def write_concept_delta_summary(df: pd.DataFrame, out_path: str, top_k: int = 10) -> None:
    if "model" not in df.columns:
        print("Concept delta summary skipped: missing 'model' column.")
        return
    summary = {}
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        missing_counter: Counter = Counter()
        hallucinated_counter: Counter = Counter()
        for _, row in sub.iterrows():
            missing_turns = _extract_turn_concepts(row, "missing_concepts_per_turn", "missing_concepts")
            for items in missing_turns:
                if isinstance(items, list):
                    missing_counter.update([str(x) for x in items])
            hallucinated_turns = _extract_turn_concepts(row, "hallucinated_concepts_per_turn", "hallucinated_concepts")
            for items in hallucinated_turns:
                if isinstance(items, list):
                    hallucinated_counter.update([str(x) for x in items])
        summary[model] = {
            "missing_concepts_total": int(sum(missing_counter.values())),
            "hallucinated_concepts_total": int(sum(hallucinated_counter.values())),
            "missing_concepts_topk": [
                {"concept": c, "count": int(n)} for c, n in missing_counter.most_common(top_k)
            ],
            "hallucinated_concepts_topk": [
                {"concept": c, "count": int(n)} for c, n in hallucinated_counter.most_common(top_k)
            ],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"Saved concept delta summary: {out_path}")


try:
    concept_delta_path = os.path.join(FIG_DIR, "concept_delta_summary.json")
    write_concept_delta_summary(df, concept_delta_path)
except Exception as e:
    print("Concept delta summary skipped due to error:", e)
