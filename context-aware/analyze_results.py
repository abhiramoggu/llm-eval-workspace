# analyze_results.py
# Visualizes quantitative & LLM-judge evaluation results across models

import os
import shutil
import json
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
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="model", y="avg_recovery_delay", data=df)
    plt.title("Distribution of Recovery Delay (Turns) Across Models")
    plt.ylabel("Turns to Recover")
    plt.xticks(rotation=30)
    plt.tight_layout() 
    save_path = os.path.join(FIG_DIR, "box_plots", "recovery_delay_boxplot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

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
