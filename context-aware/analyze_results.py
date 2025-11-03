# analyze_results.py
# Visualizes quantitative & LLM-judge evaluation results across models

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_FILE, FIG_DIR, LOG_DIR

# === Load Results ===
os.makedirs(FIG_DIR, exist_ok=True)

records = [json.loads(line) for line in open(RESULTS_FILE)]
df = pd.DataFrame(records)

# Flatten LLM-judge nested dict
judge_df = pd.json_normalize(df["judge"])
df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)

# === 1. Grouped Bar Chart: Recovery & Coherence per Model ===
# Use the correct metric names from results.jsonl
metrics = [ # avg_recovery_delay removed as its scale is different (number of turns)
    "topic_recovery_rate", "topic_interference",
    "cross_coherence", "context_retention", "context_adaptation_score"
]
# Filter out metrics that might not be in the dataframe to prevent KeyErrors
metrics = [m for m in metrics if m in df.columns]

df_mean = df.groupby("model")[metrics].mean().reset_index()

df_mean.plot(x="model", kind="bar", figsize=(10, 6))
plt.title("Model-wise Average Context Adaptation Metrics")
plt.ylabel("Score / Probability")
plt.xticks(rotation=45)
plt.tight_layout()
save_path = os.path.join(FIG_DIR, "context_adaptation_metrics_bar.png")
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

judge_metrics = ["clarity", "politeness", "recovery", "context_memory", "engagement"]
judge_metrics = [m for m in judge_metrics if m in df.columns]
if judge_metrics:
    judge_mean = df.groupby("model")[judge_metrics].mean().reset_index()
    radar_path = os.path.join(FIG_DIR, "llm_judge_radar.png")
    plot_radar(judge_mean, judge_metrics, "LLM-Judge Scores Radar Chart", radar_path)

# === 3. Box Plot: Recovery Turns Distribution ===
if "avg_recovery_delay" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="model", y="avg_recovery_delay", data=df)
    plt.title("Distribution of Recovery Delay (Turns) Across Models")
    plt.ylabel("Turns to Recover")
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "recovery_delay_boxplot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# === 4. Correlation Heatmap ===
all_metrics = [m for m in (metrics + judge_metrics) if m in df.columns]
if len(all_metrics) > 1:
    corr = df[all_metrics].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "metrics_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# === 5. Stacked Bar: Combined Recovery Metrics ===
recovery_metrics = ["topic_recovery_rate", "topic_interference"]
if all(m in df_mean.columns for m in recovery_metrics):
    stacked = df_mean.set_index("model")[recovery_metrics]
    stacked.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Stacked Recovery & Interference per Model")
    plt.ylabel("Score")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "stacked_recovery_bar.png")
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

    cas_radar_path = os.path.join(FIG_DIR, "cas_components_radar.png")
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
    save_path = os.path.join(FIG_DIR, "combined_avg_deviation_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# Generate a single, combined deviation graph for all models
plot_combined_deviation_graph(df)
