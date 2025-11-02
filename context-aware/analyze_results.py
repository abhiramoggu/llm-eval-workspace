# analyze_results.py
# Visualizes quantitative & LLM-judge evaluation results across models

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FILE = "results.jsonl"
ANALYSIS_OUTPUT_DIR = "analysis_plots"

# === Load Results ===
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

records = [json.loads(line) for line in open(RESULTS_FILE)]
df = pd.DataFrame(records)

# Flatten LLM-judge nested dict
judge_df = pd.json_normalize(df["judge"])
df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)

# === 1. Grouped Bar Chart: Recovery & Coherence per Model ===
metrics = ["text_recovery_success", "state_recovery_success", "rec_consistency", "user_coherence", "system_coherence", "avg_recovery_alignment"]
df_mean = df.groupby("model")[metrics].mean().reset_index()

df_mean.plot(x="model", kind="bar", figsize=(10, 6))
plt.title("Model-wise Average Quantitative Metrics")
plt.ylabel("Score / Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.tight_layout() # Adjust layout to make room for label
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "quantitative_metrics_bar.png"))
plt.close() # Close the figure to free memory
print(f"Saved: {ANALYSIS_OUTPUT_DIR}/quantitative_metrics_bar.png")

# === 2. Radar Chart (LLM-Judge metrics) ===
def plot_radar(df_mean, metrics, title):
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
    plt.show()
    plt.savefig(filename, bbox_inches='tight') # Use bbox_inches to prevent legend cutoff
    plt.close()
    print(f"Saved: {filename}")

judge_metrics = ["clarity", "politeness", "recovery", "context_memory", "engagement"]
judge_mean = df.groupby("model")[judge_metrics].mean().reset_index()
plot_radar(judge_mean, judge_metrics, "LLM-Judge Scores Radar Chart")
radar_chart_path = os.path.join(ANALYSIS_OUTPUT_DIR, "llm_judge_radar.png")
plot_radar(judge_mean, judge_metrics, "LLM-Judge Scores Radar Chart", radar_chart_path)

# === 3. Box Plot: Recovery Turns Distribution ===
plt.figure(figsize=(8, 5))
sns.boxplot(x="model", y="recovery_turns", data=df)
plt.title("Distribution of Recovery Turns Across Models")
plt.ylabel("Turns to Recover")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
plt.tight_layout() # Adjust layout to make room for label
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "recovery_turns_boxplot.png"))
plt.close()
print(f"Saved: {ANALYSIS_OUTPUT_DIR}/recovery_turns_boxplot.png")

# === 4. Correlation Heatmap ===
corr = df[metrics + judge_metrics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Metric Correlation Heatmap")
plt.tight_layout()
plt.show()
plt.tight_layout() # Adjust layout to make room for label
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "metrics_correlation_heatmap.png"))
plt.close()
print(f"Saved: {ANALYSIS_OUTPUT_DIR}/metrics_correlation_heatmap.png")

# === 5. Stacked Bar: Combined Recovery Metrics ===
stacked = df_mean.set_index("model")[["text_recovery_success", "state_recovery_success", "rec_consistency"]]
stacked.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Stacked Recovery Performance per Model")
plt.ylabel("Proportion of Successful Recovery")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "stacked_recovery_bar.png"))
plt.close()
print(f"Saved: {ANALYSIS_OUTPUT_DIR}/stacked_recovery_bar.png")
