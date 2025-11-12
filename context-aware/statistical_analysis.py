# statistical_analysis.py
"""
Performs statistical significance testing on the evaluation results to determine
if the observed differences between models are statistically significant.

It uses:
1. ANOVA to check for any significant difference across all models for a given metric.
2. Tukey's HSD post-hoc test to perform pairwise comparisons and identify which
   specific model pairs have significantly different means.
"""

import pandas as pd
import json
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from config import RESULTS_FILE

def perform_statistical_analysis(alpha=0.05):
    """
    Performs statistical tests (ANOVA and Tukey's HSD) to compare model performance.
    
    Args:
        alpha (float): The significance level for the tests.
    """
    print("--- Starting Statistical Significance Analysis ---")

    # Load and prepare data
    try:
        records = [json.loads(line) for line in open(RESULTS_FILE)]
        df = pd.DataFrame(records)
        judge_df = pd.json_normalize(df["judge"])
        df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{RESULTS_FILE}'. Please run `batch_run.py` first.")
        return

    all_metrics = [
        "topic_recovery_rate", "topic_interference", "cross_coherence",
        "context_retention", "context_adaptation_score", "avg_recovery_delay",
        "proactiveness", "coherence", "personalization"
    ]
    metrics_present = [m for m in all_metrics if m in df.columns]
    models = df['model'].unique()

    if len(models) < 2:
        print("Cannot perform statistical tests with fewer than two models.")
        return

    for metric in metrics_present:
        print(f"\n--- Analyzing Metric: {metric.replace('_', ' ').title()} ---")

        # Prepare data for ANOVA
        grouped_data = [df[metric][df['model'] == model].dropna() for model in models]

        # Check if there's enough data to perform the test
        if any(len(data) < 2 for data in grouped_data):
            print("Skipping test: at least one model has fewer than 2 data points.")
            continue

        # 1. Perform ANOVA test
        f_val, p_val_anova = stats.f_oneway(*grouped_data)
        print(f"ANOVA Test: F-statistic = {f_val:.4f}, p-value = {p_val_anova:.4f}")

        if p_val_anova < alpha:
            print(f"Result: A statistically significant difference exists among the models (p < {alpha}).")
            
            # 2. Perform Tukey's HSD post-hoc test for pairwise comparisons
            print("\nPairwise Comparisons (Tukey's HSD):")
            tukey_result = pairwise_tukeyhsd(endog=df[metric].dropna(), groups=df['model'], alpha=alpha)
            
            results_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
            
            # Filter to show only significant differences
            significant_pairs = results_df[results_df['p-adj'] < alpha]
            
            if not significant_pairs.empty:
                print(significant_pairs.to_string(index=False))
            else:
                print("No specific model pairs were found to be significantly different after correction.")
        else:
            print(f"Result: No statistically significant difference was found among the models (p >= {alpha}).")

if __name__ == "__main__":
    perform_statistical_analysis()