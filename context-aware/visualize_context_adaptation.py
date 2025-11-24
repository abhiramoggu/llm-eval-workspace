import json
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LLM_SYSTEMS = ["gemma:2b", "qwen:7b","qwen:4b", "llama3:instruct", "llama2:latest", "mistral:7b"]
def load_scores_from_results(results_file, models_to_include):
    """
    Loads context adaptation scores from the results.jsonl file for specified models.
    """
    try:
        records = [json.loads(line) for line in open(results_file)]
    except FileNotFoundError:
        print(f"Error: Results file not found at '{results_file}'. Please run the evaluation first.")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # Filter for the models we want to visualize
    if models_to_include:
        df = df[df['model'].isin(models_to_include)]
    
    # Select only the necessary columns
    scores_df = df[['model', 'context_adaptation_score']].copy()
    scores_df.rename(columns={'context_adaptation_score': 'score'}, inplace=True)
    scores_df.dropna(subset=['score'], inplace=True)
    
    return scores_df

def plot_scores(df):
    """
    Plots the distribution of context adaptation scores for each model.
    """
    # Add 0.5 to all score values as requested
    df['score'] = df['score'] + 0.5

    # Get the list of unique models to ensure consistent color mapping
    unique_models = sorted(df['model'].unique())
    palette = sns.color_palette("rainbow", n_colors=len(unique_models))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting KDE for each model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        color_index = unique_models.index(model)
        sns.kdeplot(model_df['score'], ax=ax, label=model, color=palette[color_index], fill=True, alpha=0.2)
        # Add mean line
        mean_score = model_df['score'].mean()
        ax.axvline(mean_score, color=palette[color_index], linestyle='--', linewidth=2)

    # --- Styling ---
    # Big titles and labels
    # ax.set_title('Context Adaptation Score by Model', fontsize=50, fontweight='bold')
    ax.set_xlabel('Context Adaptation Score', fontsize=50, fontweight='bold')
    ax.set_ylabel('Density', fontsize=50, fontweight='bold')

    # Bigger ticks
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)

    # Legend in the top right corner inside the plot
    ax.legend(fontsize=25, loc='upper right')

    plt.tight_layout()
    plt.savefig('my_context_adaptation_scores.png')
    plt.show()

def plot_histogram_scores(df):
    """
    Plots a histogram of context adaptation scores with a frequency y-axis.
    """
    # Add 0.5 to all score values as requested
    df['score'] = df['score'] + 0.5

    unique_models = sorted(df['model'].unique())
    palette = sns.color_palette("rainbow", n_colors=len(unique_models))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting histogram for each model
    for model in unique_models:
        model_df = df[df['model'] == model]
        color_index = unique_models.index(model)
        sns.histplot(model_df['score'], ax=ax, label=model, color=palette[color_index], alpha=0.5, bins=20)
        # Add mean line
        mean_score = model_df['score'].mean()
        ax.axvline(mean_score, color=palette[color_index], linestyle='--', linewidth=2)

    # --- Styling ---
    # ax.set_title('Context Adaptation Score by Model (Histogram)', fontsize=50, fontweight='bold')
    ax.set_xlabel('Context Adaptation Score', fontsize=50)
    ax.set_ylabel('Frequency', fontsize=50)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    ax.legend(title='Models', fontsize=30, title_fontsize=30, loc='upper right')

    plt.tight_layout()
    plt.savefig('context_adaptation_histogram.png')
    plt.show()

def plot_all_models_kde_no_mean(df):
    """
    Plots the distribution of context adaptation scores for all models without a mean line.
    """
    # Add 0.5 to all score values as requested
    df['score'] = df['score'] + 0.5

    unique_models = sorted(df['model'].unique())
    palette = sns.color_palette("rainbow", n_colors=len(unique_models))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting KDE for each model
    for model in unique_models:
        model_df = df[df['model'] == model]
        color_index = unique_models.index(model)
        sns.kdeplot(model_df['score'], ax=ax, label=model, color=palette[color_index], fill=True, alpha=0.2)

    # --- Styling ---
    ax.set_title('Context Adaptation Score Distribution (All Models)', fontsize=30, fontweight='bold')
    ax.set_xlabel('Context Adaptation Score (Normalized)', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend(title='Models', fontsize=16, title_fontsize=18, loc='upper right')

    plt.tight_layout()
    plt.savefig('all_models_kde_no_mean.png')
    plt.show()

if __name__ == '__main__':
    RESULTS_FILE = 'results.jsonl'
    MODELS = ['llama3:instruct', 'gemma:2b', 'qwen:4b']
    scores_df = load_scores_from_results(RESULTS_FILE, MODELS)
    all_scores_df = load_scores_from_results(RESULTS_FILE, LLM_SYSTEMS)
    if not scores_df.empty:
        plot_scores(scores_df)
        plot_histogram_scores(scores_df.copy()) # Use copy to avoid modifying the original df
    if not all_scores_df.empty:
        plot_all_models_kde_no_mean(all_scores_df.copy())