import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    palette = sns.color_palette(n_colors=len(unique_models))

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
    ax.set_title('Context Adaptation Score by Model', fontsize=20, fontweight='bold')
    ax.set_xlabel('Context Adaptation Score (Normalized)', fontsize=20)
    ax.set_ylabel('Density', fontsize=16)

    # Bigger ticks
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Legend in the top right corner inside the plot
    ax.legend(title='Models', fontsize=12, title_fontsize=13, loc='upper right')

    plt.tight_layout()
    
    # Ensure the output directory exists
    output_dir = "extra_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'context_adaptation_scores.png')
    
    plt.savefig(save_path)
    plt.show()
    print(f"Saved plot: {save_path}")

if __name__ == '__main__':
    RESULTS_FILE = 'results.jsonl'
    MODELS = ['llama3:instruct', 'gemma:2b', 'qwen:4b']
    scores_df = load_scores_from_results(RESULTS_FILE, MODELS)
    if not scores_df.empty:
        plot_scores(scores_df)