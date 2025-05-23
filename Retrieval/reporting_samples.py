import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import path
from math import ceil
import numpy as np
import pandas as pd

def print_results(results):
    print(f"Year: {results['year']['appears']} {results['year']['number']}")
    print(f"Non-year: {results['non_year_numbers']['appears']} {results['non_year_numbers']['number']}")
    

# Plot multiple results in a grid of subplots
def plot_multiple_results(results_list, names, cols=3):
    import math
    # Calculate number of rows needed
    n = len(results_list)
    rows = math.ceil(n / cols)
    # Create subplots grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    # Plot each result in its own subplot
    for idx, (results, title) in enumerate(zip(results_list, names)):
        ax = axes[idx]
        # Plot word location distributions as histograms with seaborn
        for w, stats in results['words_repeating'].items():
            locations = stats.get('locations', [])
            if locations:
                # Plot word position counts as histogram
                sns.histplot(locations, discrete=True, ax=ax, stat='count', alpha=0.5, label=w)
        ax.set_title(title)
        ax.set_xlabel('Word position')
        ax.set_ylabel('Frequency')
        ax.legend()
    # Remove any unused subplots
    for ax in axes[len(results_list) :]:
        fig.delaxes(ax)
    plt.tight_layout()
    return fig
# Analysis utility functio


def analyze_texts(texts, target_words, year_range):
    # Initialize word statistics
    word_stats = {w: {'appears': 0, 'number': 0, 'locations': []} for w in target_words}
    # Initialize numeric position lists
    year_locations = []
    non_year_locations = []
    year_appears = 0
    year_number = 0
    non_year_appears = 0
    non_year_number = 0
    for text in texts:
        found_year = False
        found_non_year = False
        text_lower = text.lower()
        test_token_list = text_lower.split()
        # Track numeric token positions
        for pos, token in enumerate(test_token_list, start=1):
            for num_str in re.findall(r"\d+", token):
                n = float(num_str)
                if year_range[0] <= n <= year_range[1]:
                    year_locations.append(pos)
                else:
                    non_year_locations.append(pos)
        # Count word occurrences
        for w in target_words:
            count_w = text_lower.count(w.lower())
            if count_w > 0:
                word_stats[w]['appears'] += 1
                word_stats[w]['number'] += count_w
                position_sum = []
                for i in range(len(test_token_list)):
                    if w.lower() in test_token_list[i]:
                        position_sum.append(i+1)
                assert len(position_sum) == count_w, f"{w} {count_w} {position_sum}: {text_lower.split()}"
                word_stats[w]['locations'].extend(position_sum)
        # Find all numeric substrings
        for num_str in re.findall(r"\d+", text):
            n = int(num_str)
            if year_range[0] <= n <= year_range[1]:
                year_number += 1
                found_year = True
            else:
                non_year_number += 1
                found_non_year = True
        # Update appears counters
        if found_year:
            year_appears += 1
        if found_non_year:
            non_year_appears += 1

    # Compute average distances between each pair of target words
    from itertools import combinations
    avg_distances = {}
    for w1, w2 in combinations(target_words, 2):
        locs1 = word_stats[w1]['locations']
        locs2 = word_stats[w2]['locations']
        # Compute all pairwise absolute distances
        distances = [abs(p2 - p1) for p1 in locs1 for p2 in locs2]
        if distances:
            avg_distances[(w1, w2)] = sum(distances) / len(distances)
        else:
            avg_distances[(w1, w2)] = 0
    # Include average distances in results

    return {
        'words_repeating': word_stats,
        'year': {'appears': year_appears, 'number': year_number, 'locations': year_locations},
        'non_year_numbers': {'appears': non_year_appears, 'number': non_year_number, 'locations': non_year_locations},
        'average_distances': avg_distances
    }



def plot_multiple_number_results(results_list, names, cols=3):
    import math
    threshold = 1000
    # Calculate number of rows needed
    rows = math.ceil(len(results_list) / cols)
    # Create subplots grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    # Plot each result in its own subplot
    for idx, (results, title) in enumerate(zip(results_list, names)):
        ax = axes[idx]
        # Year-range number positions with thresholding
        year_locs = results['year'].get('locations', [])
        if year_locs:
            bins_year = range(1, max(year_locs) + 2)
            n_year, bins_used_year, patches_year = ax.hist(year_locs, bins=bins_year, alpha=0.5, label='year numbers', color='red')
            # Cap and annotate counts above threshold
            for count, patch in zip(n_year, patches_year):
                if count > threshold:
                    patch.set_height(threshold)
                    ax.text(patch.get_x() + patch.get_width()/2, threshold, str(int(count)), ha='center', va='bottom', color='red')
        else:
            # No year numbers: plot zero bar for legend
            ax.hist([], bins=[0,1], alpha=0.5, label='year numbers', color='red')
        # Non-year number positions with thresholding
        non_year_locs = results['non_year_numbers'].get('locations', [])
       
        if non_year_locs:
            bins_non_year = range(1, max(non_year_locs) + 2)
            n_non, bins_used_non, patches_non = ax.hist(non_year_locs, bins=bins_non_year, alpha=0.5, label='other numbers', color='blue')
            # Cap and annotate counts above threshold
            for count, patch in zip(n_non, patches_non):
                if count > threshold:
                    patch.set_height(threshold)
                    ax.text(patch.get_x() + patch.get_width()/2, threshold, str(int(count)), ha='center', va='bottom', color='blue')
        else:
            # No other numbers: plot zero bar for legend
            ax.hist([], bins=[0,1], alpha=0.5, label='other numbers', color='blue')
        ax.set_title(title)
        ax.set_xlabel('Token position')
        ax.set_ylabel('Frequency')
        max_height = min(threshold + 10, max(max(n_year) if year_locs else 0, max(n_non) if non_year_locs else 0)+ 10)
        ax.set_ylim(0, max_height)  # Set y-axis limits from 0 to min(threshold+100, max value)
        ax.legend()
    # Remove unused subplots
    for ax in axes[len(results_list):]:
        fig.delaxes(ax)
    plt.tight_layout()
    return fig



def plot_distance_distribution(results_list, names, cols=3):
    import math
    import seaborn as sns
    # Calculate rows
    rows = math.ceil(len(results_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    for idx, (results, title) in enumerate(zip(results_list, names)):
        ax = axes[idx]
        # Extract average distances
        values = list(results.get('average_distances', {}).values())
        keys = list(results.get('average_distances', {}).keys())
        if values:
            # Plot each value as a column
            x_positions = range(len(values))
            ax.bar(x_positions, values, color='purple', width=0.8)
            
            # Add labels for each bar
            for i, (x, y, key) in enumerate(zip(x_positions, values, keys)):
                ax.text(x, y, f"{key}", ha='center', va='bottom')
                
            # Set x-axis limits with some padding
            ax.set_xlim(-0.5, len(values)-0.5)
        ax.set_title(title)
        ax.set_xlabel('Average distance')
        ax.set_ylabel('Count')
    # Remove unused axes
    for ax in axes[len(results_list):]:
        fig.delaxes(ax)
    plt.tight_layout()
    return fig


# Plot per-word distributions across ks and analysis types
def plot_per_word_distributions(results_matrix, names_matrix, target_words, k_list, cols=4):
    import math
    import seaborn as sns
    # Total subplots: rows = len(k_list) * len(target_words)
    rows = len(k_list) * len(target_words)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2))
    axes = axes.flatten()
    idx = 0
    # Iterate over each word and each k and analysis type
    for w in target_words:
        for i, k in enumerate(k_list):
            for j, results in enumerate(results_matrix[i]):
                ax = axes[idx]
                # Extract locations for this word
                locs = results['words_repeating'][w]['locations']
                if locs:
                    sns.histplot(locs, discrete=True, ax=ax, stat='count', alpha=0.5, label=w)
                else:
                    # Empty bar for legend
                    ax.bar([], [], alpha=0.5, label=w)
                # Title: word + analysis name
                title = names_matrix[i][j]
                ax.set_title(f"{w}: {title}")
                ax.set_xlabel('Word position')
                ax.set_ylabel('Frequency')
                ax.legend()
                idx += 1
    # Remove any extra axes
    for a in axes[idx:]:
        fig.delaxes(a)
    plt.tight_layout()
    return fig


# Centralized reporting function
def reporting_figures(dataset_file_path, retrieval_data_file_path, k_list, target_words, output_folder, year_range):
    # Prepare output directory
    os.makedirs(output_folder, exist_ok=True)
    base = path.splitext(path.basename(retrieval_data_file_path))[0]
    # Load dataset
    data = []
    with open(dataset_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    # Build all_data and gold_index
    all_data = []
    gold_index = []
    for i, item in enumerate(data):
        gold_index.append(len(all_data))
        all_data.append(item['user_response'])
        all_data.extend(item['extra_samples'])
    # Load retrieval results
    retrieval_data = []
    with open(retrieval_data_file_path, 'r') as f:
        for line in f:
            retrieval_data.append(json.loads(line))
    # Multi-k analysis
    results_matrix = []
    names_matrix = []
    for k in k_list:
        # Collect texts
        all_top_texts = []
        all_bottom_texts = []
        for row in retrieval_data:
            sorted_scores = sorted(row['index_score_tuple_list'], key=lambda x: x[1], reverse=True)
            top_idxs = [idx for idx, _ in sorted_scores[:k]]
            bottom_idxs = [idx for idx, _ in sorted_scores[-k:]]
            all_top_texts.extend(all_data[i] for i in top_idxs)
            all_bottom_texts.extend(all_data[i] for i in bottom_idxs)
        # Analyze
        r_top = analyze_texts(all_top_texts, target_words, year_range)
        r_bottom = analyze_texts(all_bottom_texts, target_words, year_range)
        # Gold presence
        present = set()
        for row in retrieval_data:
            sorted_scores = sorted(row['index_score_tuple_list'], key=lambda x: x[1], reverse=True)
            top_idxs = {idx for idx, _ in sorted_scores[:k]}
            present |= top_idxs & set(gold_index)
        all_gold = set(gold_index)
        missing = all_gold - present
        r_gp = analyze_texts([all_data[i] for i in present], target_words, year_range)
        r_gm = analyze_texts([all_data[i] for i in missing], target_words, year_range)
        results_matrix.append([r_top, r_bottom, r_gp, r_gm])
        names_matrix.append([f"Top-{k}", f"Bottom-{k}", f"Gold present (k={k})", f"Gold missing (k={k})"])
    # Flatten
    flat_results = [res for row in results_matrix for res in row]
    flat_names = [name for row in names_matrix for name in row]
    # Plot and save each figure
    fig1 = plot_multiple_results(flat_results, flat_names, cols=4)
    fig1.savefig(path.join(output_folder, f"{base}_word_dist.png"))
    fig2 = plot_multiple_number_results(flat_results, [f"{n} numbers" for n in flat_names], cols=4)
    fig2.savefig(path.join(output_folder, f"{base}_number_dist.png"))
    fig3 = plot_distance_distribution(flat_results, [f"{n} avg distances" for n in flat_names], cols=4)
    fig3.savefig(path.join(output_folder, f"{base}_avg_dist.png"))
    fig4 = plot_per_word_distributions(results_matrix, names_matrix, target_words, k_list, cols=4)
    fig4.savefig(path.join(output_folder, f"{base}_per_word.png"))


# Optionally, add CLI entry point for quick reporting
if __name__ == '__main__':
    reporting_figures(
        'Dataset_Generation/Data/A_Multi.jsonl',
        'MetatagIndexing/Experiments/Retrieval/Results/A_Multi_reasonir_index.jsonl',
        [100, 200, 300],
        ['model','cost','bought'],
        'MetatagIndexing/Experiments/Retrieval/reports/figures',
        (2012, 2024)
    )
