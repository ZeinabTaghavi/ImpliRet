import os
import time
import math
import json
import requests
import multiprocessing
from typing import Dict, List, Tuple
from collections import Counter
import string
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from p_tqdm import p_map
import random
import re
from collections import defaultdict
# for the similarity
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# open AI
from openai import OpenAI
import backoff

random.seed(42)

# Optionally restrict GPU visibility
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

# ------------------------------------------------------------------------------------------------
#
#                                  Gathering the data from Wikidata
#
# ------------------------------------------------------------------------------------------------
# 1.1 Sending the query to Wikidata
# ------------------------------------------------------------------------------------------------
def create_session() -> requests.Session:
    """Create a requests session with retry logic (3 retries, exponential backoff)."""
    retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429,500,502,503,504],
    respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def query_batch(
    session: requests.Session,
    sparql_query: str,
    headers: Dict[str, str],
    offset: int,
    limit: int
) -> List[Dict]:
    """
    Fetch a single paginated batch of SPARQL results from Wikidata.
    """
    url = "https://query.wikidata.org/sparql"
    paged = f"{sparql_query}\nOFFSET {offset}\nLIMIT {limit}" + '''}
      OPTIONAL { ?item wdt:P17 ?country.      ?country rdfs:label ?countryLabel.      FILTER(LANG(?countryLabel)="en") }
      OPTIONAL { ?item wdt:P131 ?adminEntity. FILTER NOT EXISTS { ?adminEntity wdt:P31 wd:Q515. } ?adminEntity rdfs:label ?adminEntityLabel. FILTER(LANG(?adminEntityLabel)="en") }
      OPTIONAL { ?item wdt:P131 ?city. FILTER EXISTS { ?city wdt:P31 wd:Q515. } ?city rdfs:label ?cityLabel. FILTER(LANG(?cityLabel)="en") }
      OPTIONAL { ?item wdt:P276 ?loc.       ?loc rdfs:label ?locationLabel.       FILTER(LANG(?locationLabel)="en") }
      OPTIONAL { ?item wdt:P669 ?street.     ?street rdfs:label ?streetNameLabel.     FILTER(LANG(?streetNameLabel)="en") }
    }
    '''
    try:
        resp = session.get(url, params={"query": paged, "format": "json"}, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get('results', {}).get('bindings', [])
    except Exception as err:
        print(f"Error at offset {offset}: {err}")
        return []


def simplify_binding(binding: Dict) -> Dict:
    """Convert a SPARQL binding dict into a flat record."""
    def val(var: str) -> str:
        return binding.get(var, {}).get('value', '')
    extract_id = lambda uri: uri.rsplit('/', 1)[-1] if '/' in uri else uri
    return {
        'itemQ': extract_id(val('item')),
        'itemLabel': val('itemLabel'),
        'countryQ': extract_id(val('country')),
        'countryLabel': val('countryLabel'),
        'adminEntityQ': extract_id(val('adminEntity')),
        'adminEntityLabel': val('adminEntityLabel'),
        'cityLabel': val('cityLabel'),
        'locationLabel': val('locationLabel'),
        'streetNameLabel': val('streetNameLabel')
    }


def fetch_and_process(
    offset: int,
    query: str,
    headers: Dict[str, str],
    batch_size: int
) -> List[Dict]:
    """Worker: fetch a batch, apply a brief delay, and simplify bindings."""
    sess = create_session()
    try:
        raw = query_batch(sess, query, headers, offset, batch_size)
        # Brief 10ms pause to respect API
        time.sleep(random.uniform(1, 1.5))
        return [simplify_binding(b) for b in raw]
    finally:
        sess.close()


def primal_wiki_query(
    sparql_query: str,
    output_folder: str,
    max_rows: int = 1500000,
    batch_size: int = 5000,
    max_workers: int = None,
    num_splits: int = 10
) -> List[Dict]:
    """
    Execute a full SPARQL pull from Wikidata in parallel, saving split JSONL parts.

    :param sparql_query: SPARQL query without OFFSET/LIMIT clauses
    :param output_folder: Directory to save split JSONL outputs
    :param max_rows: Maximum number of rows to fetch
    :param batch_size: Number of rows per batch
    :param max_workers: Number of parallel workers (defaults to CPU count)
    :param num_splits: Number of output files to split the records into
    :return: List of all simplified records
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Determine worker count
    cpu = multiprocessing.cpu_count()
    workers = min( 10 ,max_workers or cpu)
    print(f"Using {workers} workers for {batch_size}-row batches...")

    headers = {
        "User-Agent": "PrimalWikiQuery/1.0 (example@mail.com)",
        "Accept": "application/json"
    }

    # Build batch offsets
    offsets = list(range(0, max_rows, batch_size))
    args = [(off, sparql_query, headers, batch_size) for off in offsets]

    # Parallel fetch with progress bar
    batches = p_map(lambda a: fetch_and_process(*a), args, num_cpus=workers)

    # Flatten all records
    records = [rec for batch in batches for rec in batch]
    total = len(records)
    print(f"Fetched total {total} records.")

    # Split and save into parts
    chunk_size = math.ceil(total / num_splits)
    for i in range(num_splits):
        start = i * chunk_size
        end = start + chunk_size
        if start >= total:
            break
        part = records[start:end]
        os.makedirs(output_folder, exist_ok=True)
        part_file = os.path.join(output_folder, f"part_{i+1}.jsonl")
        with open(part_file, 'w', encoding='utf-8') as f:
            for rec in part:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"Saved part {i+1} with {len(part)} records to {part_file}")

    return records

# ------------------------------------------------------------------------------------------------
# 1.2. Filtering the records
# ------------------------------------------------------------------------------------------------
def filter_results(
    results: List[Dict],
    length_limit: int = 8
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter out spurious or ambiguous Wikidata records.

    :param results: List of record dicts with keys like 'itemLabel', 'countryLabel', etc.
    :param length_limit: Maximum number of words allowed in itemLabel.
    :return: (final_results, repetitive_samples)
    """
    # Initial filtering
    filtered = [
        item for item in results
        if not re.match(r'^Q\d+', item['itemLabel'])
        and not re.match(r'^Q\d+', item['countryLabel'])
        and not re.match(r'^Q\d+', item['adminEntityLabel'])
        and ':' not in item['itemLabel']
        and 'untitled' not in item['itemLabel'].lower()
        and not any(word.isupper() for word in item['itemLabel'].split())
        and len(item['itemLabel'].split()) < length_limit
        and item['countryQ']
        and item['countryLabel']
        and item['adminEntityQ']
        and item['adminEntityLabel']
    ]

    # Group by itemLabel to detect duplicates across countries
    item_to_countries = defaultdict(set)
    for item in filtered:
        item_to_countries[item['itemLabel']].add(item['countryLabel'])

    # Final separation
    final_results = [
        item for item in filtered
        if len(item_to_countries[item['itemLabel']]) == 1
    ]
    repetitive_samples = [
        item for item in filtered
        if len(item_to_countries[item['itemLabel']]) > 1
    ]

    return final_results, repetitive_samples

# ------------------------------------------------------------------------------------------------
#
#                       Finding the Similarity between the items and the locations
#
# ------------------------------------------------------------------------------------------------
# 2.1. Adding the 'all_location' key to the records
# ------------------------------------------------------------------------------------------------

def add_all_location(
    items: List[Dict]
) -> List[Dict]:
    """
    For each item dict, concatenate non-empty location fields into 'all_location'.
    Fields in order: countryLabel, adminEntityLabel, cityLabel, locationLabel.
    """
    for item in items:
        parts = []
        for key in ('countryLabel', 'adminEntityLabel', 'cityLabel', 'locationLabel'):
            val = item.get(key, '')
            if val:
                parts.append(val)
        item['all_location'] = ' - '.join(parts)
    return items

# ------------------------------------------------------------------------------------------------
# 2.2. Removing the lexical overlap between the items and the locations
# ------------------------------------------------------------------------------------------------

def remove_lexical_overlap(items: List[Dict]) -> List[Dict]:
    """
    Remove any item whose itemLabel shares any token with its all_location string.

    :param items: List of records, each with 'itemLabel' and 'all_location'.
    :return: Filtered list with overlaps removed.
    """
    filtered = []
    for item in items:
        label_tokens = item['itemLabel'].split()
        location_str = item.get('all_location', '')
        # If any token from the label appears in the location string, skip this item
        if any(token in location_str for token in label_tokens):
            continue
        filtered.append(item)
    return filtered

# ------------------------------------------------------------------------------------------------
# 2.3. Finding the Similarity between the items and the locations
# ------------------------------------------------------------------------------------------------

def compute_label_location_similarity_with_scores(
    records: List[Dict],
    model_name: str = 'facebook/contriever',
    batch_size: int = 128,
    output_folder: str = None
) -> List[Dict]:
    """
    Compute cosine similarity between 'itemLabel' and 'all_location' for each record,
    round to 5 decimals, store under 'score', and return the augmented list.
    """

    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts
    
    import gc

    def clean_memory():
        """
        Free up unused memory by running garbage collection and clearing PyTorch's CUDA cache.
        """
        gc.collect()            # Run Python's garbage collector
        torch.cuda.empty_cache()  # Release unoccupied cached memory back to the GPU

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if n_gpus > 1:
        model = DataParallel(model)
    model.to(device).eval()

    similarities = []
    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size)):
            batch = records[i : i + batch_size]
            texts1 = [r['itemLabel'] for r in batch]
            texts2 = [r['all_location'] for r in batch]

            enc1 = tokenizer(texts1, padding=True, truncation=True, return_tensors='pt', max_length=512)
            enc2 = tokenizer(texts2, padding=True, truncation=True, return_tensors='pt', max_length=512)
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}

            out1 = model(**enc1)[0]
            out2 = model(**enc2)[0]
            emb1 = F.normalize(mean_pooling(out1, enc1['attention_mask']), p=2, dim=1)
            emb2 = F.normalize(mean_pooling(out2, enc2['attention_mask']), p=2, dim=1)

            batch_sims = F.cosine_similarity(emb1, emb2, dim=1)
            similarities.extend(batch_sims.cpu().tolist())

    # Attach rounded scores to records
    for rec, score in zip(records, similarities):
        rec['score'] = round(score, 5)

    # Save to JSONL
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        part_file = os.path.join(output_folder, f"wikidata_step_2_scored_records.jsonl")
        with open(part_file, 'w', encoding='utf-8') as fout:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return records


# ------------------------------------------------------------------------------------------------
# 2.4. Filtering the records based on the similarity score and plot the distribution
# ------------------------------------------------------------------------------------------------

def plot_score_distribution_and_filter(
    records: List[Dict],
    threshold: float,
    output_dir: str = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records",
    filename: str = "wikidata_step_2_score_distribution.png"
) -> List[Dict]:
    """
    Plot the histogram of 'score' for each record, draw a vertical line at `threshold`,
    save the figure to the given directory, and return all records whose score < threshold.

    :param records: List of dicts, each containing a 'score' key.
    :param threshold: Score cutoff; records below this will be returned.
    :param output_dir: Directory in which to save the plot PNG.
    :param filename: Filename for the saved plot.
    :return: List of records with score < threshold.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    # Extract scores
    scores = [r['score'] for r in records]

    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True)
    plt.title('Distribution of Similarity Scores', fontsize=14)
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Plot threshold line
    plt.axvline(threshold, color='purple', linestyle='-', label=f'Threshold: {threshold:.3f}')

    # Plot mean & median
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='g', linestyle='--', label=f'Median: {median_score:.3f}')

    # Annotate stats
    below = sum(s < threshold for s in scores)
    stats_text = (
        f"Min: {min(scores):.3f}\n"
        f"Max: {max(scores):.3f}\n"
        f"Std: {np.std(scores):.3f}\n"
        f"Below threshold: {below}\n"
        f"Above threshold: {len(scores) - below}"
    )
    plt.text(0.6, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    low_score_samples = [r for r in records if r['score'] < threshold]

    # Save low scoring samples to file
    output_path = os.path.join(output_dir, "wikidata_step_2_low_scored_records.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in low_score_samples:
            json.dump(record, f)
            f.write('\n')
    # Return records below threshold
    return low_score_samples

# ------------------------------------------------------------------------------------------------
# 
#                           LLM Classification of the remaining records
#
# ------------------------------------------------------------------------------------------------
# 3.1. LLM Classification report
# ------------------------------------------------------------------------------------------------


def annotate_and_summarize_llm(
    low_scored: List[Dict],
    llm_classifications: List[str],
    output_dir: str = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records"
) -> Tuple[List[Dict], Dict[str, int], str]:
    """
    Annotate low‑scored records with LLM classifications, count and plot breakdown by four categories:
      - Generic without Hint
      - Generic with Hint
      - Specific without Hint
      - Specific with Hint
    Save annotated JSONL and a styled summary plot, and return:
      - annotated_records
      - counts dict
      - path to annotated JSONL
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters for each category
    counts = {
        "Generic_NoHint": 0,
        "Generic_WithHint": 0,
        "Specific_NoHint": 0,
        "Specific_WithHint": 0
    }

    # Annotate records and tally
    for rec, cls in zip(low_scored, llm_classifications):
        parts = [p.strip() for p in cls.split("-", 1)]
        gs = parts[0]                 # "Generic" or "Specific"
        hint = parts[1] if len(parts) > 1 else ""
        rec["LLM_GS_Clf"] = gs
        rec["LLM_Hint_Clf"] = hint

        key = f"{gs}_{'WithHint' if hint else 'NoHint'}"
        if key in counts:
            counts[key] += 1

    # Save annotated records
    annotated_file = os.path.join(output_dir, "wikidata_step_3_annotated.jsonl")
    with open(annotated_file, "w", encoding="utf-8") as f:
        for rec in low_scored:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Prepare data for plotting
    labels = [
        "Generic (no hint)",
        "Generic (hint)",
        "Specific (no hint)",
        "Specific (hint)"
    ]
    values = [
        counts["Generic_NoHint"],
        counts["Generic_WithHint"],
        counts["Specific_NoHint"],
        counts["Specific_WithHint"]
    ]

    # Create a more polished bar chart
    plt.figure(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
    max_val = max(values) if values else 0

    # Annotate bar values
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y + max_val * 0.01,
            f"{y}",
            ha='center', va='bottom',
            fontsize=10
        )

    plt.title("LLM Generic/Specific & Hint Classification", fontsize=16)
    plt.ylabel("Number of Records", fontsize=12)
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(output_dir, "wikidata_step_3_llm_classification_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save records without hints
    no_hint_file = os.path.join(output_dir, "wikidata_step_3_low_score_specific_no_hint.jsonl")
    low_scored_specific_no_hint = []
    with open(no_hint_file, "w", encoding="utf-8") as f:
        for record in low_scored:
            if record["LLM_Hint_Clf"] == "" and record["LLM_GS_Clf"] == "Specific":
                record["LLM_Hint_Clf"] = "No Hint"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                low_scored_specific_no_hint.append(record)


    return low_scored_specific_no_hint, counts, annotated_file

# ------------------------------------------------------------------------------------------------
# 
#                           Counting the number of records in C4 with WIMDB
#
# ------------------------------------------------------------------------------------------------
def count_for_sample(sample):
    return count_documents_containing_phrases(
        "c4",
        [sample['itemLabel']],
        es=es,
        all_phrases=True
    )


def plot_c4_distribution_and_filter(
    records: List[Dict],
    min_count: int,
    max_count ,
    upper_bound: int,
    output_dir: str = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records",
    fig_name: str = "wikidata_step_6_c4_distribution.png",
    filtered_name: str = "wikidata_step_5_c4_filtered.jsonl"
) -> List[Dict]:
    """
    • Plot a professional histogram of capped 'c4_count', save it.
    • Save filtered records (min_count ≤ c4_count ≤ max_count) to JSONL.
    • Print 5 samples with highest c4_count.
    • Return the filtered list.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract and cap counts
    raw_counts = np.array([rec.get('c4_count', 0) for rec in records])
    capped = np.minimum(raw_counts, upper_bound)

    # Prepare histogram bins
    bins = np.arange(0, upper_bound + 2) - 0.5

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(capped, bins=20, color="#4C72B0", edgecolor="black", alpha=0.8)

    ax.set_title("C4 Count Distribution (capped)", fontsize=16, weight='bold')
    ax.set_xlabel(f"C4 Count (≥{upper_bound} in last bin)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Draw filter lines
    if max_count == "max":
        ax_max = 5000
    else: 
        ax_max = max_count
    ax.axvline(min_count, color="#C44E52", linestyle='--', linewidth=2, label=f"Min = {min_count}")
    ax.axvline(ax_max, color="#55A868", linestyle='--', linewidth=2, label=f"Max = {ax_max}")

    # Annotate bars
    total = len(raw_counts)
    for patch in ax.patches:
        height = patch.get_height()
        if height == 0:
            continue
        pct = height / total * 100
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + total * 0.005,
            f"{int(height)}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9
        )

    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()

    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {fig_path}")

    # Filter and save JSONL
    if max_count == "max":
        filtered = [rec for rec in records if min_count <= rec.get('c4_count', 0)]
    else:
        filtered = [rec for rec in records if min_count <= rec.get('c4_count', 0) <= max_count]
    filtered_path = os.path.join(output_dir, filtered_name)
    with open(filtered_path, 'w', encoding='utf-8') as f:
        for rec in filtered:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')
    print(f"Filtered records saved to: {filtered_path} ({len(filtered)} items)")

    # Print top 5 samples by raw c4_count
    top5 = sorted(records, key=lambda r: r.get('c4_count', 0), reverse=True)[:5]
    print("\nTop 5 records by C4 count:")
    for i, rec in enumerate(top5, 1):
        print(f"{i}. {rec.get('itemLabel', 'N/A')} — c4_count: {rec.get('c4_count', 0)}")

    return filtered



def plot_top_itemlabel_words(
    records: List[Dict],
    top_n: int = 20,
    output_dir: str = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records",
    filename: str = "wikidata_step_5_top_itemlabel_words.png"
) -> None:
    """
    Count the most frequent words in `itemLabel` across records,
    plot the top N as a bar chart, and save the figure.

    :param records: List of dicts, each containing an 'itemLabel' key.
    :param top_n: How many top words to display.
    :param output_dir: Directory in which to save the plot.
    :param filename: Filename for the saved plot.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    # Build a normalized list of all words
    all_words = []
    for rec in records:
        label = rec.get('itemLabel', '')
        # Remove punctuation, lowercase, split on whitespace
        cleaned = label.translate(str.maketrans('', '', string.punctuation)).lower()
        all_words.extend(cleaned.split())

    # Count frequencies
    counter = Counter(all_words)
    most_common = counter.most_common(top_n)
    words, counts = zip(*most_common)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(words, counts, color='#4C72B0', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top {top_n} Most Frequent Words in itemLabel", fontsize=14, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel("Word", fontsize=12)
    plt.tight_layout()

    # Annotate counts above bars
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + max(counts)*0.01,
                 f"{h}", ha='center', va='bottom', fontsize=9)

    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top words plot saved to: {save_path}")

# ------------------------------------------------------------------------------------------------
# 
#                    Extracting the unique samples / Adding Type of location
#
# ------------------------------------------------------------------------------------------------

def extract_unique_samples(
    input_file: str,
    output_file: str
) -> List[Dict]:
    """
    Load records from a JSONL file, keep only the first occurrence of each 'itemLabel',
    save the unique samples to another JSONL file, and return the list of unique samples.

    :param input_file: Path to the source JSONL file with possibly duplicate itemLabel entries.
    :param output_file: Path where the unique-samples JSONL will be written.
    :return: List of unique sample dicts.
    """
    # Load all records
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)

    # Filter unique by itemLabel
    seen = set()
    unique = []
    for rec in records:
        label = rec.get('itemLabel')
        if label and label not in seen:
            seen.add(label)
            unique.append(rec)
        else:
            # Remove the previously added record with the same label
            unique = [item for item in unique if item['itemLabel'] != label]

    # Report and save
    print(f"Found {len(unique)} unique samples out of {total} total records")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in unique:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')
    print(f"Unique samples saved to {output_file}")

    return unique

def plot_country_distribution(
    records: List[Dict],
    output_dir: str = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records",
    filename: str = "wikidata_step_7_unique_country_histogram.png"
) -> None:
    """
    Plot and save a histogram of the 'countryLabel' field across the given records.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    # Count country frequencies
    countries = [rec.get('countryLabel', '') for rec in records]
    counter = Counter(countries)
    labels, counts = zip(*counter.most_common())

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='#4C72B0', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Countries in Unique Samples', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Country', fontsize=12)
    plt.tight_layout()

    # Annotate bars
    max_count = max(counts)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + max_count*0.01,
                 f"{h}", ha='center', va='bottom', fontsize=9)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Country distribution plot saved to: {save_path}")

# ------------------------------------------------------------------------------------------------
#                           Adding Type of location
# ------------------------------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def fetch_instance_of_labels_once(
    item_q: str,
    session: requests.Session,
    endpoint: str = "https://query.wikidata.org/sparql"
) -> List[str]:
    """
    Query Wikidata in a single request for all P31 values of item_q
    and return their English labels.
    """
    sparql = f"""
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?classLabel WHERE {{
  wd:{item_q} wdt:P31 ?class .
  ?class rdfs:label ?classLabel .
  FILTER(LANG(?classLabel) = "en")
}}
"""
    resp = session.get(
        endpoint,
        params={"query": sparql, "format": "json"},
        headers={"User-Agent": "InstanceLabelFetcher/2.0"},
        timeout=30
    )
    resp.raise_for_status()
    bindings = resp.json().get("results", {}).get("bindings", [])
    return [b["classLabel"]["value"] for b in bindings if "classLabel" in b]

def annotate_with_instance_labels_once(
    input_jsonl: str,
    output_jsonl: str
) -> None:
    """
    Read records from input_jsonl, fetch all P31 labels in one SPARQL call per item,
    replace 'instanceOf' with the list of labels, and write to output_jsonl.
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    session = create_session()

    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Annotating records with instance labels"):
            rec = json.loads(line)
            item_q = rec.get("itemQ")
            if item_q:
                labels = fetch_instance_of_labels_once(item_q, session)
            else:
                labels = []
            # assign the labels directly
            rec["instanceOf"] = labels

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    session.close()
# ------------------------------------------------------------------------------------------------
#                           GPT-search checking replication of the locations
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
def fetch_unique_location_with_gpt(
    input_jsonl: str,
    all_output_jsonl: str,
    unique_output_jsonl: str,
    model: str = "gpt-4o-search-preview-2025-03-11",
    max_retries: int = 3,
    request_delay: float = 0.5,
    target_unique: int = 100,
) -> None:
    """
    Query GPT‑4o for location info **while preserving country balance**.

    Workflow
    --------
    1. Load `input_jsonl` and build a dict:  countryLabel ➔ deque[record].
       (Records whose countryLabel is missing fall under key `'Unknown'`.)

    2. Iterate round‑robin across countries:
          first item from country‑1, first from country‑2, …,
          second item from country‑1, second from country‑2, …

       • Call GPT for each record’s `itemLabel`.
       • Append GPT’s answer to the record under `'gpt_location'`.
       • Collect every processed record in `all_records`.
       • Collect into `unique_records` only when `len(gpt_location)==1`.

    3. Stop as soon as `len(unique_records)` reaches `target_unique`
       **or** when every country list is exhausted.

    4. Write:
         • all_records          ➔ `all_output_jsonl`
         • unique_records (≤60) ➔ `unique_output_jsonl`

    The JSONL files are now streamed and flushed to disk every 10 processed records to avoid data loss.
    """
    from collections import defaultdict, deque

    openai_api_key = "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"
    client = OpenAI(api_key=openai_api_key)

    prompt_prefix = (
        "You will get a name of an entity, which could be a location, landmark, place, ...\n"
        "You should perform a search and check where that place is located at returning the city and country name. "
        "If there are multiple matches (could be close matches) you should return them all.\n\n"
        "In the end, return a json dictionary where the key is the full name of the found places and the value "
        "is the location of it: city, province/state, country. Don't return anything after."
    )

    @backoff.on_exception(backoff.expo, Exception, max_tries=max_retries)
    def gpt_call(label: str) -> dict:
        """Call GPT‑4o, falling back if SDK lacks web_search_options."""
        messages = [{"role": "user", "content": f"{prompt_prefix}\n\nEntity: {label}"}]
        try:
            completion = client.chat.completions.create(
                model=model,
                web_search_options={},
                messages=messages,
            )
        except TypeError:
            completion = client.chat.completions.create(model=model, messages=messages)

        content = completion.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', content, re.S)
            return json.loads(m.group(0)) if m else {}

    # --------------------------------------------------
    # 1. Load records grouped by country
    # --------------------------------------------------
    from tqdm import tqdm  # local import to avoid polluting global scope
    country_to_q = defaultdict(deque)

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            country = rec.get("countryLabel") or "Unknown"
            country_to_q[country].append(rec)

    country_keys = list(country_to_q.keys())
    num_countries = len(country_keys)
    print(f"Round‑robin across {num_countries} countries, aiming for {target_unique} unique results.")

    # --------------------------------------------------
    # Open output files for streaming; flush every 10 records
    # --------------------------------------------------
    os.makedirs(os.path.dirname(all_output_jsonl), exist_ok=True)
    f_all = open(all_output_jsonl, "w", encoding="utf-8")
    f_uni = open(unique_output_jsonl, "w", encoding="utf-8")

    # --------------------------------------------------
    # 2. Round‑robin processing
    # --------------------------------------------------
    all_records, unique_records = [], []
    idx = 0
    from tqdm import tqdm as _tqdm
    with _tqdm(total=target_unique, desc="Collecting unique GPT matches") as pbar:
        while len(unique_records) < target_unique:
            # Break when every deque is empty
            if not any(country_to_q[c] for c in country_keys):
                break

            country = country_keys[idx % num_countries]
            idx += 1
            if not country_to_q[country]:
                continue  # this country exhausted, skip

            rec = country_to_q[country].popleft()
            label = rec.get("itemLabel", "")
            if not label:
                continue

            result = gpt_call(label)
            rec["gpt_location"] = result
            all_records.append(rec)
            # Write to 'all' file and flush every 10 records
            f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(all_records) % 10 == 0:
                f_all.flush()
            if len(result) == 1:
                unique_records.append(rec)
                # Write to 'unique' file and flush every 10 unique records
                f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if len(unique_records) % 10 == 0:
                    f_uni.flush()
                pbar.update(1)

            time.sleep(request_delay)

    print(f"Processed {len(all_records)} records; obtained {len(unique_records)} unique results.")

    # Close streamed output files
    f_all.close()
    f_uni.close()

    print(
        f"Saved {len(all_records)} total GPT‑answered records → {all_output_jsonl}\n"
        f"Saved {len(unique_records)} unique‑match records → {unique_output_jsonl}"
    )
# ------------------------------------------------------------------------------------------------
# 
#                                           Main Function
#
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?item ?itemLabel ?country ?countryLabel ?adminEntity ?adminEntityLabel ?cityLabel ?locationLabel ?streetNameLabel WHERE {
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
      { SELECT DISTINCT ?item WHERE {
          { ?item p:P31 ?s0. ?s0 (ps:P31/(wdt:P279*)) wd:Q17350442. } UNION
          { ?item p:P31 ?s1. ?s1 (ps:P31/(wdt:P279*)) wd:Q4895393. } UNION
          { ?item p:P31 ?s2. ?s2 (ps:P31/(wdt:P279*)) wd:Q464980. } UNION
          { ?item p:P31 ?s3. ?s3 (ps:P31/(wdt:P279*)) wd:Q3918. } UNION
          { ?item p:P31 ?s4. ?s4 (ps:P31/(wdt:P279*)) wd:Q33506. }
      }
    """
    
    print('1. Gathering the data from Wikidata')
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_Query_Output"):
        print("1.1. Wikipedia does not exist, sending the query to Wikidata.")
        records = primal_wiki_query(
            sparql_query=QUERY,
            output_folder="./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_Query_Output",
            max_rows=1_500_000,
            batch_size=5_000,
            max_workers=40,
        num_splits=10
        )

    else: 
        print("1.1. Have already done this step.")

    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_2_low_scored_records.jsonl"):
        records = []
        # Load and concatenate all parts
        output_folder = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_Query_Output"
        for file in os.listdir(output_folder):
            if file.startswith("part_") and file.endswith(".jsonl"):
                file_path = os.path.join(output_folder, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        records.append(json.loads(line))
        print(f"Loaded {len(records)} records from existing files \n")

        print("1.2. Filtering the records")
        final_results, repetitive_samples = filter_results(records)
        print(f"Filtered {len(repetitive_samples)} repetitive samples \n")
        print(f"Final {len(final_results)} records \n")


        print("2. Finding the Similarity between the items and the locations")
        print("2.1. Adding the 'all_location' key to the records")
        final_results = add_all_location(final_results)
        print(f"Final {len(final_results)} records \n")

        print("2.2. Removing the lexical overlap between the items and the locations")
        final_results = remove_lexical_overlap(final_results)
        print(f"Final {len(final_results)} records \n")

        print("2.3. Finding the Similarity between the items and the locations")
        scored_records = compute_label_location_similarity_with_scores(final_results, 
                                                                    batch_size=512, 
                                                                    output_folder = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records")
        print(f"Final {len(scored_records)} records \n")

        print("2.4. Filtering the records based on the similarity score and plot the distribution")
        threshold = 0.25
        low_score_samples = plot_score_distribution_and_filter(
            records=scored_records,
            threshold=threshold,
            output_dir="./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records",
        )
        print(f"{len(low_score_samples)} records have score below {threshold} \n")
    
    else:
        print("1.1. Have already done this step. \n")

    print("2. Finding the Similarity between the items and the locations")
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_3_LLM_Generic_Specific_Classification.jsonl"):

        low_scored_records = []
        output_folder = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_2_low_scored_records.jsonl"
        with open(output_folder, 'r', encoding='utf-8') as f:
            for line in f:
                low_scored_records.append(json.loads(line))
        print(f"Loaded {len(low_scored_records)} records from existing file \n")

        print("3. LLM Classification of the remaining records")
        # LLM
        import Semantic_Both_1_Wikidt_Generic_Specific_Classification
        num_gpus = torch.cuda.device_count()
        Semantic_Both_1_Wikidt_Generic_Specific_Classification.main(num_gpus=num_gpus, model_name="meta-llama/Llama-3.3-70B-Instruct")

    else: 
        print("2.1. Have already done this step. \n")
        print("3. LLM Classification of the remaining records")
        print("3.1. Have already done this step. \n")


    print("4. Loading LLM Classification Results")
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_3_low_score_specific_no_hint.jsonl"):
        low_scored_records = []
        output_folder = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_2_low_scored_records.jsonl"
        with open(output_folder, 'r', encoding='utf-8') as f:
            for line in f:
                low_scored_records.append(json.loads(line))

        llm_cls = []
        with open("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_3_LLM_Generic_Specific_Classification.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                llm_cls.append(json.loads(line))

        # Annotate, summarize, plot, and save
        low_scored_specific_no_hint, summary_counts, annotated_path = annotate_and_summarize_llm(low_scored_records, llm_cls)
        print("Annotated file saved at:", annotated_path)
        print("Summary counts:", summary_counts)

    else: 
        print("4.1. Have already done this step. \n")   

    print("5. Getting counts in C4 with WIMDB")
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_4_LLM_known_c4_counts.jsonl"):
        low_scored_specific_no_hint = []
        with open("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_3_low_score_specific_no_hint.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                low_scored_specific_no_hint.append(json.loads(line))
        print(f"Loaded {len(low_scored_specific_no_hint)} specific records without hints")


        from wimbd.es import es_init, get_indices
        from wimbd.es import count_documents_containing_phrases

        es = es_init("./wimbd_keys/es_config_8.yml")
        
        # Get number of available CPUs
        num_cpus = multiprocessing.cpu_count()
        print(f"Number of CPUs available: {num_cpus}")
        start_time = time.time()
        c4_counts = p_map(
            count_for_sample,
            low_scored_specific_no_hint,
            num_cpus=min(num_cpus, 32)
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        output_filename = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_Unique_Samples.jsonl"

        with open(output_filename, 'w') as f:
            for sample, count in zip(low_scored_specific_no_hint, c4_counts):
                sample_with_count = sample.copy()
                sample_with_count['c4_count'] = count
                json.dump(sample_with_count, f)
                f.write('\n')

        print(f"Counts saved to {output_filename}")

    else:
        print("5.1. Have already done this step. \n")

    print("6. Plotting the distribution of C4 counts and filtering the records")

    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_5_c4_filtered.jsonl"):
        # Load your c4-counted records
        input_file = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_4_LLM_known_c4_counts.jsonl"
        records_with_c4 = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records_with_c4.append(json.loads(line))

        min_count = 0
        max_count = "max" # number or "max"
        upper_bound = 5000
        # Plot with capping at 20, filter between 1 and 10
        selected = plot_c4_distribution_and_filter(
            records=records_with_c4,
            min_count=min_count,
            max_count=max_count,
            upper_bound=upper_bound
        )

        plot_top_itemlabel_words(selected, top_n=30)
        print(f"{len(selected)} records have c4_count between {min_count} and {max_count}")

    print("7. Final Unique Samples and location type\n")
    print("7.1. Extracting the unique samples")
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7_unique_samples_with_c4_count.jsonl"):
        # Load your c4-counted records
        input_file = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_5_c4_filtered.jsonl"
        output_file = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7_unique_samples_with_c4_count.jsonl"
        unique_samples = extract_unique_samples(input_file, output_file)
        plot_country_distribution(unique_samples)

    else:
        print("7.1. Have already done this step. \n")

    print("7.2. Adding Type of location to the unique samples")
    if not os.path.exists("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7__unique_samples.jsonl"):
        annotate_with_instance_labels_once(
                input_jsonl="./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7_unique_samples_with_c4_count.jsonl",
                output_jsonl="./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7__unique_samples.jsonl"
            )
    else:
        print("7.2. Have already done this step. \n")
  
    # 8. Filtering unique samples using GPT‑4o search
    print("8. Filtering unique samples using GPT‑4o search")
    gpt_all_path = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_8_gpt_all_locations.jsonl"
    gpt_unique_path = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_8_gpt_unique_locations.jsonl"

    # if not (os.path.exists(gpt_all_path) and os.path.exists(gpt_unique_path)):
    fetch_unique_location_with_gpt(
        input_jsonl="./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_7__unique_samples.jsonl",
        all_output_jsonl=gpt_all_path,
        unique_output_jsonl=gpt_unique_path,
        target_unique=100,
    )
    # else:
    #     print("8.1. GPT output files already exist.\n")