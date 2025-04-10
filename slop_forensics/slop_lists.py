import os
import json
import logging
import string
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
from tqdm import tqdm
from wordfreq import word_frequency

# For phrase extraction:
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from collections import Counter
from functools import partial
from multiprocessing import Pool

# Local imports from your package:
from . import config
from .utils import (
    load_json_file,
    save_list_one_item_per_line,
    save_json_file,
    sanitize_filename,
    normalize_text,
    extract_words,
    load_jsonl_file,
    setup_logging,
)
from .analysis import (
    filter_mostly_numeric,
    merge_plural_possessive_s,
    filter_stopwords,
    filter_common_words,
    analyze_word_rarity,
    find_over_represented_words,
    find_zero_frequency_words,
    STOP_WORDS
)

logger = logging.getLogger(__name__)

# Make sure NLTK data is available:
#   nltk.download("punkt")
#   nltk.download("stopwords")

stop_words_nltk = set(stopwords.words('english'))

###############################################################################
# Additional Functions for Phrase Extraction
###############################################################################


def has_sentence_end_in_the_middle(phrase: str) -> bool:
    """
    Returns True if there is . ? or ! in the middle of 'phrase'
    (not counting the very last character).
    """
    s = phrase.strip()
    if len(s) <= 2:
        return False
    for c in ".?!":
        if c in s[:-1]:  # check everything except the last character
            return True
    return False


def save_list_jsonl(items, filename: str):
    """
    Writes each element of 'items' as JSON on its own line (JSONL).
    Example: items = [(phrase, freq), (phrase2, freq2), ...].
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in items:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Saved phrase data to: {filename}")
    except Exception as e:
        logger.error(f"Error saving phrases file {filename}: {e}")


def extract_ngrams_cleaned(texts_list: List[str], n: int, top_k: int) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Extract the top_k most frequent n-grams from a corpus after a "cleaning" step:
       - Tokenize
       - Keep alpha tokens only
       - Exclude stopwords
       - Lowercase
    Returns a list of (ngram_tuple, frequency).
    """
    ngram_counts = Counter()
    logger.info(f"Extracting cleaned {n}-grams from {len(texts_list)} combined texts...")

    for text in tqdm(texts_list, desc=f"Extracting {n}-grams", leave=False):
        if not isinstance(text, str) or not text.strip():
            continue

        tokens = [
            w.lower()
            for w in word_tokenize(text)
            if w.isalpha() and w.lower() not in stop_words_nltk
        ]
        if len(tokens) >= n:
            ngram_counts.update(ngrams(tokens, n))

    # Return the top_k most common ngrams
    return ngram_counts.most_common(top_k)


def process_one_text_for_substrings(
    text: str,
    top_ngrams_set: set,
    n: int
) -> Counter:
    """
    Worker function for multiprocessing. For each text:
      1) Tokenize text with naive offsets.
      2) Build a list of cleaned tokens + map to the offsets in the original text.
      3) For each n-length window in cleaned_tokens, if it matches something
         in top_ngrams_set, retrieve the exact substring from the original text.
      4) Return a Counter of substring -> frequency for this single text.
    """
    local_counter = Counter()

    if not isinstance(text, str) or not text.strip():
        return local_counter

    # Naive tokenization with offsets:
    tokens_with_spans = []
    offset = 0
    raw_tokens = word_tokenize(text)

    for tk in raw_tokens:
        idx = text.find(tk, offset)
        if idx == -1:
            continue
        start_idx = idx
        end_idx = idx + len(tk)
        tokens_with_spans.append((tk, start_idx, end_idx))
        offset = end_idx

    # Build cleaned_tokens + offset map
    cleaned_tokens = []
    char_index_map = []
    for (tk, st, en) in tokens_with_spans:
        lower_tk = tk.lower()
        if lower_tk.isalpha() and (lower_tk not in stop_words_nltk):
            cleaned_tokens.append(lower_tk)
            char_index_map.append((st, en))

    if len(cleaned_tokens) < n:
        return local_counter

    # Slide over the cleaned tokens
    limit = len(cleaned_tokens) - n + 1
    for i in range(limit):
        candidate = tuple(cleaned_tokens[i : i + n])
        if candidate in top_ngrams_set:
            # Retrieve the exact substring from the original text
            start_char = char_index_map[i][0]
            end_char = char_index_map[i + n - 1][1]
            exact_substring = text[start_char:end_char]
            local_counter[exact_substring] += 1

    return local_counter


def extract_and_save_slop_phrases(
    texts: List[str],
    output_dir: str,
    n: int = 3,
    top_k_ngrams: int = 1000,
    top_phrases_to_save: int = 10000,
    chunksize: int = 50
):
    """
    1) Extract top-k n-grams from the combined texts (cleaned).
    2) Use multiprocessing to find exact substring occurrences in the original text.
    3) Filter out phrases with mid-phrase punctuation.
    4) Save the top phrases to a JSONL file in output_dir.
    """
    logger.info(f"Extracting top {top_k_ngrams} {n}-grams, then retrieving phrases...")

    # Step 1: get top n-grams from the cleaned perspective
    top_ngrams = extract_ngrams_cleaned(texts, n=n, top_k=top_k_ngrams)
    logger.info(f"Found {len(top_ngrams)} unique {n}-grams after cleaning.")

    if not top_ngrams:
        logger.warning("No n-grams found; skipping phrase extraction.")
        return

    # Convert that list to a set of n-gram tuples for quick membership checks
    top_ngrams_set = set(ng for ng, _freq in top_ngrams)
    logger.info(f"Created set of {len(top_ngrams_set)} top n-gram tuples.")

    # Step 2: Use multiprocessing to process texts
    process_func = partial(
        process_one_text_for_substrings,
        top_ngrams_set=top_ngrams_set,
        n=n
    )

    num_procs = min(os.cpu_count() or 1, config.SLOP_PHRASES_MAX_PROCESSES)
    logger.info(f"Spawning up to {num_procs} worker processes for phrase extraction...")

    with Pool(processes=num_procs) as p:
        partial_counters = list(
            tqdm(
                p.imap_unordered(process_func, texts, chunksize=chunksize),
                desc="MP substring extraction",
                total=len(texts)
            )
        )

    # Merge counters
    combined_substring_counter = Counter()
    for c in partial_counters:
        combined_substring_counter.update(c)

    logger.info(f"Merged counters: {len(combined_substring_counter)} unique substrings found.")

    # Step 3: Filter out phrases with mid-phrase punctuation
    filtered_substring_counter = Counter()
    for phrase, freq in combined_substring_counter.items():
        if not has_sentence_end_in_the_middle(phrase):
            filtered_substring_counter[phrase] = freq

    logger.info(f"After filtering, we have {len(filtered_substring_counter)} unique phrases.")

    # Step 4: Keep top X phrases and save
    top_phrases = filtered_substring_counter.most_common(top_phrases_to_save)
    phrases_slop_filename = os.path.join(output_dir, 'slop_list_phrases.jsonl')
    save_list_jsonl(top_phrases, phrases_slop_filename)
    logger.info(f"Saved top {len(top_phrases)} phrases to {phrases_slop_filename}.")


###############################################################################
# Main function to create slop lists (existing logic + new phrase extraction)
###############################################################################


def create_slop_lists(
    analysis_files_dir: str = config.ANALYSIS_OUTPUT_DIR,
    output_dir: str = config.SLOP_LIST_OUTPUT_DIR,
    max_items_per_model: int = config.ANALYSIS_MAX_ITEMS_PER_MODEL
):
    """
    Combines analysis results from multiple models to create final slop lists.
    Also extracts and saves top slop phrases (multi-word substrings) in JSONL.
    """
    logger.info("Starting combined slop list generation...")
    all_model_data = []
    analysis_files = [f for f in os.listdir(analysis_files_dir) if f.endswith('.json')]

    if not analysis_files:
        logger.error(f"No analysis JSON files found in {analysis_files_dir}. Cannot create slop lists.")
        return

    logger.info(f"Found {len(analysis_files)} analysis files. Loading data...")
    for filename in tqdm(analysis_files, desc="Loading analysis files"):
        filepath = os.path.join(analysis_files_dir, filename)
        data = load_json_file(filepath)
        if data and isinstance(data, dict) and "model_name" in data:
            model_name = data["model_name"]
            sanitized_name = sanitize_filename(model_name)
            dataset_filename = os.path.join(
                config.DATASET_OUTPUT_DIR,
                f"generated_{sanitized_name}.jsonl"
            )
            if os.path.exists(dataset_filename):
                logger.debug(f"Reloading dataset for {model_name} from {dataset_filename}")
                model_dataset = load_jsonl_file(dataset_filename, max_items=max_items_per_model)
                texts = [
                    item['output']
                    for item in model_dataset
                    if 'output' in item and isinstance(item['output'], str)
                ]
                if texts:
                    all_model_data.append({"model_name": model_name, "texts": texts})
                else:
                    logger.warning(f"No text found in dataset file for {model_name}")
            else:
                logger.warning(f"Dataset file not found for {model_name}: {dataset_filename}")
        else:
            logger.warning(f"Skipping invalid analysis file: {filename}")

    if not all_model_data:
        logger.error("No valid model data loaded. Cannot create slop lists.")
        return

    logger.info(f"Processing combined text data from {len(all_model_data)} models...")

    # Aggregate text from all models
    all_texts_flat = []
    for model_data in all_model_data:
        all_texts_flat.extend(model_data.get("texts", []))

    if not all_texts_flat:
        logger.error("No text data available after combining models.")
        return

    # =======================
    # 1) WORD-BASED SLOP LIST
    # =======================

    logger.info("Counting combined words...")
    raw_combined_counts = Counter()
    for text in tqdm(all_texts_flat, desc="Counting words"):
        normalized = normalize_text(text)
        words = extract_words(normalized, config.WORD_MIN_LENGTH)
        raw_combined_counts.update(words)

    logger.info("Filtering combined counts...")
    filtered_numeric = filter_mostly_numeric(raw_combined_counts)
    merged_counts = merge_plural_possessive_s(filtered_numeric)
    filtered_stopwords = filter_stopwords(merged_counts)

    if not filtered_stopwords:
        logger.warning("No words remaining after numeric/stopword filtering. Cannot proceed.")
        return

    # Analyze rarity (for correlation, etc.) – not strictly needed for final lists, but included
    logger.info("Analyzing combined word rarity...")
    corpus_freqs, wordfreq_freqs, avg_corp, avg_wf, corr = analyze_word_rarity(filtered_stopwords)

    if not corpus_freqs:
        logger.error("Could not calculate corpus frequencies for combined data.")
        return

    # Filter out "common words" based on wordfreq
    logger.info(f"Filtering common words (wordfreq > {config.COMMON_WORD_THRESHOLD})...")
    final_counts_for_slop = filter_common_words(
        filtered_stopwords,
        wordfreq_freqs,
        config.COMMON_WORD_THRESHOLD
    )

    if not final_counts_for_slop:
        logger.warning("No words remaining after filtering common words. Slop lists will be empty.")
        over_represented_words = []
        zero_freq_words = []
    else:
        final_total_words = sum(final_counts_for_slop.values())
        final_corpus_freqs = {
            w: c / final_total_words for w, c in final_counts_for_slop.items()
        } if final_total_words > 0 else {}

        logger.info("Finding over-represented and zero-frequency words...")
        over_represented_words = find_over_represented_words(
            final_corpus_freqs,
            wordfreq_freqs,
            top_n=config.SLOP_LIST_TOP_N_OVERREP * 2
        )
        zero_freq_words = find_zero_frequency_words(
            final_counts_for_slop,
            wordfreq_freqs,
            top_n=config.SLOP_LIST_TOP_N_ZERO_FREQ
        )

    # Create & save final word-based slop lists
    logger.info("Creating final word slop lists...")
    top_over_rep_words = [item[0] for item in over_represented_words[:config.SLOP_LIST_TOP_N_OVERREP]]
    top_zero_freq_words = [item[0] for item in zero_freq_words[:config.SLOP_LIST_TOP_N_ZERO_FREQ]]
    combined_slop_word_set = set(top_over_rep_words + top_zero_freq_words)
    sorted_slop_words = sorted(list(combined_slop_word_set))
    formatted_slop_list = [[word] for word in sorted_slop_words]
    slop_list_filename = os.path.join(output_dir, 'slop_list.json')
    save_list_one_item_per_line(formatted_slop_list, slop_list_filename)
    logger.info(f"Saved standard word slop list ({len(formatted_slop_list)} words).")

    # Frequency-sorted slop list
    slop_word_frequencies = {word: final_counts_for_slop.get(word, 0) for word in combined_slop_word_set}
    sorted_by_freq = sorted(slop_word_frequencies.items(), key=lambda x: x[1], reverse=True)
    formatted_freq_slop_list = [[word, count] for word, count in sorted_by_freq]
    freq_slop_list_filename = os.path.join(output_dir, 'slop_list_by_freq.json')
    save_json_file(formatted_freq_slop_list, freq_slop_list_filename, indent=None)
    logger.info(f"Saved frequency-sorted word slop list ({len(formatted_freq_slop_list)} words).")

    # =======================
    # 2) N-GRAM (PHRASES) SLOP LIST
    # =======================

    # First, aggregate bigrams/trigrams from analysis files (as in your original approach).
    logger.info("Aggregating N-gram data for slop lists...")
    combined_bigrams = defaultdict(lambda: {'total_freq': 0, 'models': set()})
    combined_trigrams = defaultdict(lambda: {'total_freq': 0, 'models': set()})

    for filename in tqdm(analysis_files, desc="Aggregating N-grams"):
        filepath = os.path.join(analysis_files_dir, filename)
        data = load_json_file(filepath)
        if data and isinstance(data, dict):
            model_name = data.get("model_name", "unknown")
            for bg_data in data.get("top_bigrams", []):
                ngram = bg_data.get("ngram")
                freq = bg_data.get("frequency", 0)
                if ngram and freq > 0:
                    combined_bigrams[ngram]['total_freq'] += freq
                    combined_bigrams[ngram]['models'].add(model_name)
            for tg_data in data.get("top_trigrams", []):
                ngram = tg_data.get("ngram")
                freq = tg_data.get("frequency", 0)
                if ngram and freq > 0:
                    combined_trigrams[ngram]['total_freq'] += freq
                    combined_trigrams[ngram]['models'].add(model_name)

    min_models_for_ngram_slop = 2
    filtered_bigrams = {
        ng: data
        for ng, data in combined_bigrams.items()
        if len(data['models']) >= min_models_for_ngram_slop
    }
    filtered_trigrams = {
        ng: data
        for ng, data in combined_trigrams.items()
        if len(data['models']) >= min_models_for_ngram_slop
    }

    sorted_bigrams = sorted(filtered_bigrams.items(), key=lambda item: item[1]['total_freq'], reverse=True)
    sorted_trigrams = sorted(filtered_trigrams.items(), key=lambda item: item[1]['total_freq'], reverse=True)

    # Save bigram/trigram slop lists
    top_bigrams_list = [[bg[0]] for bg in sorted_bigrams[:config.SLOP_LIST_TOP_N_BIGRAMS]]
    bigram_slop_filename = os.path.join(output_dir, 'slop_list_bigrams.json')
    save_list_one_item_per_line(top_bigrams_list, bigram_slop_filename)
    logger.info(f"Saved bigram slop list ({len(top_bigrams_list)} bigrams).")

    top_trigrams_list = [[tg[0]] for tg in sorted_trigrams[:config.SLOP_LIST_TOP_N_TRIGRAMS]]
    trigram_slop_filename = os.path.join(output_dir, 'slop_list_trigrams.json')
    save_list_one_item_per_line(top_trigrams_list, trigram_slop_filename)
    logger.info(f"Saved trigram slop list ({len(top_trigrams_list)} trigrams).")

    # =======================
    # 3) EXACT PHRASE EXTRACTION
    # =======================
    # Uses multi-processing and substring matching for the top n-grams in the *combined* data.
    # This step is not reliant on the analysis JSON files; it re-processes all_texts_flat.
    logger.info("Extracting and saving slop phrases from combined data...")
    extract_and_save_slop_phrases(
        texts=all_texts_flat,
        output_dir=output_dir,
        n=config.SLOP_PHRASES_NGRAM_SIZE,
        top_k_ngrams=config.SLOP_PHRASES_TOP_NGRAMS,
        top_phrases_to_save=config.SLOP_PHRASES_TOP_PHRASES_TO_SAVE,
        chunksize=config.SLOP_PHRASES_CHUNKSIZE
    )

    logger.info("Slop list + phrase generation finished.")
