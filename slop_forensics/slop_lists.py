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
       - Normalize text (NFKC, quote normalization, lowercase)
       - Tokenize with regex [a-zA-Z']+
       - Strip leading/trailing apostrophes
       - Exclude stopwords
    Returns a list of (ngram_tuple, frequency).

    Aligns with other repo's preprocessing.
    """
    ngram_counts = Counter()
    logger.info(f"Extracting cleaned {n}-grams from {len(texts_list)} combined texts...")

    for text in tqdm(texts_list, desc=f"Extracting {n}-grams", leave=False):
        if not isinstance(text, str) or not text.strip():
            continue

        # Normalize: NFKC, quote normalization, lowercase
        normalized_text = normalize_text(text)
        # Tokenize using extract_words (regex [a-zA-Z']+, strip apostrophes)
        tokens = [
            word for word in extract_words(normalized_text, min_length=1)
            if word not in stop_words_nltk
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

    Uses aligned preprocessing (quote normalization, regex tokenization).
    """
    local_counter = Counter()

    if not isinstance(text, str) or not text.strip():
        return local_counter

    # First, normalize the text for comparison (quote normalization, lowercase)
    normalized_text = normalize_text(text)

    # Naive tokenization with offsets on ORIGINAL text:
    # We need to find tokens in the original text to preserve casing/formatting
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

    # Build cleaned_tokens + offset map using aligned tokenization
    cleaned_tokens = []
    char_index_map = []
    for (tk, st, en) in tokens_with_spans:
        # Normalize this token the same way (lowercase, strip apostrophes)
        normalized_tk = normalize_text(tk)
        # Extract using same regex pattern
        extracted = extract_words(normalized_tk, min_length=1)
        if extracted:
            word = extracted[0]  # Should only be one token from a single word
            if word not in stop_words_nltk:
                cleaned_tokens.append(word)
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
    Creates slop lists by aggregating pre-computed analysis results.

    This approach:
    - Loads analysis JSON files (which already have top_repetitive_words, bigrams, trigrams)
    - Aggregates bigrams/trigrams by summing frequencies
    - Aggregates repetitive words by averaging corpus_freq, then recomputing score
    - Stays consistent with how individual model profiles compute their lists
    """
    logger.info("Starting combined slop list generation from analysis files...")
    analysis_files = [f for f in os.listdir(analysis_files_dir) if f.endswith('.json')]

    if not analysis_files:
        logger.error(f"No analysis JSON files found in {analysis_files_dir}. Cannot create slop lists.")
        return

    logger.info(f"Found {len(analysis_files)} analysis files.")

    # =======================
    # 1) AGGREGATE BIGRAMS & TRIGRAMS
    # =======================
    logger.info("Aggregating N-gram data from analysis files...")
    combined_bigrams = defaultdict(lambda: {'total_freq': 0, 'models': set()})
    combined_trigrams = defaultdict(lambda: {'total_freq': 0, 'models': set()})

    for filename in tqdm(analysis_files, desc="Aggregating N-grams"):
        filepath = os.path.join(analysis_files_dir, filename)
        data = load_json_file(filepath)
        if data and isinstance(data, dict):
            model_name = data.get("model_name", "unknown")

            # Aggregate bigrams
            for bg_data in data.get("top_bigrams", []):
                ngram = bg_data.get("ngram")
                freq = bg_data.get("frequency", 0)
                if ngram and freq > 0:
                    combined_bigrams[ngram]['total_freq'] += freq
                    combined_bigrams[ngram]['models'].add(model_name)

            # Aggregate trigrams
            for tg_data in data.get("top_trigrams", []):
                ngram = tg_data.get("ngram")
                freq = tg_data.get("frequency", 0)
                if ngram and freq > 0:
                    combined_trigrams[ngram]['total_freq'] += freq
                    combined_trigrams[ngram]['models'].add(model_name)

    # Filter to ngrams appearing in at least N models
    min_models_for_ngram_slop = 2
    filtered_bigrams = {
        ng: data for ng, data in combined_bigrams.items()
        if len(data['models']) >= min_models_for_ngram_slop
    }
    filtered_trigrams = {
        ng: data for ng, data in combined_trigrams.items()
        if len(data['models']) >= min_models_for_ngram_slop
    }

    # Sort by frequency
    sorted_bigrams = sorted(filtered_bigrams.items(), key=lambda item: item[1]['total_freq'], reverse=True)
    sorted_trigrams = sorted(filtered_trigrams.items(), key=lambda item: item[1]['total_freq'], reverse=True)

    # Save bigram slop list
    top_bigrams_list = [[bg[0]] for bg in sorted_bigrams[:config.SLOP_LIST_TOP_N_BIGRAMS]]
    bigram_slop_filename = os.path.join(output_dir, 'slop_list_bigrams.json')
    save_list_one_item_per_line(top_bigrams_list, bigram_slop_filename)
    logger.info(f"Saved bigram slop list ({len(top_bigrams_list)} bigrams).")

    # Save trigram slop list
    top_trigrams_list = [[tg[0]] for tg in sorted_trigrams[:config.SLOP_LIST_TOP_N_TRIGRAMS]]
    trigram_slop_filename = os.path.join(output_dir, 'slop_list_trigrams.json')
    save_list_one_item_per_line(top_trigrams_list, trigram_slop_filename)
    logger.info(f"Saved trigram slop list ({len(top_trigrams_list)} trigrams).")

    # =======================
    # 2) AGGREGATE REPETITIVE WORDS
    # =======================
    logger.info("Aggregating repetitive words from analysis files...")

    # Collect all word data across models
    # word -> list of (corpus_freq, wordfreq_freq) from each model
    word_data = defaultdict(lambda: {'corpus_freqs': [], 'wordfreq_freq': None, 'models': set()})

    for filename in tqdm(analysis_files, desc="Aggregating words"):
        filepath = os.path.join(analysis_files_dir, filename)
        data = load_json_file(filepath)
        if data and isinstance(data, dict):
            model_name = data.get("model_name", "unknown")

            for word_entry in data.get("top_repetitive_words", []):
                word = word_entry.get("word")
                corpus_freq = word_entry.get("corpus_freq")
                wordfreq_freq = word_entry.get("wordfreq_freq")

                if word and corpus_freq is not None:
                    word_data[word]['corpus_freqs'].append(corpus_freq)
                    word_data[word]['models'].add(model_name)
                    # Store wordfreq_freq (should be same across models, just take the value)
                    if wordfreq_freq is not None:
                        word_data[word]['wordfreq_freq'] = wordfreq_freq

    # Compute average corpus_freq and recompute score
    logger.info("Computing averaged corpus frequencies and scores...")
    word_scores = []
    epsilon = 1e-12

    for word, data in word_data.items():
        if not data['corpus_freqs']:
            continue

        # Average corpus frequency across models
        avg_corpus_freq = np.mean(data['corpus_freqs'])
        wordfreq_freq = data['wordfreq_freq'] if data['wordfreq_freq'] is not None else 0.0

        # Recompute score: corpus_freq / wordfreq_freq
        score = avg_corpus_freq / max(wordfreq_freq, epsilon)

        word_scores.append({
            'word': word,
            'score': score,
            'avg_corpus_freq': avg_corpus_freq,
            'wordfreq_freq': wordfreq_freq,
            'num_models': len(data['models'])
        })

    # Sort by score (descending)
    word_scores.sort(key=lambda x: x['score'], reverse=True)

    # Also get zero-frequency words from analysis files
    logger.info("Aggregating zero-frequency words from analysis files...")
    zero_freq_word_data = defaultdict(lambda: {'corpus_freqs': [], 'models': set()})

    for filename in tqdm(analysis_files, desc="Aggregating zero-freq words"):
        filepath = os.path.join(analysis_files_dir, filename)
        data = load_json_file(filepath)
        if data and isinstance(data, dict):
            model_name = data.get("model_name", "unknown")

            for word_entry in data.get("zero_frequency_words", []):
                word = word_entry.get("word")
                corpus_freq = word_entry.get("corpus_freq")

                if word and corpus_freq is not None:
                    zero_freq_word_data[word]['corpus_freqs'].append(corpus_freq)
                    zero_freq_word_data[word]['models'].add(model_name)

    # Average zero-freq words
    zero_freq_scores = []
    for word, data in zero_freq_word_data.items():
        if not data['corpus_freqs']:
            continue
        avg_corpus_freq = np.mean(data['corpus_freqs'])
        zero_freq_scores.append({
            'word': word,
            'avg_corpus_freq': avg_corpus_freq,
            'num_models': len(data['models'])
        })

    # Sort by average corpus frequency
    zero_freq_scores.sort(key=lambda x: x['avg_corpus_freq'], reverse=True)

    # =======================
    # 3) CREATE FINAL WORD SLOP LISTS
    # =======================
    logger.info("Creating final word slop lists...")

    # Take top N from each category
    top_over_rep_words = [item['word'] for item in word_scores[:config.SLOP_LIST_TOP_N_OVERREP]]
    top_zero_freq_words = [item['word'] for item in zero_freq_scores[:config.SLOP_LIST_TOP_N_ZERO_FREQ]]

    # Combine and deduplicate
    combined_slop_word_set = set(top_over_rep_words + top_zero_freq_words)
    sorted_slop_words = sorted(list(combined_slop_word_set))

    # Save alphabetically sorted list
    formatted_slop_list = [[word] for word in sorted_slop_words]
    slop_list_filename = os.path.join(output_dir, 'slop_list.json')
    save_list_one_item_per_line(formatted_slop_list, slop_list_filename)
    logger.info(f"Saved standard word slop list ({len(formatted_slop_list)} words).")

    # Create frequency-sorted list (by average corpus freq)
    word_avg_freqs = {}
    for item in word_scores:
        if item['word'] in combined_slop_word_set:
            word_avg_freqs[item['word']] = item['avg_corpus_freq']
    for item in zero_freq_scores:
        if item['word'] in combined_slop_word_set and item['word'] not in word_avg_freqs:
            word_avg_freqs[item['word']] = item['avg_corpus_freq']

    sorted_by_freq = sorted(word_avg_freqs.items(), key=lambda x: x[1], reverse=True)
    formatted_freq_slop_list = [[word, freq] for word, freq in sorted_by_freq]
    freq_slop_list_filename = os.path.join(output_dir, 'slop_list_by_freq.json')
    save_json_file(formatted_freq_slop_list, freq_slop_list_filename, indent=None)
    logger.info(f"Saved frequency-sorted word slop list ({len(formatted_freq_slop_list)} words).")

    logger.info("Slop list generation finished.")
