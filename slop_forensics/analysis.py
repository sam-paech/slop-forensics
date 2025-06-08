import re
import json
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Counter as TypingCounter, Optional, Union, Any

import numpy as np
from tqdm import tqdm
from wordfreq import word_frequency
import nltk
from nltk.corpus import stopwords
from scipy.stats import spearmanr

from . import config
from .constants import KNOWN_CONTRACTIONS_S, FORBIDDEN_SUBSTRINGS
from .utils import normalize_text, extract_words

logger = logging.getLogger(__name__)

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    STOP_WORDS = set(stopwords.words(config.STOPWORD_LANG))
    logger.info(f"Loaded {len(STOP_WORDS)} NLTK stopwords for '{config.STOPWORD_LANG}'.")
except LookupError:
    logger.warning(f"NLTK 'punkt' or 'stopwords' not found. Run nltk.download('punkt') and nltk.download('stopwords').")
    STOP_WORDS = set()
except ImportError:
    logger.warning("NLTK not installed. Stopword filtering will be skipped.")
    STOP_WORDS = set()


# --- Core Word Counting and Filtering ---

def get_word_counts(texts: List[str], min_length: int = config.WORD_MIN_LENGTH) -> TypingCounter[str]:
    """Counts overall word frequencies in a list of texts."""
    word_counts = Counter()
    for text in texts: # No tqdm here, usually called within another loop
        normalized_text = normalize_text(text) # Use utility function
        words = extract_words(normalized_text, min_length) # Use utility function
        word_counts.update(words)
    return word_counts

def get_word_prompt_map(texts_with_ids: List[Tuple[str, str]], min_length: int = config.WORD_MIN_LENGTH) -> Dict[str, Set[str]]:
    """Creates a map of words to the set of prompt IDs they appear in."""
    word_prompts = defaultdict(set)
    for text, prompt_id in texts_with_ids: # No tqdm here
        normalized_text = normalize_text(text)
        words = extract_words(normalized_text, min_length)
        for word in words:
            word_prompts[word].add(prompt_id)
    return dict(word_prompts)

def filter_mostly_numeric(word_counts: TypingCounter[str]) -> TypingCounter[str]:
    """Filters out words containing a high proportion of digits."""
    def is_mostly_numbers(word):
        if not word: return False
        digit_count = sum(c.isdigit() for c in word)
        # Avoid division by zero for empty words if they somehow get here
        return (digit_count / len(word) > 0.2) if len(word) > 0 else False

    return Counter({word: count for word, count in word_counts.items() if not is_mostly_numbers(word)})

def merge_plural_possessive_s(word_counts: TypingCounter[str]) -> TypingCounter[str]:
    """Merges counts of possessive words ending in 's with their base words."""
    merged_counts = Counter()
    for word, count in word_counts.items():
        if word.endswith("'s") and word not in KNOWN_CONTRACTIONS_S:
            base_word = word[:-2]
            # Ensure base_word is not empty (e.g., "'s")
            if base_word:
                merged_counts[base_word] += count
            else:
                 merged_counts[word] += count # Keep original if base is empty
        else:
            merged_counts[word] += count
    return merged_counts

def filter_forbidden_words(word_counts: TypingCounter[str]) -> TypingCounter[str]:
    """Filters out words containing any forbidden substrings."""
    if not FORBIDDEN_SUBSTRINGS:
        return word_counts
    return Counter({
        word: count for word, count in word_counts.items()
        if not any(sub in word for sub in FORBIDDEN_SUBSTRINGS) # Assumes word is already lowercase
    })

def filter_by_minimum_count(word_counts: TypingCounter[str], min_count: int) -> TypingCounter[str]:
    """Filters out words appearing <= min_count times."""
    if min_count <= 0:
        return word_counts
    return Counter({word: count for word, count in word_counts.items() if count > min_count})

def filter_stopwords(word_counts: TypingCounter[str]) -> TypingCounter[str]:
    """Filters out common English stopwords."""
    if not STOP_WORDS:
        return word_counts
    return Counter({word: count for word, count in word_counts.items() if word not in STOP_WORDS})

def filter_common_words(word_counts: TypingCounter[str], wordfreq_freqs: Dict[str, float], threshold: float) -> TypingCounter[str]:
    """Filters out words whose general frequency (wordfreq) is above a threshold."""
    return Counter({
        word: count for word, count in word_counts.items()
        if wordfreq_freqs.get(word, 0) <= threshold
    })


# --- Rarity Analysis ---

def analyze_word_rarity(word_counts: TypingCounter[str]) -> Tuple[Dict[str, float], Dict[str, float], float, float, float]:
    """Analyzes word rarity based on corpus and wordfreq frequencies."""
    corpus_frequencies = {}
    wordfreq_frequencies = {}
    avg_corpus_rarity = np.nan
    avg_wordfreq_rarity = np.nan
    correlation = np.nan

    if not word_counts:
        return corpus_frequencies, wordfreq_frequencies, avg_corpus_rarity, avg_wordfreq_rarity, correlation

    total_words = sum(word_counts.values())
    if total_words == 0:
        return corpus_frequencies, wordfreq_frequencies, avg_corpus_rarity, avg_wordfreq_rarity, correlation

    corpus_frequencies = {word: count / total_words for word, count in word_counts.items()}

    # Fetch wordfreq data (can be slow for large vocabularies)
    logger.debug(f"Fetching wordfreq data for {len(word_counts)} unique words...")
    for word in word_counts.keys():
        try:
            wordfreq_frequencies[word] = word_frequency(word, 'en')
        except Exception as e:
            logger.warning(f"Error fetching word frequency for '{word}': {e}")
            wordfreq_frequencies[word] = 0.0 # Assign 0 if error

    # Calculate rarity metrics
    valid_words = [word for word, freq in wordfreq_frequencies.items() if freq > 0]
    if valid_words:
        corpus_freq_list = [corpus_frequencies[word] for word in valid_words]
        wordfreq_freq_list = [wordfreq_frequencies[word] for word in valid_words]

        # Use np.log10, handle potential log(0) with small epsilon or filter zeros
        epsilon = 1e-12 # Small value to avoid log10(0)
        safe_corpus_freqs = [max(f, epsilon) for f in corpus_freq_list]
        safe_wordfreq_freqs = [max(f, epsilon) for f in wordfreq_freq_list]

        avg_corpus_rarity = np.mean([-np.log10(f) for f in safe_corpus_freqs])
        avg_wordfreq_rarity = np.mean([-np.log10(f) for f in safe_wordfreq_freqs])

        if len(valid_words) >= 2:
            try:
                correlation, _ = spearmanr(corpus_freq_list, wordfreq_freq_list)
                if np.isnan(correlation): correlation = 0.0 # Handle NaN result from spearmanr
            except Exception as e:
                 logger.warning(f"Could not calculate correlation: {e}")
                 correlation = np.nan

    return corpus_frequencies, wordfreq_frequencies, avg_corpus_rarity, avg_wordfreq_rarity, correlation


# --- Finding Specific Word/N-gram Lists ---

def find_over_represented_words(
    corpus_frequencies: Dict[str, float],
    wordfreq_frequencies: Dict[str, float],
    top_n: int = 50000
) -> List[Tuple[str, float, float, float]]:
    """
    Finds words most over-represented compared to wordfreq.
    Returns list of (word, ratio, corpus_freq, wordfreq_freq).
    Sorts by ratio descending.
    """
    over_representation = {}
    epsilon = 1e-12 # Avoid division by zero for wordfreq
    for word, corpus_freq in corpus_frequencies.items():
        wordfreq_freq = wordfreq_frequencies.get(word, 0)
        ratio = corpus_freq / max(wordfreq_freq, epsilon)
        over_representation[word] = (ratio, corpus_freq, wordfreq_freq)

    # Sort by ratio (descending)
    sorted_words = sorted(
        over_representation.items(),
        key=lambda item: item[1][0], # Sort by ratio
        reverse=True
    )

    # Return top_n with full info: (word, (ratio, corpus_freq, wordfreq_freq)) -> (word, ratio, corpus_freq, wordfreq_freq)
    return [(word, data[0], data[1], data[2]) for word, data in sorted_words[:top_n]]


def find_zero_frequency_words(
    word_counts: TypingCounter[str],
    wordfreq_frequencies: Dict[str, float],
    top_n: int = 20000
) -> List[Tuple[str, int]]:
    """Finds most frequent words with zero wordfreq frequency."""
    zero_freq_words = {
        word: count for word, count in word_counts.items()
        if wordfreq_frequencies.get(word, -1) == 0 # Check if wordfreq is exactly 0
    }
    # Sort by count (descending)
    return sorted(zero_freq_words.items(), key=lambda item: item[1], reverse=True)[:top_n]


def get_ngrams(
    prompts_data: Dict[str, List[str]],
    n: int,
    top_k: int,
    min_prompt_ids: int
) -> List[Dict[str, Union[str, int]]]:
    """
    Extracts top_k N-grams appearing across min_prompt_ids unique prompts.
    Returns list of dicts: [{'ngram': 'word1 word2', 'frequency': count}, ...]
    """
    ngram_counts = Counter()
    ngram_prompt_map = defaultdict(set)

    if len(prompts_data) < min_prompt_ids:
        return []

    logger.debug(f"Extracting {n}-grams (min prompts: {min_prompt_ids})...")
    total_texts_processed = 0
    for prompt_id, texts in prompts_data.items():
        for text in texts:
            total_texts_processed += 1
            normalized_text = normalize_text(text) # Normalize first
            try:
                # Tokenize, remove punctuation/stopwords
                tokens = [
                    word for word in nltk.word_tokenize(normalized_text)
                    if word.isalpha() and word not in STOP_WORDS
                ]
            except LookupError:
                 logger.warning("NLTK 'punkt' tokenizer not found. Using basic split for ngrams.")
                 tokens = [w for w in normalized_text.split() if w.isalpha() and w not in STOP_WORDS]


            if len(tokens) < n:
                continue

            try:
                current_ngrams = nltk.ngrams(tokens, n)
                for ngram_tuple in current_ngrams:
                    ngram_counts[ngram_tuple] += 1
                    ngram_prompt_map[ngram_tuple].add(prompt_id)
            except Exception as e:
                 logger.warning(f"Error generating {n}-grams for text snippet '{normalized_text[:50]}...': {e}")


    logger.debug(f"Processed {total_texts_processed} texts for {n}-grams.")

    # Filter by min_prompt_ids
    filtered_ngrams = {
        ngram: count for ngram, count in ngram_counts.items()
        if len(ngram_prompt_map[ngram]) >= min_prompt_ids
    }

    if not filtered_ngrams:
        logger.debug(f"No {n}-grams found meeting the minimum prompt ID criterion ({min_prompt_ids}).")
        return []

    # Sort by frequency and format output
    sorted_filtered = sorted(filtered_ngrams.items(), key=lambda item: item[1], reverse=True)

    # Format as list of dictionaries
    formatted_output = [
        {"ngram": " ".join(ngram_tuple), "frequency": count}
        for ngram_tuple, count in sorted_filtered[:top_k]
    ]
    logger.debug(f"Found {len(formatted_output)} top {n}-grams meeting criteria.")
    return formatted_output


# --- Main Analysis Orchestration ---

def analyze_texts(
    model_name: str,
    texts_with_ids: List[Tuple[str, str]],          # (text, prompt_id)
    prompts_data: Dict[str, List[str]]              # {prompt_id: [text1, …]}
) -> Dict[str, Any]:
    """
    Comprehensive one-model analysis.

    Key change: we now split high-ratio words into
        • top_repetitive_words  (wordfreq > 0)
        • zero_frequency_words  (wordfreq == 0)
    so that nonce tokens no longer swamp the preview list, while
    later stages that recompute from raw text stay unaffected.
    """
    logger.info(f"Starting analysis for model: {model_name}")
    analysis_results = {"model_name": model_name}

    # ---------- basic counts ----------
    num_texts = len(texts_with_ids)
    num_prompts = len(prompts_data)
    analysis_results["num_texts_analyzed"]   = num_texts
    analysis_results["num_unique_prompts"]  = num_prompts
    if num_texts == 0:
        logger.warning("No texts supplied; returning empty result.")
        return analysis_results

    all_texts_flat  = [t for t, _ in texts_with_ids]
    all_text_concat = "\n\n".join(all_texts_flat)

    # ---------- simple metrics ----------
    total_chars = sum(len(t) for t in all_texts_flat)
    analysis_results["avg_length"] = round(total_chars / num_texts, 2)

    from .metrics import calculate_complexity_index, calculate_slop_index_new
    analysis_results["vocab_complexity"] = calculate_complexity_index(all_text_concat)
    analysis_results["slop_score"]       = calculate_slop_index_new(all_text_concat)

    # ---------- word counting & filters ----------
    raw_counts          = get_word_counts(all_texts_flat)
    counts_no_numbers   = filter_mostly_numeric(raw_counts)
    counts_merged_s     = merge_plural_possessive_s(counts_no_numbers)

    if num_prompts >= config.WORD_MIN_PROMPT_IDS:
        word_prompt_map = get_word_prompt_map(texts_with_ids)
        eligible_words  = {
            w for w, ids in word_prompt_map.items()
            if len(ids) >= config.WORD_MIN_PROMPT_IDS
        }
        counts_multi_prompt = Counter({
            w: c for w, c in counts_merged_s.items() if w in eligible_words
        })
    else:
        counts_multi_prompt = counts_merged_s

    counts_no_forbidden = filter_forbidden_words(counts_multi_prompt)
    final_word_counts   = filter_by_minimum_count(
        counts_no_forbidden, config.WORD_MIN_REPETITION_COUNT
    )
    analysis_results["total_unique_words_after_filters"] = len(final_word_counts)

    # ---------- rarity / over-rep split ----------
    if final_word_counts:
        (
            corpus_freqs, wordfreq_freqs,
            avg_corp_rarity, avg_wf_rarity, corr
        ) = analyze_word_rarity(final_word_counts)

        analysis_results["avg_corpus_rarity"]  = round(avg_corp_rarity, 4) if avg_corp_rarity else None
        analysis_results["avg_wordfreq_rarity"] = round(avg_wf_rarity, 4) if avg_wf_rarity else None
        analysis_results["rarity_correlation"]  = round(corr, 4) if corr else None

        over_rep_words = find_over_represented_words(
            corpus_freqs, wordfreq_freqs, top_n=config.TOP_N_WORDS_REPETITION * 3
        )

        top_nonzero : List[Dict[str, Any]] = []
        zero_freq   : List[Dict[str, Any]] = []

        for word, ratio, cf, wf in over_rep_words:
            if wf == 0.0:
                zero_freq.append(
                    {"word": word, "corpus_freq": cf}
                )
            else:
                top_nonzero.append(
                    {
                        "word":           word,
                        "score":          ratio,
                        "corpus_freq":    cf,
                        "wordfreq_freq":  wf
                    }
                )

        analysis_results["top_repetitive_words"] = top_nonzero[:config.TOP_N_WORDS_REPETITION]
        analysis_results["zero_frequency_words"] = zero_freq[:config.SLOP_LIST_TOP_N_ZERO_FREQ]

        # simple repetition score for logging consistency
        top_n_for_score = 100
        repetition_score_val = sum(
            item["corpus_freq"] for item in top_nonzero[:top_n_for_score]
        ) * 100
        analysis_results["repetition_score"] = round(repetition_score_val, 4)

    else:
        # nothing survived filtering
        analysis_results["avg_corpus_rarity"]   = None
        analysis_results["avg_wordfreq_rarity"] = None
        analysis_results["rarity_correlation"]  = None
        analysis_results["top_repetitive_words"] = []
        analysis_results["zero_frequency_words"] = []
        analysis_results["repetition_score"]     = 0.0

    # ---------- N-gram analysis (unchanged) ----------
    if num_prompts >= config.NGRAM_MIN_PROMPT_IDS:
        analysis_results["top_bigrams"]  = get_ngrams(
            prompts_data, 2, config.TOP_N_BIGRAMS, config.NGRAM_MIN_PROMPT_IDS
        )
        analysis_results["top_trigrams"] = get_ngrams(
            prompts_data, 3, config.TOP_N_TRIGRAMS, config.NGRAM_MIN_PROMPT_IDS
        )
    else:
        analysis_results["top_bigrams"]  = []
        analysis_results["top_trigrams"] = []

    logger.info(f"Analysis complete for model: {model_name}")
    return analysis_results

