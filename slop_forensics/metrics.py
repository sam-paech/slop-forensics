import nltk
from nltk.corpus import cmudict
import string
import logging
import json
import re
import os
from typing import  Set

# Attempt to load NLTK resources, warn if missing
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/cmudict')
    pronunciation_dict = nltk.corpus.cmudict.dict()
    NLTK_AVAILABLE = True
except LookupError:
    logging.warning("NLTK resources ('punkt', 'cmudict') not found. Complexity calculation will be basic.")
    logging.warning("Run: nltk.download('punkt'); nltk.download('cmudict')")
    pronunciation_dict = {}
    NLTK_AVAILABLE = False
except ImportError:
     logging.warning("NLTK not installed. Complexity calculation will be basic.")
     pronunciation_dict = {}
     NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Vocabulary Complexity ---

# Load CMU Pronouncing Dictionary
pronunciation_dict = cmudict.dict()

def syllable_count(word):
    """Determine the number of syllables in a word."""
    word = word.lower()
    if word in pronunciation_dict:
        return max([len([phoneme for phoneme in phonetic if phoneme[-1].isdigit()]) for phonetic in pronunciation_dict[word]])
    return 1  # Assume one syllable if the word isn't found

def is_polysyllabic(word):
    """Identify if a word is polysyllabic (i.e., has 3 or more syllables)."""
    return syllable_count(word) >= 3

def calculate_complexity_index(text: str) -> float:
    """Calculate complexity index (0-100) based on FK grade and complex words."""
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    try:
        sentences = nltk.sent_tokenize(text)
        tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()] # Keep only alphanumeric
    except LookupError:
         logger.warning("NLTK 'punkt' tokenizer not found. Using basic splitting for complexity.")
         sentences = [s for s in text.split('.') if s] # Very basic sentence split
         tokens = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]

    sentence_count = max(1, len(sentences))
    word_count = max(1, len(tokens))

    # Flesch-Kincaid Grade Level
    total_syllables = sum(syllable_count(token) for token in tokens)
    if word_count == 0: return 0.0 # Avoid division by zero

    try:
        fk_grade_level = (0.39 * (word_count / sentence_count) +
                          11.8 * (total_syllables / word_count) - 15.59)
    except ZeroDivisionError:
        fk_grade_level = 0.0

    # Percentage of complex words
    complex_word_count = sum(1 for token in tokens if is_polysyllabic(token))
    percent_complex_words = (complex_word_count / word_count) * 100 if word_count > 0 else 0

    # Normalize and combine (cap values)
    fk_capped = min(max(0, fk_grade_level), 14) # Cap between 0 and 14
    complex_capped = min(percent_complex_words, 20) # Cap at 20%

    fk_normalized = (fk_capped / 14) * 100
    complex_normalized = (complex_capped / 20) * 100

    complexity_index = (fk_normalized + complex_normalized) / 2
    return round(complexity_index, 4)

# Global cache for slop lists to avoid reloading repeatedly within a script run
_slop_list_cache = {}

def _load_slop_list_to_set(list_type: str) -> Set[str]:
    """Loads a specific slop list (word, bigram, trigram) into a set, using cache."""
    global _slop_list_cache
    if list_type in _slop_list_cache:
        return _slop_list_cache[list_type]

    filename_map = {
        'word': 'data/slop_list.json',
        'bigram': 'data/slop_list_bigrams.json',
        'trigram': 'data/slop_list_trigrams.json',
    }
    filename = filename_map.get(list_type)
    if not filename or not os.path.exists(filename):
        logger.warning(f"Slop file for type '{list_type}' not found at {filename}. Returning empty set.")
        _slop_list_cache[list_type] = set()
        return set()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assumes format [["item"], ["item phrase"], ...]
        slop_items = {item[0].lower() for item in data if item and isinstance(item, list) and item[0]}
        logger.info(f"Loaded {len(slop_items)} {list_type} items from {filename}")
        _slop_list_cache[list_type] = slop_items
        return slop_items
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}. Returning empty set.")
        _slop_list_cache[list_type] = set()
        return set()
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}. Returning empty set.")
        _slop_list_cache[list_type] = set()
        return set()

def calculate_slop_index_new(text: str, debug: bool = False) -> float:
    """Calculates the 'new' slop index based on hits in word, bigram, and trigram lists."""
    # 1. Load Slop Lists (uses cache)
    slop_words_set = _load_slop_list_to_set('word')
    slop_bigrams_set = _load_slop_list_to_set('bigram')
    slop_trigrams_set = _load_slop_list_to_set('trigram')

    if not slop_words_set and not slop_bigrams_set and not slop_trigrams_set:
        logger.warning("No slop lists loaded. Returning slop index 0.")
        return 0.0

    if not text or not isinstance(text, str) or not text.strip():
        if debug: logger.debug("Slop Index New: Input text is empty or invalid.")
        return 0.0

    # 2. Preprocess Text and Tokenize
    lower_text = text.lower()
    try:
        # Keep only alphanumeric tokens
        tokens = [token for token in nltk.word_tokenize(lower_text) if token.isalnum()]
    except LookupError:
        logger.warning("NLTK 'punkt' tokenizer not found. Using basic regex split for slop index.")
        tokens = re.findall(r'\b\w+\b', lower_text)

    total_words = len(tokens)
    if total_words == 0:
        if debug: logger.debug("Slop Index New: No valid words found after tokenization.")
        return 0.0

    # 3. Count Hits
    word_hits = 0
    bigram_hits = 0
    trigram_hits = 0

    if slop_words_set:
        word_hits = sum(1 for token in tokens if token in slop_words_set)

    if slop_bigrams_set and len(tokens) >= 2:
        try:
            text_bigrams = nltk.ngrams(tokens, 2)
            for bigram_tuple in text_bigrams:
                if ' '.join(bigram_tuple) in slop_bigrams_set:
                    bigram_hits += 1
        except Exception as e:
             logger.warning(f"Error generating bigrams: {e}")


    if slop_trigrams_set and len(tokens) >= 3:
        try:
            text_trigrams = nltk.ngrams(tokens, 3)
            for trigram_tuple in text_trigrams:
                if ' '.join(trigram_tuple) in slop_trigrams_set:
                    trigram_hits += 1
        except Exception as e:
             logger.warning(f"Error generating trigrams: {e}")

    # 4. Calculate Final Score (Weights: 1 for word, 2 for bigram, 8 for trigram)
    # Weights are chosen based on the original snippet's implied logic, adjust if needed
    total_slop_score = word_hits + (2 * bigram_hits) + (8 * trigram_hits)
    slop_index = (total_slop_score / total_words) * 1000 if total_words > 0 else 0.0

    if debug:
        logger.debug(f"--- Slop Index New Debug ---")
        logger.debug(f"Total Words Analyzed: {total_words}")
        logger.debug(f"Word Hits: {word_hits} (using {len(slop_words_set)} slop words)")
        logger.debug(f"Bigram Hits: {bigram_hits} (using {len(slop_bigrams_set)} slop bigrams)")
        logger.debug(f"Trigram Hits: {trigram_hits} (using {len(slop_trigrams_set)} slop trigrams)")
        logger.debug(f"Weighted Hit Score: {total_slop_score}")
        logger.debug(f"Calculated Slop Index: {slop_index:.4f}")
        logger.debug("------------------------")

    return round(slop_index, 4)
