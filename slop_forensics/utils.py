import json
import logging
import os
import re
import unicodedata
from typing import List, Dict, Any, Set, Tuple, Union, Counter as TypingCounter
from collections import Counter

from .constants import WORD_PATTERN

logger = logging.getLogger(__name__)

# --- File I/O ---

def load_json_file(filename: str) -> Union[Dict, List, None]:
    """Loads data from a JSON file."""
    if not os.path.exists(filename):
        logger.warning(f"File not found: {filename}")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {filename}", exc_info=True)
        return None
    except IOError as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        return None

def save_json_file(data: Union[Dict, List], filename: str, indent: int = 2):
    """Saves data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Saved data to: {filename}")
    except IOError as e:
        logger.error(f"Error writing JSON to file {filename}: {e}", exc_info=True)
    except TypeError as e:
        logger.error(f"Data is not JSON serializable for file {filename}: {e}", exc_info=True)


def load_jsonl_file(filename: str, max_items: int = -1) -> List[Dict]:
    """Loads data from a JSON Lines file."""
    data = []
    if not os.path.exists(filename):
        logger.warning(f"JSONL file not found: {filename}")
        return data
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_items > 0 and i >= max_items:
                    logger.info(f"Reached max_items limit ({max_items}) for {filename}.")
                    break
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {filename}: {line}")
        logger.debug(f"Loaded {len(data)} items from {filename}.")
    except IOError as e:
        logger.error(f"Error reading JSONL file {filename}: {e}", exc_info=True)
    return data

def save_jsonl_file(data: List[Dict], filename: str):
    """Saves data to a JSON Lines file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.debug(f"Saved {len(data)} items to JSONL: {filename}")
    except IOError as e:
        logger.error(f"Error writing JSONL to file {filename}: {e}", exc_info=True)
    except TypeError as e:
         logger.error(f"Data contains non-JSON serializable items for file {filename}: {e}", exc_info=True)


def save_list_one_item_per_line(data: List[Any], filename: str):
    """Saves a list to JSON with each item on its own line (for slop lists)."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("[\n")
            if data:
                item_strs = [json.dumps(item, separators=(',', ':'), ensure_ascii=False) for item in data]
                f.write(",\n".join(item_strs))
            f.write("\n]")
        logger.info(f"Saved list with one item per line to: {filename}")
    except Exception as e:
        logger.error(f"Error saving list file {filename}: {e}", exc_info=True)

# --- Text Processing ---

def normalize_text(text: str) -> str:
    """Normalizes text: lowercase, unicode normalization, apostrophe standardization."""
    if not isinstance(text, str):
        return ""
    try:
        # Unicode normalization (NFKC recommended for compatibility)
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Standardize apostrophes
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace("ʼ", "'")
        # Optional: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.warning(f"Error during text normalization: {e}. Returning original text snippet: '{text[:50]}...'")
        return text # Return original on error


def extract_words(normalized_text: str, min_length: int = 4) -> List[str]:
    """Extracts words meeting criteria from normalized text using precompiled pattern."""
    if not isinstance(normalized_text, str):
        return []
    words = WORD_PATTERN.findall(normalized_text)
    return [
        word for word in words
        if len(word) >= min_length or "'" in word
    ]

# --- Misc ---

def sanitize_filename(name: str) -> str:
    """Sanitizes a string for use as a filename."""
    # Replace slashes first
    sanitized = name.replace("/", "__")
    # Remove other invalid characters
    sanitized = re.sub(r'[<>:"|?*\\ ]', '-', sanitized)
    # Remove leading/trailing hyphens/underscores
    sanitized = sanitized.strip('-_')
    # sanitized = sanitized[:max_len]
    return sanitized if sanitized else "invalid_name"

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')