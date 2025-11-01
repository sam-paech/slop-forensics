import json
import logging
import os
import re
import unicodedata
from typing import List, Dict, Any, Union

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


# --- Text Processing ---------------------------------------------------
#
#  All tokenisation / normalisation now aligns with the other repo's
#  preprocessing: quote normalization, regex tokenization with apostrophes
#  preserved internally, then stripped from leading/trailing positions.

# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #
_SPACES_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z']+")

# Quote normalization mappings (align with other repo)
_QUOTE_MAP = {
    # Single quotes
    ''': "'", ''': "'", '‛': "'", '‚': "'",
    '′': "'", 'ʼ': "'", '＇': "'", '`': "'",
    # Double quotes
    '"': '"', '"': '"', '„': '"', '‟': '"',
    '″': '"', '«': '"', '»': '"', '＂': '"'
}

def _normalize_quotes(text: str) -> str:
    """
    Normalize exotic/curly quotes to ASCII quotes.
    Maps all variants of single quotes to ' and double quotes to "
    """
    for exotic, ascii_char in _QUOTE_MAP.items():
        text = text.replace(exotic, ascii_char)
    return text

# ------------------------------------------------------------------ #
# Public functions
# ------------------------------------------------------------------ #

def normalize_text(text: str) -> str:
    """
    Normalize text for ngram extraction and word analysis.

    Steps:
    1. NFKC Unicode normalization
    2. Quote normalization (curly → straight)
    3. Lowercase conversion

    Does NOT tokenize - use extract_words() or tokenize directly with TOKEN_RE.
    """
    if not isinstance(text, str):
        return ""
    try:
        # NFC / NFKC keeps composed + decomposed chars comparable
        text = unicodedata.normalize("NFKC", text)
        # Normalize quotes BEFORE tokenization
        text = _normalize_quotes(text)
        # Lowercase
        text = text.lower()
        return text
    except Exception as exc:
        logger.warning("Error during text normalization: %s. "
                       "Returning raw snippet '%s…'",
                       exc, text[:50])
        return text


def extract_words(normalized_text: str,
                  min_length: int = 1) -> List[str]:
    """
    Tokenize normalized text using regex pattern [a-zA-Z']+

    Steps:
    1. Extract tokens matching [a-zA-Z']+ pattern
    2. Strip leading/trailing apostrophes from each token
    3. Filter by min_length (default 1 to keep all words)

    Args:
        normalized_text: Text already passed through normalize_text()
        min_length: Minimum token length (default 1)

    Returns:
        List of tokens meeting the length requirement
    """
    if not isinstance(normalized_text, str):
        return []

    tokens = []
    for match in _TOKEN_RE.findall(normalized_text):
        # Strip leading/trailing apostrophes
        token = match.strip("'")
        if len(token) >= min_length:
            tokens.append(token)

    return tokens

# ------------------------------------------------------------------ #
# End TEXT-PROCESSING section                                        #
# ------------------------------------------------------------------ #



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