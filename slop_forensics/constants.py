import re

# For word counting and analysis
WORD_PATTERN = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?")

KNOWN_CONTRACTIONS_S = {
    "it's", "that's", "what's", "who's", "he's", "she's",
    "there's", "here's", "where's", "when's", "why's", "how's",
    "let's"
}

# For filtering during repetition analysis
FORBIDDEN_SUBSTRINGS = {

}