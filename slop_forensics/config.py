import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & Paths ---
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PHYLIP_PATH = os.getenv("PHYLIP_PATH") # Optional, used if PHYLIP not in system PATH

# --- Base Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Output Subdirectories ---
DATASET_OUTPUT_DIR = os.path.join(RESULTS_DIR, "datasets")
ANALYSIS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
SLOP_LIST_OUTPUT_DIR = os.path.join(RESULTS_DIR, "slop_lists")
PHYLOGENY_OUTPUT_DIR = os.path.join(RESULTS_DIR, "phylogeny")
PHYLOGENY_CHARTS_DIR = os.path.join(PHYLOGENY_OUTPUT_DIR, "charts")

# --- Output Files ---
COMBINED_METRICS_FILE = os.path.join(RESULTS_DIR, "slop_profile_results.json") # Output of analysis script

# --- Dataset Generation ---
# Hugging Face Datasets for prompts
DATASET_SOURCES = {
    "Nitral-AI": "Nitral-AI/Reddit-SFW-Writing_Prompts_ShareGPT",
    #"llm-aes": "llm-aes/writing-prompts"
}



# Generation Parameters
SYSTEM_PROMPT = "You are a helpful writing assistant. Your goal is to write compelling story chapters based on user prompts."
USER_PROMPT_TEMPLATE = "write one chapter in a larger story, using this prompt as general inspiration. Approximately 800 words. Only output the chapter text, with no extra commentary before or after."
TEMPERATURE = 0.7
MAX_TOKENS = 4096 # Adjust based on expected word count and model limits
MIN_OUTPUT_LENGTH = 500 # Minimum character length for generated output

# Concurrency & Saving
MAX_WORKERS = 500 # Adjust based on API rate limits and system resources
SAVE_EVERY_N = 20
API_RETRIES = 5
API_TIMEOUT = 180 # seconds
TARGET_RECORDS_PER_MODEL = 1000 # Target number of records to generate per model

# --- Analysis Settings ---
# For slop list generation and repetition metrics
ANALYSIS_MAX_ITEMS_PER_MODEL = 10000 # Max items to load from dataset for analysis
WORD_MIN_LENGTH = 4 # Min length for word counting (unless it has an apostrophe)
WORD_MIN_REPETITION_COUNT = 5 # Min times a word must appear overall to be considered repetitive
WORD_MIN_PROMPT_IDS = 2 # Min unique prompts a word must appear in to be considered repetitive
NGRAM_MIN_PROMPT_IDS = 2 # Min unique prompts an N-gram must appear in
TOP_N_WORDS_REPETITION = 1000 # How many top repetitive words to store in analysis file
TOP_N_BIGRAMS = 200
TOP_N_TRIGRAMS = 200
COMMON_WORD_THRESHOLD = 1.2e-5 # Wordfreq threshold to filter common words in slop lists
STOPWORD_LANG = 'english'

# --- Slop List Creation Settings ---
SLOP_LIST_TOP_N_OVERREP = 1500 # Number of over-represented words for final slop list
SLOP_LIST_TOP_N_ZERO_FREQ = 500 # Number of zero-frequency words for final slop list
SLOP_LIST_TOP_N_BIGRAMS = 1000
SLOP_LIST_TOP_N_TRIGRAMS = 1000
# Phrase-extraction settings:
SLOP_PHRASES_NGRAM_SIZE = 3             # e.g. 3 for trigrams
SLOP_PHRASES_TOP_NGRAMS = 1000          # top n-grams to keep from the "cleaned" analysis
SLOP_PHRASES_TOP_PHRASES_TO_SAVE = 10000
SLOP_PHRASES_CHUNKSIZE = 50
SLOP_PHRASES_MAX_PROCESSES = 8          # or any limit you want, e.g. min(cpu_count, 8)

# --- Phylogeny Settings ---
PHYLO_TOP_N_FEATURES = 1500 # Total features (words+bigrams+trigrams) per model for tree building
PHYLO_RUN_CONSENSE = True # Whether to run PHYLIP 'consense'
PHYLO_MODELS_TO_IGNORE = [ # Models to exclude from phylogeny

]
# Model name substitutions for cleaner tree labels
PHYLO_MODEL_NAME_SUBS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    # Add more substitutions as needed
}
# Colors for model families in trees
PHYLO_FAMILY_COLORS = {
    "Llama": "#1f77b4",
    "Gemma": "#ff7f0e",
    "Mistral": "#2ca02c",
    "Qwen": "#d62728",
    "GPT": "#9467bd",
    "Claude": "#8c564b",
    "Other": "#7f7f7f",
}
# Function to determine family from model name (customize as needed)
def get_model_family(model_name):
    name_lower = model_name.lower()
    if "llama" in name_lower: return "Llama"
    if "gemma" in name_lower: return "Gemma"
    if "mistral" in name_lower or "mixtral" in name_lower: return "Mistral"
    if "qwen" in name_lower: return "Qwen"
    if "gpt" in name_lower: return "GPT"
    if "claude" in name_lower: return "Claude"
    return "Other"

# --- Ensure Output Directories Exist ---
os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(SLOP_LIST_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHYLOGENY_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHYLOGENY_CHARTS_DIR, exist_ok=True)

# --- Validation ---
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables or .env file.")