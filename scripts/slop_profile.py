import sys
import os
import argparse
import logging
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tqdm import tqdm

from slop_forensics import config
from slop_forensics.analysis import analyze_texts
from slop_forensics.utils import (
    setup_logging, load_jsonl_file, save_json_file,
    sanitize_filename, load_json_file
)

def log_top_patterns(logger, analysis_results, top_n=5):
    """Concise console preview of each model’s most interesting patterns."""
    model = analysis_results.get("model_name", "Unknown")
    logger.info(f"MODEL: {model}")

    # ---------- non-zero over-rep ----------
    words = analysis_results.get("top_repetitive_words", [])
    if words:
        wlist = [f"'{w['word']}'" for w in words[:top_n]]
        logger.info("WORDS: " + ", ".join(wlist))
    else:
        logger.info("WORDS: None")

    # ---------- bigrams ----------
    bigrams = analysis_results.get("top_bigrams", [])
    if bigrams:
        blist = [f"'{b['ngram']}'" for b in bigrams[:top_n]]
        logger.info("BIGRAMS: " + ", ".join(blist))
    else:
        logger.info("BIGRAMS: None")

    # ---------- trigrams ----------
    trigrams = analysis_results.get("top_trigrams", [])
    if trigrams:
        tlist = [f"'{t['ngram']}'" for t in trigrams[:top_n]]
        logger.info("TRIGRAMS: " + ", ".join(tlist))
    else:
        logger.info("TRIGRAMS: None")

    # ---------- zero-freq count summary ----------
    zf_count = len(analysis_results.get("zero_frequency_words", []))
    logger.info(f"ZERO-FREQ WORDS: {zf_count}")
    logger.info("---")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Analyze generated model outputs for metrics and features.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.DATASET_OUTPUT_DIR,
        help=f"Directory containing generated .jsonl datasets (default: {config.DATASET_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=config.ANALYSIS_OUTPUT_DIR,
        help=f"Directory to save per-model analysis JSON files (default: {config.ANALYSIS_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--combined-output-file",
        type=str,
        default=config.COMBINED_METRICS_FILE,
        help=f"File to save combined metrics (merged with existing ELO if applicable) (default: {config.COMBINED_METRICS_FILE})"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=config.ANALYSIS_MAX_ITEMS_PER_MODEL,
        help=f"Maximum number of items to load per model dataset for analysis (default: {config.ANALYSIS_MAX_ITEMS_PER_MODEL})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top words/bigrams/trigrams to log (default: 5)"
    )
    args = parser.parse_args()

    logger.info(f"Starting analysis of datasets in: {args.input_dir}")
    logger.info(f"Analysis output directory: {args.analysis_output_dir}")
    logger.info(f"Combined metrics output file: {args.combined_output_file}")
    logger.info(f"Max items per model: {args.max_items}")
    logger.info(f"Will log top {args.top_n} patterns per model")

    os.makedirs(args.analysis_output_dir, exist_ok=True)

    # --- Find dataset files ---
    dataset_files = [f for f in os.listdir(args.input_dir) if f.startswith("generated_") and f.endswith(".jsonl")]
    if not dataset_files:
        logger.error(f"No dataset files found in {args.input_dir}. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(dataset_files)} dataset files to analyze.")

    all_models_metrics = {}
    all_models_analysis = {}  # Store analysis results for summary at the end

    # --- Load existing combined data (e.g., ELO scores) if it exists ---
    # This allows merging new metrics with previous results.
    existing_combined_data = load_json_file(args.combined_output_file) or {}
    if existing_combined_data:
         logger.info(f"Loaded existing combined data for {len(existing_combined_data)} models from {args.combined_output_file}")
         # Copy existing data to start with
         all_models_metrics = existing_combined_data.copy()


    # --- Process each dataset file ---
    for filename in tqdm(dataset_files, desc="Analyzing Models"):
        filepath = os.path.join(args.input_dir, filename)
        logger.info(f"Processing file: {filename}")

        # Load data for the model
        model_data = load_jsonl_file(filepath, max_items=args.max_items)
        if not model_data:
            logger.warning(f"No data loaded from {filename}. Skipping.")
            continue

        # Infer model name from filename or data (prefer data if available)
        model_name = model_data[0].get("model")
        if not model_name:
            # Fallback: try to parse from filename "generated_provider__model_name.jsonl"
            try:
                sanitized_name = filename.replace("generated_", "").replace(".jsonl", "")
                model_name = sanitized_name.replace("__", "/") # Simple reverse sanitization
                logger.warning(f"Model name not found in data, inferred from filename: {model_name}")
            except Exception:
                 logger.error(f"Could not determine model name for {filename}. Skipping.")
                 continue

        logger.info(f"Analyzing model: {model_name} ({len(model_data)} items)")

        # Prepare data for analysis functions
        # texts_with_ids: List[Tuple[str, str]] - (text, prompt_id)
        # prompts_data: Dict[str, List[str]] - {prompt_id: [text1, text2]}
        texts_with_ids = []
        prompts_data = defaultdict(list)
        unique_prompts = set()

        for item in model_data:
            text = item.get("output")
            prompt_id = f"{item.get('source', 'unknown')}_{item.get('id', 'unknown')}" # Create unique prompt ID
            if text and isinstance(text, str):
                texts_with_ids.append((text, prompt_id))
                prompts_data[prompt_id].append(text)
                unique_prompts.add(prompt_id)

        if not texts_with_ids:
            logger.warning(f"No valid text entries found for {model_name}. Skipping analysis.")
            continue

        logger.debug(f"Model {model_name}: {len(texts_with_ids)} texts from {len(unique_prompts)} unique prompts.")

        # Perform analysis
        try:
            analysis_results = analyze_texts(model_name, texts_with_ids, prompts_data)
            # Store analysis results for summary
            all_models_analysis[model_name] = analysis_results
        except Exception as e:
            logger.error(f"Error during analysis for {model_name}: {e}", exc_info=True)
            continue # Skip to next model on error

        # Save individual analysis file
        analysis_filename = os.path.join(args.analysis_output_dir, f"slop_profile__{sanitize_filename(model_name)}.json")
        save_json_file(analysis_results, analysis_filename)

        # Merge results into the combined dictionary
        # If model already exists (from loaded ELO), update its dict, otherwise add it
        if model_name in all_models_metrics:
            all_models_metrics[model_name].update(analysis_results)
        else:
            all_models_metrics[model_name] = analysis_results

    # --- Save the final combined metrics file ---
    if all_models_metrics:
        logger.info(f"Saving combined metrics for {len(all_models_metrics)} models to {args.combined_output_file}")
        save_json_file(all_models_metrics, args.combined_output_file)
    else:
        logger.warning("No models were successfully analyzed. Combined metrics file not saved.")

    # --- Log summary of top patterns for each model ---
    logger.info("\n========== SUMMARY OF TOP PATTERNS ==========")
    for model_name, analysis_results in all_models_analysis.items():
        log_top_patterns(logger, analysis_results, top_n=args.top_n)
    logger.info("============== END SUMMARY ===============")

    logger.info("Analysis script finished.")
    
    # Display file locations for reference
    logger.info("\nFull results are available at:")
    logger.info(f"- Combined metrics file: {os.path.abspath(args.combined_output_file)}")
    logger.info(f"- Individual analysis files: {os.path.abspath(args.analysis_output_dir)}/analysis_*.json")

if __name__ == "__main__":
    main()