import sys
import os
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from slop_forensics import config
from slop_forensics.phylogeny import generate_phylogenetic_trees
from slop_forensics.utils import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Generate phylogenetic trees based on model features.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=config.COMBINED_METRICS_FILE,
        help=f"Path to the combined metrics JSON file (output of analyze_outputs.py) (default: {config.COMBINED_METRICS_FILE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.PHYLOGENY_OUTPUT_DIR,
        help=f"Directory to save phylogeny results (trees, charts) (default: {config.PHYLOGENY_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=config.PHYLO_TOP_N_FEATURES,
        help=f"Total number of top features (words+bigrams+trigrams) per model to use for tree building (default: {config.PHYLO_TOP_N_FEATURES})"
    )

    args = parser.parse_args()

    logger.info(f"Starting phylogenetic tree generation using data from: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Top N features per model: {args.top_n_features}")

    if not os.path.exists(args.input_file):
        logger.error(f"Input metrics file not found: {args.input_file}")
        logger.error("Please run the analysis script (02_analyze_outputs.py) first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.PHYLOGENY_CHARTS_DIR, exist_ok=True) # Ensure charts subdir exists

    try:
        generate_phylogenetic_trees(
            metrics_file=args.input_file,
            output_dir=args.output_dir,
            charts_dir=config.PHYLOGENY_CHARTS_DIR, # Pass explicitly
            top_n_features=args.top_n_features,
            models_to_ignore=config.PHYLO_MODELS_TO_IGNORE
        )
    except Exception as e:
        logger.error(f"Error during phylogenetic tree generation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Phylogenetic tree generation script finished.")

if __name__ == "__main__":
    main()