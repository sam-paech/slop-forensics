import sys
import os
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from slop_forensics import config
from slop_forensics.slop_lists import create_slop_lists
from slop_forensics.utils import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Combine model analyses to create slop lists (and slop phrases).")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.ANALYSIS_OUTPUT_DIR,
        help=f"Directory containing per-model analysis JSON files (default: {config.ANALYSIS_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.SLOP_LIST_OUTPUT_DIR,
        help=f"Directory to save the final slop lists (default: {config.SLOP_LIST_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=config.ANALYSIS_MAX_ITEMS_PER_MODEL, # Reuse analysis limit for reloading
        help=f"Maximum number of items to reload per model dataset (default: {config.ANALYSIS_MAX_ITEMS_PER_MODEL})"
    )

    args = parser.parse_args()

    logger.info(f"Starting slop list creation from analysis files in: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        create_slop_lists(
            analysis_files_dir=args.input_dir,
            output_dir=args.output_dir,
            max_items_per_model=args.max_items
        )
    except Exception as e:
        logger.error(f"Error during slop list creation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Slop list creation script finished.")

if __name__ == "__main__":
    main()
