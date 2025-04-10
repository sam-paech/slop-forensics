import sys
import os
import argparse
import logging

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from slop_forensics import config
from slop_forensics.dataset_generator import generate_for_model
from slop_forensics.utils import setup_logging, sanitize_filename

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Generate story chapter datasets using LLMs.")
    parser.add_argument(
        "--model-ids",
        type=str,
        default=None,
        help="Specify one or more model ids to process, comma separated."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.DATASET_OUTPUT_DIR,
        help=f"Directory to save generated datasets (default: {config.DATASET_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--generate-n",
        type=int,
        default=config.TARGET_RECORDS_PER_MODEL,
        help=f"Target number of records to generate per model (default: {config.TARGET_RECORDS_PER_MODEL})"
    )
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Please configure it in your .env file.")
        sys.exit(1)

    models_to_process = []
    if args.model_ids:
        models_to_process = args.model_ids.split(',')
    else:
        logger.error('--model-ids must be specified')
        sys.exit(1)

    logger.info(f"Starting dataset generation for models: {', '.join(models_to_process)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target records per model: {args.generate_n}")

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in models_to_process:
        try:
            generate_for_model(model_name, args.output_dir, args.generate_n)
        except Exception as e:
            logger.error(f"Critical error during generation for model {model_name}: {e}", exc_info=True)
            logger.error(f"Skipping remaining generation for {model_name} due to error.")
            continue # Move to the next model

    logger.info("Dataset generation script finished.")

if __name__ == "__main__":
    main()