import os
import requests
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple, Optional

from datasets import load_dataset
from tqdm import tqdm # Use standard tqdm here

from . import config
from .utils import save_jsonl_file, sanitize_filename, load_jsonl_file

logger = logging.getLogger(__name__)

# Lock for thread-safe file writing
file_lock = threading.Lock()

def _load_processed_ids(output_filename: str) -> Set[Tuple[str, int]]:
    """Loads processed (source, id) tuples from an existing output file."""
    processed_ids = set()
    if os.path.exists(output_filename):
        logger.info(f"Output file '{output_filename}' found. Loading processed IDs.")
        try:
            existing_data = load_jsonl_file(output_filename)
            for item in existing_data:
                 if isinstance(item, dict) and 'source' in item and 'id' in item:
                     processed_ids.add((item['source'], item['id']))
            logger.info(f"Loaded {len(processed_ids)} previously processed prompt IDs.")
        except Exception as e:
            logger.error(f"Error reading existing output file '{output_filename}': {e}. Continuing without resume.", exc_info=True)
            processed_ids = set() # Reset if file is corrupt
    return processed_ids

def load_and_prepare_prompts(output_filename: str) -> Tuple[List[Dict], Set[Tuple[str, int]]]:
    """Loads datasets and extracts prompts, handling resume logic."""
    all_prompts = []
    processed_ids = _load_processed_ids(output_filename)
    total_loaded = 0

    for source_name, dataset_id in config.DATASET_SOURCES.items():
        logger.info(f"Loading dataset: {dataset_id} (Source: {source_name})")
        try:
            # Load non-streaming first for easier length check and iteration
            ds = load_dataset(dataset_id, split='train', streaming=False)
            dataset_len = len(ds) # Get length if possible
            total_loaded += dataset_len
            logger.info(f"Loaded {dataset_len} rows from {source_name}.")

            for i, row in enumerate(tqdm(ds, desc=f"Processing {source_name}", unit="prompts")):
                if (source_name, i) in processed_ids:
                    continue
                try:
                    prompt_text = None
                    # Adapt based on dataset structure
                    if source_name == "Nitral-AI":
                        conversations = row.get('conversations')
                        if isinstance(conversations, list):
                            for msg in conversations:
                                if isinstance(msg, dict) and msg.get('from') == 'human':
                                    prompt_text = msg.get('value')
                                    break
                        elif isinstance(conversations, str): # Handle potential stringified JSON
                             try:
                                 conv_list = json.loads(conversations)
                                 for msg in conv_list:
                                     if isinstance(msg, dict) and msg.get('from') == 'human':
                                         prompt_text = msg.get('value')
                                         break
                             except json.JSONDecodeError:
                                 logger.warning(f"Could not parse 'conversations' string in {source_name} row {i}.")

                    elif source_name == "llm-aes":
                        prompt_text = row.get('prompt')

                    # Validate and add
                    if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                        all_prompts.append({
                            "source": source_name,
                            "id": i,
                            "prompt": prompt_text.strip()
                        })
                    else:
                        logger.debug(f"No valid prompt found in {source_name} row {i}. Content: {row}")

                except Exception as e:
                    logger.error(f"Error processing row {i} from {source_name}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to load or process dataset {dataset_id}: {e}", exc_info=True)

    logger.info(f"Total prompts loaded across sources: {total_loaded} (before filtering/resume)")
    logger.info(f"Total prompts to process (after filtering/resume): {len(all_prompts)}")
    if not all_prompts:
        logger.warning("No prompts available to process. Check dataset loading and resume logic.")

    return all_prompts, processed_ids

def _call_api(prompt_details: dict, model_name: str) -> Optional[Dict]:
    """Internal function to call the API with retries."""
    source = prompt_details['source']
    row_id = prompt_details['id']
    prompt_text = prompt_details['prompt']

    user_prompt = f"{config.USER_PROMPT_TEMPLATE}\n\n[writing prompt]: {prompt_text}"

    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Optional, some models might require it
        "X-Title": "Slop Forensics",      # Optional
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
        "stream": False
    }

    for attempt in range(1, config.API_RETRIES + 1):
        try:
            response = requests.post(
                f"{config.OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.API_TIMEOUT
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            if "choices" not in data or not data["choices"]:
                 logger.warning(f"API response for {source}-{row_id} missing 'choices'. Response: {data}")
                 time.sleep(3 * attempt)
                 continue

            llm_response = data["choices"][0].get("message", {}).get("content")
            if not llm_response or not isinstance(llm_response, str):
                logger.warning(f"API response for {source}-{row_id} missing content. Response: {data}")
                time.sleep(3 * attempt)
                continue

            llm_response_stripped = llm_response.strip()

            if len(llm_response_stripped) < config.MIN_OUTPUT_LENGTH:
                logger.debug(f"Output for {source}-{row_id} too short ({len(llm_response_stripped)} chars). Discarding.")
                return None # Success, but too short

            logger.debug(f"Successfully generated for {source}-{row_id} (attempt {attempt})")
            return {
                "source": source,
                "id": row_id,
                "prompt": prompt_text,
                "model": model_name, # Add model name to output
                "output": llm_response_stripped
            }

        except requests.exceptions.Timeout:
            logger.warning(f"API request timed out for {source}-{row_id} on attempt {attempt}/{config.API_RETRIES}.")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logger.warning(f"API request failed for {source}-{row_id} (Attempt {attempt}/{config.API_RETRIES}, Status: {status_code}): {e}")
            try:
                error_details = e.response.json()
                logger.warning(f"Error details: {error_details}")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse error response body: {e.response.text}")

            if status_code == 429: # Rate limit
                wait_time = 15 * attempt
                logger.warning(f"Rate limit likely hit. Waiting {wait_time}s before retry.")
                time.sleep(wait_time)
            elif status_code >= 500: # Server error
                wait_time = 5 * attempt
                logger.warning(f"Server error encountered. Waiting {wait_time}s before retry.")
                time.sleep(wait_time)
            elif status_code == 401: # Unauthorized
                 logger.error("API Key invalid or missing. Stopping generation.")
                 raise ValueError("Invalid API Key") # Stop the process
            elif status_code == 400: # Bad request (e.g., model not found, bad params)
                 logger.error(f"Bad request (400) for {model_name}. Check model name and parameters. Stopping generation for this model.")
                 # Decide whether to stop all or just this model. Here, we stop for this model.
                 return {"error": "Bad Request", "source": source, "id": row_id} # Signal error
            else: # Other client errors
                time.sleep(3 * attempt)
        except requests.exceptions.RequestException as e:
             logger.warning(f"General request error for {source}-{row_id} (Attempt {attempt}/{config.API_RETRIES}): {e}")
             time.sleep(3 * attempt)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response for {source}-{row_id} (Attempt {attempt}): {e}. Response text: {response.text if 'response' in locals() else 'N/A'}", exc_info=True)
            time.sleep(3 * attempt)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during API call for {source}-{row_id} (Attempt {attempt}): {e}", exc_info=True)
            time.sleep(5 * attempt)


    logger.error(f"Failed to generate story for {source}-{row_id} after {config.API_RETRIES} attempts.")
    return None # Failed after retries

def _save_results_batch(results_batch: List[Dict], filename: str):
    """Appends a batch of results to the JSON Lines file thread-safely."""
    if not results_batch:
        return
    with file_lock:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'a', encoding='utf-8') as f_out:
                for result in results_batch:
                    # Don't save error markers
                    if not isinstance(result, dict) or "error" not in result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.debug(f"Saved batch of {len(results_batch)} results to {filename}")
        except IOError as e:
            logger.error(f"Error writing to output file {filename}: {e}", exc_info=True)
        except TypeError as e:
             logger.error(f"Data contains non-JSON serializable items for file {filename}: {e}", exc_info=True)


def generate_for_model(model_name: str, output_dir: str = config.DATASET_OUTPUT_DIR, 
                   target_records: int = config.TARGET_RECORDS_PER_MODEL,
                   max_workers: int = config.MAX_WORKERS):
    """Generates dataset for a single specified model.
    
    Args:
        model_name: The name of the model to use for generation
        output_dir: Directory to save the generated dataset
        target_records: Target number of records to generate
        max_workers: Number of worker threads to use
    """
    logger.info(f"Starting generation process for model: {model_name}")
    sanitized_model_name = sanitize_filename(model_name)
    output_filename = os.path.join(output_dir, f"generated_{sanitized_model_name}.jsonl")

    # 1. Load prompts and handle resuming
    prompts_to_process, already_processed_ids = load_and_prepare_prompts(output_filename)
    total_prompts_available = len(prompts_to_process)
    already_saved_count = len(already_processed_ids)

    prompts_needed = max(0, target_records - already_saved_count)

    if prompts_needed == 0:
        logger.info(f"Target of {target_records} records already met or exceeded ({already_saved_count} saved). Skipping generation for {model_name}.")
        return

    if total_prompts_available == 0:
        logger.warning(f"No new prompts available to process for {model_name}. Cannot reach target.")
        return

    # Limit prompts to what's needed and available
    prompts_to_process = prompts_to_process[:min(prompts_needed, total_prompts_available)]
    num_tasks = len(prompts_to_process)
    logger.info(f"Need {prompts_needed} more records. Will process {num_tasks} available prompts.")


    logger.info(f"Initializing ThreadPoolExecutor with {max_workers} workers.")
    results_buffer = []
    futures = []
    processed_count_session = 0
    saved_count_session = 0
    encountered_error = False

    # Use try-with-resources for the executor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit tasks
            for prompt_detail in prompts_to_process:
                future = executor.submit(_call_api, prompt_detail, model_name)
                futures.append(future)
            logger.info(f"Submitted {len(futures)} tasks to the executor for {model_name}.")

            # Process completed tasks
            with tqdm(total=num_tasks, desc=f"Generating ({model_name})", unit="prompt") as pbar:
                for future in as_completed(futures):
                    result = None
                    try:
                        result = future.result()
                    except ValueError as e: # Catch specific errors like Invalid API Key
                         logger.error(f"Stopping generation for {model_name} due to critical error: {e}")
                         encountered_error = True
                         # Attempt to cancel remaining futures
                         for f in futures:
                             if not f.done():
                                 f.cancel()
                         break # Exit the loop
                    except Exception as e:
                        logger.error(f"Error retrieving result from future: {e}", exc_info=True)
                        # Optionally mark this prompt as failed if needed

                    processed_count_session += 1
                    pbar.update(1)

                    if result:
                        if isinstance(result, dict) and result.get("error") == "Bad Request":
                            logger.error(f"Encountered Bad Request (400) for model {model_name}. Stopping generation for this model.")
                            encountered_error = True
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break # Exit the loop
                        elif isinstance(result, dict): # Valid result (not None, not error marker)
                            results_buffer.append(result)
                            saved_count_session += 1
                            if len(results_buffer) >= config.SAVE_EVERY_N:
                                logger.info(f"Buffer full ({len(results_buffer)} items). Saving batch...")
                                _save_results_batch(results_buffer, output_filename)
                                results_buffer = [] # Clear buffer after saving

                    total_saved = already_saved_count + saved_count_session
                    pbar.set_postfix({"saved_total": total_saved, "processed_session": processed_count_session})

                    # Check if target reached
                    if total_saved >= target_records:
                        logger.info(f"Target of {target_records} records reached for {model_name}. Stopping processing.")
                        # Cancel remaining futures gracefully
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break # Exit the as_completed loop

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Shutting down gracefully...")
            # Executor is shut down automatically by context manager
        except Exception as e:
            logger.error(f"An unexpected error occurred during processing for {model_name}: {e}", exc_info=True)
            # Executor is shut down automatically
        finally:
            # Final save of any remaining results in the buffer
            if results_buffer:
                logger.info(f"Performing final save of {len(results_buffer)} remaining results for {model_name}...")
                _save_results_batch(results_buffer, output_filename)

            total_saved_final = already_saved_count + saved_count_session
            logger.info(f"Generation process finished for {model_name}. "
                        f"Prompts processed in this run: {processed_count_session}. "
                        f"Results saved in this run: {saved_count_session}. "
                        f"Total results saved: {total_saved_final}.")
            if encountered_error:
                 logger.warning(f"Generation for {model_name} stopped prematurely due to errors.")