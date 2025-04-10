import os
import json
import subprocess
import tempfile
import shutil
import logging
from typing import Dict, Optional, List, Tuple, Set

import os
# set this env var so we can render images without a screen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree
from ete3 import Tree, TreeStyle, NodeStyle, TextFace, faces
import numpy as np
from tqdm import tqdm

from . import config
from .utils import load_json_file, save_json_file, sanitize_filename

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_updated_model_name(original: str) -> str:
    """Get updated model name from config substitutions."""
    return config.PHYLO_MODEL_NAME_SUBS.get(original, original)

def _layout_fn_with_highlight(node, focus_model_name: Optional[str] = None, highlight_color: str = "#FF0000"):
    """ETE3 layout function for coloring nodes by family and highlighting."""
    if not node.is_leaf():
        # Internal node style
        style = NodeStyle(size=0, hz_line_width=1, vt_line_width=1) # Thinner lines
        node.set_style(style)
        return

    leaf_label = node.name
    # Use original_name if PHYLIP code was mapped back
    if hasattr(node, 'original_name') and node.original_name:
        leaf_label = node.original_name

    updated_label = _get_updated_model_name(leaf_label)
    family = config.get_model_family(leaf_label) # Use function from config
    circle_color = config.PHYLO_FAMILY_COLORS.get(family, "#cccccc") # Use colors from config

    # Highlight focus model
    if leaf_label == focus_model_name:
        circle_color = highlight_color
        text_face = TextFace(updated_label, fsize=8, fgcolor=highlight_color, bold=True) # Smaller font
    else:
        text_face = TextFace(updated_label, fsize=8, fgcolor="black") # Smaller font

    # Add label slightly offset from node
    faces.add_face_to_node(text_face, node, column=0, position="branch-right")

    # Node style
    style = NodeStyle(size=6, fgcolor=circle_color, shape="circle", hz_line_width=1, vt_line_width=1) # Smaller nodes/lines
    node.set_style(style)


def _render_ete_tree_focus(ete_tree: Tree, focus_model_name: str, output_image: str, layout: str = "c"):
    """Renders ETE3 tree with highlighting."""
    ts = TreeStyle()
    ts.mode = layout
    ts.show_leaf_name = False # Labels are added by layout function
    ts.show_branch_length = False
    ts.show_scale = False
    ts.scale = None # Explicitly disable scale bar drawing

    if layout == 'c':  # For circular layout
        # Fix minimum branch length for very small trees
        if len(ete_tree.get_leaves()) <= 10:
            min_branch_length = 4
            for node in ete_tree.traverse():
                if node.dist < min_branch_length:
                    node.dist = min_branch_length

    if layout == 'r':
        ts.branch_vertical_margin = 8 # Adjust spacing for smaller font
        width = 600 # Adjust width for rectangular
    else: # circular
        width = 1200 # Adjust width for circular

    # Define the custom layout function dynamically
    def dynamic_layout(node):
        _layout_fn_with_highlight(node, focus_model_name)

    ts.layout_fn = dynamic_layout

    try:
        ete_tree.render(output_image, w=width, units="px", tree_style=ts)
        logger.info(f"Saved {layout.upper()} tree '{os.path.basename(output_image)}' (highlight: {focus_model_name})")
    except Exception as e:
        logger.error(f"Failed to render tree to {output_image}: {e}", exc_info=True)


def _run_phylip_command(command: List[str], input_text: str, env: Optional[Dict] = None, timeout: int = 300, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    logger.debug(f"Running PHYLIP command: {' '.join(command)} in {cwd or os.getcwd()}")
    logger.debug(f"Environment PATH: {env.get('PATH', 'Not set')}")
    logger.debug(f"Files in working directory before command: {os.listdir(cwd or os.getcwd())}")
    
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
            cwd=cwd,
            check=False
        )
        
        logger.debug(f"{command[0]} STDOUT FULL:\n{result.stdout}")
        if result.stderr:
            logger.error(f"{command[0]} STDERR FULL:\n{result.stderr}")
            
        # Check files after command execution
        logger.debug(f"Files in working directory after command: {os.listdir(cwd or os.getcwd())}")
        
        if result.returncode != 0:
            logger.error(f"PHYLIP command '{command[0]}' failed with exit code {result.returncode}")
            
        return result
    except FileNotFoundError:
        logger.error(f"PHYLIP command '{command[0]}' not found. Is PHYLIP installed and in PATH or PHYLIP_PATH?")
        raise # Re-raise to signal failure
    except subprocess.TimeoutExpired:
        logger.error(f"PHYLIP command '{command[0]}' timed out after {timeout} seconds.")
        # Create a dummy result to indicate timeout
        return subprocess.CompletedProcess(command, -1, stdout="Timeout", stderr="Timeout")
    except Exception as e:
        logger.error(f"Error running PHYLIP command '{command[0]}': {e}", exc_info=True)
        return subprocess.CompletedProcess(command, -1, stdout=str(e), stderr=str(e))


def _build_parsimony_tree(
    model_features: Dict[str, Set[str]],
    output_dir: str = config.PHYLOGENY_OUTPUT_DIR,
    charts_dir: str = config.PHYLOGENY_CHARTS_DIR,
    phylip_path: Optional[str] = config.PHYLIP_PATH,
    run_consense: bool = config.PHYLO_RUN_CONSENSE
) -> Optional[Tree]:
    """Attempts to build a tree using PHYLIP parsimony."""
    logger.info("Attempting parsimony tree construction using PHYLIP...")

    all_models = sorted(model_features.keys())
    if len(all_models) < 3: # Parsimony needs at least 3 taxa
        logger.warning("Parsimony requires at least 3 models with features. Skipping.")
        return None

    # Build global vocabulary
    global_vocab = sorted(list(set.union(*model_features.values())))
    n_taxa = len(all_models)
    n_chars = len(global_vocab)
    logger.info(f"Parsimony analysis: {n_taxa} models, {n_chars} features.")

    temp_dir = tempfile.mkdtemp(prefix="phylip_pars_")
    logger.debug(f"Using temporary directory for PHYLIP: {temp_dir}")
    final_tree_path = None # Path to the final tree file (outtree or outtree_consensus)

    try:
        # --- Create PHYLIP input file and translation map ---
        code_to_model = {}
        model_to_code = {}
        # Use shorter codes if many models, ensure fixed length
        code_len = max(4, len(str(n_taxa)))
        code_format = f"M{{:0{code_len}d}}" # e.g., M0001 or M01

        for i, model in enumerate(all_models):
            code = code_format.format(i + 1)
            # Ensure code is <= 10 chars for PHYLIP
            if len(code) > 10:
                 logger.error("Generated model code exceeds 10 characters. PHYLIP might fail.")
                 # Use a simple index as fallback, hoping it's unique enough
                 code = str(i+1).zfill(10)

            code_to_model[code] = model
            model_to_code[model] = code

        translation_file = os.path.join(output_dir, "parsimony_model_codes.json")
        save_json_file({"code_to_model": code_to_model, "model_to_code": model_to_code}, translation_file)

        phylip_infile_path = os.path.join(temp_dir, "infile")
        with open(phylip_infile_path, "w") as f:
            f.write(f" {n_taxa} {n_chars}\n") # Note space padding
            for model in all_models:
                code = model_to_code[model]
                feats = model_features[model]
                bitstring = "".join(["1" if feat in feats else "0" for feat in global_vocab])
                # Pad code to 10 chars
                f.write(f"{code:<10}{bitstring}\n")
        shutil.copy(phylip_infile_path, os.path.join(output_dir, "parsimony_infile")) # Save copy

        # --- Prepare PHYLIP environment ---
        env = os.environ.copy()
        paths_to_check = [phylip_path] if phylip_path else []
        paths_to_check.extend(["/usr/local/bin", "/usr/lib/phylip/bin", "/opt/homebrew/bin"]) # Common locations
        found_phylip_dir = None
        for p_dir in paths_to_check:
             if p_dir and os.path.exists(os.path.join(p_dir, "pars")):
                 found_phylip_dir = p_dir
                 break

        if found_phylip_dir:
            logger.info(f"Using PHYLIP executables from: {found_phylip_dir}")
            env["PATH"] = f"{found_phylip_dir}{os.pathsep}{env.get('PATH', '')}"
        else:
            # Check if 'pars' is already in PATH
            which_result = subprocess.run(["which", "pars"], capture_output=True, text=True, env=env)
            if not which_result.stdout.strip():
                 logger.error("Could not find PHYLIP 'pars' executable in system PATH or configured PHYLIP_PATH.")
                 return None # Cannot proceed
            logger.info("Found PHYLIP 'pars' in system PATH.")


        # --- Run 'pars' ---
        logger.info("Running PHYLIP 'pars'...")
        # Input 'Y\n' accepts defaults (no jumbling, etc.)        
        pars_input = "Y\n"
        pars_result = _run_phylip_command(["pars"], pars_input, env=env, cwd=temp_dir)

        if pars_result.returncode != 0:
            logger.error("'pars' command failed.")
            return None

        # Check for output files
        pars_outfile = os.path.join(temp_dir, "outfile")
        pars_outtree = os.path.join(temp_dir, "outtree")

        if not os.path.exists(pars_outtree):
            logger.error("'pars' did not produce an 'outtree' file.")
            if os.path.exists(pars_outfile): shutil.copy(pars_outfile, os.path.join(output_dir, "parsimony_outfile_error"))
            return None

        # Copy results
        shutil.copy(pars_outfile, os.path.join(output_dir, "parsimony_outfile"))
        shutil.copy(pars_outtree, os.path.join(output_dir, "parsimony_outtree_raw"))
        final_tree_path = os.path.join(output_dir, "parsimony_outtree_raw") # Default if no consense

        # --- Run 'consense' to merge multiple discovered trees ---
        if run_consense:
            logger.info("Running PHYLIP 'consense'...")
            # consense expects input tree(s) in 'intree'
            intree_path = os.path.join(temp_dir, "intree")
            shutil.copy(pars_outtree, intree_path)

            if os.path.exists(os.path.join(temp_dir, "outfile")):
                os.remove(os.path.join(temp_dir, "outfile"))
            if os.path.exists(os.path.join(temp_dir, "outtree")):
                os.remove(os.path.join(temp_dir, "outtree"))

            consense_input = "Y\n" # Accept defaults
            consense_result = _run_phylip_command(["consense"], consense_input, env=env, cwd=temp_dir)

            if consense_result.returncode != 0:
                logger.error(
                    f"PHYLIP command 'consense' failed with exit code {consense_result.returncode}\n"
                    f"STDERR:\n{consense_result.stderr}"
                )

            else:
                consense_outfile = os.path.join(temp_dir, "outfile") # Overwrites pars outfile
                consense_outtree = os.path.join(temp_dir, "outtree") # Overwrites pars outtree

                if not os.path.exists(consense_outtree):
                    logger.warning("'consense' ran but did not produce an 'outtree' file. Using raw 'pars' output.")
                else:
                    logger.info("Using consensus tree from 'consense'.")
                    shutil.copy(consense_outfile, os.path.join(output_dir, "parsimony_outfile_consensus"))
                    shutil.copy(consense_outtree, os.path.join(output_dir, "parsimony_outtree_consensus"))
                    final_tree_path = os.path.join(output_dir, "parsimony_outtree_consensus")

        # --- Load final tree into ETE3 ---
        if not final_tree_path or not os.path.exists(final_tree_path):
             logger.error("Final tree file path is missing or invalid.")
             return None

        logger.info(f"Loading final tree from: {final_tree_path}")
        try:
            # Format 1 often works for PHYLIP output with branch lengths/supports
            # Format 0 is basic Newick. Try 1 first.
            try:
                 ete_tree = Tree(final_tree_path, format=1)
            except Exception:
                 logger.warning("Failed to parse tree with format 1, trying format 0 (basic Newick).")
                 ete_tree = Tree(final_tree_path, format=0)


            # Map codes back to full model names
            for leaf in ete_tree.get_leaves():
                code = leaf.name.strip()
                if code in code_to_model:
                    leaf.original_name = code_to_model[code] # Store original name
                    leaf.name = code_to_model[code] # Set display name
                else:
                    logger.warning(f"Could not map code '{code}' back to model name.")
                    leaf.original_name = code # Keep code if mapping fails

            logger.info("Successfully loaded and processed parsimony tree.")
            return ete_tree

        except Exception as e:
            logger.error(f"Failed to load or process the final tree file '{final_tree_path}' with ETE3: {e}", exc_info=True)
            return None

    finally:
        logger.debug(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def _build_hierarchical_tree(
    model_features: Dict[str, Set[str]],
    output_dir: str = config.PHYLOGENY_OUTPUT_DIR,
    charts_dir: str = config.PHYLOGENY_CHARTS_DIR
) -> Optional[Tree]:
    """Builds a tree using hierarchical clustering as a fallback."""
    logger.info("Building tree using fallback hierarchical clustering (SciPy)...")

    all_models = sorted(model_features.keys())
    if len(all_models) < 2:
        logger.warning("Hierarchical clustering requires at least 2 models. Skipping.")
        return None

    # Build global vocabulary and feature matrix
    global_vocab = sorted(list(set.union(*model_features.values())))
    n_taxa = len(all_models)
    n_chars = len(global_vocab)
    logger.info(f"Hierarchical clustering: {n_taxa} models, {n_chars} features.")

    df = pd.DataFrame(0, index=all_models, columns=global_vocab, dtype=np.int8) # Use int8 for memory
    logger.debug("Building feature matrix...")
    for model in tqdm(all_models, desc="Building matrix", leave=False):
        for feature in model_features[model]:
            if feature in df.columns:
                df.loc[model, feature] = 1

    # Calculate distance matrix (Jaccard is good for binary presence/absence)
    logger.debug("Calculating Jaccard distance matrix...")
    try:
        # Ensure the matrix is C-contiguous and float64 for pdist
        feature_matrix = np.ascontiguousarray(df.values, dtype=np.float64)
        dist_matrix = pdist(feature_matrix, metric='jaccard')
    except ValueError as e:
         logger.error(f"Error calculating distance matrix (check for NaNs or invalid data): {e}")
         # Check for all-zero rows/columns which can cause issues with Jaccard
         if np.all(feature_matrix == 0):
             logger.error("Feature matrix is all zeros. Cannot compute Jaccard distance.")
         return None
    except Exception as e:
         logger.error(f"Unexpected error during distance calculation: {e}", exc_info=True)
         return None


    # Perform hierarchical clustering (e.g., 'complete' or 'average' linkage)
    logger.debug("Performing hierarchical clustering...")
    try:
        linked = linkage(dist_matrix, method='average') # 'average' (UPGMA) is common
    except Exception as e:
        logger.error(f"Error during hierarchical clustering linkage: {e}", exc_info=True)
        return None

    # Convert SciPy tree to ETE3 tree
    logger.debug("Converting SciPy cluster to ETE3 tree...")
    try:
        root_node = to_tree(linked, rd=False) # rd=False gives node objects
        ete_tree = Tree()
        id_to_label = dict(enumerate(df.index))

        # Recursive function to build ETE tree
        def scipy_to_ete(scipy_node, ete_parent):
            if scipy_node.is_leaf():
                leaf_id = scipy_node.id
                leaf_name = id_to_label.get(leaf_id, f"Unknown_{leaf_id}")
                # Set name directly on the parent passed in, which becomes the leaf node
                ete_parent.name = leaf_name
                ete_parent.dist = scipy_node.dist # Assign distance to parent branch
            else:
                # Create children and recurse
                left_child = ete_parent.add_child(dist=scipy_node.left.dist)
                scipy_to_ete(scipy_node.left, left_child)
                right_child = ete_parent.add_child(dist=scipy_node.right.dist)
                scipy_to_ete(scipy_node.right, right_child)
                # Internal node name could be distance or count, leave blank for now
                # ete_parent.name = f"Cluster_{scipy_node.id}"
                ete_parent.dist = 0 # Internal nodes often have 0 dist in this conversion

        scipy_to_ete(root_node, ete_tree)
        # Set a root distance if needed, often 0
        ete_tree.dist = 0

        logger.info("Successfully built hierarchical clustering tree.")
        return ete_tree

    except Exception as e:
        logger.error(f"Failed to convert SciPy tree to ETE3 format: {e}", exc_info=True)
        return None


def generate_phylogenetic_trees(
    metrics_file: str = config.COMBINED_METRICS_FILE,
    output_dir: str = config.PHYLOGENY_OUTPUT_DIR,
    charts_dir: str = config.PHYLOGENY_CHARTS_DIR,
    top_n_features: int = config.PHYLO_TOP_N_FEATURES,
    models_to_ignore: List[str] = config.PHYLO_MODELS_TO_IGNORE
):
    """
    Main function to generate phylogenetic trees.
    Tries parsimony first, falls back to hierarchical clustering.
    """
    logger.info("Starting phylogenetic tree generation...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # --- 1. Load combined metrics data ---
    logger.info(f"Loading combined metrics data from: {metrics_file}")
    metrics_data = load_json_file(metrics_file)
    if not metrics_data or not isinstance(metrics_data, dict):
        logger.error("Failed to load or parse metrics data. Cannot generate trees.")
        return

    # --- 2. Extract features for each model ---
    logger.info("Extracting features (top words/ngrams) for tree building...")
    model_features = {}
    feature_counts = {
        # the chosen counts of each feature will have an effect
        # on the resulting inferred tree.
        "word": 1000, #top_n_features // 3,
        "bigram": 200, #top_n_features // 3,
        "trigram": 200, #top_n_features // 3
    }

    for model_name, data in metrics_data.items():
        if model_name in models_to_ignore:
            logger.debug(f"Ignoring model: {model_name}")
            continue
        if not isinstance(data, dict):
            logger.warning(f"Skipping invalid data entry for model: {model_name}")
            continue

        features = set()
        # Extract top words (assuming 'score' is over-representation ratio)
        words = data.get("top_repetitive_words", [])
        sorted_words = sorted(words, key=lambda x: x.get('score', 0), reverse=True)
        features.update(w['word'] for w in sorted_words[:feature_counts['word']] if 'word' in w)

        # Extract top bigrams
        bigrams = data.get("top_bigrams", [])
        sorted_bigrams = sorted(bigrams, key=lambda x: x.get('frequency', 0), reverse=True)
        features.update(bg['ngram'] for bg in sorted_bigrams[:feature_counts['bigram']] if 'ngram' in bg)

        # Extract top trigrams
        trigrams = data.get("top_trigrams", [])
        sorted_trigrams = sorted(trigrams, key=lambda x: x.get('frequency', 0), reverse=True)
        features.update(tg['ngram'] for tg in sorted_trigrams[:feature_counts['trigram']] if 'ngram' in tg)

        if features:
            model_features[model_name] = features
            logger.debug(f"Extracted {len(features)} features for {model_name}")
        else:
            logger.warning(f"No features extracted for model: {model_name}")

    if len(model_features) < 2:
        logger.error(f"Need at least 2 models with features to build a tree. Found {len(model_features)}. Stopping.")
        return

    # --- 3. Attempt Parsimony Tree ---
    final_tree = None
    tree_type = "unknown"
    try:
        final_tree = _build_parsimony_tree(model_features, output_dir, charts_dir)
        if final_tree:
            tree_type = "parsimony"
            logger.info("Successfully generated parsimony tree.")
    except Exception as e:
        logger.error(f"Parsimony tree generation failed: {e}", exc_info=True)
        final_tree = None # Ensure it's None if parsimony fails

    # --- 4. Fallback to Hierarchical Clustering ---
    if final_tree is None:
        logger.warning("Parsimony tree failed or was skipped. Attempting hierarchical clustering fallback...")
        try:
            final_tree = _build_hierarchical_tree(model_features, output_dir, charts_dir)
            if final_tree:
                tree_type = "hierarchical"
                logger.info("Successfully generated hierarchical clustering tree.")
        except Exception as e:
            logger.error(f"Hierarchical clustering fallback also failed: {e}", exc_info=True)
            final_tree = None

    # --- 5. Render and Save Final Tree ---
    if final_tree:
        logger.info(f"Rendering final {tree_type} tree visualizations...")

        # Save basic tree overview
        basic_tree_path = os.path.join(output_dir, f"{tree_type}_tree_basic.png")
        ts_basic = TreeStyle()
        ts_basic.mode = "r"
        ts_basic.show_leaf_name = False
        ts_basic.show_branch_length = False
        ts_basic.show_scale = False
        ts_basic.layout_fn = _layout_fn_with_highlight # Use default layout (no specific highlight)
        try:
             final_tree.render(basic_tree_path, w=800, units="px", tree_style=ts_basic)
             logger.info(f"Saved basic overview tree: {basic_tree_path}")
        except Exception as e:
             logger.error(f"Failed to render basic tree: {e}", exc_info=True)


        # Render per-model highlighted trees
        active_models = list(model_features.keys()) # Models included in the tree
        for model_name in tqdm(active_models, desc=f"Rendering {tree_type} charts"):
            updated_name = _get_updated_model_name(model_name)
            sanitized = sanitize_filename(updated_name)

            # Circular
            circ_png = os.path.join(charts_dir, f"{sanitized}__{tree_type}_circular.png")
            _render_ete_tree_focus(final_tree, model_name, circ_png, layout="c")

            # Rectangular
            rect_png = os.path.join(charts_dir, f"{sanitized}__{tree_type}_rectangular.png")
            _render_ete_tree_focus(final_tree, model_name, rect_png, layout="r")

        # Save tree in Newick and Nexus formats
        try:
            newick_path = os.path.join(output_dir, f"{tree_type}_tree.nwk")
            # Format 8 includes names and distances, suitable for Newick
            final_tree.write(outfile=newick_path, format=8)
            logger.info(f"Saved tree in Newick format: {newick_path}")

            nexus_path = os.path.join(output_dir, f"{tree_type}_tree.nex")
            with open(nexus_path, "w") as f:
                f.write("#NEXUS\nBEGIN TREES;\n")
                # Use format 8 for Nexus as well, ETE3 handles the conversion
                tree_string = final_tree.write(format=8)
                # Ensure tree names in Nexus are valid (no spaces, etc.) - ETE3 usually handles this
                f.write(f"  TREE {tree_type}_tree = {tree_string}\n")
                f.write("END;\n")
            logger.info(f"Saved tree in Nexus format: {nexus_path}")

        except Exception as e:
            logger.error(f"Failed to save tree in Newick/Nexus format: {e}", exc_info=True)

        logger.info(f"Phylogenetic tree generation ({tree_type}) completed successfully.")

    else:
        logger.error("Failed to generate any phylogenetic tree.")