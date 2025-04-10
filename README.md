# Slop Forensics Toolkit

A toolkit for generating & analyzing "slop" — over-represented lexical patterns — in LLM outputs.

### 🧬 Dataset Generation
Generate a standardised set of outputs from several models for downstream analysis.

### 🔍 Slop Profiling
Analyze a model's outputs for repetitive words, bigrams, trigrams, vocabulary complexity, and slop scores.

### 🧪 Slop List Creation  
Aggregate findings across models to build canonical slop lists of of over-represented words and phrases.

### 🌳 Phylogenetic Tree Building  
Cluster models based on slop profile similarity using parsimony (PHYLIP) or hierarchical clustering.

--- 

## Example Notebook

https://colab.research.google.com/drive/1SQfnHs4wh87yR8FZQpsCOBL5h5MMs8E6?usp=sharing

![image](https://github.com/user-attachments/assets/29a81001-a611-4bf9-b472-ff3c4697cb49)

## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)  
2. [Project Structure](#project-structure)  
3. [Configuration / Environment Setup](#configuration--environment-setup)  
4. [Usage](#quickstart-usage)  
   - [1. Generate Dataset](#1-generate-dataset)  
   - [2. Analyze Outputs & Profile Slop](#2-analyze-outputs--profile-slop)  
   - [3. Create Slop Lists](#3-create-slop-lists)  
   - [4. Generate Phylogenetic Trees](#4-generate-phylogenetic-trees)  
5. [Script Descriptions](#script-reference)
6. [Analysis Pipeline Explanation](#analysis-pipeline)
7. [License](#license)  
8. [Contact](#contact)

---

## Prerequisites & Installation

1. **Python 3.7+**  
2. The required Python dependencies are listed in `requirements.txt`. Install them via:
   ```bash
   pip install -r requirements.txt
   ```
3. **PHYLIP** (optional)  
   - PHYLIP is required if you want to run the phylogenetic/parsimony analysis.  
   - On Debian/Ubuntu: 
     ```bash
     sudo apt-get install phylip
     ```
   - Or build from source.  
   - Make sure the `pars` and `consense` executables are in your `PATH` or specify `PHYLIP_PATH` in `.env`.

4. **NLTK data** (recommended):  
   We use `punkt`, `punkt_tab`, `stopwords`, and `cmudict` for parts of the analysis. Download via:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('cmudict')
   ```

---

## Project Structure

```
slop-forensics/
  ├─ scripts/
  │   ├─ generate_dataset.py
  │   ├─ slop_profile.py
  │   ├─ create_slop_lists.py
  │   ├─ generate_phylo_trees.py
  │   └─ ...
  ├─ slop_forensics/
  │   ├─ config.py
  │   ├─ dataset_generator.py
  │   ├─ analysis.py
  │   ├─ metrics.py
  │   ├─ phylogeny.py
  │   ├─ slop_lists.py
  │   ├─ utils.py
  │   └─ ...
  ├─ data/
  │   └─ (internal data files for slop lists, e.g. slop_list.json, etc.)
  ├─ results/
  │   ├─ datasets/
  │   ├─ analysis/
  │   ├─ slop_lists/
  │   ├─ phylogeny/
  │   └─ ...
  ├─ .env.example
  ├─ requirements.txt
  ├─ README.md  ← You are here!
  └─ ...
```

- **scripts/** contains the runnable scripts that tie the pipeline together.  
- **slop_forensics/** contains the main library code.  
- **results/** is where output files from each step will be saved by default.  
- **data/** is where any static data or references (including existing slop lists) live.  

---

## Configuration / Environment Setup

1. Copy `.env.example` to `.env` and update the variables:
   ```bash
   cp .env.example .env
   ```
2. In `.env`, set `OPENAI_API_KEY` to an [OpenRouter](https://openrouter.ai/) or OpenAI-compatible key.  
3. (Optional) Set `PHYLIP_PATH` if the `pars`/`consense` binaries are not in your `PATH`.  

**Example `.env` contents**:
```ini
# .env
OPENAI_API_KEY=sk-or-v1-xxxxxx
OPENAI_BASE_URL="https://openrouter.ai/api/v1"
PHYLIP_PATH="/usr/local/bin"
```
  
**Note**: If you are not using OpenRouter, you can point to another OpenAI-compatible service by changing `OPENAI_BASE_URL`.

---

## Quickstart Usage

Below is a typical workflow, using **mostly defaults**. Adjust paths/arguments as desired.

Note: several default parameters are pre-configured in `slop_forensics/config.py`.

### 1. Generate Dataset

Use `generate_dataset.py` to prompt the specified LLMs for story outputs.  

```bash
python3 scripts/generate_dataset.py \
  --model-ids x-ai/grok-3-mini-beta,meta-llama/llama-4-maverick,meta-llama/llama-4-scout,google/gemma-3-4b-it \
  --generate-n 100
```
- **Description**: Prompts each listed model for ~100 outputs.  
- **Output**: `.jsonl` files in `results/datasets`, named like `generated_x-ai__grok-3-mini-beta.jsonl`, etc.

### 2. Analyze Outputs & Profile Slop

Once data is generated, run `slop_profile.py` to calculate word/bigram/trigram usage, repetition scores, slop scores, etc.

```bash
python3 scripts/slop_profile.py
```

- **Description**: Reads all `generated_*.jsonl` in `results/datasets`, analyzes each, and writes results to:  
  - `results/analysis/slop_profile__{model}.json` (per-model detailed analysis)  
  - `results/slop_profile_results.json` (combined data for all models).  
- **CLI Options**: You can specify `--input-dir`, `--analysis-output-dir`, and so on if you want to override defaults.

### 3. Create Slop Lists

Use `create_slop_lists.py` to combine analysis results from multiple models and generate final “slop lists” of suspicious words and phrases.

```bash
python3 scripts/create_slop_lists.py
```

- **Description**: Loads all per-model `.json` files from `results/analysis/`, re-reads the corresponding model `.jsonl` files, and creates aggregated slop lists.  
- **Outputs**:  
  - `results/slop_lists/slop_list.json` → top suspicious single words  
  - `results/slop_lists/slop_list_bigrams.json` → suspicious bigrams  
  - `results/slop_lists/slop_list_trigrams.json` → suspicious trigrams  
  - `results/slop_lists/slop_list_phrases.jsonl` → top multi-word substrings actually extracted from text

### 4. Generate Phylogenetic Trees

Finally, `generate_phylo_trees.py` can attempt to create phylogenetic trees (via PHYLIP parsimony or hierarchical clustering fallback).

```bash
python3 scripts/generate_phylo_trees.py
```

- **Description**:  
  1. Loads the combined metrics (`results/slop_profile_results.json`).  
  2. Extracts the top repeated words, bigrams, and trigrams for each model (up to `PHYLO_TOP_N_FEATURES` total).  
  3. Tries to run PHYLIP’s `pars` (parsimony) and optionally `consense`.  
  4. If PHYLIP is unavailable or fails, falls back to a hierarchical clustering approach.  
- **Outputs**:  
  - Basic newick / nexus tree files in `results/phylogeny/`  
  - `.png` images (both circular & rectangular) per model highlighting that model on the tree.

---

## Script Reference

Below is a quick reference to each major script:

1. **`scripts/generate_dataset.py`**  
   - **Purpose**: Calls an OpenAI-compatible API to generate text outputs from specified models.  
   - **Key Args**:  
     - `--model-ids`: Comma-separated model IDs.  
     - `--output-dir`: Where to write `.jsonl` dataset files.  
     - `--generate-n`: Target number of outputs per model.  
   - **Dependencies**:  
     - Requires a valid `OPENAI_API_KEY` in your `.env`.

2. **`scripts/slop_profile.py`**  
   - **Purpose**: Analyzes each `.jsonl` file from `generate_dataset.py` for repetitive words, bigrams, trigrams, slop scores, etc.  
   - **Key Args**:  
     - `--input-dir`: Directory containing generated `.jsonl` files.  
     - `--analysis-output-dir`: Directory to write per-model JSON analysis.  
     - `--combined-output-file`: JSON file for combined results.  
   - **Outputs**:  
     - Per-model `.json` analysis in `results/analysis/`  
     - Merged summary in `results/slop_profile_results.json`

3. **`scripts/create_slop_lists.py`**  
   - **Purpose**: Combines analysis from many models and re-reads the full text to build global slop lists (words, bigrams, trigrams, and multi-word phrases).  
   - **Outputs**:  
     - `slop_list.json`, `slop_list_bigrams.json`, `slop_list_trigrams.json`, and `slop_list_phrases.jsonl` in `results/slop_lists/`.

4. **`scripts/generate_phylo_trees.py`**  
   - **Purpose**: Uses the combined metrics file (slop_profile_results.json) to cluster or build parsimony trees across all models.  
   - **Key Args**:  
     - `--input-file`: Path to `slop_profile_results.json`.  
     - `--output-dir`: Directory for final trees.  
     - `--top-n-features`: Control how many top repeated terms (words+bigrams+trigrams) to use.  
   - **Outputs**:  
     - Tree files in `results/phylogeny/` (Newick, Nexus)  
     - PNG visualizations in `results/phylogeny/charts/`


Below is a concise overview explaining how each part of the analysis works, from slop profiling and list creation to phylogenetic tree building.

---

## Analysis Pipeline

### 1. Slop Profiling

- **Goal**: Examine each model’s generated outputs to identify **over-represented** words, bigrams, and trigrams (“slop”).  
- **Process**:  
  1. **Token & Frequency Analysis**: We count word occurrences per model, filtering out stopwords, numerical tokens, etc.  
  2. **Slop Scores**: We compute metrics like repetition score, average vocabulary complexity, and a slop index (based on shared “slop lists”) to quantify how often each model uses repetitive patterns.  
  3. **N-gram Analysis**: We look at bigrams/trigrams for broader repetitive phrases, focusing on those appearing in multiple prompts.  

**Outcome**: Per-model JSON files that summarize word/bigram/trigram usage, repetition scores, and other key metrics.

---

### 2. Slop List Creation

- **Goal**: Combine findings across all models to create reference lists of frequently over-used words and phrases.  
- **Process**:  
  1. **Aggregation**: We merge each model’s top words, bigrams, and trigrams.  
  2. **Filtering**: We remove high-frequency “normal” words (using external frequency data) so the final lists focus on genuinely over-represented (slop) items.  
  3. **Phrase Extraction**: We also identify multi-word substrings from the original outputs for an additional “slop phrases” list.  

**Outcome**:  
- `slop_list.json` → Over-represented single words  
- `slop_list_bigrams.json` / `slop_list_trigrams.json` → Over-represented bigrams & trigrams  
- `slop_list_phrases.jsonl` → Extended multi-word slop phrases  

---

### 3. Phylogenetic Tree Building

- **Goal**: Visualize how models cluster based on slop usage.  
- **Process**:  
  1. **Presence/Absence Vectors**: We take each model’s top slop words/phrases and record a ‘1’ if present or ‘0’ otherwise.  
  2. **Parsimony (PHYLIP)**: We repurpose a bioinformatics tool (originally for DNA/protein sequences) to build a parsimony tree from these presence/absence patterns. Each model is treated like a “taxon,” and each slop feature is treated like a “mutation.”  
  3. **Hierarchical Clustering Fallback**: If PHYLIP is unavailable, we perform standard hierarchical clustering (using Jaccard distance) to produce a similar tree.  

**Example PHYLIP input**:
```
M0001      010000010001100010000...
M0002      000100110000010100001...
```

**Outcome**:  
- Newick/Nexus trees (`.nwk`, `.nex`)  
- Circular and rectangular tree charts for each model, highlighting how they branch relative to others.

---

This end-to-end pipeline reveals which lexical patterns each model tends to repeat, aggregates them into global slop lists, and then visually clusters the models based on these patterns.


---

## License

MIT License

---

## Contact

For questions or feedback:
- **Maintainer**: [Sam Paech](https://x.com/sam_paech)
- Or create an [issue in this repo](https://github.com/sam-paech/slop-forensics/issues).


## How to Cite

If you use Slop Forensics in your research or work, please cite it as:

```bibtex
@software{paech2025slopforensics,
  author = {Paech, Samuel J},
  title = {Slop Forensics: A Toolkit for Generating \& Analyzing Lexical Patterns in LLM Outputs},
  url = {https://github.com/sam-paech/slop-forensics},
  year = {2025},
}
