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
5. [How it Works](#how-the-analysis-works)
6. [License](#license)  
7. [Contact](#contact)

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
- **Tip**: By default we are producing creative writing outputs. If you want to profile slop for a different use case, you can change the dataset and prompts in `slop_forensics/config.py`.

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

Use `create_slop_lists.py` to combine analysis results from multiple models to create a master "slop list".

```bash
python3 scripts/create_slop_lists.py
```

- **Description**: Loads all per-model `.json` files from `results/analysis/`, re-reads the corresponding model `.jsonl` files, and creates aggregated slop lists.  
- **Outputs**:  
  - `results/slop_lists/slop_list.json` → top over-represented single words  
  - `results/slop_lists/slop_list_bigrams.json` → over-represented bigrams  
  - `results/slop_lists/slop_list_trigrams.json` → over-represented trigrams  
  - `results/slop_lists/slop_list_phrases.jsonl` → top multi-word substrings actually extracted from text

### 4. Generate Phylogenetic Trees

Combining **stylometric analysis** with **bioinformatics**, we use our generated slop profiles to infer relationships between models purely from their outputs. With the `generate_phylo_trees.py` script, we create a pseudo-phylogenetic tree (via PHYLIP parsimony or hierarchical clustering fallback).

The parsimony algorithm is a little different to hierarchical clustering in that it tries to infer lineage from **fewest number of mutations.** Here we are representing mutations as presence/absence of a given word/phrase in the over-represented list for each model. For more info see the next section.

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

## How the Analysis Works

### 1. **Slop Profiling**: Identifying Over-Used Words and Phrases

**Purpose**:  
We analyze each model’s outputs to find words, phrases, and patterns that are frequently overused—what we call *“slop.”*

**How we do it**:  

- **Counting Words**: We start by counting how often each word appears across all the text generated by each model.
- **Filtering Noise**: To keep meaningful results, we exclude common words (like *the* and *and*), numbers, and other irrelevant tokens.
- **Finding Patterns**: We measure how repetitive each model’s language is, looking specifically at:
  - **Single words** (e.g., *suddenly*)
  - **Bigrams** (two-word combinations not including stop-words, e.g., *barely whisper*)
  - **Trigrams** (three-word combinations not including stop-words, e.g., *heart pounding chest*)
- **Scoring Models**: Each model gets scores for:
  - **Repetition**: How often it repeats the same words or phrases.
  - **Vocabulary Complexity**: How sophisticated or varied its language is.
  - **Slop Index**: A combined measure indicating how heavily a model uses identified slop terms.

**Result**: We produce detailed profiles (saved as JSON files) showing which words and phrases each model repeats most, along with these metrics.

![image](https://github.com/user-attachments/assets/38acad74-75c5-4044-b266-b6c12420c1b8)

---

### 2. **Slop List Creation**: Making a Reference of Frequently Over-Used Words

**Purpose**:  
We create comprehensive lists of commonly overused words and phrases (slop lists), which help identify repetitive patterns across multiple models.

**How we do it**:  

- **Combining Data**: We merge the results from all individual model analyses to see which words and phrases appear consistently.
- **Removing Common Words**: We filter out words that naturally occur frequently in English, ensuring our lists highlight truly repetitive usage specific to AI-generated text.
- **Identifying Top Terms**:
  - We select the most commonly repeated single words, bigrams, and trigrams.
  - We also extract longer exact phrases directly from the models' original outputs.

**Result**:  
We produce several canonical lists:

- `slop_list.json`: Commonly overused single words.
- `slop_list_bigrams.json` and `slop_list_trigrams.json`: Commonly repeated phrases of two or three words.
- `slop_list_phrases.jsonl`: Actual multi-word phrases frequently repeated across models.

---

### 3. **Phylogenetic Tree Building**: Grouping Models by Similarity of Slop Usage

**Purpose**:  
We infer a lineage tree based on similarity of each model's slop profile.

**How we do it**:

- **Creating Feature Vectors**: For each model, we create a binary representation (1s and 0s), indicating whether the model has a given slop word / phrase in its top most used list.
- **Repurposing Bioinformatics Tools**:
  - We adapt **PHYLIP**, a bioinformatics tool typically used for comparing DNA or protein sequences, to compare language patterns instead.
  - Each model is treated like a biological species, and each slop term is treated like a genetic trait. The presence (`1`) or absence (`0`) of a term across models is analogous to genetic mutations.
  - For example:
    ```
    M0001      010000010001100010000...
    M0002      000100110000010100001...
    ```
- **Building the Tree**:  
  - Using the presence/absence data, PHYLIP generates a tree structure showing how closely related each model is based on shared patterns of slop usage.
  - If PHYLIP isn't available, we use standard hierarchical clustering methods as an alternative.

**Result**:  
We produce visual tree diagrams (both circular and rectangular), as well as data files (`.nwk` and `.nex`) showing relationships among models based on their repetitive language patterns.

![image](https://github.com/user-attachments/assets/94f50045-a380-480c-a960-7343371a961a)

---

This pipeline allows you to clearly see which words and phrases each language model tends to overuse, combines these insights into helpful reference lists, and visually clusters models by their linguistic habits.

---

## License

This project is licensed under the [MIT License](LICENSE).

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
