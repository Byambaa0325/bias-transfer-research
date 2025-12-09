# Reproducing Bias Transfer Research

This repository contains code for reproducing the research on **Bias Transfer in Multi-Turn LLM Dialogue**. This guide will help you set up and run the experiments from scratch.

## Overview

This research investigates how Large Language Models (LLMs) adopt users' cognitive biases in multi-turn conversations and subsequently "leak" stereotypes in unrelated turns. The pipeline consists of:

1. **Dataset Generation**: Creating multi-turn bias injection questions from the EMGSD dataset
2. **Model Evaluation**: Testing models on the generated dataset to measure bias transfer
3. **Analysis**: Comparing biased vs. control responses to quantify stereotype leakage

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **GPU** (optional but recommended for Ollama models): NVIDIA GPU with CUDA support
- **RAM**: 8GB minimum (16GB+ recommended)
- **Disk Space**: ~10GB for datasets and models

### Required Accounts/API Keys

1. **AWS Bedrock** (for Bedrock models):
   - Bedrock API credentials (Team ID and API Token)
   - Place credentials in `.env.bedrock` file (see Setup section)

2. **Ollama** (for self-hosted models, optional):
   - Install Ollama from https://ollama.ai
   - No API keys needed for local inference

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bias-transfer-research
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install core dependencies:

```bash
pip install pandas numpy requests python-dotenv tqdm
```

### Step 4: Configure Credentials

#### For Bedrock Models

Create `.env.bedrock` in the project root:

```bash
BEDROCK_TEAM_ID=your_team_id
BEDROCK_API_TOKEN=your_api_token
BEDROCK_API_ENDPOINT=https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke
```

#### For Ollama Models

1. **Install Ollama**:
   - Windows: Download from https://ollama.ai/download
   - Linux/macOS: `curl -fsSL https://ollama.ai/install.sh | sh`

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

3. **Verify installation**:
   ```bash
   ollama list
   ```

## Dataset Setup

### Step 1: Prepare EMGSD Dataset

The research uses the Expanded Multi-Grain Stereotype Dataset (EMGSD). Place your EMGSD CSV file in the expected location or update the path in the configuration.

The dataset should contain columns:
- `emgsd_text`: Original stereotype sentence
- `emgsd_stereotype_type`: Type of stereotype
- `emgsd_category`: Category (profession, gender, etc.)
- `emgsd_trait`: Trait being stereotyped
- `emgsd_target_word`: Target word (if applicable)
- `target_question`: The question to be biased

### Step 2: Generate Multi-Turn Dataset

Generate bias injection questions for the dataset:

```bash
# Generate dataset with default settings (all 8 bias types)
python -m dataset_generation.main

# Generate with specific number of samples
python -m dataset_generation.main --samples 5000

# Generate with specific bias types
python -m dataset_generation.main --bias-types confirmation_bias anchoring_bias

# Use specific model for bias generation
python -m dataset_generation.main --model us.amazon.nova-pro-v1:0
```

**Output**: Dataset saved to `dataset_generation/data/multiturn_emgsd_dataset_YYYYMMDD_HHMMSS.csv`

**Configuration**: Edit `dataset_generation/config.py` to customize:
- Bias generator model
- Bias types to include
- Parallel processing settings

## Running Model Evaluations

### Step 1: Choose Models to Evaluate

You can evaluate models from two sources:

#### Bedrock Models (Cloud-based)

Available models include:
- `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- `us.anthropic.claude-3-haiku-20240307-v1:0`
- `us.meta.llama3-1-70b-instruct-v1:0`
- `us.amazon.nova-pro-v1:0`
- `us.amazon.nova-lite-v1:0`
- `us.amazon.nova-micro-v1:0`

#### Ollama Models (Self-hosted)

First, pull the models you want to use:

```bash
# Pull models (one-time setup)
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:14b
ollama pull gpt-oss:20b-cloud
```

See `model_evaluations/README.md` for a complete list of available models and their sizes.

### Step 2: Run Evaluation

#### Basic Evaluation

```bash
# Evaluate all default benchmark models
python -m model_evaluations.main

# Evaluate specific models
python -m model_evaluations.main --models llama3.1:8b mistral:7b

# Mix Bedrock and Ollama models
python -m model_evaluations.main --models \
  us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  llama3.1:8b \
  gemma2:9b
```

#### Advanced Options

```bash
# Limit to specific number of entries (for testing)
python -m model_evaluations.main --samples 100

# Evaluate specific bias types only
python -m model_evaluations.main --bias-types confirmation_bias anchoring_bias

# Custom output directory
python -m model_evaluations.main --output-dir my_results

# Adjust parallel processing
python -m model_evaluations.main --workers 10 --rate-limit 5.0

# Disable parallel processing (sequential)
python -m model_evaluations.main --no-parallel
```

### Step 3: Monitor Progress

The evaluation process will:
- Show progress bars for each model
- Save checkpoints periodically in `results/checkpoints/`
- Display model source (Bedrock/Ollama) for each model
- Continue with other models if one fails

### Step 4: Check Results

Results are saved as JSON files in `model_evaluations/results/`:
- `evaluation_{model_id}.json`: Final results for each model
- `checkpoints/checkpoint_{model_id}_{timestamp}.json`: Checkpoint files

Each result file contains an array of evaluation objects with:
- `entry_index`: Dataset entry index
- `model_id`: Model evaluated
- `bias_type`: Type of cognitive bias
- `turn1_question`: Bias injection question
- `turn1_response`: Model's response to bias
- `turn2_response`: Model's response to target question (after bias)
- `control_response`: Model's response to target question (no bias)
- `error`: Error message if evaluation failed

## Project Structure

```
bias-transfer-research/
├── core/                          # Core modules
│   ├── bedrock_client.py          # Bedrock API client
│   ├── bedrock_llm_service.py    # Multi-turn conversation service
│   ├── ollama_client.py           # Ollama API client
│   ├── ollama_llm_service.py      # Ollama service
│   ├── model_selector.py          # Model selection logic
│   └── model_source_mapper.py     # Maps models to sources (Bedrock/Ollama)
├── dataset_generation/             # Dataset generation pipeline
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Configuration
│   ├── dataset_builder.py         # Dataset building logic
│   ├── bias_generator.py          # Bias question generation
│   └── data/                      # Generated datasets
├── model_evaluations/              # Model evaluation pipeline
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Configuration
│   ├── evaluator.py               # Model evaluator
│   ├── evaluation_runner.py       # Evaluation orchestration
│   └── results/                   # Evaluation results
│       └── checkpoints/           # Checkpoint files
├── notebooks/                     # Jupyter notebooks
│   └── ollama_integration.ipynb   # Ollama setup and testing
├── .env.bedrock                   # Bedrock credentials (create this)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Common Workflows

### Workflow 1: Quick Test Run

Test the pipeline with a small sample:

```bash
# 1. Generate small dataset
python -m dataset_generation.main --samples 10 --bias-types confirmation_bias

# 2. Evaluate one model
python -m model_evaluations.main --models llama3.1:8b --samples 10

# 3. Check results
cat model_evaluations/results/evaluation_llama3_1_8b.json | head -50
```

### Workflow 2: Full Reproduction

Reproduce the full experiment:

```bash
# 1. Generate full dataset (all bias types)
python -m dataset_generation.main --samples 5000

# 2. Evaluate all benchmark models
python -m model_evaluations.main

# 3. Results will be in model_evaluations/results/
```

### Workflow 3: Custom Model Evaluation

Evaluate your own models:

```bash
# For Ollama models: pull first
ollama pull your-model:tag

# Then evaluate
python -m model_evaluations.main --models your-model:tag
```

## Troubleshooting

### Issue: "Could not connect to Ollama server"

**Solution**:
1. Make sure Ollama is installed and running
2. Start Ollama: `ollama serve`
3. Verify: `ollama list` should show your models

### Issue: "Bedrock API call failed"

**Solution**:
1. Check `.env.bedrock` file exists and has correct credentials
2. Verify your Bedrock API token is valid
3. Check your internet connection

### Issue: "Dataset not found"

**Solution**:
1. Make sure you've generated the dataset first
2. Check the path in `model_evaluations/config.py`
3. Or specify path: `python -m model_evaluations.main --dataset path/to/dataset.csv`

### Issue: "Model not found" (Ollama)

**Solution**:
1. Pull the model: `ollama pull model-name`
2. Verify it's installed: `ollama list`
3. Check the model name matches exactly (case-sensitive)

### Issue: Out of Memory (Ollama)

**Solution**:
1. Use smaller models (e.g., `llama3.1:8b` instead of `llama3.1:70b`)
2. Close other applications
3. Reduce batch size in parallel processing

## Configuration Files

### `dataset_generation/config.py`

Key settings:
- `bias_generator_model`: Model used to generate bias questions
- `bias_types`: List of bias types to generate
- `max_workers`: Parallel processing workers
- `max_requests_per_second`: Rate limiting

### `model_evaluations/config.py`

Key settings:
- `dataset_path`: Path to dataset (default: `dataset_generation/data`)
- `output_dir`: Results output directory
- `models`: List of models to evaluate
- `checkpoint_interval`: How often to save checkpoints

## Understanding Results

### Result Structure

Each evaluation result contains:
- **turn1_response**: Model's response to bias injection (measures alignment)
- **turn2_response**: Model's response to target question after bias (measures leakage)
- **control_response**: Model's response without bias (baseline)

### Key Metrics

1. **Alignment Score**: How much the model agrees with the bias in turn1_response
2. **Leakage Score**: Stereotype probability in turn2_response
3. **Drift Score**: Difference between biased and control responses

### Analysis

To analyze results:
1. Load JSON files: `pd.read_json('evaluation_model.json')`
2. Compare turn2_response vs control_response
3. Calculate drift scores
4. Aggregate by bias type and model

See `notebooks/ollama_integration.ipynb` for example analysis code.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bias-transfer-2024,
  title={The Imperfect User: Quantifying Stereotype Leakage Driven by Everyday Cognitive Biases in Multi-Turn LLM Dialogue},
  author={...},
  year={2024}
}
```

## Support

For detailed methodology and research questions, see `research_readme.md`.

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration files
3. Check the individual module READMEs:
   - `dataset_generation/README.md` (if exists)
   - `model_evaluations/README.md`

## License

[Add your license information here]
