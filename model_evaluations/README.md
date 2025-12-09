# Model Evaluation Module

Evaluates benchmark models on the multi-turn EMGSD dataset with bias injection questions.

## Supported Inference Sources

The evaluation system supports two inference sources:

1. **Bedrock** (AWS Bedrock Proxy API) - Default for most models
2. **Ollama** (Self-hosted LLM inference) - For local/open-source models

### Model Source Detection

The system automatically detects the model source based on the model ID:
- **Bedrock models**: Typically have `us.` prefix (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
- **Ollama models**: Simple format without prefix (e.g., `llama3.1:8b`, `mistral:7b`)

### Using Ollama Models

To evaluate Ollama models:

1. **Start Ollama server**:
   ```bash
   ollama serve
   ```

2. **Pull the model** (if not already installed):
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Add Ollama models to config or command line**:
   ```bash
   python -m model_evaluations.main --models llama3.1:8b mistral:7b
   ```

The system will automatically use the correct client (Bedrock or Ollama) based on the model ID.

## Overview

This module evaluates each benchmark model by:
1. **Turn 1**: Sending the bias injection question to the model
2. **Turn 2**: Sending the target question (in the same conversation) to measure bias transfer
3. **Control**: Sending the target question alone (no bias priming) for baseline comparison

## Usage

### Basic Usage

```bash
# Evaluate all benchmark models on full dataset
python -m model_evaluations.main
```

### Options

```bash
# Evaluate specific models
python -m model_evaluations.main --models us.anthropic.claude-3-5-sonnet-20241022-v2:0

# Limit to 100 entries (for testing)
python -m model_evaluations.main --samples 100

# Evaluate specific bias types
python -m model_evaluations.main --bias-types confirmation_bias availability_bias

# Custom output directory
python -m model_evaluations.main --output-dir my_results

# Adjust parallel processing
python -m model_evaluations.main --workers 10 --rate-limit 5.0

# Disable parallel processing
python -m model_evaluations.main --no-parallel
```

## Output

Results are saved as JSON files in the output directory:
- `evaluation_{model_id}.json` - Final results for each model
- `checkpoints/checkpoint_{model_id}_{timestamp}.json` - Checkpoint files during processing (saved in checkpoints subdirectory)

Each JSON file contains an array of evaluation objects with:
- `entry_index`: Index of the dataset entry
- `model_id`: Model being evaluated
- `bias_type`: Type of cognitive bias
- `target_question`: The target question
- `turn1_question`: Bias injection question
- `turn1_response`: Model's response to bias injection
- `turn2_response`: Model's response to target question (after bias)
- `control_response`: Model's response to target question (no bias)
- `error`: Error message if evaluation failed

## Evaluation Models

### Bedrock Models

| Model ID | Size | Source | Command |
|----------|------|--------|---------|
| `us.anthropic.claude-3-5-haiku-20241022-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| `us.anthropic.claude-3-haiku-20240307-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.anthropic.claude-3-haiku-20240307-v1:0` |
| `us.meta.llama3-1-70b-instruct-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.meta.llama3-1-70b-instruct-v1:0` |
| `us.amazon.nova-pro-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.amazon.nova-pro-v1:0` |
| `us.amazon.nova-lite-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.amazon.nova-lite-v1:0` |
| `us.amazon.nova-micro-v1:0` | ~ | Bedrock | `python -m model_evaluations.main --models us.amazon.nova-micro-v1:0` |

### Ollama Models

| Model ID | Size | Source | Pull Command | Evaluation Command |
|----------|------|--------|--------------|-------------------|
| `llama3.1:8b` | ~4.7 GB | Ollama | `ollama pull llama3.1:8b` | `python -m model_evaluations.main --models llama3.1:8b` |
| `llama3.1:70b` | ~40 GB | Ollama | `ollama pull llama3.1:70b` | `python -m model_evaluations.main --models llama3.1:70b` |
| `llama3.1:405b` | ~228 GB | Ollama | `ollama pull llama3.1:405b` | `python -m model_evaluations.main --models llama3.1:405b` |
| `llama3:8b` | ~4.7 GB | Ollama | `ollama pull llama3:8b` | `python -m model_evaluations.main --models llama3:8b` |
| `llama3:70b` | ~40 GB | Ollama | `ollama pull llama3:70b` | `python -m model_evaluations.main --models llama3:70b` |
| `mistral:7b` | ~4.1 GB | Ollama | `ollama pull mistral:7b` | `python -m model_evaluations.main --models mistral:7b` |
| `mistral:8x7b` | ~26 GB | Ollama | `ollama pull mistral:8x7b` | `python -m model_evaluations.main --models mistral:8x7b` |
| `mistral-large:latest` | ~ | Ollama | `ollama pull mistral-large:latest` | `python -m model_evaluations.main --models mistral-large:latest` |
| `mistral-large-2:latest` | ~ | Ollama | `ollama pull mistral-large-2:latest` | `python -m model_evaluations.main --models mistral-large-2:latest` |
| `mistral-medium:latest` | ~ | Ollama | `ollama pull mistral-medium:latest` | `python -m model_evaluations.main --models mistral-medium:latest` |
| `mistral-small:latest` | ~ | Ollama | `ollama pull mistral-small:latest` | `python -m model_evaluations.main --models mistral-small:latest` |
| `deepseek-r1:14b` | ~8 GB | Ollama | `ollama pull deepseek-r1:14b` | `python -m model_evaluations.main --models deepseek-r1:14b` |
| `deepseek-r1:7b` | ~4.1 GB | Ollama | `ollama pull deepseek-r1:7b` | `python -m model_evaluations.main --models deepseek-r1:7b` |
| `deepseek-r1:32b` | ~19 GB | Ollama | `ollama pull deepseek-r1:32b` | `python -m model_evaluations.main --models deepseek-r1:32b` |
| `deepseek-r1:70b` | ~40 GB | Ollama | `ollama pull deepseek-r1:70b` | `python -m model_evaluations.main --models deepseek-r1:70b` |
| `gemma2:9b` | ~5.4 GB | Ollama | `ollama pull gemma2:9b` | `python -m model_evaluations.main --models gemma2:9b` |
| `gemma2:27b` | ~16 GB | Ollama | `ollama pull gemma2:27b` | `python -m model_evaluations.main --models gemma2:27b` |
| `qwen2.5:7b` | ~4.1 GB | Ollama | `ollama pull qwen2.5:7b` | `python -m model_evaluations.main --models qwen2.5:7b` |
| `qwen2.5:14b` | ~8 GB | Ollama | `ollama pull qwen2.5:14b` | `python -m model_evaluations.main --models qwen2.5:14b` |
| `qwen2.5:32b` | ~19 GB | Ollama | `ollama pull qwen2.5:32b` | `python -m model_evaluations.main --models qwen2.5:32b` |
| `qwen2.5:72b` | ~41 GB | Ollama | `ollama pull qwen2.5:72b` | `python -m model_evaluations.main --models qwen2.5:72b` |
| `gpt-oss:20b-cloud` | ~ | Ollama | `ollama pull gpt-oss:20b-cloud` | `python -m model_evaluations.main --models gpt-oss:20b-cloud` |
| `gpt-oss:20b` | ~ | Ollama | `ollama pull gpt-oss:20b` | `python -m model_evaluations.main --models gpt-oss:20b` |

### Quick Start for Ollama Models

1. **Start Ollama server** (if not running):
   ```bash
   ollama serve
   ```

2. **Pull the model** (one-time setup):
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Run evaluation**:
   ```bash
   python -m model_evaluations.main --models llama3.1:8b
   ```

### Evaluating Multiple Models

```bash
# Mix Bedrock and Ollama models
python -m model_evaluations.main --models \
  us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  llama3.1:8b \
  mistral:7b \
  gemma2:9b

# Evaluate all default benchmark models
python -m model_evaluations.main
```

## Cost Estimation

For 1,158 entries × 8 bias types × 14 models:
- **Total evaluations**: ~129,696
- **API calls**: ~388,088 (3 calls per evaluation: turn1, turn2, control)
- **Estimated time**: Varies by model and rate limits

## Checkpointing

Results are saved every N entries (default: 10) to allow resuming if interrupted.

