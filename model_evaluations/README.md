# Model Evaluation Module

Evaluates benchmark models on the multi-turn EMGSD dataset with bias injection questions.

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
- `checkpoint_{model_id}_{timestamp}.json` - Checkpoint files during processing

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

## Benchmark Models

The default configuration evaluates 14 models:
- 4 required models (from your CSV files)
- 10 representative models (cheap options from each family)

See `config.py` for the complete list.

## Cost Estimation

For 1,158 entries × 8 bias types × 14 models:
- **Total evaluations**: ~129,696
- **API calls**: ~388,088 (3 calls per evaluation: turn1, turn2, control)
- **Estimated time**: Varies by model and rate limits

## Checkpointing

Results are saved every N entries (default: 10) to allow resuming if interrupted.

