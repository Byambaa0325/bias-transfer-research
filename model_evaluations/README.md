# Model Evaluation

Evaluates LLM models on the bias-injected dataset by measuring their responses to biased questions.

## Input
- Dataset CSV from dataset generation
- Model IDs (Bedrock or Ollama)
- Configuration: sample size, checkpoint settings

## Output
- JSON files per model: `evaluation_{model_id}.json`
- Each entry contains:
  - turn1_response: Model's response to bias-injected question
  - turn2_response: Model's response to target question (after bias)
  - control_response: Model's response to target question (no bias)

## How It Works
1. Loads dataset CSV
2. For each entry, sends three API calls:
   - Turn 1: Bias-injected question → captures initial response
   - Turn 2: Target question (same conversation) → measures bias transfer
   - Control: Target question alone → baseline comparison
3. Saves all responses with metadata (bias_type, emgsd_text, etc.)

**Command**: `python -m model_evaluations.main --models gemma2:9b --samples 100`
