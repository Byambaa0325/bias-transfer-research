# Dataset Generation

Generates multi-turn conversation datasets from EMGSD stereotypes with cognitive bias injection questions.

## Input
- EMGSD dataset CSV file (stereotypes with traits, target words, categories)
- Configuration: bias types, sample size, stereotype filters

## Output
- CSV file with multi-turn conversations:
  - Original EMGSD stereotype text
  - Turn 1: Bias-injected question (e.g., "Isn't it true that...")
  - Turn 2: Target question (e.g., "The supervisor is...")
  - Control question (target question without bias priming)

## How It Works
1. Loads EMGSD stereotypes from CSV
2. For each stereotype, generates 8 different bias injection questions (confirmation bias, anchoring, etc.)
3. Creates target questions using the stereotype text
4. Generates control questions (same target, no bias)
5. Saves as CSV with columns: emgsd_text, bias_type, turn1_question, target_question

**Command**: `python -m dataset_generation.main --samples 100`
