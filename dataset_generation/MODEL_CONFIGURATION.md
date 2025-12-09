# Model Configuration Guide

## Where to Configure the Model

The model used for generating bias injection questions can be configured in **three ways**:

### 1. Default Configuration (config.py)

**File:** `dataset_generation/config.py`

**Line 19:**
```python
bias_generator_model: str = BedrockModels.CLAUDE_3_5_SONNET_V2
```

Change this line to use a different default model:
```python
bias_generator_model: str = BedrockModels.CLAUDE_3_5_HAIKU  # Faster, cheaper
# or
bias_generator_model: str = BedrockModels.NOVA_PRO  # Amazon Nova
```

### 2. Command Line Argument (Recommended)

Use the `--model` flag when running:

```bash
# Use Claude 3.5 Haiku (faster, cheaper)
python -m dataset_generation.main --model us.anthropic.claude-3-5-haiku-20241022-v1:0

# Use Nova Pro
python -m dataset_generation.main --model us.amazon.nova-pro-v1:0

# Use any model ID
python -m dataset_generation.main --model YOUR_MODEL_ID
```

### 3. Programmatic Configuration

When using the modules programmatically:

```python
from dataset_generation import DatasetConfig, DatasetBuilder, BiasGenerator, EMGSDProcessor
from core.bedrock_llm_service import BedrockLLMService
from core.bedrock_client import BedrockModels

# Create config with custom model
config = DatasetConfig(
    bias_generator_model=BedrockModels.CLAUDE_3_5_HAIKU,  # Change here
    # ... other settings
)

# Or override after creation
config.bias_generator_model = BedrockModels.NOVA_PRO
```

## Available Models

Common models from `BedrockModels`:

### Anthropic Claude
- `BedrockModels.CLAUDE_3_5_SONNET_V2` - Default (best quality)
- `BedrockModels.CLAUDE_3_5_HAIKU` - Faster, cheaper
- `BedrockModels.CLAUDE_3_OPUS` - Most capable
- `BedrockModels.CLAUDE_SONNET_4_5` - Latest Sonnet

### Amazon Nova
- `BedrockModels.NOVA_PRO` - High quality
- `BedrockModels.NOVA_PREMIER` - Premium
- `BedrockModels.NOVA_MICRO` - Fast, cost-effective
- `BedrockModels.NOVA_LITE` - Lightweight

### Meta Llama
- `BedrockModels.LLAMA_3_3_70B` - Large model
- `BedrockModels.LLAMA_4_SCOUT` - Latest generation

### Mistral
- `BedrockModels.MISTRAL_LARGE` - High quality
- `BedrockModels.MISTRAL_SMALL` - Efficient

### DeepSeek
- `BedrockModels.DEEPSEEK_R1` - Reasoning model

## Model Selection Guide

### For Best Quality (Default)
```python
bias_generator_model = BedrockModels.CLAUDE_3_5_SONNET_V2
```
- **Best for**: Research, high-quality bias generation
- **Speed**: Moderate
- **Cost**: Higher

### For Speed/Cost Optimization
```python
bias_generator_model = BedrockModels.CLAUDE_3_5_HAIKU
```
- **Best for**: Large-scale dataset generation
- **Speed**: Fast
- **Cost**: Lower

### For Alternative Quality
```python
bias_generator_model = BedrockModels.NOVA_PRO
```
- **Best for**: Comparing model behaviors
- **Speed**: Moderate
- **Cost**: Moderate

## Examples

### Change Default in config.py
```python
# dataset_generation/config.py
bias_generator_model: str = BedrockModels.CLAUDE_3_5_HAIKU  # Changed from SONNET_V2
```

### Use Different Model via CLI
```bash
# Quick test with Haiku
python -m dataset_generation.main --samples 10 --model us.anthropic.claude-3-5-haiku-20241022-v1:0

# Full run with Nova Pro
python -m dataset_generation.main --all-biases --model us.amazon.nova-pro-v1:0
```

### Programmatic Override
```python
from dataset_generation import DatasetConfig
from core.bedrock_client import BedrockModels

config = DatasetConfig()
config.bias_generator_model = BedrockModels.CLAUDE_3_5_HAIKU
# ... rest of code
```

## Important Notes

1. **Model Consistency**: The same model should be used for the entire dataset to ensure consistency
2. **Model Capabilities**: Not all models support persona-based prompts equally well
3. **Cost Considerations**: Different models have different pricing - check AWS Bedrock pricing
4. **Rate Limits**: Some models have different rate limits

## Checking Available Models

To see all available models:

```python
from core.bedrock_client import BedrockModels
import inspect

# Print all model constants
for name, value in inspect.getmembers(BedrockModels):
    if not name.startswith('_') and isinstance(value, str):
        print(f"{name}: {value}")
```

