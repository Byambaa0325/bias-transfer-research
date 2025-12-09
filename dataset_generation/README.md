# Dataset Generation Module

A maintainable, modular system for generating multi-turn EMGSD datasets with cognitive bias injection questions.

## Architecture

```
dataset_generation/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── dataset_builder.py   # Main orchestrator
├── bias_generator.py    # Bias question generation
├── emgsd_processor.py   # EMGSD data processing
├── utils.py             # Utility functions
├── main.py              # CLI entry point
└── README.md            # This file
```

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Type Safety**: Uses type hints and dataclasses for clarity
4. **Error Handling**: Comprehensive error handling and validation
5. **Testability**: Modular design enables unit testing
6. **Maintainability**: Clear interfaces and documentation

## Usage

### Basic Usage

```bash
# Generate dataset for all stereotypes
python -m dataset_generation.main

# Limit to 100 samples
python -m dataset_generation.main --samples 100

# Filter by type
python -m dataset_generation.main --stereotype-type profession

# Use all 8 bias types
python -m dataset_generation.main --all-biases

# Custom bias types
python -m dataset_generation.main --bias-types availability_bias confirmation_bias

# Resume from checkpoint
python -m dataset_generation.main --resume data/checkpoint_multiturn_emgsd_*.csv
```

### Programmatic Usage

```python
from dataset_generation import DatasetConfig, DatasetBuilder, BiasGenerator, EMGSDProcessor
from core.bedrock_llm_service import BedrockLLMService

# Create configuration
config = DatasetConfig(
    bias_types=["availability_bias", "confirmation_bias"],
    sample_limit=100,
    output_dir=Path("results")
)

# Initialize components
llm_service = BedrockLLMService()
bias_generator = BiasGenerator(
    llm_service=llm_service,
    bias_generator_model=config.bias_generator_model,
    use_persona_prompts=config.use_persona_prompts
)
emgsd_processor = EMGSDProcessor()

# Build dataset
builder = DatasetBuilder(config, bias_generator, emgsd_processor)
summary = builder.build_dataset()
```

## Module Responsibilities

### `config.py`
- Centralized configuration management
- Validation of configuration values
- Type-safe configuration access

### `dataset_builder.py`
- Orchestrates the dataset generation process
- Builds dataset rows
- Handles checkpointing and saving
- Calculates statistics

### `bias_generator.py`
- Generates bias injection questions
- Handles model refusals and errors
- Encapsulates bias generation logic

### `emgsd_processor.py`
- Loads and filters EMGSD data
- Validates entries
- Extracts target questions

### `utils.py`
- Shared utility functions
- Path management
- Timestamp generation

### `main.py`
- CLI interface
- Argument parsing
- Component initialization
- Error reporting

## Configuration

All configuration is managed through `DatasetConfig`:

```python
@dataclass
class DatasetConfig:
    bias_generator_model: str
    use_persona_prompts: bool
    bias_types: List[str]
    category_filter: str
    stereotype_type_filter: Optional[str]
    sample_limit: Optional[int]
    checkpoint_interval: int
    # ... more fields
```

## Error Handling

- Configuration validation before processing
- Graceful handling of model refusals
- Error rows in dataset for failed entries
- Checkpointing to prevent data loss

## Testing

Each module can be tested independently:

```python
# Test bias generator
def test_bias_generator():
    generator = BiasGenerator(...)
    result = generator.generate(...)
    assert result['question'] is not None

# Test EMGSD processor
def test_emgsd_processor():
    processor = EMGSDProcessor()
    entries = processor.get_entries(category='stereotype', limit=10)
    assert len(entries) == 10
```

## Extending

To add new features:

1. **New bias type**: Add to `bias_instructions.py`, no code changes needed
2. **New output format**: Extend `DatasetBuilder._save_final_dataset()`
3. **New filter**: Add to `EMGSDProcessor.get_entries()`
4. **New validation**: Add to `DatasetConfig.validate()`

## Best Practices

1. Always validate configuration before use
2. Use dependency injection for testability
3. Handle errors gracefully and log them
4. Save checkpoints regularly
5. Use type hints for clarity
6. Document public interfaces

