"""Test that short bias names work in BedrockLLMService"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.bedrock_llm_service import BedrockLLMService
from core.bias_instructions import get_sentence_generation_guide

print("=" * 70)
print("TEST: Short Bias Names in BedrockLLMService")
print("=" * 70)

# Test that templates are available for short names
print("\n1. Testing template availability:")
print("-" * 70)

short_names = [
    "confirmation",
    "availability",
    "anchoring",
    "demographic",
    "leading",
    "stereotypical"
]

for name in short_names:
    template = get_sentence_generation_guide(name)
    if template:
        persona_line = [line for line in template.split('\n') if 'USER PERSONA' in line][0]
        print(f"[OK] {name:20} -> {persona_line.strip()}")
    else:
        print(f"[FAIL] {name:20} -> No template found")

# Test service initialization
print("\n2. Testing service initialization:")
print("-" * 70)

try:
    service = BedrockLLMService()
    print("[OK] BedrockLLMService initialized successfully")
except Exception as e:
    print(f"[FAIL] Could not initialize service: {e}")
    sys.exit(1)

# Test that create_persona_based_prompt works with short names
print("\n3. Testing create_persona_based_prompt with short names:")
print("-" * 70)

test_cases = [
    ("confirmation", "The supervisor is", "bossy"),
    ("availability", "The teenager is", "distracted"),
    ("anchoring", "The worker is", "lazy"),
]

for bias_type, prompt, trait in test_cases:
    try:
        result = service.prompt_generator.create_persona_based_prompt(
            bias_type=bias_type,
            original_prompt=prompt,
            target_trait=trait
        )
        if result:
            print(f"[OK] {bias_type:20} -> Prompt generated ({len(result)} chars)")
        else:
            print(f"[FAIL] {bias_type:20} -> Returned None")
    except Exception as e:
        print(f"[FAIL] {bias_type:20} -> Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\nYou can now use short bias names like 'confirmation' instead of")
print("'confirmation_bias' in both persona-based and psycholinguistic approaches:")
print()
print("  service.inject_bias_llm(")
print("      prompt='The supervisor is',")
print("      bias_type='confirmation',  # Short name works!")
print("      use_persona_prompts=True")
print("  )")
