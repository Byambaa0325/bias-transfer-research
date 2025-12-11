"""Test bias type name normalization"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.bias_instructions import get_sentence_generation_guide

# Test cases: (input, expected_to_work)
tests = [
    ('confirmation', True),
    ('confirmation_bias', True),
    ('availability', True),
    ('availability_bias', True),
    ('anchoring', True),
    ('anchoring_bias', True),
    ('demographic', True),
    ('demographic_bias', True),
    ('leading', True),
    ('leading_question', True),
    ('stereotypical', True),
    ('stereotypical_assumption', True),
    ('framing', False),  # No persona template for framing
    ('framing_bias', False),
]

print("Testing Bias Type Name Normalization")
print("=" * 60)

success_count = 0
fail_count = 0

for bias_type, should_work in tests:
    result = get_sentence_generation_guide(bias_type)
    found = result is not None

    if found == should_work:
        status = "[OK]"
        success_count += 1
    else:
        status = "[FAIL]"
        fail_count += 1

    expected = "Found" if should_work else "None"
    actual = "Found" if found else "None"

    print(f"{status} {bias_type:30} Expected: {expected:10} Got: {actual:10}")

print("=" * 60)
print(f"Results: {success_count} passed, {fail_count} failed")

if fail_count == 0:
    print("\nAll tests passed!")
    sys.exit(0)
else:
    print(f"\n{fail_count} tests failed")
    sys.exit(1)
