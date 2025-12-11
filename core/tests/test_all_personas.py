"""Test all persona templates including new ones"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.bias_instructions import get_sentence_generation_guide, SENTENCE_GENERATION_GUIDES

print("=" * 70)
print("ALL PERSONA TEMPLATES")
print("=" * 70)

count = 0
for bias_type in sorted(SENTENCE_GENERATION_GUIDES.keys()):
    template = get_sentence_generation_guide(bias_type)
    if template and 'USER PERSONA' in template:
        persona_line = [line for line in template.split('\n') if 'USER PERSONA' in line][0]
        print(f"[OK] {bias_type:30} -> {persona_line.strip()}")
        count += 1
    else:
        print(f"[FAIL] {bias_type:30} -> No persona found")

print("=" * 70)
print(f"Total: {count} persona templates")

print()
print("=" * 70)
print("TESTING SHORT NAMES")
print("=" * 70)

short_tests = [
    ('confirmation', True),
    ('availability', True),
    ('anchoring', True),
    ('demographic', True),
    ('leading', True),
    ('stereotypical', True),
    ('framing', True),         # NEW
    ('negativity', True),       # NEW
]

all_pass = True
for short_name, expected in short_tests:
    result = get_sentence_generation_guide(short_name)
    found = result is not None

    if found == expected:
        status = '[OK]'
    else:
        status = '[FAIL]'
        all_pass = False

    found_text = "Found" if result else "Not found"
    print(f"{status} {short_name:20} -> {found_text}")

print("=" * 70)

if all_pass:
    print("\nAll tests passed!")
    sys.exit(0)
else:
    print("\nSome tests failed")
    sys.exit(1)
