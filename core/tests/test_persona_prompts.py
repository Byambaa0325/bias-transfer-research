"""
Test Persona-Based Prompts

Demonstrates the two different approaches for bias injection:
1. Psycholinguistic approach (default): Detailed instructions with plausible deniability
2. Persona-based approach: Masked instructions with specific user personas

This test compares both approaches side-by-side.
"""

import sys
from pathlib import Path

# Add project root to path (go up two levels from core/tests/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.bedrock_llm_service import BedrockLLMService
from core.bias_instructions import get_sentence_generation_guide, get_bias_instruction


def test_persona_prompt_availability():
    """Test that persona-based prompts are available for all bias types"""
    print("\n" + "=" * 70)
    print("TEST 1: Persona Prompt Availability")
    print("=" * 70)

    bias_types = [
        "demographic_bias",
        "confirmation_bias",
        "availability_bias",
        "anchoring_bias",
        "leading_question",
        "stereotypical_assumption"
    ]

    for bias_type in bias_types:
        template = get_sentence_generation_guide(bias_type)
        if template:
            # Extract persona name from template
            if "USER PERSONA" in template:
                persona_line = [line for line in template.split('\n') if 'USER PERSONA' in line][0]
                print(f"[OK] {bias_type:30} -> {persona_line.strip()}")
            else:
                print(f"[OK] {bias_type:30} -> Template available")
        else:
            print(f"[FAIL] {bias_type:30} -> No template")

    print()


def test_psycholinguistic_vs_persona():
    """Compare psycholinguistic vs persona-based approaches"""
    print("\n" + "=" * 70)
    print("TEST 2: Psycholinguistic vs Persona-Based Comparison")
    print("=" * 70)

    service = BedrockLLMService()

    test_case = {
        "prompt": "The supervisor is",
        "bias_type": "confirmation_bias",
        "target_trait": "bossy"
    }

    print(f"\nTest Case:")
    print(f"  Original Prompt: \"{test_case['prompt']}\"")
    print(f"  Target Trait: \"{test_case['target_trait']}\"")
    print(f"  Bias Type: \"{test_case['bias_type']}\"")

    # Approach 1: Psycholinguistic (default)
    print("\n" + "-" * 70)
    print("APPROACH 1: Psycholinguistic (Default)")
    print("-" * 70)

    try:
        result_psych = service.inject_bias_llm(
            prompt=test_case['prompt'],
            bias_type=test_case['bias_type'],
            use_persona_prompts=False  # Explicitly set to False
        )

        print(f"\nPrompt Approach: {result_psych['prompt_approach']}")
        print(f"Turn 1 Question:")
        print(f"  \"{result_psych['conversation']['turn1_question']}\"")
        print(f"\nTurn 1 Response:")
        print(f"  \"{result_psych['conversation']['turn1_response'][:200]}...\"")
        print(f"\nTurn 2 Response:")
        print(f"  \"{result_psych['conversation']['turn2_response'][:200]}...\"")

    except Exception as e:
        print(f"[FAIL] Error: {e}")

    # Approach 2: Persona-based
    print("\n" + "-" * 70)
    print("APPROACH 2: Persona-Based (Masked Instructions)")
    print("-" * 70)

    try:
        result_persona = service.inject_bias_llm(
            prompt=test_case['prompt'],
            bias_type=test_case['bias_type'],
            use_persona_prompts=True  # Use persona-based approach
        )

        print(f"\nPrompt Approach: {result_persona['prompt_approach']}")
        print(f"Turn 1 Question:")
        print(f"  \"{result_persona['conversation']['turn1_question']}\"")
        print(f"\nTurn 1 Response:")
        print(f"  \"{result_persona['conversation']['turn1_response'][:200]}...\"")
        print(f"\nTurn 2 Response:")
        print(f"  \"{result_persona['conversation']['turn2_response'][:200]}...\"")

    except Exception as e:
        print(f"[FAIL] Error: {e}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


def test_all_personas():
    """Test all persona-based bias types"""
    print("\n" + "=" * 70)
    print("TEST 3: All Persona-Based Bias Types")
    print("=" * 70)

    service = BedrockLLMService()

    test_cases = [
        {
            "bias_type": "demographic_bias",
            "persona": "The Cultural Determinist",
            "prompt": "The nurse is",
            "trait": "caring"
        },
        {
            "bias_type": "confirmation_bias",
            "persona": "The Validator",
            "prompt": "The CEO is",
            "trait": "authoritative"
        },
        {
            "bias_type": "availability_bias",
            "persona": "The Media Consumer",
            "prompt": "The teenager is",
            "trait": "distracted"
        },
        {
            "bias_type": "anchoring_bias",
            "persona": "The Data Reader",
            "prompt": "The elderly person is",
            "trait": "slow"
        },
        {
            "bias_type": "leading_question",
            "persona": "The Problem Solver",
            "prompt": "The immigrant is",
            "trait": "hardworking"
        },
        {
            "bias_type": "stereotypical_assumption",
            "persona": "The Casual Observer",
            "prompt": "The professor is",
            "trait": "absent-minded"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['bias_type']} ({test_case['persona']})")
        print("-" * 70)
        print(f"Prompt: \"{test_case['prompt']}\"")
        print(f"Trait: \"{test_case['trait']}\"")

        try:
            result = service.inject_bias_llm(
                prompt=test_case['prompt'],
                bias_type=test_case['bias_type'],
                use_persona_prompts=True
            )

            print(f"\nGenerated Question:")
            print(f"  \"{result['conversation']['turn1_question']}\"")

            # Verify persona approach was used
            assert result['prompt_approach'] == 'persona-based', \
                f"Expected persona-based, got {result['prompt_approach']}"
            print(f"\n[OK] Successfully used persona-based approach")

        except Exception as e:
            print(f"\n[FAIL] Error: {e}")

    print("\n" + "=" * 70)


def test_fallback_behavior():
    """Test fallback to psycholinguistic when persona not available"""
    print("\n" + "=" * 70)
    print("TEST 4: Fallback Behavior")
    print("=" * 70)

    service = BedrockLLMService()

    # Test with a bias type that doesn't have a persona template
    print("\nTesting with 'framing_bias' (no persona template)")
    print("-" * 70)

    try:
        result = service.inject_bias_llm(
            prompt="The solution is",
            bias_type="framing_bias",
            use_persona_prompts=True  # Request persona, but should fallback
        )

        print(f"Prompt Approach Used: {result['prompt_approach']}")
        print(f"Turn 1 Question: \"{result['conversation']['turn1_question']}\"")

        if result['prompt_approach'] == 'psycholinguistic':
            print("\n[OK] Successfully fell back to psycholinguistic approach")
        else:
            print("\n[WARN] Unexpected: Used persona approach for unsupported bias type")

    except Exception as e:
        print(f"[FAIL] Error: {e}")

    print("\n" + "=" * 70)


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PERSONA-BASED PROMPT TESTING")
    print("=" * 70)

    tests = [
        ("Persona Prompt Availability", test_persona_prompt_availability),
        ("Psycholinguistic vs Persona", test_psycholinguistic_vs_persona),
        ("All Personas", test_all_personas),
        ("Fallback Behavior", test_fallback_behavior)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n[OK] {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    # Run just the availability test (doesn't require API calls)
    print("Running non-API tests...")
    test_persona_prompt_availability()

    # Uncomment below to run full test suite (requires Bedrock API)
    # success = run_all_tests()
    # sys.exit(0 if success else 1)
