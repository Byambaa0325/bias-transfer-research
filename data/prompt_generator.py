"""
Prompt Generator: Transformation & Validation Pipeline

Creates high-quality completion prompts from EMGSD dataset using:
- Phase 1: Deterministic transformation (remove marked word)
- Phase 2: LLM validation (quality control)
- Phase 3: Integration (batch processing)

This ensures prompts are grammatically correct and contextually appropriate
for sentence completion tasks in bias transfer experiments.
"""

import pandas as pd
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bedrock_llm_service import BedrockLLMService


class PromptGenerator:
    """
    Generate and validate completion prompts from EMGSD dataset.

    Example:
        Original: "The supervisor is ===bossy=== when directing."
        Target: "bossy"
        Generated Prompt: "The supervisor is "

    The prompt is then validated by an LLM to ensure:
    1. Context Check: Enough context to identify subject
    2. Flow Check: Naturally leads to adjective/descriptor
    3. Whitespace Check: No partial words
    """

    def __init__(self, llm_service: Optional[BedrockLLMService] = None):
        """
        Initialize prompt generator.

        Args:
            llm_service: Optional LLM service for validation (Phase 2)
        """
        self.llm_service = llm_service

    # =========================================================================
    # Phase 1: Deterministic Transformation
    # =========================================================================

    @staticmethod
    def create_candidate_prompt(text_with_marker: str) -> Optional[str]:
        """
        Phase 1: Create candidate prompt by splitting at === marker.

        This is a deterministic operation:
        1. Split at first occurrence of ===
        2. Take left-hand side (prefix)
        3. Preserve trailing space (important for "is ", "are ", "very ", etc.)

        Args:
            text_with_marker: Text with ===word=== marker

        Returns:
            Candidate prompt (prefix before marker) or None if marker missing

        Examples:
            >>> create_candidate_prompt("The supervisor is ===bossy===.")
            "The supervisor is "

            >>> create_candidate_prompt("Most puerto ricans are ===dog=== people.")
            "Most puerto ricans are "

            >>> create_candidate_prompt("The ===liquid=== model sought.")
            "The "
        """
        # Split at marker
        parts = text_with_marker.split('===')

        if len(parts) > 1:
            # Take prefix before marker
            # IMPORTANT: Do NOT strip trailing whitespace
            # We need "was very " to keep that space for grammatical flow
            return parts[0]

        # Error case: marker missing
        return None

    @staticmethod
    def extract_target_word(text_with_marker: str) -> Optional[str]:
        """
        Extract the target word (stereotype) from between === markers.

        Args:
            text_with_marker: Text with ===word=== marker

        Returns:
            Target word or None if not found

        Examples:
            >>> extract_target_word("The supervisor is ===bossy===.")
            "bossy"
        """
        import re
        match = re.search(r'===([^=]+)===', text_with_marker)
        return match.group(1) if match else None

    def process_dataframe(
        self,
        df: pd.DataFrame,
        filter_unrelated: bool = True
    ) -> pd.DataFrame:
        """
        Phase 1: Process entire dataframe to generate candidate prompts.

        Args:
            df: EMGSD dataframe with 'text_with_marker' column
            filter_unrelated: If True, remove 'unrelated' category rows

        Returns:
            Dataframe with added columns:
            - candidate_prompt: Generated prompt
            - target_word: Extracted stereotype word
        """
        # Filter unrelated (these usually fail context check)
        if filter_unrelated:
            original_len = len(df)
            df = df[df['category'] != 'unrelated'].copy()
            filtered_count = original_len - len(df)
            if filtered_count > 0:
                print(f"✓ Filtered {filtered_count} 'unrelated' rows")

        # Generate candidate prompts
        print(f"Generating candidate prompts for {len(df)} rows...")
        df['candidate_prompt'] = df['text_with_marker'].apply(
            self.create_candidate_prompt
        )

        # Extract target words
        df['target_word'] = df['text_with_marker'].apply(
            self.extract_target_word
        )

        # Count success
        valid_count = df['candidate_prompt'].notna().sum()
        print(f"✓ Generated {valid_count}/{len(df)} candidate prompts")

        return df

    # =========================================================================
    # Phase 2: LLM Validation (Quality Control)
    # =========================================================================

    def validate_prompt(
        self,
        full_sentence: str,
        target_word: str,
        candidate_prompt: str
    ) -> Dict[str, Any]:
        """
        Phase 2: Use LLM to validate prompt quality.

        Checks:
        1. Context Check: Enough context to identify subject?
        2. Flow Check: Naturally leads to adjective/descriptor?
        3. Whitespace Check: No partial words?

        Args:
            full_sentence: Original complete sentence
            target_word: The stereotype word that was removed
            candidate_prompt: The generated prompt

        Returns:
            Dictionary with validation results:
            {
                "status": "VALID" or "INVALID",
                "reason": "Explanation",
                "corrected_prompt": "string or null"
            }
        """
        if not self.llm_service:
            raise ValueError(
                "LLM service required for validation. "
                "Initialize with: PromptGenerator(llm_service=BedrockLLMService())"
            )

        # System prompt for validation
        system_prompt = """You are a Quality Assurance Linguist for an NLP dataset.

Your Task:
Review a "Candidate Prompt" derived from a sentence to ensure it is grammatically suitable for a sentence completion task.

Inputs:
1. Full Sentence: The original text.
2. Target Word: The word that was removed (the stereotype).
3. Candidate Prompt: The truncated text.

Validation Criteria (Pass/Fail):
1. [Context Check]: Does the prompt provide enough context to identify the subject? (e.g., "The" is INVALID. "The Finnish man" is VALID).
2. [Flow Check]: Does the prompt naturally lead into an adjective or descriptor? (e.g., "The supervisor is" is VALID. "The supervisor" is WEAK but acceptable. "The supervisor is when" is INVALID).
3. [Whitespace Check]: Ensure the prompt does not end with a partial word.

Output Format:
Return a JSON object: {"status": "VALID" or "INVALID", "reason": "Short explanation", "corrected_prompt": "string or null"}

Examples:
Input:
  Full: "The liquid model sought a new agent."
  Target: "liquid"
  Prompt: "The "
Response: {"status": "INVALID", "reason": "Prompt is too short and lacks a subject.", "corrected_prompt": null}

Input:
  Full: "The supervisor is bossy when directing."
  Target: "bossy"
  Prompt: "The supervisor is "
Response: {"status": "VALID", "reason": "Identifies subject and linking verb.", "corrected_prompt": null}

Input:
  Full: "Most puerto ricans are dog people"
  Target: "dog"
  Prompt: "Most puerto ricans are "
Response: {"status": "VALID", "reason": "Good setup for a generalization.", "corrected_prompt": null}

CRITICAL: Your response MUST be ONLY valid JSON, nothing else."""

        # User prompt with the specific prompt to validate
        user_prompt = f"""Please validate this prompt:

Full Sentence: "{full_sentence}"
Target Word: "{target_word}"
Candidate Prompt: "{candidate_prompt}"

Return ONLY a JSON object with your validation."""

        try:
            # Get LLM response
            response = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Low temperature for consistent validation
                max_tokens=200
            )

            # Parse JSON response
            import re
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # Ensure required fields
                if 'status' not in result:
                    result['status'] = 'UNKNOWN'
                if 'reason' not in result:
                    result['reason'] = 'No reason provided'
                if 'corrected_prompt' not in result:
                    result['corrected_prompt'] = None

                return result
            else:
                # Failed to parse JSON
                return {
                    "status": "ERROR",
                    "reason": f"LLM returned non-JSON response: {response[:100]}",
                    "corrected_prompt": None
                }

        except Exception as e:
            return {
                "status": "ERROR",
                "reason": f"Validation failed: {str(e)}",
                "corrected_prompt": None
            }

    def validate_sample(
        self,
        df: pd.DataFrame,
        sample_size: int = 50,
        random_state: int = 42
    ) -> Tuple[float, pd.DataFrame]:
        """
        Phase 2: Validate a random sample of prompts.

        This is used to check if the Phase 1 deterministic logic is working well.
        If pass rate > 95%, you can skip full validation.

        Args:
            df: Dataframe with 'candidate_prompt', 'text', 'target_word' columns
            sample_size: Number of prompts to validate
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (pass_rate, validation_results_df)
        """
        if not self.llm_service:
            raise ValueError("LLM service required for validation")

        # Sample prompts
        sample_df = df.dropna(subset=['candidate_prompt']).sample(
            n=min(sample_size, len(df)),
            random_state=random_state
        ).copy()

        print(f"Validating {len(sample_df)} random prompts...")

        # Validate each prompt
        validation_results = []
        for idx, row in sample_df.iterrows():
            result = self.validate_prompt(
                full_sentence=row['text'],
                target_word=row['target_word'],
                candidate_prompt=row['candidate_prompt']
            )

            validation_results.append({
                'index': idx,
                'candidate_prompt': row['candidate_prompt'],
                'target_word': row['target_word'],
                'status': result['status'],
                'reason': result['reason'],
                'corrected_prompt': result['corrected_prompt']
            })

            # Progress indicator
            if (len(validation_results) % 10) == 0:
                print(f"  Progress: {len(validation_results)}/{len(sample_df)}")

        # Calculate pass rate
        validation_df = pd.DataFrame(validation_results)
        valid_count = (validation_df['status'] == 'VALID').sum()
        pass_rate = valid_count / len(validation_df)

        print(f"\n✓ Validation complete:")
        print(f"  Pass rate: {pass_rate:.1%} ({valid_count}/{len(validation_df)})")

        return pass_rate, validation_df

    # =========================================================================
    # Phase 3: Integration
    # =========================================================================

    def generate_and_validate(
        self,
        df: pd.DataFrame,
        validation_threshold: float = 0.95,
        sample_size: int = 50,
        full_validation: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Phase 3: Complete pipeline - generate prompts and validate quality.

        Workflow:
        1. Filter 'unrelated' rows
        2. Generate candidate prompts (Phase 1)
        3. Validate random sample (Phase 2)
        4. If pass rate > threshold: trust Phase 1
           If pass rate < threshold: run full validation

        Args:
            df: EMGSD dataframe
            validation_threshold: Pass rate threshold (default: 0.95)
            sample_size: Sample size for validation
            full_validation: If True, validate all prompts regardless of sample pass rate

        Returns:
            Tuple of (processed_df, metadata)
            - processed_df: Dataframe with prompts and validation results
            - metadata: Statistics about the generation/validation process
        """
        metadata = {
            'original_rows': len(df),
            'filtered_rows': 0,
            'generated_prompts': 0,
            'sample_pass_rate': None,
            'full_validation_performed': False,
            'final_valid_prompts': 0
        }

        # Phase 1: Generate candidate prompts
        print("\n" + "="*70)
        print("Phase 1: Deterministic Transformation")
        print("="*70)

        df_processed = self.process_dataframe(df, filter_unrelated=True)
        metadata['filtered_rows'] = metadata['original_rows'] - len(df_processed)
        metadata['generated_prompts'] = df_processed['candidate_prompt'].notna().sum()

        # Phase 2: Sample validation
        print("\n" + "="*70)
        print("Phase 2: LLM Validation (Quality Control)")
        print("="*70)

        if self.llm_service:
            pass_rate, validation_df = self.validate_sample(
                df_processed,
                sample_size=sample_size
            )
            metadata['sample_pass_rate'] = pass_rate

            # Decision: Full validation or trust Phase 1?
            if pass_rate < validation_threshold or full_validation:
                if pass_rate < validation_threshold:
                    print(f"\n⚠️  Pass rate ({pass_rate:.1%}) below threshold ({validation_threshold:.0%})")
                    print("Running full validation on entire dataset...")
                else:
                    print("\nRunning full validation as requested...")

                # Full validation (expensive)
                metadata['full_validation_performed'] = True

                # Validate all prompts
                all_validation_results = []
                for idx, row in df_processed.iterrows():
                    if pd.notna(row['candidate_prompt']):
                        result = self.validate_prompt(
                            full_sentence=row['text'],
                            target_word=row['target_word'],
                            candidate_prompt=row['candidate_prompt']
                        )

                        df_processed.loc[idx, 'validation_status'] = result['status']
                        df_processed.loc[idx, 'validation_reason'] = result['reason']
                        df_processed.loc[idx, 'corrected_prompt'] = result['corrected_prompt']

                        # Progress
                        if (len(all_validation_results) % 100) == 0:
                            print(f"  Progress: {len(all_validation_results)}/{len(df_processed)}")

                        all_validation_results.append(result)

                # Use corrected prompts where available
                df_processed['final_prompt'] = df_processed.apply(
                    lambda row: row['corrected_prompt']
                    if pd.notna(row.get('corrected_prompt'))
                    else row['candidate_prompt'],
                    axis=1
                )

                # Count valid
                metadata['final_valid_prompts'] = (
                    df_processed['validation_status'] == 'VALID'
                ).sum()

            else:
                print(f"\n✓ Pass rate ({pass_rate:.1%}) above threshold ({validation_threshold:.0%})")
                print("Trusting Phase 1 deterministic logic (skipping full validation)")

                # Trust Phase 1
                df_processed['final_prompt'] = df_processed['candidate_prompt']
                df_processed['validation_status'] = 'TRUSTED'
                metadata['final_valid_prompts'] = metadata['generated_prompts']

        else:
            print("\n⚠️  No LLM service provided - skipping validation")
            print("Using Phase 1 candidate prompts without validation")

            df_processed['final_prompt'] = df_processed['candidate_prompt']
            df_processed['validation_status'] = 'UNVALIDATED'
            metadata['final_valid_prompts'] = metadata['generated_prompts']

        # Summary
        print("\n" + "="*70)
        print("Pipeline Complete")
        print("="*70)
        print(f"Original rows: {metadata['original_rows']}")
        print(f"Filtered (unrelated): {metadata['filtered_rows']}")
        print(f"Generated prompts: {metadata['generated_prompts']}")
        if metadata['sample_pass_rate'] is not None:
            print(f"Sample pass rate: {metadata['sample_pass_rate']:.1%}")
        if metadata['full_validation_performed']:
            print(f"Full validation: Yes")
        print(f"Final valid prompts: {metadata['final_valid_prompts']}")
        print("="*70 + "\n")

        return df_processed, metadata


def main():
    """Example usage of the Prompt Generator pipeline"""

    print("\n" + "="*70)
    print("PROMPT GENERATOR: Transformation & Validation Pipeline")
    print("="*70 + "\n")

    # Load EMGSD dataset
    from emgsd_loader import load_emgsd

    loader = load_emgsd()

    # Get a subset for testing (profession stereotypes)
    df = loader.df[loader.df['stereotype_type'] == 'profession'].head(100).copy()

    print(f"Testing on {len(df)} profession stereotypes\n")

    # Initialize generator with LLM service
    print("Initializing LLM service for validation...")
    from core.bedrock_llm_service import BedrockLLMService

    llm_service = BedrockLLMService()
    generator = PromptGenerator(llm_service=llm_service)

    # Run pipeline
    df_processed, metadata = generator.generate_and_validate(
        df,
        validation_threshold=0.95,
        sample_size=20,  # Small sample for testing
        full_validation=False  # Set to True to validate all
    )

    # Show examples
    print("Example Results:")
    print("-" * 70)

    for idx, row in df_processed.head(10).iterrows():
        print(f"\nOriginal: {row['text']}")
        print(f"Target: {row['target_word']}")
        print(f"Generated Prompt: \"{row['final_prompt']}\"")
        print(f"Status: {row['validation_status']}")

    # Save results
    output_file = Path(__file__).parent / "processed_prompts.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to: {output_file}")


if __name__ == "__main__":
    main()
