import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import random


@dataclass
class EMGSDEntry:
    """A single entry from the EMGSD dataset"""
    text: str
    text_with_marker: str
    stereotype_type: str  # nationality, profession, gender, religion, race
    category: str  # stereotype, neutral, unrelated
    data_source: str  # stereoset_intrasentence, etc.
    label: str  # Combined label (e.g., "stereotype_profession")

    # Extracted components
    target_group: Optional[str] = None  # e.g., "software developer", "Kenya"
    trait: Optional[str] = None  # The marked word in === ===
    
    # Prompt transformation (from prompt_transformer.py)
    final_prompt: Optional[str] = None  # High-quality prompt for experiments
    target_word: Optional[str] = None  # The word removed from prompt

    def is_stereotype(self) -> bool:
        """Check if this entry is a stereotype"""
        return self.category == "stereotype"

    def is_neutral(self) -> bool:
        """Check if this entry is neutral (anti-stereotype)"""
        return self.category == "neutral"

    def is_unrelated(self) -> bool:
        """Check if this entry is unrelated/nonsensical"""
        return self.category == "unrelated"


class EMGSDLoader:
    """
    Loader for the EMGSD dataset from the HEARTS project.

    The dataset contains stereotypes across multiple categories:
    - Nationality (e.g., "Finnish people are energetic")
    - Profession (e.g., "Supervisors are bossy")
    - Gender (e.g., "His grandmother is wise")
    - Religion, Race, Age, etc.

    Each entry has:
    - Stereotype version (reinforces stereotype)
    - Neutral version (counters stereotype)
    - Unrelated version (nonsensical control)
    """

    DEFAULT_DATASET_PATH = Path(
        r"E:\UCL-Workspaces\ai-sustainable-dev\project\HEARTS-Text-Stereotype-Detection"
        r"\Exploratory Data Analysis\MGSD - Expanded.csv"
    )

    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Initialize the EMGSD loader.

        Args:
            dataset_path: Path to the EMGSD CSV file (defaults to HEARTS location)
        """
        self.dataset_path = dataset_path or self.DEFAULT_DATASET_PATH

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"EMGSD dataset not found at: {self.dataset_path}\n"
                f"Please ensure the HEARTS project is at the expected location."
            )

        print(f"Loading EMGSD dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Loaded {len(self.df)} entries")

        # Parse dataset
        self.entries = self._parse_entries()
        print(f"✓ Parsed {len(self.entries)} valid entries")

    def _extract_trait_from_marker(self, text_with_marker: str) -> Optional[str]:
        """Extract the trait from the marked text (between === ===)"""
        import re
        match = re.search(r'===([^=]+)===', text_with_marker)
        return match.group(1) if match else None

    def _parse_entries(self) -> List[EMGSDEntry]:
        """Parse the dataframe into EMGSDEntry objects"""
        entries = []

        for _, row in self.df.iterrows():
            try:
                # Extract trait from marked text
                trait = self._extract_trait_from_marker(row['text_with_marker'])

                # Check if transformed prompts are available
                final_prompt = row.get('final_prompt', None)
                target_word = row.get('target_word', None)

                entry = EMGSDEntry(
                    text=row['text'],
                    text_with_marker=row['text_with_marker'],
                    stereotype_type=row['stereotype_type'],
                    category=row['category'],
                    data_source=row['data_source'],
                    label=row['label'],
                    trait=trait,
                    final_prompt=final_prompt,
                    target_word=target_word
                )

                entries.append(entry)
            except Exception as e:
                print(f"Warning: Could not parse row: {e}")
                continue

        return entries

    def get_stereotypes(
        self,
        stereotype_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[EMGSDEntry]:
        """
        Get stereotype entries (category='stereotype').

        Args:
            stereotype_type: Filter by type (nationality, profession, gender, etc.)
            limit: Maximum number of entries to return

        Returns:
            List of stereotype entries
        """
        stereotypes = [e for e in self.entries if e.is_stereotype()]

        if stereotype_type:
            stereotypes = [e for e in stereotypes if e.stereotype_type == stereotype_type]

        if limit:
            stereotypes = stereotypes[:limit]

        return stereotypes

    def get_neutral(
        self,
        stereotype_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[EMGSDEntry]:
        """
        Get neutral (anti-stereotype) entries.

        Args:
            stereotype_type: Filter by type
            limit: Maximum number of entries

        Returns:
            List of neutral entries
        """
        neutral = [e for e in self.entries if e.is_neutral()]

        if stereotype_type:
            neutral = [e for e in neutral if e.stereotype_type == stereotype_type]

        if limit:
            neutral = neutral[:limit]

        return neutral

    def get_stereotype_pairs(
        self,
        stereotype_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, EMGSDEntry]]:
        """
        Get pairs of stereotype + neutral (anti-stereotype) entries.

        This is useful for calculating drift scores:
        Drift = Similarity(Output, Stereotype) - Similarity(Output, Anti-Stereotype)

        Args:
            stereotype_type: Filter by type
            limit: Maximum number of pairs

        Returns:
            List of dicts with 'stereotype' and 'neutral' keys
        """
        # Group by stereotype_type and similar content
        # For EMGSD, we can group by the base text pattern

        # Simplified: Just pair stereotypes with neutrals of same type
        stereotypes = self.get_stereotypes(stereotype_type=stereotype_type)
        neutrals = self.get_neutral(stereotype_type=stereotype_type)

        pairs = []
        for i, stereotype_entry in enumerate(stereotypes):
            if i < len(neutrals):
                pairs.append({
                    'stereotype': stereotype_entry,
                    'neutral': neutrals[i]
                })

        if limit:
            pairs = pairs[:limit]

        return pairs

    def get_random_stereotype(
        self,
        stereotype_type: Optional[str] = None,
        exclude_traits: Optional[List[str]] = None
    ) -> EMGSDEntry:
        """
        Get a random stereotype entry.

        Args:
            stereotype_type: Filter by type
            exclude_traits: Exclude entries with these traits

        Returns:
            Random stereotype entry
        """
        stereotypes = self.get_stereotypes(stereotype_type=stereotype_type)

        if exclude_traits:
            stereotypes = [
                s for s in stereotypes
                if s.trait not in exclude_traits
            ]

        return random.choice(stereotypes)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'total_entries': len(self.entries),
            'stereotypes': len([e for e in self.entries if e.is_stereotype()]),
            'neutrals': len([e for e in self.entries if e.is_neutral()]),
            'unrelated': len([e for e in self.entries if e.is_unrelated()]),
            'by_type': {}
        }

        # Count by stereotype type
        for entry in self.entries:
            if entry.stereotype_type not in stats['by_type']:
                stats['by_type'][entry.stereotype_type] = {
                    'stereotype': 0,
                    'neutral': 0,
                    'unrelated': 0
                }

            if entry.is_stereotype():
                stats['by_type'][entry.stereotype_type]['stereotype'] += 1
            elif entry.is_neutral():
                stats['by_type'][entry.stereotype_type]['neutral'] += 1
            elif entry.is_unrelated():
                stats['by_type'][entry.stereotype_type]['unrelated'] += 1

        return stats

    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("EMGSD Dataset Statistics")
        print("="*60)
        print(f"Total entries: {stats['total_entries']}")
        print(f"  - Stereotypes: {stats['stereotypes']}")
        print(f"  - Neutrals (anti-stereotypes): {stats['neutrals']}")
        print(f"  - Unrelated (nonsensical): {stats['unrelated']}")
        print("\nBy stereotype type:")
        for stype, counts in stats['by_type'].items():
            print(f"  {stype:15} → {counts['stereotype']:4} stereotypes, "
                  f"{counts['neutral']:4} neutrals, {counts['unrelated']:4} unrelated")
        print("="*60 + "\n")


def load_emgsd(dataset_path: Optional[Path] = None) -> EMGSDLoader:
    """
    Convenience function to load the EMGSD dataset.

    Args:
        dataset_path: Optional path to dataset CSV

    Returns:
        EMGSDLoader instance
    """
    return EMGSDLoader(dataset_path=dataset_path)


if __name__ == "__main__":
    # Example usage
    loader = load_emgsd()

    # Print statistics
    loader.print_statistics()

    # Get some examples
    print("\n" + "="*60)
    print("Example Stereotypes")
    print("="*60)

    # Profession stereotypes
    profession_stereotypes = loader.get_stereotypes(stereotype_type="profession", limit=5)
    for i, entry in enumerate(profession_stereotypes, 1):
        print(f"\n{i}. {entry.text}")
        print(f"   Type: {entry.stereotype_type}")
        print(f"   Trait: {entry.trait}")

    # Gender stereotypes
    print("\n" + "="*60)
    print("Example Gender Stereotypes")
    print("="*60)
    gender_stereotypes = loader.get_stereotypes(stereotype_type="gender", limit=5)
    for i, entry in enumerate(gender_stereotypes, 1):
        print(f"\n{i}. {entry.text}")
        print(f"   Trait: {entry.trait}")

    # Stereotype-Neutral pairs
    print("\n" + "="*60)
    print("Example Stereotype-Neutral Pairs")
    print("="*60)
    pairs = loader.get_stereotype_pairs(stereotype_type="profession", limit=3)
    for i, pair in enumerate(pairs, 1):
        print(f"\n{i}. Stereotype: {pair['stereotype'].text}")
        print(f"   Neutral:     {pair['neutral'].text}")
