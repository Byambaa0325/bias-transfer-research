"""
Bias Injection Module

Demonstrates different ways to inject bias into prompts with explicit callouts.
This helps users understand how subtle changes can introduce bias.
"""

from typing import List, Dict, Any
import re


class BiasInjector:
    """Injects various types of biases into prompts to demonstrate their effects"""
    
    def inject_biases(self, original_prompt: str) -> List[Dict[str, Any]]:
        """
        Generate multiple biased versions of a prompt.
        
        Args:
            original_prompt: The original user prompt
            
        Returns:
            List of dictionaries, each containing a biased version and explanation
        """
        biased_versions = []
        
        # 1. Add demographic bias
        demographic_biased = self._inject_demographic_bias(original_prompt)
        if demographic_biased:
            biased_versions.append(demographic_biased)
        
        # 2. Add confirmation bias
        confirmation_biased = self._inject_confirmation_bias(original_prompt)
        if confirmation_biased:
            biased_versions.append(confirmation_biased)
        
        # 3. Add availability bias
        availability_biased = self._inject_availability_bias(original_prompt)
        if availability_biased:
            biased_versions.append(availability_biased)
        
        # 4. Add anchoring bias
        anchoring_biased = self._inject_anchoring_bias(original_prompt)
        if anchoring_biased:
            biased_versions.append(anchoring_biased)
        
        # 5. Add leading question bias
        leading_biased = self._inject_leading_question(original_prompt)
        if leading_biased:
            biased_versions.append(leading_biased)
        
        # 6. Add stereotypical assumption
        stereotype_biased = self._inject_stereotypical_assumption(original_prompt)
        if stereotype_biased:
            biased_versions.append(stereotype_biased)
        
        return biased_versions
    
    def _inject_demographic_bias(self, prompt: str) -> Dict[str, Any]:
        """Inject demographic bias by adding demographic qualifiers"""
        # Check if prompt already has demographic info
        if any(word in prompt.lower() for word in ['demographic', 'race', 'gender', 'age', 'religion']):
            return None
        
        # Add a demographic qualifier
        biased_prompt = f"When considering [demographic group], {prompt.lower()}"
        
        return {
            'type': 'demographic_bias',
            'biased_prompt': biased_prompt,
            'bias_added': 'Demographic Bias',
            'explanation': 'Added a demographic qualifier that may cause the LLM to make assumptions based on group identity rather than individual characteristics. This can lead to stereotyping and unfair generalizations.',
            'how_it_works': 'By introducing demographic information, the prompt primes the model to consider group-level characteristics, which can override individual assessment.'
        }
    
    def _inject_confirmation_bias(self, prompt: str) -> Dict[str, Any]:
        """Inject confirmation bias by adding leading phrases"""
        if any(phrase in prompt.lower() for phrase in ['isn\'t it', 'clearly', 'obviously', 'proves']):
            return None
        
        biased_prompt = f"Isn't it true that {prompt.lower()}? This clearly shows the pattern."
        
        return {
            'type': 'confirmation_bias',
            'biased_prompt': biased_prompt,
            'bias_added': 'Confirmation Bias',
            'explanation': 'Added phrases that suggest a particular conclusion, making the LLM more likely to confirm the premise rather than critically evaluate it.',
            'how_it_works': 'The phrases "isn\'t it true" and "clearly shows" prime the model to seek confirming evidence rather than considering alternative perspectives.'
        }
    
    def _inject_availability_bias(self, prompt: str) -> Dict[str, Any]:
        """Inject availability bias by referencing recent/visible examples"""
        if 'recent' in prompt.lower() or 'you\'ve probably' in prompt.lower():
            return None
        
        biased_prompt = f"Given recent examples you've probably heard about, {prompt.lower()}"
        
        return {
            'type': 'availability_bias',
            'biased_prompt': biased_prompt,
            'bias_added': 'Availability Bias',
            'explanation': 'Added reference to "recent examples" which relies on easily recalled information rather than comprehensive evidence, potentially skewing the response.',
            'how_it_works': 'By emphasizing recent or memorable examples, the prompt biases the model toward information that is easily accessible in memory, which may not be representative.'
        }
    
    def _inject_anchoring_bias(self, prompt: str) -> Dict[str, Any]:
        """Inject anchoring bias by providing a reference point"""
        if any(word in prompt.lower() for word in ['compared to', 'relative to', 'more than', 'less than']):
            return None
        
        biased_prompt = f"Compared to a baseline of [X], {prompt.lower()}"
        
        return {
            'type': 'anchoring_bias',
            'biased_prompt': biased_prompt,
            'bias_added': 'Anchoring Bias',
            'explanation': 'Added a comparison point that may unduly influence the LLM\'s assessment, causing it to anchor to the provided reference.',
            'how_it_works': 'The comparison frame sets an anchor point that can skew subsequent judgments, even if the anchor is arbitrary or irrelevant.'
        }
    
    def _inject_leading_question(self, prompt: str) -> Dict[str, Any]:
        """Inject leading question bias"""
        if prompt.strip().endswith('?') and any(word in prompt.lower() for word in ['why', 'isn\'t', 'don\'t']):
            return None
        
        # Convert statement to leading question
        if not prompt.strip().endswith('?'):
            biased_prompt = f"Why is it that {prompt.lower()}? Don't you think this is problematic?"
        else:
            biased_prompt = f"Don't you think {prompt.lower()} This seems like a major issue."
        
        return {
            'type': 'leading_question',
            'biased_prompt': biased_prompt,
            'bias_added': 'Leading Question Bias',
            'explanation': 'Rephrased as a leading question that suggests a particular answer, making the LLM more likely to agree with the implied conclusion.',
            'how_it_works': 'Leading questions contain assumptions that guide the respondent toward a specific answer, reducing critical evaluation.'
        }
    
    def _inject_stereotypical_assumption(self, prompt: str) -> Dict[str, Any]:
        """Inject stereotypical assumptions"""
        if any(word in prompt.lower() for word in ['typically', 'usually', 'always', 'never', 'all']):
            return None
        
        biased_prompt = f"Typically, {prompt.lower()} This is usually the case."
        
        return {
            'type': 'stereotypical_assumption',
            'biased_prompt': biased_prompt,
            'bias_added': 'Stereotypical Assumption',
            'explanation': 'Added language that makes broad generalizations, which can lead to stereotyping and overlook individual variation.',
            'how_it_works': 'Words like "typically" and "usually" prime the model to make group-level generalizations that may not apply to individuals, reinforcing stereotypes.'
        }


if __name__ == "__main__":
    # Example usage
    injector = BiasInjector()
    test_prompt = "What are the best programming languages to learn?"
    biased_versions = injector.inject_biases(test_prompt)
    
    for version in biased_versions:
        print(f"\n{version['bias_added']}:")
        print(f"  Prompt: {version['biased_prompt']}")
        print(f"  Explanation: {version['explanation']}")

