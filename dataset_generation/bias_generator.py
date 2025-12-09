"""
Bias injection question generator.

Handles generation of Turn 1 bias injection questions for different cognitive bias types.
"""

from typing import Dict, Optional, Any
from core.bedrock_llm_service import BedrockLLMService
from core.bias_instructions import get_bias_instruction
from data.emgsd_loader import EMGSDEntry


class BiasGenerator:
    """
    Generates bias injection questions for cognitive bias types.
    
    This class encapsulates the logic for generating Turn 1 questions
    that prime the conversation with specific cognitive biases.
    """
    
    def __init__(
        self,
        llm_service: BedrockLLMService,
        bias_generator_model: str,
        use_persona_prompts: bool = True
    ):
        """
        Initialize bias generator.
        
        Args:
            llm_service: Bedrock LLM service instance
            bias_generator_model: Model ID for generating bias questions
            use_persona_prompts: Whether to use persona-based prompts
        """
        self.llm_service = llm_service
        self.bias_generator_model = bias_generator_model
        self.use_persona_prompts = use_persona_prompts
    
    def generate(
        self,
        target_question: str,
        bias_type: str,
        stereotype_entry: EMGSDEntry
    ) -> Dict[str, Any]:
        """
        Generate a Turn 1 bias injection question.
        
        Args:
            target_question: The target question to prime for
            bias_type: Cognitive bias type (e.g., 'availability_bias')
            stereotype_entry: EMGSD entry with target trait
            
        Returns:
            Dictionary with:
            - question: Generated bias injection question (or None)
            - model: Model used for generation
            - bias_type: Cognitive bias type
            - bias_name: Name of the bias
            - bias_description: Description of the bias
            - bias_source: Source of bias definition
            - refusal_detected: Whether model refused
            - refusal_reason: Reason for refusal (if any)
            - error: Error message (if any)
        """
        # Get bias instruction metadata
        instruction = get_bias_instruction(bias_type)
        if not instruction:
            return self._create_error_result(
                bias_type=bias_type,
                error=f'Bias instruction not found for {bias_type}'
            )
        
        bias_name = instruction.get('name', bias_type)
        bias_description = instruction.get('description', '')
        bias_source = instruction.get('framework', 'bias_instructions.py')
        
        try:
            # Generate bias injection question only (no conversation execution)
            bias_result = self.llm_service.inject_bias_llm(
                prompt=target_question,
                bias_type=bias_type,
                target_model_id=None,  # None = only generate question, don't execute conversation
                stereotype_entry=stereotype_entry,
                existing_conversation=None,
                use_persona_prompts=self.use_persona_prompts,
                bias_generator_model_id=self.bias_generator_model  # Use configured model
            )
            
            # Check for refusals
            if bias_result.get('refusal_detected', False):
                return {
                    'question': None,
                    'model': self.bias_generator_model,
                    'bias_type': bias_type,
                    'bias_name': bias_name,
                    'bias_description': bias_description,
                    'bias_source': bias_source,
                    'refusal_detected': True,
                    'refusal_reason': bias_result.get('refusal_reason', 'Unknown'),
                    'error': 'Model refused to generate question'
                }
            
            # Extract the bias injection question
            bias_question = bias_result['conversation']['turn1_question']
            
            return {
                'question': bias_question,
                'model': self.bias_generator_model,
                'bias_type': bias_type,
                'bias_name': bias_name,
                'bias_description': bias_description,
                'bias_source': bias_source,
                'refusal_detected': False,
                'refusal_reason': None,
                'error': None
            }
        
        except Exception as e:
            return self._create_error_result(
                bias_type=bias_type,
                bias_name=bias_name,
                bias_description=bias_description,
                bias_source=bias_source,
                error=str(e)
            )
    
    def _create_error_result(
        self,
        bias_type: str,
        bias_name: Optional[str] = None,
        bias_description: Optional[str] = None,
        bias_source: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a standardized error result dictionary."""
        instruction = get_bias_instruction(bias_type)
        if instruction:
            bias_name = instruction.get('name', bias_type)
            bias_description = instruction.get('description', '')
            bias_source = instruction.get('framework', 'bias_instructions.py')
        
        return {
            'question': None,
            'model': self.bias_generator_model,
            'bias_type': bias_type,
            'bias_name': bias_name or bias_type,
            'bias_description': bias_description or '',
            'bias_source': bias_source or 'unknown',
            'refusal_detected': False,
            'refusal_reason': None,
            'error': error or 'Unknown error'
        }

