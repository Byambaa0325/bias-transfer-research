"""
Model Selector

Handles model selection logic for different tasks in bias injection experiments.
Ensures consistent model usage for bias generation vs. evaluation.
"""

from typing import Optional, List

try:
    from bedrock_client import BedrockModels
    BEDROCK_AVAILABLE = True
except ImportError:
    try:
        from .bedrock_client import BedrockModels
        BEDROCK_AVAILABLE = True
    except ImportError:
        BEDROCK_AVAILABLE = False


class ModelSelector:
    """
    Handles model selection logic for different tasks.

    Responsibilities:
    - Select bias generator model (always Claude 3.5 Sonnet V2 for consistency)
    - Select target model for evaluation (the model being tested)
    - Determine model capabilities (temperature support, etc.)

    Key Design Decision:
    - **Bias Generator (Turn 1):** Always Claude 3.5 Sonnet V2
      → Ensures consistent, high-quality priming across all experiments
    - **Target Model (Evaluation):** User-specified or default
      → The model being tested for bias susceptibility

    Example:
        >>> selector = ModelSelector()
        >>> bias_gen = selector.get_bias_generator_model()
        >>> target = selector.get_target_model(
        ...     target_model_id="llama-4-scout",
        ...     legacy_model_id=None,
        ...     default_model="claude-3-5-sonnet"
        ... )
        >>> print(f"Bias gen: {bias_gen}, Target: {target}")
    """

    # Model capabilities cache (avoids repeated string matching)
    _temperature_support_cache = {}

    @staticmethod
    def get_bias_generator_model(override_model_id: Optional[str] = None) -> str:
        """
        Get the model to use for generating bias prompts (Turn 1).

        By default returns Claude 3.5 Sonnet V2 for consistency and quality,
        but can be overridden for experimental purposes.

        Why Claude 3.5 Sonnet V2 (default)?
        1. **Consistency:** Same model generates all priming questions
        2. **Quality:** Strong instruction following for subtle bias generation
        3. **Reproducibility:** Reduces variability in research experiments

        Args:
            override_model_id: Optional model ID to use instead of default.
                             Useful for testing different bias generators.

        Returns:
            Model ID for bias generation

        Example:
            >>> selector = ModelSelector()
            >>> # Use default (Claude 3.5 Sonnet V2)
            >>> model = selector.get_bias_generator_model()
            >>> "claude-3-5-sonnet" in model
            True
            >>> # Override with custom model
            >>> model = selector.get_bias_generator_model("llama-4-maverick")
            >>> model
            'llama-4-maverick'
        """
        # If override provided, use it
        if override_model_id:
            return override_model_id

        # Otherwise use default (Claude 3.5 Sonnet V2)
        if not BEDROCK_AVAILABLE:
            return "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

        return BedrockModels.CLAUDE_3_5_SONNET_V2

    @staticmethod
    def get_target_model(
        target_model_id: Optional[str],
        legacy_model_id: Optional[str],
        default_model: str
    ) -> str:
        """
        Get the target model to evaluate for bias susceptibility.

        This is the model being TESTED in the experiment (not the one generating bias).

        Priority order:
        1. target_model_id (preferred, explicit parameter)
        2. legacy_model_id (deprecated, for backward compatibility)
        3. default_model (fallback)

        Args:
            target_model_id: Explicitly specified target model
            legacy_model_id: Legacy parameter (deprecated, use target_model_id)
            default_model: Default model from service

        Returns:
            Model ID to use for evaluation

        Example:
            >>> selector = ModelSelector()
            >>> # Explicit target
            >>> model = selector.get_target_model("llama-4", None, "claude")
            >>> model
            'llama-4'
            >>> # Fallback to default
            >>> model = selector.get_target_model(None, None, "claude")
            >>> model
            'claude'
        """
        return target_model_id or legacy_model_id or default_model

    @staticmethod
    def supports_temperature(model_id: str) -> bool:
        """
        Check if a model supports the temperature parameter.

        Different model families have different parameter support:
        - Claude: Supports temperature ✓
        - Mistral: Supports temperature ✓
        - Llama: Conservative (may or may not) ✗
        - Nova: Conservative ✗
        - DeepSeek: Conservative ✗

        Uses caching to avoid repeated string matching.

        Args:
            model_id: Model identifier

        Returns:
            True if model likely supports temperature, False otherwise

        Example:
            >>> selector = ModelSelector()
            >>> selector.supports_temperature("claude-3-5-sonnet")
            True
            >>> selector.supports_temperature("llama-3-70b")
            False
        """
        if not model_id:
            return False

        # Check cache first
        if model_id in ModelSelector._temperature_support_cache:
            return ModelSelector._temperature_support_cache[model_id]

        model_lower = model_id.lower()

        # Known to support temperature
        if 'claude' in model_lower or 'anthropic' in model_lower:
            result = True
        elif 'mistral' in model_lower:
            result = True
        # Conservative for others (may or may not support)
        elif 'llama' in model_lower or 'meta' in model_lower:
            result = False
        elif 'nova' in model_lower or 'amazon' in model_lower:
            result = False
        elif 'deepseek' in model_lower:
            result = False
        else:
            # Default: be conservative, don't include temperature
            result = False

        # Cache result
        ModelSelector._temperature_support_cache[model_id] = result
        return result

    @staticmethod
    def get_model_family(model_id: str) -> str:
        """
        Get the model family from model ID.

        Args:
            model_id: Model identifier

        Returns:
            Model family name (e.g., "claude", "llama", "nova")

        Example:
            >>> selector = ModelSelector()
            >>> selector.get_model_family("claude-3-5-sonnet")
            'claude'
            >>> selector.get_model_family("llama-4-scout")
            'llama'
        """
        if not model_id:
            return "unknown"

        model_lower = model_id.lower()

        if 'claude' in model_lower or 'anthropic' in model_lower:
            return 'claude'
        elif 'llama' in model_lower or 'meta' in model_lower:
            return 'llama'
        elif 'nova' in model_lower or 'amazon' in model_lower:
            return 'nova'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'deepseek' in model_lower:
            return 'deepseek'
        else:
            return 'unknown'

    @staticmethod
    def get_stop_sequences(model_id: str) -> Optional[List[str]]:
        """
        Get recommended stop sequences for a model to prevent runaway generation.

        Llama models in particular tend to continue conversations with themselves,
        so we add stop sequences to prevent this behavior.

        Args:
            model_id: Model identifier

        Returns:
            List of stop sequences or None if not needed

        Example:
            >>> selector = ModelSelector()
            >>> stops = selector.get_stop_sequences("llama-3-70b")
            >>> len(stops) > 0
            True
            >>> stops = selector.get_stop_sequences("claude-3-5-sonnet")
            >>> stops is None
            True
        """
        if not model_id:
            return None

        model_family = ModelSelector.get_model_family(model_id)

        # Llama models need stop sequences to prevent conversation loops
        if model_family == 'llama':
            return [
                "\n\nUser:",
                "\n\nAssistant:",
                "\nUser:",
                "\nAssistant:",
                "\n\nHuman:",
                "\n\nAI:",
                "\nHuman:",
                "\nAI:",
            ]
        
        # Other models generally handle this better
        return None

    @staticmethod
    def get_repetition_penalty(model_id: str) -> Optional[float]:
        """
        Get recommended repetition penalty for a model to reduce repetitive outputs.

        Llama models benefit from a repetition penalty > 1.0 to prevent repetitive
        conversation loops and text repetition.

        Args:
            model_id: Model identifier

        Returns:
            Repetition penalty value (typically 1.1-1.3) or None if not needed

        Example:
            >>> selector = ModelSelector()
            >>> penalty = selector.get_repetition_penalty("llama-3-70b")
            >>> penalty > 1.0
            True
            >>> penalty = selector.get_repetition_penalty("claude-3-5-sonnet")
            >>> penalty is None
            True
        """
        if not model_id:
            return None

        model_family = ModelSelector.get_model_family(model_id)

        # Llama models benefit from repetition penalty to reduce loops
        # Recommended range: 1.1-1.3 (too high can cause incoherence)
        if model_family == 'llama':
            return 1.15  # Balanced value to reduce repetition without harming quality
        
        # Other models generally don't need this
        return None

    @staticmethod
    def get_recommended_temperature(model_id: str) -> Optional[float]:
        """
        Get recommended temperature for a model.

        Different models have different optimal temperature ranges:
        - Claude: 0.7 (balanced creativity and consistency)
        - Mistral: 0.7
        - Others: None (don't use temperature)

        Args:
            model_id: Model identifier

        Returns:
            Recommended temperature (0.0-1.0) or None if not supported

        Example:
            >>> selector = ModelSelector()
            >>> selector.get_recommended_temperature("claude-3-5")
            0.7
            >>> selector.get_recommended_temperature("llama-4")
            None
        """
        if not ModelSelector.supports_temperature(model_id):
            return None

        # Conservative default for models that support it
        return 0.7

    @staticmethod
    def build_invoke_params(
        messages: list,
        model_id: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        **kwargs
    ) -> dict:
        """
        Build parameters for model invocation with proper temperature handling.

        Automatically handles temperature parameter based on model support.
        Also adds stop sequences for models that need them (e.g., Llama).

        Args:
            messages: Conversation messages
            model_id: Model identifier
            max_tokens: Maximum tokens in response
            temperature: Desired temperature (if supported)
            **kwargs: Additional parameters (will override defaults)

        Returns:
            Dictionary of invoke parameters

        Example:
            >>> selector = ModelSelector()
            >>> params = selector.build_invoke_params(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     model_id="claude-3-5-sonnet",
            ...     temperature=0.7
            ... )
            >>> 'temperature' in params
            True
            >>> params = selector.build_invoke_params(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     model_id="llama-4-scout",
            ...     temperature=0.7
            ... )
            >>> 'temperature' in params
            False
            >>> 'stop_sequences' in params
            True
        """
        params = {
            "messages": messages,
            "model": model_id,
            "max_tokens": max_tokens
        }

        # Only add temperature if model supports it and value provided
        if temperature is not None and ModelSelector.supports_temperature(model_id):
            params["temperature"] = temperature

        # Add stop sequences for models that need them (unless overridden in kwargs)
        if "stop_sequences" not in kwargs:
            stop_sequences = ModelSelector.get_stop_sequences(model_id)
            if stop_sequences:
                params["stop_sequences"] = stop_sequences

        # Add repetition penalty for models that need it (unless overridden in kwargs)
        # This helps prevent repetitive conversation loops in Llama models
        if "repetition_penalty" not in kwargs:
            repetition_penalty = ModelSelector.get_repetition_penalty(model_id)
            if repetition_penalty is not None:
                params["repetition_penalty"] = repetition_penalty

        # Add any additional parameters (kwargs override defaults)
        params.update(kwargs)

        return params

    @staticmethod
    def clear_cache():
        """
        Clear the temperature support cache.

        Useful for testing or if model capabilities change.

        Example:
            >>> selector = ModelSelector()
            >>> selector.clear_cache()
        """
        ModelSelector._temperature_support_cache.clear()
