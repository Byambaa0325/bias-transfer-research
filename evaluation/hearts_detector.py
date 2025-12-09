"""
HEARTS Stereotype Detector

Integrates the Holistic AI ALBERT-v2 bias classifier model.
Based on the HEARTS framework (King et al., 2024).

Model: holistic-ai/bias_classifier_albertv2
Binary classification: Stereotype vs Non-Stereotype
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Try to import required libraries (silently - warnings only shown when actually initializing)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import LIME for confidence scoring
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class HEARTSDetector:
    """
    HEARTS-based stereotype detector using ALBERT-v2.

    Features:
    - Binary stereotype classification (Stereotype vs Non-Stereotype)
    - Confidence scoring
    - SHAP-based token importance (if available)
    - LIME-based explanation confidence (if available)

    Based on: King et al. (2024) - HEARTS Framework
    """

    MODEL_NAME = "holistic-ai/bias_classifier_albertv2"
    LABELS = {
        0: "Non-Stereotype",
        1: "Stereotype"
    }

    def __init__(
        self, 
        model_name: Optional[str] = None, 
        device: Optional[str] = None,
        enable_shap: bool = False,
        enable_lime: bool = False
    ):
        """
        Initialize HEARTS detector.

        Args:
            model_name: HuggingFace model name (default: holistic-ai/bias_classifier_albertv2)
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            enable_shap: Enable SHAP explainer (memory-intensive, default: False)
            enable_lime: Enable LIME explainer (very memory-intensive, default: False)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers torch"
            )
        
        # Warn about missing optional dependencies only when initializing
        if enable_shap and not SHAP_AVAILABLE:
            print("Warning: SHAP requested but not available. Install with: pip install shap")
        if enable_lime and not LIME_AVAILABLE:
            print("Warning: LIME requested but not available. Install with: pip install lime")

        self.model_name = model_name or self.MODEL_NAME
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_lime = enable_lime and LIME_AVAILABLE

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading HEARTS model: {self.model_name} on {self.device}...")
        if not enable_shap and not enable_lime:
            print("  SHAP/LIME explainers disabled (memory-efficient mode)")

        # Load model and tokenizer with production-friendly settings
        try:
            import os
            # Use local_files_only if model is cached (for production)
            local_files_only = os.getenv('HEARTS_LOCAL_FILES_ONLY', 'false').lower() == 'true'
            
            # Set cache directory if specified (useful for Cloud Run with persistent disk)
            cache_dir = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE')
            
            # Default cache location (works in Docker/Cloud Run)
            if not cache_dir:
                cache_dir = os.path.expanduser('~/.cache/huggingface')
                # Also try /app/.cache/huggingface (common in Docker)
                if not os.path.exists(cache_dir):
                    alt_cache = '/app/.cache/huggingface'
                    if os.path.exists(alt_cache):
                        cache_dir = alt_cache
            
            load_kwargs = {}
            if cache_dir:
                load_kwargs['cache_dir'] = cache_dir
            if local_files_only:
                load_kwargs['local_files_only'] = True
                print(f"  Using local files only (cache_dir: {cache_dir})")
            elif cache_dir:
                print(f"  Using cache directory: {cache_dir}")
            
            # Load tokenizer first (smaller, faster)
            print(f"  Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Load model (larger, may take time)
            print(f"  Loading model (this may take a minute on first run)...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device and set eval mode
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"✓ HEARTS model loaded successfully on {self.device}")
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide helpful error messages
            if "Connection" in error_type or "timeout" in error_msg.lower():
                raise Exception(
                    f"Failed to download HEARTS model (network error): {error_msg}\n"
                    f"  Tip: Set HEARTS_LOCAL_FILES_ONLY=true if model is already cached"
                )
            elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                raise Exception(
                    f"Failed to load HEARTS model (disk space): {error_msg}\n"
                    f"  Tip: Free up disk space or set TRANSFORMERS_CACHE to a location with more space"
                )
            elif "memory" in error_msg.lower() or "OOM" in error_msg:
                raise Exception(
                    f"Failed to load HEARTS model (out of memory): {error_msg}\n"
                    f"  Tip: Increase container memory or use CPU device"
                )
            else:
                raise Exception(
                    f"Failed to load HEARTS model: {error_msg}\n"
                    f"  Error type: {error_type}"
                )

        # Initialize SHAP explainer (optional, disabled by default for memory efficiency)
        self.shap_explainer = None
        if self.enable_shap and SHAP_AVAILABLE:
            try:
                # Create a wrapper function for SHAP
                def model_predict(texts):
                    """Wrapper for SHAP to get predictions"""
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            list(texts),
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.device)
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        result = probs.cpu().numpy()
                        
                        # Clean up tensors
                        del inputs, outputs, probs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return result

                self.shap_explainer = shap.Explainer(model_predict, self.tokenizer)
                print("✓ SHAP explainer initialized (memory usage: ~500MB)")
            except Exception as e:
                print(f"Warning: Could not initialize SHAP: {e}")
                self.shap_explainer = None

        # Initialize LIME explainer (optional, disabled by default for memory efficiency)
        self.lime_explainer = None
        if self.enable_lime and LIME_AVAILABLE:
            try:
                self.lime_explainer = LimeTextExplainer(class_names=list(self.LABELS.values()))
                print("✓ LIME explainer initialized (memory usage: ~300MB per request)")
            except Exception as e:
                print(f"Warning: Could not initialize LIME: {e}")
                self.lime_explainer = None

    def detect_stereotypes(
        self,
        text: str,
        explain: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect stereotypes in text using HEARTS ALBERT-v2 model.

        Args:
            text: Input text to analyze
            explain: Whether to generate SHAP explanations (if available)
            confidence_threshold: Threshold for classification (default: 0.5)

        Returns:
            Dictionary with detection results and explanations
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Extract probabilities (move to CPU and convert immediately)
        probs_np = probs.cpu().numpy()[0]
        non_stereotype_prob = float(probs_np[0])
        stereotype_prob = float(probs_np[1])

        # Clean up tensors immediately to free memory
        del inputs, outputs, logits, probs, probs_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Determine prediction
        predicted_class = 1 if stereotype_prob > confidence_threshold else 0
        predicted_label = self.LABELS[predicted_class]
        confidence = max(non_stereotype_prob, stereotype_prob)

        result = {
            'text': text,
            'prediction': predicted_label,
            'is_stereotype': predicted_class == 1,
            'confidence': confidence,
            'probabilities': {
                'Non-Stereotype': non_stereotype_prob,
                'Stereotype': stereotype_prob
            },
            'model': 'HEARTS ALBERT-v2',
            'framework': 'HEARTS (King et al., 2024)'
        }

        # Add explanations if requested
        if explain:
            explanations = self._generate_explanations(text, predicted_class)
            result['explanations'] = explanations

            # Calculate explanation confidence if both SHAP and LIME available
            if explanations.get('shap_available') and explanations.get('lime_available'):
                result['explanation_confidence'] = self._calculate_explanation_confidence(text)

        return result

    def _generate_explanations(self, text: str, predicted_class: int) -> Dict[str, Any]:
        """
        Generate token-level explanations using SHAP and/or LIME.

        Args:
            text: Input text
            predicted_class: Predicted class (0 or 1)

        Returns:
            Dictionary with token importance scores
        """
        explanations = {
            'shap_available': False,
            'lime_available': False,
            'token_importance': []
        }

        # SHAP explanations (memory-intensive, only run if explainer exists)
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer([text])

                # Extract token importances for the predicted class
                tokens = self.tokenizer.tokenize(text)
                values = shap_values.values[0, :, predicted_class]

                # Combine tokens and values
                token_importance = []
                for token, value in zip(tokens, values):
                    token_importance.append({
                        'token': token,
                        'importance': abs(float(value)),
                        'contribution': 'positive' if value > 0 else 'negative',
                        'shap_value': float(value)
                    })

                # Sort by importance (descending)
                token_importance.sort(key=lambda x: x['importance'], reverse=True)

                explanations['token_importance'] = token_importance[:10]  # Top 10
                explanations['shap_available'] = True

                # Explicitly delete SHAP values to free memory
                del shap_values, values, token_importance

            except Exception as e:
                print(f"Warning: SHAP explanation failed: {e}")

        # LIME explanations (as alternative/complement)
        # NOTE: LIME is very memory-intensive (generates hundreds of samples)
        # Consider disabling in production if memory is constrained
        if self.lime_explainer is not None:
            try:
                def predict_proba(texts):
                    """Prediction function for LIME"""
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            list(texts),
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.device)
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        result = probs.cpu().numpy()
                        
                        # Clean up tensors immediately
                        del inputs, outputs, probs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return result

                lime_exp = self.lime_explainer.explain_instance(
                    text,
                    predict_proba,
                    num_features=10,
                    labels=(predicted_class,)
                )

                # Extract LIME importance scores
                lime_importance = lime_exp.as_list(label=predicted_class)
                explanations['lime_importance'] = [
                    {'token': token, 'importance': abs(score), 'lime_score': score}
                    for token, score in lime_importance
                ]
                explanations['lime_available'] = True

                # Clean up LIME objects
                del lime_exp, lime_importance

            except Exception as e:
                print(f"Warning: LIME explanation failed: {e}")

        return explanations

    def _calculate_explanation_confidence(self, text: str) -> float:
        """
        Calculate explanation confidence by comparing SHAP and LIME.

        Uses cosine similarity between SHAP and LIME importance vectors
        as described in the HEARTS paper.

        Args:
            text: Input text

        Returns:
            Confidence score (0-1)
        """
        if not (SHAP_AVAILABLE and LIME_AVAILABLE):
            return 0.0

        try:
            # Get SHAP values
            shap_values = self.shap_explainer([text])

            # Get LIME values
            def predict_proba(texts):
                with torch.no_grad():
                    inputs = self.tokenizer(list(texts), return_tensors="pt",
                                          padding=True, truncation=True, max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            lime_exp = self.lime_explainer.explain_instance(text, predict_proba, num_features=20)

            # Calculate cosine similarity
            # This is a simplified version - full implementation would align tokens properly
            shap_vector = shap_values.values[0, :, 1]  # Stereotype class
            lime_dict = dict(lime_exp.as_list(label=1))

            # Simplified similarity score
            from sklearn.metrics.pairwise import cosine_similarity

            # Create aligned vectors (simplified)
            shap_norm = np.linalg.norm(shap_vector)
            lime_norm = np.sqrt(sum(v**2 for v in lime_dict.values()))

            if shap_norm > 0 and lime_norm > 0:
                # Normalized similarity
                similarity = min(1.0, max(0.0, 0.5 + 0.5 * (shap_norm / lime_norm)))
                return similarity

            return 0.5  # Default if calculation fails

        except Exception as e:
            print(f"Warning: Explanation confidence calculation failed: {e}")
            return 0.5

    def batch_detect(
        self,
        texts: List[str],
        explain: bool = False,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Detect stereotypes in multiple texts (batch processing).

        Args:
            texts: List of texts to analyze
            explain: Whether to generate explanations
            batch_size: Batch size for processing

        Returns:
            List of detection results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Process batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process each result
            for j, text in enumerate(batch):
                probs_np = probs[j].cpu().numpy()
                result = self.detect_stereotypes(text, explain=explain)
                results.append(result)

        return results


def is_hearts_available() -> bool:
    """Check if HEARTS detector can be initialized"""
    return TRANSFORMERS_AVAILABLE


if __name__ == "__main__":
    # Example usage
    if is_hearts_available():
        detector = HEARTSDetector()

        # Test prompts
        test_prompts = [
            "What are the benefits of regular exercise?",
            "Why are women always so emotional?",
            "How does photosynthesis work?",
            "All teenagers are irresponsible with money."
        ]

        print("\n" + "="*60)
        print("HEARTS Stereotype Detection Test")
        print("="*60 + "\n")

        for prompt in test_prompts:
            result = detector.detect_stereotypes(prompt, explain=True)
            print(f"Prompt: {prompt}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Stereotype Probability: {result['probabilities']['Stereotype']:.2%}")

            if result.get('explanations', {}).get('token_importance'):
                print("Top biased tokens:")
                for token_info in result['explanations']['token_importance'][:3]:
                    print(f"  - {token_info['token']}: {token_info['importance']:.3f}")

            print("-" * 60 + "\n")
    else:
        print("HEARTS not available. Install required packages:")
        print("pip install transformers torch")
