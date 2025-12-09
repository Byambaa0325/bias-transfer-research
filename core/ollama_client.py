"""
Ollama API Client
A client for interacting with Ollama (self-hosted LLM inference server).
Matches BedrockClient interface for compatibility.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import requests


class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors"""
    def __init__(self, status_code: int, message: str, response: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"[{status_code}] {message}")


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Matches BedrockClient interface for compatibility with existing code.
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Ollama API client.
        
        Args:
            api_base: Ollama API base URL (defaults to http://localhost:11434)
            default_model: Default model to use
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_base = api_base or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        self.default_model = default_model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Test connection
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ValueError(f"Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Could not connect to Ollama server at {self.api_base}\n"
                "Make sure Ollama is running. On Windows, it should start automatically.\n"
                "On Linux/macOS, start with: ollama serve"
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Ollama client: {e}")
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the Ollama model with the given messages.
        
        Matches BedrockClient.invoke() signature for compatibility.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use (defaults to default_model)
            max_tokens: Maximum tokens in response (Ollama uses "num_predict")
            temperature: Sampling temperature
            stop_sequences: Stop sequences
            **kwargs: Additional parameters (e.g., repetition_penalty)
            
        Returns:
            Response dict with 'content' key containing list of dicts with 'text'
            Format matches BedrockClient response structure
        """
        model = model or self.default_model
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "stream": False,  # Explicitly disable streaming
            "options": {
                "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
            }
        }
        
        if temperature is not None:
            params["options"]["temperature"] = temperature
        
        if stop_sequences:
            params["options"]["stop"] = stop_sequences
        
        # Add any additional options from kwargs
        if kwargs:
            params["options"].update(kwargs)
        
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/api/chat",
                    json=params,
                    timeout=120
                )
                response.raise_for_status()
                
                # Handle JSON parsing
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    # Try to extract first valid JSON object
                    response_text = response.text.strip()
                    first_brace = response_text.find('{')
                    if first_brace != -1:
                        brace_count = 0
                        end_pos = first_brace
                        for i, char in enumerate(response_text[first_brace:], start=first_brace):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                        
                        if end_pos > first_brace:
                            json_str = response_text[first_brace:end_pos]
                            result = json.loads(json_str)
                        else:
                            raise json_err
                    else:
                        raise json_err
                
                # Extract text from response
                text = result.get("message", {}).get("content", "")
                
                # Format to match Bedrock response structure
                return {
                    "content": [{"text": text}],
                    "metadata": {
                        "model": model,
                        "usage": {
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                        }
                    }
                }
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise OllamaAPIError(
                        response.status_code if hasattr(e, 'response') and e.response else 500,
                        f"Ollama API call failed: {e}",
                        response=None
                    )
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise OllamaAPIError(500, f"Ollama API call failed: {e}", response=None)
        
        raise OllamaAPIError(500, f"Ollama API call failed after {self.max_retries} retries: {last_exception}")
    
    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        except Exception as e:
            raise OllamaAPIError(500, f"Could not list models: {e}", response=None)
    
    def pull_model(self, model_name: str):
        """
        Pull a model from Ollama library.
        
        Args:
            model_name: Model name (e.g., "llama3.1:8b")
        """
        print(f"Pulling model: {model_name}")
        print("This may take a while depending on model size...")
        
        try:
            response = requests.post(
                f"{self.api_base}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            # Stream the response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"  {data['status']}")
                    except:
                        pass
            
            print(f"✓ Model {model_name} pulled successfully")
        except Exception as e:
            raise OllamaAPIError(500, f"Failed to pull model: {e}", response=None)

