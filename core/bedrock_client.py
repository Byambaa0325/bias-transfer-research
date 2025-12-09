"""
Bedrock Proxy API Client
A comprehensive client for interacting with the AWS Bedrock Proxy API.
Supports multiple model families: Claude, Llama, Nova, Mistral, DeepSeek
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    import requests
except ImportError:
    raise ImportError("requests library is required. Install with: pip install requests")


class ModelFamily(Enum):
    """Available model families"""
    CLAUDE = "claude"
    LLAMA = "llama"
    NOVA = "nova"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"


class BedrockModels:
    """Available Bedrock models organized by family"""

    # Anthropic Claude Series
    CLAUDE_3_5_SONNET_V2 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_3_OPUS = "us.anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_3_SONNET = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU = "us.anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_OPUS_4 = "us.anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_SONNET_4 = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Meta Llama Series
    LLAMA_3_2_90B = "us.meta.llama3-2-90b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"
    LLAMA_3_2_3B = "us.meta.llama3-2-3b-instruct-v1:0"
    LLAMA_3_2_1B = "us.meta.llama3-2-1b-instruct-v1:0"
    LLAMA_3_1_70B = "us.meta.llama3-1-70b-instruct-v1:0"
    LLAMA_3_1_8B = "us.meta.llama3-1-8b-instruct-v1:0"
    LLAMA_3_3_70B = "us.meta.llama3-3-70b-instruct-v1:0"
    LLAMA_4_SCOUT = "us.meta.llama4-scout-17b-instruct-v1:0"
    LLAMA_4_MAVERICK = "us.meta.llama4-maverick-17b-instruct-v1:0"

    # Amazon Nova Series
    NOVA_PREMIER = "us.amazon.nova-premier-v1:0"
    NOVA_PRO = "us.amazon.nova-pro-v1:0"
    NOVA_LITE = "us.amazon.nova-lite-v1:0"
    NOVA_MICRO = "us.amazon.nova-micro-v1:0"

    # Mistral Series
    PIXTRAL_LARGE = "us.mistral.pixtral-large-2502-v1:0"
    MISTRAL_LARGE = "mistral.mistral-large-2402-v1:0"
    MISTRAL_SMALL = "mistral.mistral-small-2402-v1:0"
    MISTRAL_7B = "mistral.mistral-7b-instruct-v0:2"
    MIXTRAL_8X7B = "mistral.mixtral-8x7b-instruct-v0:1"

    # DeepSeek Series
    DEEPSEEK_R1 = "us.deepseek.r1-v1:0"


@dataclass
class QuotaInfo:
    """Budget and quota information"""
    llm_cost: float
    gpu_cost: float
    total_cost: float
    budget_limit: float
    remaining_budget: float
    budget_usage_percent: float

    def __str__(self):
        return (
            f"Budget: ${self.remaining_budget:.2f} / ${self.budget_limit:.2f} "
            f"({self.budget_usage_percent:.1f}% used)"
        )


class BedrockAPIError(Exception):
    """Custom exception for Bedrock API errors"""
    def __init__(self, status_code: int, message: str, response: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"[{status_code}] {message}")


class BedrockClient:
    """Client for interacting with the Bedrock Proxy API"""

    def __init__(
        self,
        team_id: Optional[str] = None,
        api_token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Bedrock API client.

        Args:
            team_id: Your team ID (defaults to BEDROCK_TEAM_ID env var)
            api_token: Your API token (defaults to BEDROCK_API_TOKEN env var)
            api_endpoint: API endpoint URL (defaults to BEDROCK_API_ENDPOINT env var)
            default_model: Default model to use (defaults to BEDROCK_DEFAULT_MODEL env var)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.team_id = team_id or os.getenv("BEDROCK_TEAM_ID")
        self.api_token = api_token or os.getenv("BEDROCK_API_TOKEN")
        self.api_endpoint = api_endpoint or os.getenv(
            "BEDROCK_API_ENDPOINT",
            "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
        )
        self.default_model = default_model or os.getenv(
            "BEDROCK_DEFAULT_MODEL",
            BedrockModels.CLAUDE_3_5_SONNET_V2
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.team_id or not self.api_token:
            raise ValueError(
                "team_id and api_token are required. "
                "Set them via constructor or BEDROCK_TEAM_ID/BEDROCK_API_TOKEN env vars."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "X-Team-ID": self.team_id,
            "X-API-Token": self.api_token
        }

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors"""
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise BedrockAPIError(
                response.status_code,
                f"Invalid JSON response: {response.text}",
                None
            )

        if response.status_code == 401:
            raise BedrockAPIError(
                401,
                "Unauthorized - Check your team_id and api_token",
                data
            )
        elif response.status_code == 403:
            raise BedrockAPIError(
                403,
                "Forbidden - Model not allowed or access denied",
                data
            )
        elif response.status_code == 429:
            raise BedrockAPIError(
                429,
                "Budget exhausted - Check your remaining quota",
                data
            )
        elif response.status_code == 400:
            raise BedrockAPIError(
                400,
                f"Bad Request - {data.get('message', 'Invalid request format')}",
                data
            )
        elif not response.ok:
            raise BedrockAPIError(
                response.status_code,
                f"API Error: {data.get('message', response.text)}",
                data
            )

        return data

    def invoke(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the LLM with the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID to use (defaults to default_model)
            max_tokens: Maximum tokens in response
            tools: Optional list of tool definitions for function calling
            tool_choice: Tool choice strategy ("auto", "required", or specific tool)
            response_format: Optional structured output format (JSON schema)
            **kwargs: Additional parameters to pass to the API

        Returns:
            API response dict containing 'content' and 'metadata'

        Raises:
            BedrockAPIError: If the API request fails
        """
        model = model or self.default_model

        # Build base payload
        payload = {
            "team_id": self.team_id,
            "api_token": self.api_token,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        # Filter kwargs to only include supported parameters
        # Some models may not support all parameters (e.g., temperature)
        # We'll try to include them but the API will reject if not supported
        # For now, we'll include common parameters and let the API validate
        supported_params = ['temperature', 'top_p', 'top_k', 'stop_sequences', 'stream', 'repetition_penalty']
        for key, value in kwargs.items():
            if key in supported_params or key.startswith('_'):
                # Include known parameters or internal parameters (prefixed with _)
                payload[key] = value

        # Add optional parameters
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        if response_format:
            payload["response_format"] = response_format

        # Retry logic with parameter validation handling
        last_error = None
        original_payload = payload.copy()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120
                )
                return self._handle_response(response)

            except BedrockAPIError as e:
                # Handle parameter validation errors - retry without unsupported parameters
                if e.status_code == 500 and 'extraneous key' in str(e.message).lower():
                    # Extract the unsupported parameter from error message
                    error_msg = str(e.message).lower()
                    if 'temperature' in error_msg and 'temperature' in payload:
                        # Remove temperature and retry
                        payload.pop('temperature', None)
                        print(f"Warning: Model {model} doesn't support temperature parameter, retrying without it")
                        continue
                    elif 'extraneous key' in error_msg:
                        # Try to identify and remove the problematic parameter
                        # This is a fallback for other unsupported parameters
                        for param in ['temperature', 'top_p', 'top_k']:
                            if param in payload and param in error_msg:
                                payload.pop(param, None)
                                print(f"Warning: Model {model} doesn't support {param} parameter, retrying without it")
                                break
                        continue
                
                # Don't retry on client errors (4xx except 429)
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    raise
                last_error = e

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Reset payload for final attempt
                    payload = original_payload.copy()

            except requests.exceptions.RequestException as e:
                last_error = BedrockAPIError(0, f"Request failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise last_error or BedrockAPIError(0, "Unknown error occurred")

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Simple chat interface - send a message and get a text response.

        Args:
            message: User message
            model: Model to use
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Text response from the model
        """
        messages = [{"role": "user", "content": message}]
        response = self.invoke(messages, model=model, max_tokens=max_tokens, **kwargs)
        return response["content"][0]["text"]

    def multi_turn_chat(
        self,
        conversation: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Multi-turn conversation interface.

        Args:
            conversation: List of message dicts with 'role' and 'content'
            model: Model to use
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Text response from the model
        """
        response = self.invoke(conversation, model=model, max_tokens=max_tokens, **kwargs)
        return response["content"][0]["text"]

    def get_quota_info(self, response: Dict[str, Any]) -> Optional[QuotaInfo]:
        """
        Extract quota information from API response.

        Args:
            response: API response dict

        Returns:
            QuotaInfo object or None if not available
        """
        metadata = response.get("metadata", {})
        quota_data = metadata.get("remaining_quota")

        if not quota_data:
            return None

        return QuotaInfo(
            llm_cost=quota_data.get("llm_cost", 0.0),
            gpu_cost=quota_data.get("gpu_cost", 0.0),
            total_cost=quota_data.get("total_cost", 0.0),
            budget_limit=quota_data.get("budget_limit", 0.0),
            remaining_budget=quota_data.get("remaining_budget", 0.0),
            budget_usage_percent=quota_data.get("budget_usage_percent", 0.0)
        )

    def structured_output(
        self,
        message: str,
        schema: Dict[str, Any],
        schema_name: str = "structured_output",
        model: Optional[str] = None,
        strict: bool = True,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get structured output matching a JSON schema.

        Args:
            message: User message
            schema: JSON schema for the output
            schema_name: Name for the schema
            model: Model to use (must support tool calling)
            strict: Whether to enforce strict schema compliance
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Parsed JSON object matching the schema
        """
        messages = [{"role": "user", "content": message}]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": strict,
                "schema": schema
            }
        }

        response = self.invoke(
            messages,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs
        )

        return json.loads(response["content"][0]["text"])

    def function_calling(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        tool_choice: str = "auto",
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM with function/tool calling capabilities.

        Args:
            message: User message
            tools: List of tool definitions
            model: Model to use (must support tool calling)
            tool_choice: Tool choice strategy
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Full API response including tool calls
        """
        messages = [{"role": "user", "content": message}]

        return self.invoke(
            messages,
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )


def load_env_file(filepath: str = ".env.bedrock") -> None:
    """
    Load environment variables from a file.

    Args:
        filepath: Path to .env file
    """
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
