"""Qwen Chat Model Provider.

This module provides a QwenChatModel class that implements the chat model interface
for GraphRAG, using HTTP requests to communicate with a local Qwen model server.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

import requests
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ValidationError

from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class QwenChatModel:
    """Chat model provider for Qwen3-30b using URL-based HTTP requests.

    This provider implements the chat model interface expected by GraphRAG,
    making HTTP POST requests to a local Qwen server endpoint.
    """

    def __init__(
        self,
        name: str,
        config: Any,  # LanguageModelConfig
        **kwargs: Any,
    ):
        """Initialize QwenChatModel.

        Args:
            name: Model name identifier
            config: LanguageModelConfig with api_base, api_key, and model settings
            **kwargs: Additional configuration options
        """
        self.name = name
        self.config = config

        # Extract configuration values
        self.api_base = getattr(config, "api_base", "http://192.168.60.202:8333")
        self.api_key = getattr(config, "api_key", "sk-7966098172664c7f832496c33cfb86b8")
        self.model_name = getattr(config, "model", "qwen3-30b")

        # Ensure api_base doesn't end with /
        if self.api_base.endswith("/"):
            self.api_base = self.api_base.rstrip("/")

        # Construct endpoint URL
        self.endpoint = f"{self.api_base}/v1/chat/completions"

        # Thread pool executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(
            f"Initialized QwenChatModel '{name}' with endpoint: {self.endpoint}"
        )

    def _build_messages(
        self,
        prompt: str,
        history: list | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build message list in OpenAI-compatible format.

        Args:
            prompt: Current user prompt
            history: Conversation history (list of strings or dicts)
            system_prompt: Optional system prompt

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history if provided
        if history:
            for item in history:
                if isinstance(item, dict):
                    # Already in message format
                    messages.append(item)
                elif isinstance(item, str):
                    # Assume alternating user/assistant pattern
                    # This is a simplified assumption - in practice, you might need
                    # more sophisticated history handling
                    role = "user" if len(messages) % 2 == (1 if system_prompt else 0) else "assistant"
                    messages.append({"role": role, "content": item})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _clean_content(self, content: str) -> str:
        """Remove Qwen-specific reasoning tags from response content.

        Args:
            content: Raw response content that may contain reasoning tags

        Returns:
            Cleaned content with reasoning tags removed
        """
        # Remove </think> tags by splitting and taking last part
        # This matches the pattern from model_url_import_example.py
        if "</think>" in content:
            content = content.split("</think>")[-1]

        # Remove any remaining reasoning tags (cleanup)
        content = content.replace("</think>", "")
        content = content.replace("<think>", "")

        return content.strip()

    def _validate_extraction_response(self, content: str) -> str:
        """Validate and clean extraction response format for GraphRAG.

        Args:
            content: The response content to validate

        Returns:
            Cleaned and validated content
        """
        # Check if response contains expected format
        if not (content.startswith('("entity"') or content.startswith('("relationship"')):
            logger.warning(f"Response doesn't start with expected format. Content: {content[:200]}...")

        # Ensure completion delimiter is present
        if not content.endswith('<|><|>'):
            logger.warning("Response missing completion delimiter, adding it")
            content = content.rstrip() + '\n<|><|>'

        return content

    def _make_request(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP POST request to Qwen API endpoint.

        Args:
            messages: List of message dictionaries
            json_mode: Whether to request JSON format response
            json_model: Optional Pydantic model for structured output

        Returns:
            JSON response from API

        Raises:
            requests.RequestException: If request fails
            ValueError: If response format is invalid
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "enable_thinking": False,
        }

        # Add JSON mode if requested
        if json_mode:
            # Qwen API uses response_format parameter for JSON mode
            # Check if json_model is a Pydantic BaseModel
            if json_model and inspect.isclass(json_model) and issubclass(json_model, BaseModel):
                # For Pydantic models, we request JSON object format
                # The model validation will happen during parsing
                data["response_format"] = {"type": "json_object"}
            else:
                # General JSON object format
                data["response_format"] = {"type": "json_object"}

        try:
            logger.debug(f"Making request to {self.endpoint} with {len(messages)} messages")
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=data,
                timeout=30,  # 30 second timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request to {self.endpoint} timed out")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to {self.endpoint}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {self.endpoint}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error to {self.endpoint}: {e}")
            raise

    def _parse_response(
        self,
        response: dict[str, Any],
        json_model: type[BaseModel] | None = None,
    ) -> tuple[str, BaseModel | None]:
        """Parse API response and extract content, optionally parsing JSON.

        Args:
            response: JSON response from API
            json_model: Optional Pydantic model to validate parsed JSON against

        Returns:
            Tuple of (content_string, parsed_model_or_none)

        Raises:
            ValueError: If response format is invalid
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("Response contains no choices")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise ValueError("Response message contains no content")

            # Clean reasoning tags
            content = self._clean_content(content)

            # Validate response format for GraphRAG extraction responses only
            # Only apply to responses that look like tuple format, not JSON responses
            if (content.startswith('("entity"') or content.startswith('("relationship"')) and not content.strip().startswith('{'):
                content = self._validate_extraction_response(content)

            # Parse JSON if json_model is provided
            parsed_response: BaseModel | None = None
            if json_model:
                # Don't try to parse JSON if content is an error message
                if content.startswith("Error:"):
                    logger.warning(f"Skipping JSON parsing for error response: {content}")
                    parsed_response = None
                else:
                    try:
                        # Try to parse JSON from content
                        parsed_dict = json.loads(content)

                        # Validate against Pydantic model if provided
                        if inspect.isclass(json_model) and issubclass(json_model, BaseModel):
                            try:
                                parsed_response = json_model(**parsed_dict)
                                logger.debug(
                                    f"Successfully parsed JSON response into {json_model.__name__}"
                                )
                            except ValidationError as e:
                                logger.warning(
                                    f"JSON response does not match {json_model.__name__} schema: {e}. "
                                    f"Content: {content[:200]}..."
                                )
                                parsed_response = None
                        else:
                            # json_model is not a valid Pydantic model
                            parsed_response = None

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse JSON from response: {e}. "
                            f"Content: {content[:200]}..."
                        )
                        parsed_response = None

            return content, parsed_response
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response: {e}, response: {response}")
            raise ValueError(f"Invalid response format: {e}") from e

    def chat(
        self,
        prompt: str,
        history: list | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronous chat completion.

        Args:
            prompt: User prompt text
            history: Optional conversation history
            system_prompt: Optional system prompt
            **kwargs: Additional parameters including:
                - json: bool - Request JSON format response
                - json_model: type[BaseModel] - Pydantic model for structured output

        Returns:
            ModelResponse with the model's response and parsed_response if JSON mode used
        """
        try:
            # Extract JSON mode parameters
            json_mode = kwargs.get("json", False)
            json_model = kwargs.get("json_model", None)

            # Build messages
            messages = self._build_messages(prompt, history, system_prompt)

            # Make request with JSON mode if requested
            response = self._make_request(
                messages,
                json_mode=json_mode,
                json_model=json_model,
            )

            # Parse response (with JSON parsing if json_model provided)
            content, parsed_response = self._parse_response(response, json_model=json_model)

            # Return ModelResponse with parsed_response
            return BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=response,
                ),
                parsed_response=parsed_response,
            )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            # Return error response
            return BaseModelResponse(
                output=BaseModelOutput(
                    content=f"Error: {str(e)}",
                    full_response=None,
                ),
                parsed_response=None,
            )

    async def achat(
        self,
        prompt: str,
        history: list | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Asynchronous chat completion.

        Uses ThreadPoolExecutor to run synchronous requests in a thread.

        Args:
            prompt: User prompt text
            history: Optional conversation history
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (ignored for now)

        Returns:
            ModelResponse with the model's response
        """
        import asyncio
        from functools import partial

        # For GraphRAG entity extraction, force exact formatting with system prompt
        effective_system_prompt = system_prompt
        if not history or not any(msg.get("role") == "system" for msg in history if isinstance(msg, dict)):
            # Check if this is a GraphRAG entity extraction prompt (not community reporting)
            is_entity_extraction = (
                'entity_name' in prompt and
                'entity_type' in prompt and
                'entity_description' in prompt and
                'tuple_delimiter' in prompt and
                'record_delimiter' in prompt and
                'completion_delimiter' in prompt and
                'Return output as a single list' in prompt  # Entity extraction specific
            )

            # Exclude community reporting prompts (they expect JSON output)
            is_community_reporting = (
                'JSON-formatted string' in prompt or
                'Return output as a well-formed JSON' in prompt
            )

            if is_entity_extraction and not is_community_reporting and not system_prompt:
                effective_system_prompt = """You are a helpful assistant that follows formatting instructions with extreme precision. When extracting entities and relationships for knowledge graphs, you MUST:

1. Use EXACTLY the specified tuple format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)
2. Use EXACTLY the specified delimiters: <|>| as tuple delimiter, ## as record delimiter
3. End with the completion delimiter: <|><|>
4. Capitalize entity names and use only allowed entity types
5. Do NOT include any explanations, markdown, or additional text
6. Follow the format precisely, even if it seems unusual

Example format:
("entity"<|>"JOHN DOE"<|>"PERSON"<|>"John Doe is a software engineer")
("relationship"<|>"JOHN DOE"<|>"COMPANY X"<|>"John Doe works at Company X"<|>"8")
<|><|>"""

        # Run synchronous chat in executor
        # Use functools.partial to properly bind arguments since run_in_executor
        # doesn't accept keyword arguments directly
        loop = asyncio.get_event_loop()
        chat_func = partial(
            self.chat,
            prompt=prompt,
            history=history,
            system_prompt=effective_system_prompt,
            **kwargs
        )
        return await loop.run_in_executor(self._executor, chat_func)

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion.

        Note: Qwen API streaming is not yet implemented. This method
        falls back to non-streaming response and yields the content.

        Args:
            prompt: User prompt text
            history: Optional conversation history
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Response content chunks (currently yields full response as single chunk)
        """
        # For now, streaming is not implemented - return non-streaming response
        # This maintains interface compatibility
        response = await self.achat(prompt, history=history, system_prompt=system_prompt, **kwargs)
        yield response.output.content

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Generator[str, None]:
        """Synchronous streaming chat completion.

        Note: Streaming is not supported for synchronous execution.

        Args:
            prompt: User prompt text
            history: Optional conversation history
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Response content chunks

        Raises:
            NotImplementedError: Streaming not supported for synchronous execution
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)