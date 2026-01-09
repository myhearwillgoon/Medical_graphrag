# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing fnllm model provider definitions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import requests

from graphrag.language_model.providers.fnllm.utils import run_coroutine_sync
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import (
        LanguageModelConfig,
    )


class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using direct HTTP requests."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.api_base = config.api_base or "https://api.openai.com"
        self.api_key = config.api_key
        self.model_name = config.model or "gpt-3.5-turbo"
        self.timeout = 120  # Increased from 30s for complex tasks
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced to prevent overload

    def _make_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make HTTP POST request to the OpenAI chat completions endpoint with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            The API response as a dictionary.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
        }
        url = f"{self.api_base}/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff with jitter
                    logger.warning(f"OpenAI request attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} OpenAI attempts failed. Last error: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI request failed: {e}")
                raise

    def _parse_response(self, response: dict[str, Any]) -> str:
        """Parse the API response to extract the content.

        Args:
            response: The API response dictionary.

        Returns:
            The response content string.
        """
        if "choices" not in response or not response["choices"]:
            msg = "Invalid response format: missing or empty 'choices' field"
            raise ValueError(msg)

        return response["choices"][0]["message"]["content"].strip()

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        # Build messages list
        messages = []
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Handle other formats if needed
                    messages.append({"role": "user", "content": str(msg)})

        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(self.executor, self._make_request, messages)
        content = self._parse_response(response)

        return BaseModelResponse(
            output=BaseModelOutput(
                content=content,
                full_response=response,
            ),
            parsed_response=None,
            history=messages + [{"role": "assistant", "content": content}],
            cache_hit=False,
            tool_calls=[],
            metrics=None,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        # For now, just return the full response as streaming isn't implemented
        # Could be enhanced to support actual streaming if needed
        response = await self.achat(prompt, history, **kwargs)
        yield response.output.content

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        # For now, just yield the full response as streaming isn't implemented
        response = self.chat(prompt, history, **kwargs)
        yield response.output.content


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using direct HTTP requests."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.api_base = config.api_base or "https://api.openai.com"
        self.api_key = config.api_key
        self.model_name = config.model or "text-embedding-ada-002"
        self.timeout = 120  # Increased from 30s for complex tasks
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced to prevent overload

    def _make_request(self, input_texts: list[str]) -> dict[str, Any]:
        """Make HTTP POST request to the OpenAI embeddings endpoint with retry logic.

        Args:
            input_texts: List of texts to embed.

        Returns:
            The API response as a dictionary.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model_name,
            "input": input_texts
        }
        url = f"{self.api_base}/v1/embeddings"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff with jitter
                    logger.warning(f"OpenAI embedding request attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} OpenAI embedding attempts failed. Last error: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI embedding request failed: {e}")
                raise

    def _parse_response(self, response: dict[str, Any]) -> list[list[float]]:
        """Parse the API response to extract embeddings.

        Args:
            response: The API response dictionary.

        Returns:
            List of embedding vectors.
        """
        if "data" not in response:
            msg = "Invalid response format: missing 'data' field"
            raise ValueError(msg)

        embeddings = []
        for item in response["data"]:
            if "embedding" not in item:
                msg = "Invalid response format: missing 'embedding' field in data item"
                raise ValueError(msg)
            embeddings.append(item["embedding"])

        return embeddings

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text_list: List of texts to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the texts.
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(self.executor, self._make_request, text_list)
        return self._parse_response(response)

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0]

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using direct HTTP requests."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.name = name
        self.config = config
        # Use Qwen endpoints instead of Azure OpenAI
        self.api_base = "http://192.168.60.202:8333"
        self.api_key = "sk-7966098172664c7f832496c33cfb86b8"
        self.model_name = "qwen3-30b"
        self.timeout = 120  # Increased from 30s for complex tasks
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced to prevent overload

    def _make_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make HTTP POST request to the Qwen chat completions endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            The API response as a dictionary.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "enable_thinking": False
        }
        url = f"{self.api_base}/v1/chat/completions"

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict[str, Any]) -> str:
        """Parse the API response to extract the content.

        Args:
            response: The API response dictionary.

        Returns:
            The response content string.
        """
        if "choices" not in response or not response["choices"]:
            msg = "Invalid response format: missing or empty 'choices' field"
            raise ValueError(msg)

        content = response["choices"][0]["message"]["content"]
        # Remove thinking tags if present
        content = content.split("</think>")[-1] if "</think>" in content else content
        return content.strip()

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        # Build messages list
        messages = []
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Handle other formats if needed
                    messages.append({"role": "user", "content": str(msg)})

        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(self.executor, self._make_request, messages)
        content = self._parse_response(response)

        return BaseModelResponse(
            output=BaseModelOutput(
                content=content,
                full_response=response,
            ),
            parsed_response=None,
            history=messages + [{"role": "assistant", "content": content}],
            cache_hit=False,
            tool_calls=[],
            metrics=None,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        # For now, just return the full response as streaming isn't implemented
        response = await self.achat(prompt, history, **kwargs)
        yield response.output.content

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        # For now, just yield the full response as streaming isn't implemented
        response = self.chat(prompt, history, **kwargs)
        yield response.output.content


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using direct HTTP requests."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        self.name = name
        self.config = config
        # Use Qwen endpoints instead of Azure OpenAI
        self.api_base = "http://192.168.60.202:7890"
        self.api_key = "sk-7966098172664c7f832496c33cfb86b8"
        self.model_name = "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B"
        self.timeout = 240  # Increased from 30s for complex tasks
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced to prevent overload

    def _make_request(self, input_texts: list[str]) -> dict[str, Any]:
        """Make HTTP POST request to the Qwen embeddings endpoint.

        Args:
            input_texts: List of texts to embed.

        Returns:
            The API response as a dictionary.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model_name,
            "input": input_texts
        }
        url = f"{self.api_base}/v1/embeddings"

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict[str, Any]) -> list[list[float]]:
        """Parse the API response to extract embeddings.

        Args:
            response: The API response dictionary.

        Returns:
            List of embedding vectors.
        """
        if "data" not in response:
            msg = "Invalid response format: missing 'data' field"
            raise ValueError(msg)

        embeddings = []
        for item in response["data"]:
            if "embedding" not in item:
                msg = "Invalid response format: missing 'embedding' field in data item"
                raise ValueError(msg)
            embeddings.append(item["embedding"])

        return embeddings

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(self.executor, self._make_request, text_list)
        return self._parse_response(response)

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0]

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))
