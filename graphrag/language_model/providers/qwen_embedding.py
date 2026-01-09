"""Qwen Embedding Model Provider.

This module provides a QwenEmbeddingModel class that implements the embedding model interface
for GraphRAG, using HTTP requests to communicate with a local Qwen model server.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class QwenEmbeddingModel:
    """Embedding model provider for Qwen using URL-based HTTP requests.

    This provider implements the embedding model interface expected by GraphRAG,
    making HTTP POST requests to a local Qwen server endpoint.
    """

    def __init__(
        self,
        name: str,
        config: Any,  # LanguageModelConfig
        **kwargs: Any,
    ):
        """Initialize QwenEmbeddingModel.

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
        self.model_name = getattr(config, "model", "qwen3-embedding")

        # Ensure api_base doesn't end with /
        if self.api_base.endswith("/"):
            self.api_base = self.api_base.rstrip("/")

        # Construct endpoint URL
        self.endpoint = f"{self.api_base}/v1/embeddings"

        # Thread pool executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(
            f"Initialized QwenEmbeddingModel '{name}' with endpoint: {self.endpoint}"
        )

    def _make_request(self, input_texts: list[str]) -> dict[str, Any]:
        """Make HTTP POST request to Qwen embeddings API endpoint.

        Args:
            input_texts: List of text strings to embed

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
            "input": input_texts,
        }

        try:
            logger.debug(f"Making embedding request to {self.endpoint} for {len(input_texts)} texts")
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

    def _parse_response(self, response: dict[str, Any]) -> list[list[float]]:
        """Parse API response and extract embeddings.

        Args:
            response: JSON response from API

        Returns:
            List of embedding vectors (list of floats)

        Raises:
            ValueError: If response format is invalid
        """
        try:
            data = response.get("data", [])
            if not data:
                raise ValueError("Response contains no embedding data")

            embeddings = []
            for item in data:
                embedding = item.get("embedding", [])
                if not embedding:
                    raise ValueError("Embedding item contains no embedding vector")
                embeddings.append(embedding)

            return embeddings
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing response: {e}, response: {response}")
            raise ValueError(f"Invalid response format: {e}") from e

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Synchronous single text embedding.

        Args:
            text: Text to embed
            **kwargs: Additional parameters (ignored for now)

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Make request with single text
            response = self._make_request([text])

            # Parse response
            embeddings = self._parse_response(response)

            # Return first (and only) embedding
            if not embeddings:
                raise ValueError("No embeddings returned from API")

            return embeddings[0]
        except Exception as e:
            logger.error(f"Error in embedding: {e}")
            raise

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Synchronous batch text embeddings.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters (ignored for now)

        Returns:
            List of embedding vectors (list of floats)
        """
        try:
            # Make request with batch of texts
            response = self._make_request(text_list)

            # Parse response
            embeddings = self._parse_response(response)

            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronous single text embedding.

        Uses ThreadPoolExecutor to run synchronous requests in a thread.

        Args:
            text: Text to embed
            **kwargs: Additional parameters (ignored for now)

        Returns:
            Embedding vector as list of floats
        """
        import asyncio

        # Run synchronous embed in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed,
            text,
            **kwargs,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Asynchronous batch text embeddings.

        Uses ThreadPoolExecutor to run synchronous requests in a thread.

        Args:
            text_list: List of texts to embed
            **kwargs: Additional parameters (ignored for now)

        Returns:
            List of embedding vectors (list of floats)
        """
        import asyncio

        # Run synchronous embed_batch in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed_batch,
            text_list,
            **kwargs,
        )

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)