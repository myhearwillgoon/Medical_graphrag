# Requirements: GraphRAG OpenAI to Qwen Integration

## Overview

Replace OpenAI SDK-based API calls with URL-based HTTP requests to Qwen model endpoints throughout the GraphRAG codebase. This enables the use of local Qwen model servers (Qwen 3-30B for chat, Qwen 3 Embedding 0.6B for embeddings) instead of OpenAI services while maintaining compatibility with the existing GraphRAG architecture.

## User Stories

### US-1: Qwen Model Type Support
**As a** GraphRAG user
**I want to** use Qwen chat and embedding models
**So that** I can leverage local Qwen model servers instead of OpenAI

**Acceptance Criteria:**
- QwenChat and QwenEmbedding model types are available in ModelType enum
- ModelFactory can create QwenChatModel and QwenEmbeddingModel instances
- Configuration files can specify Qwen model types

### US-2: Qwen Embedding Model Provider
**As a** GraphRAG user
**I want to** use Qwen embedding models for text embeddings
**So that** I can generate embeddings using local Qwen servers

**Acceptance Criteria:**
- QwenEmbeddingModel class implements EmbeddingModel protocol
- Supports sync (embed, embed_batch) and async (aembed, aembed_batch) methods
- Uses URL-based HTTP requests to `/v1/embeddings` endpoint
- Handles errors appropriately

### US-3: Default Qwen Configuration
**As a** GraphRAG user
**I want to** have Qwen models as default
**So that** I don't need to manually configure Qwen for each project

**Acceptance Criteria:**
- Default chat model type is QwenChat
- Default embedding model type is QwenEmbedding
- Default model names are qwen3-30b and Qwen3-Embedding-0.6B
- Initialization templates use Qwen by default

### US-4: Replace OpenAI Providers with Qwen URL Calls
**As a** GraphRAG developer
**I want to** replace OpenAI SDK calls with URL-based HTTP requests
**So that** all model calls use the same URL-based pattern

**Acceptance Criteria:**
- OpenAIChatFNLLM uses URL-based HTTP requests instead of OpenAI SDK
- OpenAIEmbeddingFNLLM uses URL-based HTTP requests instead of OpenAI SDK
- AzureOpenAIChatFNLLM uses Qwen URL calls
- AzureOpenAIEmbeddingFNLLM uses Qwen URL calls
- All providers maintain the same interface and return types

## Technical Requirements

### TR-1: URL Calling Pattern
- Chat endpoint: `{api_base}/v1/chat/completions` with model "qwen3-30b"
- Embedding endpoint: `{api_base}/v1/embeddings` with model "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B"
- Headers: `Content-Type: application/json`, `Authorization: Bearer sk-7966098172664c7f832496c33cfb86b8`
- Request format: OpenAI-compatible JSON
- Response parsing: Extract from `response['choices'][0]['message']['content']` for chat
- Special handling: Remove `</think>` tags from chat responses

### TR-2: Model Type Enum
- Add `QwenChat = "qwen_chat"` to ModelType enum
- Add `QwenEmbedding = "qwen_embedding"` to ModelType enum

### TR-3: Factory Registration
- Register QwenChatModel for ModelType.QwenChat.value
- Register QwenEmbeddingModel for ModelType.QwenEmbedding.value

### TR-4: Default Configuration
- DEFAULT_CHAT_MODEL = "qwen3-30b"
- DEFAULT_EMBEDDING_MODEL = "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B"
- DEFAULT_CHAT_MODEL_TYPE = ModelType.QwenChat
- DEFAULT_EMBEDDING_MODEL_TYPE = ModelType.QwenEmbedding
- DEFAULT_MODEL_PROVIDER = "qwen"

### TR-5: Initialization Templates
- INIT_YAML uses qwen_chat and qwen_embedding model types
- INIT_YAML uses qwen3-30b and Qwen3-Embedding-0.6B model names

## Constraints

- Must maintain backward compatibility with existing OpenAI configurations
- Must follow existing code patterns (similar to QwenChatModel implementation)
- Must handle errors gracefully
- Must support both sync and async operations
- Must maintain the same interface as existing providers

## Out of Scope

- Removing OpenAI SDK dependencies (keep for backward compatibility)
- Modifying QwenChatModel (already exists and works correctly)
- Changing the overall GraphRAG architecture
- Adding new features beyond URL-based Qwen support
