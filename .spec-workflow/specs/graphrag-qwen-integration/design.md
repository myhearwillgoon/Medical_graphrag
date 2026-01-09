# Design: GraphRAG OpenAI to Qwen Integration

## Architecture Overview

This design replaces OpenAI SDK calls with URL-based HTTP requests to Qwen endpoints while maintaining the existing GraphRAG architecture and interfaces.

## Components

### 1. Model Type Enum Extension
**File:** `graphrag/config/enums.py`

Add two new enum values to `ModelType`:
- `QwenChat = "qwen_chat"`
- `QwenEmbedding = "qwen_embedding"`

**Rationale:** Provides type-safe model type identifiers for Qwen models.

### 2. Qwen Embedding Model Provider
**File:** `graphrag/language_model/providers/qwen_embedding.py` (new)

**Class:** `QwenEmbeddingModel`

**Structure:**
- Follows the same pattern as `QwenChatModel`
- Implements `EmbeddingModel` protocol
- Uses URL-based HTTP requests to `/v1/embeddings` endpoint

**Methods:**
- `__init__(name, config, **kwargs)`: Initialize with api_base, api_key, model_name
- `embed(text, **kwargs) -> list[float]`: Synchronous single embedding
- `embed_batch(text_list, **kwargs) -> list[list[float]]`: Synchronous batch embeddings
- `aembed(text, **kwargs) -> list[float]`: Async single embedding
- `aembed_batch(text_list, **kwargs) -> list[list[float]]`: Async batch embeddings
- `_make_request(input_texts) -> dict`: Internal HTTP request method
- `_parse_response(response) -> list[list[float]]`: Parse API response

**Request Format:**
```json
{
  "model": "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B",
  "input": ["text1", "text2", ...]
}
```

**Response Format:**
```json
{
  "data": [
    {"embedding": [...]},
    {"embedding": [...]}
  ]
}
```

### 3. Factory Registration
**File:** `graphrag/language_model/factory.py`

**Changes:**
- Import `QwenChatModel` from `graphrag.language_model.providers.qwen_chat`
- Import `QwenEmbeddingModel` from `graphrag.language_model.providers.qwen_embedding`
- Register `ModelType.QwenChat.value -> QwenChatModel`
- Register `ModelType.QwenEmbedding.value -> QwenEmbeddingModel`

**Rationale:** Enables ModelFactory to create Qwen model instances.

### 4. Replace OpenAI Providers with URL-based Calls
**File:** `graphrag/language_model/providers/fnllm/models.py`

**Strategy:** Replace OpenAI SDK calls with direct HTTP requests using the `requests` library, following the pattern from `QwenChatModel._make_request()`.

**Classes to Modify:**

#### 4.1 OpenAIChatFNLLM
- Remove `create_openai_client()` call
- Remove `create_openai_chat_llm()` call
- Replace with direct HTTP POST to `{api_base}/v1/chat/completions`
- Use same request/response format as QwenChatModel
- Maintain same interface (chat, achat, chat_stream, achat_stream)

#### 4.2 OpenAIEmbeddingFNLLM
- Remove `create_openai_client()` call
- Remove `create_openai_embeddings_llm()` call
- Replace with direct HTTP POST to `{api_base}/v1/embeddings`
- Use same request/response format as QwenEmbeddingModel
- Maintain same interface (embed, embed_batch, aembed, aembed_batch)

#### 4.3 AzureOpenAIChatFNLLM
- Replace with Qwen URL calls (same endpoint pattern)
- Or maintain Azure-specific logic if needed for compatibility

#### 4.4 AzureOpenAIEmbeddingFNLLM
- Replace with Qwen URL calls (same endpoint pattern)
- Or maintain Azure-specific logic if needed for compatibility

**Implementation Pattern:**
```python
def _make_request(self, endpoint_suffix: str, data: dict) -> dict:
    """Make HTTP POST request to API endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
    }
    url = f"{self.api_base}/{endpoint_suffix}"
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()
```

### 5. Default Configuration Updates
**File:** `graphrag/config/defaults.py`

**Changes:**
- `DEFAULT_CHAT_MODEL = "qwen3-30b"`
- `DEFAULT_EMBEDDING_MODEL = "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B"`
- `DEFAULT_CHAT_MODEL_TYPE = ModelType.QwenChat`
- `DEFAULT_EMBEDDING_MODEL_TYPE = ModelType.QwenEmbedding`
- `DEFAULT_MODEL_PROVIDER = "qwen"`

**Rationale:** Makes Qwen the default model provider for new projects.

### 6. Initialization Template Updates
**File:** `graphrag/config/init_content.py`

**Changes:**
- Update `INIT_YAML` template to use `qwen_chat` and `qwen_embedding` model types
- Update model names to `qwen3-30b` and `/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B`
- Keep api_base and api_key configuration options

**Rationale:** New projects initialized with `graphrag init` will use Qwen by default.

## Data Flow

### Chat Model Request Flow
1. User calls `model.chat(prompt)`
2. Model builds messages list
3. Model makes HTTP POST to `{api_base}/v1/chat/completions`
4. Response parsed: `response['choices'][0]['message']['content']`
5. Clean reasoning tags: `content.split('</think>')[-1]`
6. Return `ModelResponse`

### Embedding Model Request Flow
1. User calls `model.embed(text)` or `model.embed_batch(texts)`
2. Model builds request: `{"model": "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B", "input": [...]}`
3. Model makes HTTP POST to `{api_base}/v1/embeddings`
4. Response parsed: `[item['embedding'] for item in response['data']]`
5. Return `list[float]` or `list[list[float]]`

## Error Handling

- Network errors: Log and raise appropriate exceptions
- HTTP errors: Use `response.raise_for_status()` and handle 4xx/5xx responses
- Invalid responses: Validate response structure and raise `ValueError` with descriptive message
- Timeout: Set 30-second timeout, raise `TimeoutError` on timeout

## Testing Considerations

- Unit tests for each provider class
- Mock HTTP requests using `responses` library or `unittest.mock`
- Test error handling paths
- Test both sync and async methods
- Verify response parsing logic

## Migration Strategy

1. Add Qwen enum values (non-breaking)
2. Create QwenEmbeddingModel (new file, non-breaking)
3. Register Qwen models in factory (non-breaking)
4. Update defaults (breaking for new projects, but existing configs still work)
5. Replace OpenAI providers (breaking change, but maintains interface)
6. Update init templates (only affects new projects)

**Backward Compatibility:**
- Existing configurations continue to work
- OpenAI model types remain available
- Users can still use OpenAI by explicitly configuring it

## Dependencies

- `requests` library (already used by QwenChatModel)
- No new dependencies required

## Performance Considerations

- HTTP requests add network latency
- Batch operations reduce number of requests
- Consider connection pooling for high-throughput scenarios
- Timeout settings prevent hanging requests
