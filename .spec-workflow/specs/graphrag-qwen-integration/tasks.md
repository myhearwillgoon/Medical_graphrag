# Tasks: GraphRAG OpenAI to Qwen Integration

## Task 1: Add Qwen Model Types to Enum
**Status:** [x]

**Files:**
- `graphrag/config/enums.py`

**Description:**
Add `QwenChat` and `QwenEmbedding` enum values to the `ModelType` enum class.

**Implementation:**
- Add `QwenChat = "qwen_chat"` after `Chat = "chat"`
- Add `QwenEmbedding = "qwen_embedding"` after `Embedding = "embedding"`

**Requirements:** US-1, TR-2

**Success Criteria:**
- ModelType.QwenChat and ModelType.QwenEmbedding are available
- Enum values serialize correctly to strings

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Configuration Developer specializing in enum and type definitions

Task: Add QwenChat and QwenEmbedding enum values to the ModelType enum class in graphrag/config/enums.py. Add the values after the existing Chat and Embedding entries following the same naming pattern.

Restrictions:
- Only add the two new enum values
- Do not modify existing enum values
- Follow exact same pattern as existing entries

_Leverage:
- graphrag/config/enums.py - Existing ModelType enum structure
- graphrag/language_model/factory.py - How enum values are used

_Requirements: US-1, TR-2

Success: ModelType enum contains QwenChat and QwenEmbedding values and can be imported successfully.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Add the enum values to graphrag/config/enums.py
3. Test that the enum values can be imported and used
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 2: Create Qwen Embedding Model Provider
**Status:** [x]

**Files:**
- `graphrag/language_model/providers/qwen_embedding.py` (new)

**Description:**
Create QwenEmbeddingModel class that implements EmbeddingModel protocol using URL-based HTTP requests.

**Implementation:**
- Create new file `qwen_embedding.py` in `providers` directory
- Implement `QwenEmbeddingModel` class following QwenChatModel pattern
- Implement `__init__` method with api_base, api_key, model_name configuration
- Implement `_make_request` method for HTTP POST to `/v1/embeddings`
- Implement `_parse_response` method to extract embeddings from response
- Implement `embed` method (sync single)
- Implement `embed_batch` method (sync batch)
- Implement `aembed` method (async single)
- Implement `aembed_batch` method (async batch)
- Add error handling and logging
- Use ThreadPoolExecutor for async operations (like QwenChatModel)

**Requirements:** US-2, TR-1

**Success Criteria:**
- QwenEmbeddingModel implements all EmbeddingModel protocol methods
- Successfully makes HTTP requests to `/v1/embeddings` endpoint
- Correctly parses response format: `{"data": [{"embedding": [...]}, ...]}`
- Handles errors appropriately
- Supports both sync and async operations

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in HTTP API integration and async programming

Task: Create QwenEmbeddingModel class in graphrag/language_model/providers/qwen_embedding.py that implements the EmbeddingModel protocol. The class should use URL-based HTTP requests to communicate with a Qwen embedding endpoint at {api_base}/v1/embeddings. Follow the same pattern as QwenChatModel (located at graphrag/language_model/providers/qwen_chat.py).

The class must implement:
- __init__(name, config, **kwargs): Initialize with api_base, api_key, and model_name from config
- embed(text, **kwargs) -> list[float]: Synchronous single text embedding
- embed_batch(text_list, **kwargs) -> list[list[float]]: Synchronous batch embeddings
- aembed(text, **kwargs) -> list[float]: Async single text embedding
- aembed_batch(text_list, **kwargs) -> list[list[float]]: Async batch embeddings
- _make_request(input_texts): Internal method for HTTP POST requests
- _parse_response(response): Internal method to parse API response

Request format: {"model": "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B", "input": ["text1", "text2", ...]}
Response format: {"data": [{"embedding": [...]}, {"embedding": [...}], ...]}

Restrictions:
- Do not use OpenAI SDK - use requests library directly
- Must match the interface of existing EmbeddingModel implementations (see graphrag/language_model/providers/fnllm/models.py OpenAIEmbeddingFNLLM)
- Must handle errors gracefully with appropriate logging
- Use ThreadPoolExecutor for async operations (like QwenChatModel)

_Leverage:
- graphrag/language_model/providers/qwen_chat.py - Reference implementation pattern
- graphrag/language_model/providers/fnllm/models.py - Reference interface patterns
- graphrag/language_model/protocol/base.py - EmbeddingModel protocol definition

_Requirements: US-2, TR-1

Success: QwenEmbeddingModel class exists, implements all required methods, successfully makes HTTP requests, and correctly parses responses. All methods match the EmbeddingModel protocol interface.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Create the qwen_embedding.py file with QwenEmbeddingModel class
3. Test the implementation
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 3: Register Qwen Models in Factory
**Status:** [x]

**Files:**
- `graphrag/language_model/factory.py`

**Description:**
Import and register QwenChatModel and QwenEmbeddingModel in ModelFactory.

**Implementation:**
- Import `QwenChatModel` from `graphrag.language_model.providers.qwen_chat`
- Import `QwenEmbeddingModel` from `graphrag.language_model.providers.qwen_embedding`
- Add registration: `ModelFactory.register_chat(ModelType.QwenChat.value, lambda **kwargs: QwenChatModel(**kwargs))`
- Add registration: `ModelFactory.register_embedding(ModelType.QwenEmbedding.value, lambda **kwargs: QwenEmbeddingModel(**kwargs))`

**Requirements:** US-1, TR-3

**Success Criteria:**
- ModelFactory.create_chat_model(ModelType.QwenChat.value) returns QwenChatModel instance
- ModelFactory.create_embedding_model(ModelType.QwenEmbedding.value) returns QwenEmbeddingModel instance
- Factory registration follows same pattern as existing registrations

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in factory patterns and dependency injection

Task: Register QwenChatModel and QwenEmbeddingModel in the ModelFactory class located at graphrag/language_model/factory.py. Import the Qwen model classes and add factory registrations following the existing pattern used for OpenAI and LiteLLM models.

Restrictions:
- Follow the exact same registration pattern as existing models
- Use ModelType.QwenChat.value and ModelType.QwenEmbedding.value for registration keys
- Do not modify existing registrations

_Leverage:
- graphrag/language_model/factory.py - Existing factory registration pattern
- graphrag/config/enums.py - ModelType enum definitions

_Requirements: US-1, TR-3

Success: QwenChatModel and QwenEmbeddingModel are registered in ModelFactory and can be created via factory methods.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Add imports and registrations to factory.py
3. Verify factory can create Qwen model instances
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 4: Replace OpenAIChatFNLLM with URL-based Calls
**Status:** [-]

**Files:**
- `graphrag/language_model/providers/fnllm/models.py`

**Description:**
Replace OpenAI SDK calls in OpenAIChatFNLLM with direct HTTP requests using requests library, following QwenChatModel pattern.

**Implementation:**
- Remove `create_openai_client()` call from `__init__`
- Remove `create_openai_chat_llm()` call from `__init__`
- Add api_base, api_key, model_name attributes from config
- Replace `achat` method to use direct HTTP POST to `/v1/chat/completions`
- Replace `chat` method to use direct HTTP POST (or call achat via run_coroutine_sync)
- Update `achat_stream` and `chat_stream` if needed
- Maintain same return types (ModelResponse)
- Add error handling

**Requirements:** US-4, TR-1

**Success Criteria:**
- OpenAIChatFNLLM no longer uses OpenAI SDK
- Makes HTTP requests to configured endpoint
- Maintains same interface and return types
- Handles errors appropriately

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in API integration and refactoring

Task: Replace OpenAI SDK calls in OpenAIChatFNLLM class (graphrag/language_model/providers/fnllm/models.py) with direct HTTP requests using the requests library. Follow the pattern from QwenChatModel._make_request() method. Remove create_openai_client() and create_openai_chat_llm() calls, and replace with direct HTTP POST requests to {api_base}/v1/chat/completions endpoint.

Restrictions:
- Must maintain the exact same interface (methods, parameters, return types)
- Do not break existing functionality
- Use requests library, not OpenAI SDK
- Follow QwenChatModel pattern for HTTP requests
- Maintain error handling and logging

_Leverage:
- graphrag/language_model/providers/qwen_chat.py - Reference HTTP request pattern
- graphrag/language_model/providers/fnllm/models.py - Existing OpenAIChatFNLLM implementation
- graphrag/language_model/providers/fnllm/utils.py - run_coroutine_sync utility

_Requirements: US-4, TR-1

Success: OpenAIChatFNLLM uses URL-based HTTP requests instead of OpenAI SDK, maintains same interface, and all methods work correctly.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Modify OpenAIChatFNLLM class to use HTTP requests
3. Test chat and achat methods
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 5: Replace OpenAIEmbeddingFNLLM with URL-based Calls
**Status:** [-]

**Files:**
- `graphrag/language_model/providers/fnllm/models.py`

**Description:**
Replace OpenAI SDK calls in OpenAIEmbeddingFNLLM with direct HTTP requests using requests library, following QwenEmbeddingModel pattern.

**Implementation:**
- Remove `create_openai_client()` call from `__init__`
- Remove `create_openai_embeddings_llm()` call from `__init__`
- Add api_base, api_key, model_name attributes from config
- Replace `aembed_batch` method to use direct HTTP POST to `/v1/embeddings`
- Replace `aembed` method to use direct HTTP POST
- Replace `embed_batch` and `embed` methods (or call async versions via run_coroutine_sync)
- Maintain same return types (list[float], list[list[float]])
- Add error handling

**Requirements:** US-4, TR-1

**Success Criteria:**
- OpenAIEmbeddingFNLLM no longer uses OpenAI SDK
- Makes HTTP requests to configured endpoint
- Maintains same interface and return types
- Handles errors appropriately

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in API integration and refactoring

Task: Replace OpenAI SDK calls in OpenAIEmbeddingFNLLM class (graphrag/language_model/providers/fnllm/models.py) with direct HTTP requests using the requests library. Follow the pattern from QwenEmbeddingModel._make_request() method. Remove create_openai_client() and create_openai_embeddings_llm() calls, and replace with direct HTTP POST requests to {api_base}/v1/embeddings endpoint.

Restrictions:
- Must maintain the exact same interface (methods, parameters, return types)
- Do not break existing functionality
- Use requests library, not OpenAI SDK
- Follow QwenEmbeddingModel pattern for HTTP requests
- Maintain error handling and logging

_Leverage:
- graphrag/language_model/providers/qwen_embedding.py - Reference HTTP request pattern (created in Task 2)
- graphrag/language_model/providers/fnllm/models.py - Existing OpenAIEmbeddingFNLLM implementation
- graphrag/language_model/providers/fnllm/utils.py - run_coroutine_sync utility

_Requirements: US-4, TR-1

Success: OpenAIEmbeddingFNLLM uses URL-based HTTP requests instead of OpenAI SDK, maintains same interface, and all methods work correctly.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Modify OpenAIEmbeddingFNLLM class to use HTTP requests
3. Test embed, embed_batch, aembed, and aembed_batch methods
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 6: Replace Azure OpenAI Providers with Qwen URL Calls
**Status:** [-]

**Files:**
- `graphrag/language_model/providers/fnllm/models.py`

**Description:**
Replace Azure OpenAI SDK calls in AzureOpenAIChatFNLLM and AzureOpenAIEmbeddingFNLLM with Qwen URL-based calls.

**Implementation:**
- Apply same changes as Tasks 4 and 5 to Azure variants
- Use same endpoint pattern (Azure endpoints may differ, but use Qwen pattern)
- Or maintain Azure-specific logic if compatibility is needed

**Requirements:** US-4

**Success Criteria:**
- Azure providers use URL-based HTTP requests
- Maintain same interface
- Handle Azure-specific configuration if needed

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in API integration and Azure services

Task: Replace Azure OpenAI SDK calls in AzureOpenAIChatFNLLM and AzureOpenAIEmbeddingFNLLM classes (graphrag/language_model/providers/fnllm/models.py) with Qwen URL-based HTTP requests. Follow the same pattern as Tasks 4 and 5, but handle Azure-specific configuration if needed.

Restrictions:
- Must maintain the exact same interface
- Use Qwen URL pattern (same as regular OpenAI replacements)
- Handle Azure-specific config attributes if they exist
- Maintain error handling

_Leverage:
- Tasks 4 and 5 implementations as reference
- graphrag/language_model/providers/fnllm/models.py - Azure provider classes

_Requirements: US-4

Success: Azure providers use URL-based HTTP requests, maintain same interface, and work correctly.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Modify AzureOpenAIChatFNLLM and AzureOpenAIEmbeddingFNLLM classes
3. Test all methods
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 7: Update Default Configurations
**Status:** [-]

**Files:**
- `graphrag/config/defaults.py`

**Description:**
Update default model configurations to use Qwen models.

**Implementation:**
- Change `DEFAULT_CHAT_MODEL = "qwen3-30b"`
- Change `DEFAULT_EMBEDDING_MODEL = "/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B"`
- Change `DEFAULT_CHAT_MODEL_TYPE = ModelType.QwenChat`
- Change `DEFAULT_EMBEDDING_MODEL_TYPE = ModelType.QwenEmbedding`
- Change `DEFAULT_MODEL_PROVIDER = "qwen"`

**Requirements:** US-3, TR-4

**Success Criteria:**
- Defaults point to Qwen models
- New projects use Qwen by default
- Existing configurations still work (backward compatible)

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Configuration Developer specializing in default settings

Task: Update default model configurations in graphrag/config/defaults.py to use Qwen models instead of OpenAI. Change DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_CHAT_MODEL_TYPE, DEFAULT_EMBEDDING_MODEL_TYPE, and DEFAULT_MODEL_PROVIDER to point to Qwen.

Restrictions:
- Only change default values, not the structure
- Ensure backward compatibility (existing configs should still work)
- Use ModelType.QwenChat and ModelType.QwenEmbedding enum values

_Leverage:
- graphrag/config/defaults.py - Current default values
- graphrag/config/enums.py - ModelType enum

_Requirements: US-3, TR-4

Success: Default configurations use Qwen models, and the changes are backward compatible.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Update default values in defaults.py
3. Verify changes don't break existing code
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```

---

## Task 8: Update Initialization Templates
**Status:** [x]

**Files:**
- `graphrag/config/init_content.py`

**Description:**
Update INIT_YAML template to use Qwen models by default.

**Implementation:**
- Change model type references to use `qwen_chat` and `qwen_embedding`
- Change model names to `qwen3-30b` and `/home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B`
- Update model_provider to `qwen` if needed
- Keep api_base and api_key configuration options

**Requirements:** US-3, TR-5

**Success Criteria:**
- `graphrag init` creates configs with Qwen models
- Template uses correct model types and names
- Configuration is valid and works

**Prompt:**
```
Implement the task for spec graphrag-qwen-integration, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Configuration Developer specializing in initialization templates

Task: Update INIT_YAML template in graphrag/config/init_content.py to use Qwen models by default. Change model types to qwen_chat and qwen_embedding, and update model names to qwen3-30b and /home/huanghong/displace/Qwen/Qwen3-Embedding-0.6B.

Restrictions:
- Only modify the template string, not the Python code structure
- Ensure the YAML is valid
- Keep api_base and api_key configuration options
- Use defs.DEFAULT_CHAT_MODEL_TYPE.value and defs.DEFAULT_EMBEDDING_MODEL_TYPE.value for model types

_Leverage:
- graphrag/config/init_content.py - Current INIT_YAML template
- graphrag/config/defaults.py - Default values (updated in Task 7)

_Requirements: US-3, TR-5

Success: INIT_YAML template uses Qwen models, and graphrag init creates valid Qwen configurations.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Update INIT_YAML template in init_content.py
3. Verify template generates valid YAML
4. Log implementation details using log-implementation tool
5. Mark task as complete [x] in tasks.md
```
