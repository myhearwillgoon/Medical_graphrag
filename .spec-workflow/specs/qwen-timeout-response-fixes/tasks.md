# Tasks: Qwen Timeout and Response Format Fixes

## Task 1: Increase Timeout and Add Retry Logic
**Status:** [x]

**Files:**
- `graphrag/language_model/providers/qwen_chat.py`
- `graphrag/language_model/providers/qwen_embedding.py`

**Description:**
Increase default timeout from 30s to 120s and add retry logic with exponential backoff for Qwen models.

**Implementation:**
- Change `self.timeout = 30` to `self.timeout = 120`
- Add retry decorator or logic for HTTP requests
- Implement exponential backoff with jitter
- Add connection pooling configuration

**Requirements:** US-1, TR-1, TR-2

**Success Criteria:**
- Timeout increased to 120 seconds
- Retry logic implemented with exponential backoff
- Network errors trigger appropriate retries
- No timeout errors in normal operation

**Prompt:**
```
Implement the task for spec qwen-timeout-response-fixes, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in HTTP client resilience and error handling

Task: Increase timeout from 30s to 120s and add retry logic with exponential backoff for Qwen chat and embedding models. Modify graphrag/language_model/providers/qwen_chat.py and graphrag/language_model/providers/qwen_embedding.py to handle long-running GraphRAG extraction tasks.

Restrictions:
- Only modify timeout and add retry logic
- Keep existing interfaces unchanged
- Use exponential backoff with reasonable limits
- Add appropriate logging for retries

_Leverage:
- Existing requests library usage
- Python tenacity library if available, otherwise implement simple retry
- Current error handling patterns in codebase

_Requirements: US-1, TR-1, TR-2

Success: Qwen models can handle complex GraphRAG prompts without timing out, with automatic retry on network failures.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Increase timeout to 120s in both Qwen providers
3. Add retry logic with exponential backoff
4. Test timeout handling
5. Log implementation details using log-implementation tool
6. Mark task as complete [x] in tasks.md
```

---

## Task 2: Enhance Response Format Compliance
**Status:** [x]

**Files:**
- `graphrag/language_model/providers/qwen_chat.py`

**Description:**
Improve system instructions and response parsing to ensure Qwen follows GraphRAG's expected output format.

**Implementation:**
- Enhance system message for formatting compliance
- Add response validation and cleanup
- Improve debug logging for troubleshooting
- Add fallback parsing for non-standard responses

**Requirements:** US-2, TR-3

**Success Criteria:**
- System instructions include formatting guidance
- Response validation catches format issues
- Debug logging shows raw responses
- Better compliance with GraphRAG expectations

**Prompt:**
```
Implement the task for spec qwen-timeout-response-fixes, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: AI Integration Specialist focusing on prompt engineering and response parsing

Task: Enhance Qwen chat model's response format compliance by improving system instructions and adding response validation. Modify graphrag/language_model/providers/qwen_chat.py to ensure Qwen follows GraphRAG's tuple-based output format for entity extraction.

Restrictions:
- Focus on response format improvement
- Add system instructions for precise formatting
- Implement response validation and cleanup
- Maintain backward compatibility

_Leverage:
- Existing system message handling
- Current response parsing logic
- GraphRAG's expected output format (tuple_delimiter, record_delimiter)

_Requirements: US-2, TR-3

Success: Qwen responses are more compliant with GraphRAG's formatting requirements, with better error handling and debugging.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Enhance system instructions for formatting
3. Add response validation and cleanup
4. Improve debug logging
5. Log implementation details using log-implementation tool
6. Mark task as complete [x] in tasks.md
```

---

## Task 3: Optimize Model Parameters
**Status:** [x]

**Files:**
- `graphrag/language_model/providers/qwen_chat.py`
- `graphrag/language_model/providers/qwen_embedding.py`

**Description:**
Optimize model parameters for GraphRAG tasks including temperature, max tokens, and concurrency settings.

**Implementation:**
- Set temperature=0.1 for consistent formatting
- Configure max_tokens appropriately
- Adjust concurrent request handling
- Add model-specific optimizations

**Requirements:** US-4, TR-4

**Success Criteria:**
- Temperature set to 0.1 for consistency
- Appropriate max_tokens configuration
- Optimized concurrency settings
- Better performance on GraphRAG tasks

**Prompt:**
```
Implement the task for spec qwen-timeout-response-fixes, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Machine Learning Engineer specializing in model parameter optimization

Task: Optimize Qwen model parameters for GraphRAG tasks by adjusting temperature, max_tokens, and other settings in both chat and embedding providers. Focus on improving consistency and performance for entity extraction tasks.

Restrictions:
- Only modify model parameters and request settings
- Keep existing interfaces intact
- Focus on GraphRAG-specific optimizations
- Test parameter changes

_Leverage:
- Current model parameter handling
- GraphRAG's typical usage patterns
- Performance requirements for entity extraction

_Requirements: US-4, TR-4

Success: Qwen models are optimized for GraphRAG tasks with appropriate parameters for consistent, high-quality responses.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Optimize temperature and max_tokens settings
3. Adjust concurrency and other parameters
4. Test parameter effectiveness
5. Log implementation details using log-implementation tool
6. Mark task as complete [x] in tasks.md
```

---

## Task 4: Apply Fixes to FNLlm Providers
**Status:** [x]

**Files:**
- `graphrag/language_model/providers/fnllm/models.py`

**Description:**
Apply the same timeout and retry improvements to the fnllm providers that are used by OpenAI/Azure OpenAI.

**Implementation:**
- Increase timeout in OpenAI/Azure providers
- Add retry logic to fnllm models
- Improve error handling and logging
- Ensure consistency across all providers

**Requirements:** TR-1, TR-2

**Success Criteria:**
- FNLlm providers have extended timeouts
- Retry logic implemented consistently
- Better error handling across all providers
- Consistent behavior between Qwen and OpenAI providers

**Prompt:**
```
Implement the task for spec qwen-timeout-response-fixes, first run spec-workflow-guide to get the workflow guide then implement the task:

Role: Backend Developer specializing in provider abstraction and consistency

Task: Apply timeout and retry improvements to the fnllm providers (OpenAI, Azure OpenAI) to ensure consistent behavior across all model providers. Modify graphrag/language_model/providers/fnllm/models.py with the same enhancements.

Restrictions:
- Apply same timeout/retry logic as Qwen providers
- Maintain existing fnllm architecture
- Ensure backward compatibility
- Test with existing OpenAI configurations

_Leverage:
- Changes made to Qwen providers
- Existing fnllm error handling patterns
- Current timeout configurations

_Requirements: TR-1, TR-2

Success: All model providers have consistent timeout and retry behavior, improving reliability across different LLM services.

Instructions:
1. Mark this task as in-progress [-] in tasks.md
2. Apply timeout extensions to fnllm providers
3. Add retry logic to fnllm models
4. Ensure consistent error handling
5. Log implementation details using log-implementation tool
6. Mark task as complete [x] in tasks.md
```
