# Design: Qwen Timeout and Response Format Fixes

## Architecture Overview

This design addresses timeout issues and response format problems when using Qwen models with GraphRAG. The fixes focus on increasing timeouts, improving response compliance, and adding retry logic while maintaining backward compatibility.

## Components

### 1. Timeout and Retry Enhancement
**Files:** `graphrag/language_model/providers/qwen_chat.py`, `graphrag/language_model/providers/qwen_embedding.py`

**Changes:**
- Increase default timeout from 30s to 120s
- Add exponential backoff retry logic
- Make timeout configurable via model config
- Add connection pooling for better performance

### 2. Response Format Improvement
**Files:** `graphrag/language_model/providers/qwen_chat.py`

**Changes:**
- Enhanced system instructions for formatting compliance
- Better response parsing and cleanup
- Debug logging for troubleshooting
- Validation before response processing

### 3. Model Parameter Optimization
**Files:** `graphrag/language_model/providers/qwen_chat.py`, `graphrag/language_model/providers/qwen_embedding.py`

**Changes:**
- Set temperature=0.1 for consistent formatting
- Configure max_tokens=4096
- Optimize concurrent request handling

### 4. Error Handling Enhancement
**Files:** `graphrag/language_model/providers/fnllm/models.py`

**Changes:**
- Apply same timeout/retry improvements to fnllm providers
- Better error logging and recovery
- Graceful degradation on failures

## Implementation Strategy

1. **Timeout Extension**: Increase timeouts and add retry logic
2. **Response Enhancement**: Improve system instructions and parsing
3. **Parameter Tuning**: Optimize model parameters for GraphRAG tasks
4. **Error Recovery**: Add robust error handling and logging

## Data Flow

1. Model request with extended timeout and retry logic
2. Enhanced system instructions improve response quality
3. Response parsing and validation before processing
4. Fallback handling for format issues

## Error Handling

- Timeout errors trigger retry with exponential backoff
- Response format issues logged for debugging
- Graceful fallback when possible
- Clear error messages for troubleshooting

## Testing Considerations

- Test timeout scenarios with mock slow responses
- Validate response format compliance
- Test retry logic with network failures
- Performance testing with different concurrency levels

## Migration Strategy

- Changes are backward compatible
- Only affect Qwen model providers
- No breaking changes to existing interfaces
- Gradual rollout with monitoring

## Dependencies

- No new dependencies required
- Uses existing requests library
- Leverages existing retry utilities

## Performance Considerations

- Extended timeouts may increase total processing time
- Retry logic adds resilience but may increase latency
- Reduced concurrency prevents server overload
- Connection pooling improves efficiency
