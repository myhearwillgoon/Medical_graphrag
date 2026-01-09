# Requirements: Qwen Timeout and Response Format Fixes

## Overview

Fix timeout issues and response format problems that occur when using Qwen models with GraphRAG's complex entity extraction prompts. The current 30-second timeout is insufficient for Qwen to process detailed extraction tasks, and response formatting needs improvement for better compliance with GraphRAG's expected output structure.

## User Stories

### US-1: Extend Timeout for Complex Prompts
**As a** GraphRAG user with Qwen models
**I want to** configure longer timeouts for complex tasks
**So that** Qwen has sufficient time to process detailed extraction prompts

**Acceptance Criteria:**
- Timeout configurable via settings (default 120s for Qwen)
- Different timeouts for different task types
- Graceful timeout handling with retry logic

### US-2: Improve Response Format Compliance
**As a** GraphRAG user with Qwen models
**I want to** better system instructions for formatting
**So that** Qwen produces responses that match GraphRAG expectations

**Acceptance Criteria:**
- Enhanced system prompts for Qwen models
- Better formatting instructions in prompts
- Response validation before processing

### US-3: Add Retry and Error Recovery
**As a** GraphRAG user with Qwen models
**I want to** automatic retry on timeouts
**So that** transient network issues don't break indexing

**Acceptance Criteria:**
- Exponential backoff retry logic
- Configurable retry attempts
- Proper error logging and recovery

### US-4: Optimize Qwen Model Parameters
**As a** GraphRAG user with Qwen models
**I want to** optimized parameters for GraphRAG tasks
**So that** Qwen performs better on entity extraction

**Acceptance Criteria:**
- Lower temperature for consistent formatting
- Appropriate max_tokens settings
- Reduced concurrent requests to prevent overload

## Technical Requirements

### TR-1: Timeout Configuration
- Default timeout: 120 seconds for Qwen models
- Configurable via model settings
- Different timeouts for chat vs embedding

### TR-2: Retry Logic
- Exponential backoff with jitter
- Maximum 3 retry attempts
- Only retry on timeout/network errors

### TR-3: Response Enhancement
- System instructions for precise formatting
- Debug logging of raw responses
- Response cleanup and validation

### TR-4: Parameter Optimization
- Temperature: 0.1 for consistency
- Max tokens: 4096 for completion
- Concurrent requests: 5-10 for Qwen

## Constraints

- Must maintain backward compatibility
- Only modify source code, not configuration files
- Keep existing interfaces intact
- Focus on Qwen-specific optimizations

## Out of Scope

- Changing GraphRAG's core prompt structure
- Modifying default configuration files
- Adding new model types
- Breaking changes to existing functionality
