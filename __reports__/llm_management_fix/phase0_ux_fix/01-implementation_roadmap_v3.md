# LLM Management UX Fix – Implementation Roadmap v3

**Project**: Hatchling – LLM Configuration UX Fix  
**Roadmap Date**: 2025-11-22  
**Phase**: Implementation  
**Source**: Implementation Roadmap v2 + Critical Feedback  
**Branch**: `fix/llm-management`  
**Target**: Bug fix release (patch version bump)  
**Timeline**: 10-15 hours (1.25-2 days)  
**Approach**: Incremental fixes with testing at each step

---

## Executive Summary

This roadmap addresses critical clarifications to v2:

**Testing Approach Change**: Use `unittest.TestCase` with `self.assert*()` methods following Wobble framework patterns, not bare Python assertions.

**Status Indicators Simplification**: Remove `DOWNLOADING` status and `?` (Unknown) indicator. Only use actual statuses:
- `✓ AVAILABLE` - Model confirmed working at provider
- `✗ UNAVAILABLE` - Model configured but not accessible

**Model Discovery Behavior**: `llm:model:discover` and `llm:model:add` only work with **already-available** models:
- For **Ollama**: Model must already be pulled locally via `ollama pull`
- For **OpenAI**: Model must be in API's available list
- **Critical**: Documentation and tutorials must include manual pull step before discovery

**Provider Commands Consolidation**: Analyze whether `llm:provider:supported` provides value vs `llm:provider:status`.

**Documentation Handoff**: Task 6 requires stakeholder interaction—defer actual writing to later phase.

---

## Key Changes from v2

| Aspect | v2 | v3 | Reason |
|--------|----|----|--------|
| **Test Assertions** | Direct Python `assert` | `self.assert*()` methods | Wobble/unittest standard |
| **Model Statuses** | AVAILABLE, UNAVAILABLE, DOWNLOADING, UNKNOWN | AVAILABLE, UNAVAILABLE only | No download tracking; unknown never occurs |
| **Status Indicators** | ✓ ✗ ↓ ? | ✓ ✗ only | Simplified, reflects reality |
| **Completer Values** | Empty list comment | Method reference | Actual model list for completions |
| **Discover Behavior** | Auto-download on add | Manual pull first, then discover | User controls when to pull |
| **Task 6** | Full autonomous docs | Deferred—requires stakeholder input | Avoid incorrect documentation |

---

## Git Workflow

**Branch Strategy**:
```
main (production)
  └── fix/llm-management (fix branch)
      ├── task/1-clean-defaults
      ├── task/2-discovery-command
      ├── task/3-enhance-add
      ├── task/4-list-display
      └── task/5-error-messages
```

**Merge Criteria**:
- Task → Fix branch: Success gates met, task tests pass
- Fix branch → main: All tasks complete, all tests pass (unit + integration + manual)

---

## Task Overview

**5 focused tasks, 10-15 hours total** (Task 6 deferred):

| Task | Description | Effort | Pre-conditions |
|------|-------------|--------|----------------|
| 1 | Clean Up Default Configuration | 1-2h | None |
| 2 | Implement Model Discovery Command | 4-6h | Task 1 |
| 3 | Enhance Model Add Command | 2-3h | Task 2 |
| 4 | Improve Model List Display | 2-3h | Task 1 |
| 5 | Better Error Messages | 1-2h | Tasks 2, 3 |

---

## Task 1: Clean Up Default Configuration

**Branch**: `task/1-clean-defaults`  
**Effort**: 1-2 hours

### Goal
Remove hard-coded phantom models while preserving environment variable support.

### Files to Modify
1. `hatchling/config/llm_settings.py` - Remove phantom model list
2. `hatchling/config/ollama_settings.py` - Document env var precedence
3. `hatchling/config/openai_settings.py` - Document env var precedence
4. `hatchling/config/languages/en.toml` - Update descriptions

### Implementation Notes

```python
# llm_settings.py - Remove hard-coded models
models: List[ModelInfo] = Field(
    default_factory=list,  # ← Empty list, no phantoms
    # Keep env var support:
    # Persistent settings override this
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
)

# Keep env var support for provider (Ollama by default)
provider_enum: ELLMProvider = Field(
    default_factory=lambda: LLMSettings.to_provider_enum(
        os.environ.get("LLM_PROVIDER", "ollama")
    ),
    # Config precedence: Persistent > Env Var > Code Default
)

# Model field: make optional, no env var default
model: Optional[str] = Field(
    default=None,  # ← Users must explicitly select/discover
)
```

### Success Gates
- ✅ Hard-coded model list removed
- ✅ Default `models` = empty list
- ✅ Default `model` = None
- ✅ Environment variable support preserved for deployment
- ✅ Existing tests pass

---

## Task 2: Implement Model Discovery Command

**Branch**: `task/2-discovery-command`  
**Effort**: 4-6 hours  
**Pre-conditions**: Task 1

### Goal
Add `llm:model:discover` command to bulk-add available models to curated list.

### Key Behavior Change from v2
**Discovery only works with already-available models.** No auto-download:
- **Ollama**: User must `ollama pull model-name` first
- **OpenAI**: Model must be in API's list (user has API access)

### Files to Modify
1. `hatchling/ui/model_commands.py` - Add command + helpers

### Implementation Notes

```python
# Add to command registry:
'llm:model:discover': {
    'handler': self._cmd_model_discover,
    'description': 'Discover available models from provider and add to curated list',
    'is_async': True,
    'args': {
        '--provider': {
            'completer_type': 'suggestions',
            'values': self.settings.llm.provider_names,
            'required': False
        }
    }
}

# Command handler logic (pseudocode):
async def _cmd_model_discover(self, args: str) -> bool:
    # Parse --provider flag (or use current)
    # Check provider health
    # List available models from provider API
    # Add to curated list (with uniqueness check)
    # Skip any already in list
    # Update command completions
    # Return success/failure count
```

### Success Gates
- ✅ Command lists all available models from provider
- ✅ Adds each to curated list (skips duplicates)
- ✅ Provider health check before discovery
- ✅ Clear feedback: added count, skipped duplicates, failures
- ✅ `--provider` flag works
- ✅ Command completions updated after discovery
- ✅ Tests use `self.assert*()` methods (unittest style)

### Test Strategy
Use `unittest.TestCase` with standard assertions:

```python
class TestModelDiscovery(unittest.TestCase):
    @integration_test(scope="component")
    def test_discover_adds_all_available_models(self):
        # Arrange: Mock provider with 3 available models
        # Act: Run discover
        # Assert: self.assertEqual(len(settings.llm.models), 3)
        
    @integration_test(scope="component")
    def test_discover_skips_existing_models(self):
        # Arrange: 1 model already in list, 2 new available
        # Act: Run discover
        # Assert: self.assertEqual(len(settings.llm.models), 3)
        # Assert: skipped_count == 1
        
    @regression_test
    def test_discover_with_unhealthy_provider(self):
        # Arrange: Provider not accessible
        # Act: Run discover
        # Assert: self.assertTrue('not accessible' in output)
```

---

## Task 3: Enhance Model Add Command

**Branch**: `task/3-enhance-add`  
**Effort**: 2-3 hours  
**Pre-conditions**: Task 2

### Goal
Add validation before adding individual models to curated list.

### Key Behavior (Updated from v2)
**Add validates the model exists** at the provider before adding:
- Check model in available list (no download triggered)
- Reject if not found
- Suggest similar models or available models as fallback

### Files to Modify
1. `hatchling/ui/model_commands.py` - Update `_cmd_model_add` validation

### Implementation Notes

```python
# Validation logic (pseudocode):
async def _cmd_model_add(self, args: str) -> bool:
    # Parse model-name and optional --provider
    # Determine provider (from flag or current)
    # Get provider health
    # If unhealthy: show troubleshooting
    
    # Fetch available models from provider
    # Check if model in available list
    # If found: add to curated list (skip if duplicate)
    # If NOT found:
    #   - Show "model not found" message
    #   - List available models
    #   - DON'T download—user must do that manually first
```

### Success Gates
- ✅ Validates model exists in provider's available list
- ✅ Rejects models not found (no download triggered)
- ✅ Shows available models when model not found
- ✅ Prevents duplicates
- ✅ Changes persisted to settings
- ✅ `--provider` flag works
- ✅ Error handling for inaccessible provider

### Test Strategy

```python
class TestModelAdd(unittest.TestCase):
    @regression_test
    def test_add_existing_available_model(self):
        # Arrange: Model available at provider
        # Act: Add model
        # Assert: self.assertIn(model, settings.llm.models)
        
    @regression_test
    def test_add_nonexistent_model_rejected(self):
        # Arrange: Model NOT available at provider
        # Act: Try to add
        # Assert: self.assertNotIn(model, settings.llm.models)
        # Assert: self.assertIn('not found', output)
        
    @integration_test(scope="component")
    def test_add_prevents_duplicates(self):
        # Arrange: Model already in list
        # Act: Add same model again
        # Assert: len(settings.llm.models) == 1 (unchanged)
```

---

## Task 4: Improve Model List Display

**Branch**: `task/4-list-display`  
**Effort**: 2-3 hours  
**Pre-conditions**: Task 1

### Goal
Show curated models with availability status (2 statuses only).

### Key Changes from v2
**Remove `DOWNLOADING` and `UNKNOWN` statuses:**
- Only use `AVAILABLE` (✓) and `UNAVAILABLE` (✗)
- Don't check download status—user responsibility
- Unknown status never occurs (all in list are either available or not)

### Files to Modify
1. `hatchling/ui/model_commands.py` - Update `_cmd_model_list` method

### Implementation Notes

```python
# Display logic (pseudocode):
async def _cmd_model_list(self) -> bool:
    if not models:
        # Show helpful guidance to discover/add models
        return True
    
    # Group by provider
    # For each provider's models:
    #   Check health (skip if unhealthy)
    #   Fetch available models from provider
    #   For each curated model:
    #     if in available list: status = ✓ AVAILABLE
    #     else: status = ✗ UNAVAILABLE
    #   Mark current model with indicator
    #   Show model name + status
    
    # Legend (simplified):
    # ✓ Available   - Ready to use
    # ✗ Unavailable - Configured but not accessible
```

### Success Gates
- ✅ Empty list shows helpful guidance
- ✅ Models grouped by provider
- ✅ Status indicators: ✓ and ✗ only
- ✅ Current model clearly marked
- ✅ Sorted alphabetically within provider
- ✅ Clear, readable formatting
- ✅ Legend explains statuses

### Test Strategy

```python
class TestModelListDisplay(unittest.TestCase):
    @regression_test
    def test_model_list_empty_shows_guidance(self):
        # Arrange: No models in list
        # Act: Run list command
        # Assert: self.assertIn('discover', output)
        
    @regression_test
    def test_model_list_shows_status_indicators(self):
        # Arrange: Models with different availability
        # Act: Run list command
        # Assert: self.assertIn('✓', output)  # Available marker
        # Assert: self.assertIn('✗', output)  # Unavailable marker
        
    @regression_test
    def test_model_list_marks_current_model(self):
        # Arrange: Current model set
        # Act: Run list command
        # Assert: current marker shown for active model
```

---

## Task 5: Better Error Messages

**Branch**: `task/5-error-messages`  
**Effort**: 1-2 hours  
**Pre-conditions**: Tasks 2, 3

### Goal
Improve error messages with actionable guidance.

### Files to Modify
1. `hatchling/ui/model_commands.py` - Enhanced error messages
2. `hatchling/ui/cli_chat.py` - Provider initialization errors

### Implementation Notes

```python
# Provider initialization error in cli_chat.py:
try:
    provider = ProviderRegistry.get_provider(settings.llm.provider_enum)
except Exception as e:
    msg = f"Failed to initialize {provider.value}: {e}\n"
    msg += "Troubleshooting:\n"
    
    if provider == OLLAMA:
        msg += "  1. Check if Ollama is running\n"
        msg += "  2. Verify IP/Port:\n"
        msg += f"     settings:set ollama:ip <ip>\n"
        msg += f"     settings:set ollama:port <port>\n"
    elif provider == OPENAI:
        msg += "  1. Verify OPENAI_API_KEY is set\n"
        msg += "  2. Check internet connection\n"
    
    logger.warning(msg)

# Model not found error in model_commands.py:
# When model not in available list:
# - Show "Model not found" message
# - List 5-10 available models
# - DON'T suggest auto-download
```

### Success Gates
- ✅ Provider errors include troubleshooting steps
- ✅ Model not found shows available models
- ✅ All errors include actionable next steps
- ✅ Provider-specific guidance (Ollama vs OpenAI)
- ✅ Clear formatting with symbols (✓ ✗)

---

## Task 6: Update Documentation (Deferred)

**Status**: Deferred to stakeholder interaction phase

This task requires close collaboration with users/stakeholders to ensure documentation reflects:
- Actual workflow post-implementation
- Manual `ollama pull` step before discovery
- Provider configuration precedence
- Troubleshooting guidance

**Action**: Plan stakeholder review meeting after Tasks 1-5 complete and manual testing validates the workflow.

---

## Critical Documentation Updates (Post-Implementation)

Once Tasks 1-5 are done, documentation must clarify:

1. **For Ollama Users**:
   - Pull models locally first: `ollama pull model-name`
   - Then discover: `llm:model:discover`
   - Or add directly: `llm:model:add model-name`

2. **For OpenAI Users**:
   - Set API key: `settings:set openai:api_key ...`
   - Discover available models: `llm:model:discover --provider openai`
   - Or add specific model: `llm:model:add gpt-4 --provider openai`

3. **Configuration Precedence**:
   - Persistent Settings (`.toml` file) > Environment Variables > Code Defaults
   - Environment variables still work for Docker/CI/CD deployments

4. **Provider Commands**:
   - Revisit `llm:provider:supported` vs `llm:provider:status` for redundancy
   - May consolidate if no unique value

---

## Acceptance Criteria

### Code Quality
- ✅ Tests use `unittest.TestCase` with `self.assert*()` methods
- ✅ All task success gates met
- ✅ No breaking changes to existing commands
- ✅ Environment variable support preserved

### Testing Coverage
- ✅ Unit tests for validation logic
- ✅ Integration tests for command workflows
- ✅ Manual tests for UX clarity
- ✅ All tests pass with clean output

### User Experience
- ✅ Clear error messages with next steps
- ✅ Model discovery works as documented
- ✅ Status indicators accurate and simple
- ✅ No confusion about phantom/unavailable models

---

## Parallel Work Opportunities

- Tasks 2 and 4 can develop in parallel (after Task 1)
- Task 5 can run alongside Tasks 2-3
- Task 6 deferred—start stakeholder engagement while implementing Tasks 1-5
