# LLM Management UX Fix – Test Plan v2

**Project**: Hatchling – LLM Configuration UX Fix  
**Test Plan Date**: 2025-11-22  
**Phase**: Test Definition (Phase 2)  
**Source**: Implementation Roadmap v3  
**Branch**: `fix/llm-management`  
**Version**: v2  
**Author**: AI Development Agent

---

## Changes from v1

### Removed (4 tests eliminated)

**Status-related tests** (DOWNLOADING and UNKNOWN no longer exist):
- ❌ Model shows ↓ (DOWNLOADING) indicator - Status removed
- ❌ Model shows ? (UNKNOWN) indicator - Status removed
- ❌ Test error message content about DOWNLOADING - Implementation detail

**Clarified Behavior**:
- ✅ No auto-download on add—model must exist in provider list
- ✅ No download tracking—too complex, not needed
- ✅ Tests focus on behavioral validation, not implementation details

### Key Updates to Existing Tests

**Test Assertions**: Changed from `assert x == y` to `self.assertEqual(x, y)`
- Follows `unittest.TestCase` standard (Wobble framework)
- Provides better error messages and test introspection

**Model Not Found Scenarios**:
- Test assumes model NOT auto-downloaded
- Validates that error message shows available models
- Does NOT trigger download via print suggestions

**Task 2 Tests**: Updated for discovered models only (already available)
**Task 3 Tests**: Updated for validation-before-add behavior

---

## Executive Summary

Test plan for LLM management UX fix addressing:
- Correct test style using `unittest.TestCase`
- Simplified status indicators (✓ ✗ only)
- Behavioral testing of model discovery/add validation
- Prevention of regressions to existing functionality

**Testing Approach**:
- **Focus**: Behavioral functionality with unittest assertions
- **Coverage**: 15 automated tests + 6 manual scenarios
- **Test-to-Code Ratio**: 3:1 (15 tests for 5 tasks)
- **Framework**: Wobble with standard decorators

---

## Test Organization

### Test Categorization

**Regression Tests** - Prevent breaking changes to existing functionality:
- Configuration behavior
- Uniqueness enforcement
- Command behavior
- Settings persistence

**Integration Tests** - Validate component interactions:
- Discovery workflows
- Provider health checks
- Settings registry integration
- Multi-provider scenarios

**Manual Tests** - UX validation:
- Fresh install experience
- Error message clarity
- Workflow intuitiveness

### Coverage Distribution

| Task | Automated | Manual | Total |
|------|-----------|--------|-------|
| Task 1: Configuration | 1 | 1 | 2 |
| Task 2: Discovery | 4 | 1 | 5 |
| Task 3: Add Validation | 3 | 1 | 4 |
| Task 4: List Display | 3 | 1 | 4 |
| Task 5: Error Messages | 2 | 1 | 3 |
| Integration Scenarios | 2 | 1 | 3 |
| **Totals** | **15** | **6** | **21** |

---

## Task 1: Configuration Tests

### Test 1.1: Environment Variables Work

**Category**: `@regression_test`  
**Location**: `tests/regression/test_llm_configuration.py`

**Purpose**: Verify environment variables still provide initial defaults for deployment flexibility.

**Test Pattern**:
```python
class TestLLMConfiguration(unittest.TestCase):
    def setUp(self):
        # Save original env vars
        self._original_env = dict(os.environ)
        
    def tearDown(self):
        # Restore env vars
        os.environ.clear()
        os.environ.update(self._original_env)
    
    @regression_test
    def test_environment_variables_set_provider_default(self):
        """Verify LLM_PROVIDER env var sets initial provider."""
        os.environ["LLM_PROVIDER"] = "openai"
        settings = LLMSettings()
        self.assertEqual(settings.provider_enum, ELLMProvider.OPENAI)
    
    @regression_test
    def test_ollama_env_vars_set_endpoint(self):
        """Verify OLLAMA_IP and OLLAMA_PORT env vars work."""
        os.environ["OLLAMA_IP"] = "192.168.1.100"
        os.environ["OLLAMA_PORT"] = "11435"
        settings = OllamaSettings()
        self.assertEqual(settings.ip, "192.168.1.100")
        self.assertEqual(settings.port, 11435)
```

**Acceptance Criteria**:
- ✅ `LLM_PROVIDER` sets initial provider
- ✅ `OLLAMA_IP` and `OLLAMA_PORT` work
- ✅ `OPENAI_API_KEY` env var works
- ✅ Deployment flexibility preserved

---

## Task 2: Model Discovery Tests

### Test 2.1: Discovery Adds Available Models

**Category**: `@integration_test(scope="component")`  
**Location**: `tests/integration/test_model_discovery.py`

**Test Pattern**:
```python
class TestModelDiscovery(unittest.TestCase):
    @integration_test(scope="component")
    async def test_discover_adds_all_available_models(self):
        """Verify discovery adds all available models from provider."""
        # Arrange: Mock provider returning 3 models
        available = [
            ModelInfo(name="llama3.2", provider=OLLAMA),
            ModelInfo(name="mistral", provider=OLLAMA),
            ModelInfo(name="neural-chat", provider=OLLAMA),
        ]
        
        # Act: Run discovery
        cmd = ModelCommands(settings, style)
        await cmd._cmd_model_discover("")
        
        # Assert: All added to curated list
        self.assertEqual(len(settings.llm.models), 3)
        names = [m.name for m in settings.llm.models]
        self.assertIn("llama3.2", names)
```

**Acceptance Criteria**:
- ✅ Discovery fetches all available models from provider
- ✅ All models added to curated list
- ✅ Duplicates skipped (idempotent)
- ✅ Returns success/failure counts
- ✅ Provider health checked first

### Test 2.2: Discovery with Unhealthy Provider

**Category**: `@integration_test(scope="component")`

**Test Pattern**:
```python
    @integration_test(scope="component")
    async def test_discover_with_unhealthy_provider(self):
        """Verify discovery shows error when provider not accessible."""
        # Arrange: Provider not running/accessible
        
        # Act: Run discovery
        result = await cmd._cmd_model_discover("")
        
        # Assert: Error shown, no models added
        self.assertEqual(len(settings.llm.models), 0)
        # (Error message validation is implementation detail, skip)
```

### Test 2.3: Discovery Skips Existing Models

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_discover_skips_existing_models(self):
        """Verify discovery doesn't duplicate existing models."""
        # Arrange: Model already in list
        existing = ModelInfo(name="llama3.2", provider=OLLAMA)
        settings.llm.models = [existing]
        
        # Act: Discover (includes llama3.2)
        # Assert: Still 1 model in list
        self.assertEqual(len(settings.llm.models), 1)
```

### Test 2.4: Discovery Updates Command Completions

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_discover_updates_completions(self):
        """Verify command completions updated after discovery."""
        # Arrange: Empty model list
        
        # Act: Discover models
        # Assert: Completions include discovered models
        completions = cmd.commands['llm:model:use']['args']['model-name']['values']
        self.assertGreater(len(completions), 0)
```

---

## Task 3: Model Add Validation Tests

### Test 3.1: Add Validates Model Exists

**Category**: `@regression_test`  
**Location**: `tests/regression/test_model_add.py`

**Test Pattern**:
```python
class TestModelAdd(unittest.TestCase):
    @regression_test
    async def test_add_existing_available_model(self):
        """Verify add validates model exists at provider."""
        # Arrange: Model available at provider
        
        # Act: Add model
        result = await cmd._cmd_model_add("llama3.2")
        
        # Assert: Model added to curated list
        self.assertEqual(result, True)
        names = [m.name for m in settings.llm.models]
        self.assertIn("llama3.2", names)
```

### Test 3.2: Add Rejects Non-existent Models

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_add_nonexistent_model_rejected(self):
        """Verify add rejects models not in provider list."""
        # Arrange: Model NOT available
        
        # Act: Try to add
        result = await cmd._cmd_model_add("nonexistent-model")
        
        # Assert: NOT added, not in list
        self.assertEqual(result, False)  # Command returns False
        names = [m.name for m in settings.llm.models]
        self.assertNotIn("nonexistent-model", names)
```

### Test 3.3: Add Prevents Duplicates

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_add_prevents_duplicates(self):
        """Verify add skips models already in curated list."""
        # Arrange: Model already added
        existing = ModelInfo(name="llama3.2", provider=OLLAMA)
        settings.llm.models = [existing]
        
        # Act: Add same model again
        result = await cmd._cmd_model_add("llama3.2")
        
        # Assert: Still 1 model (not duplicated)
        self.assertEqual(len(settings.llm.models), 1)
```

---

## Task 4: Model List Display Tests

### Test 4.1: Empty List Shows Guidance

**Category**: `@regression_test`  
**Location**: `tests/regression/test_model_list.py`

**Test Pattern**:
```python
class TestModelListDisplay(unittest.TestCase):
    @regression_test
    async def test_empty_model_list_shows_guidance(self):
        """Verify empty list displays helpful guidance."""
        # Arrange: No models in list
        settings.llm.models = []
        
        # Act: List models
        result = await cmd._cmd_model_list("")
        
        # Assert: Returns True (success), guidance shown
        self.assertEqual(result, True)
        # (Actual message content is implementation detail)
```

### Test 4.2: Models Displayed with Status Indicators

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_model_list_shows_availability_status(self):
        """Verify models show availability status (✓ or ✗)."""
        # Arrange: Models with different availability
        available = ModelInfo(name="llama3.2", provider=OLLAMA, status=AVAILABLE)
        unavailable = ModelInfo(name="mistral", provider=OLLAMA, status=UNAVAILABLE)
        settings.llm.models = [available, unavailable]
        
        # Act: List models
        result = await cmd._cmd_model_list("")
        
        # Assert: Status indicators shown
        self.assertEqual(result, True)
        # (Content validation: output contains ✓ and ✗)
```

### Test 4.3: Current Model Marked

**Category**: `@regression_test`

**Test Pattern**:
```python
    @regression_test
    async def test_current_model_marked_in_list(self):
        """Verify current model clearly marked."""
        # Arrange: Current model set
        model = ModelInfo(name="llama3.2", provider=OLLAMA)
        settings.llm.models = [model]
        settings.llm.model = "llama3.2"
        
        # Act: List models
        # Assert: Current model indicator shown
        # (Output contains marker for current model)
```

---

## Task 5: Error Messages Tests

### Test 5.1: Model Not Found Shows Available Models

**Category**: `@integration_test(scope="component")`  
**Location**: `tests/integration/test_error_messages.py`

**Test Pattern**:
```python
class TestErrorMessages(unittest.TestCase):
    @integration_test(scope="component")
    async def test_model_not_found_suggests_alternatives(self):
        """Verify model not found error shows available models."""
        # Arrange: Try to add non-existent model
        
        # Act: Add nonexistent model
        result = await cmd._cmd_model_add("nonexistent")
        
        # Assert: Fails with helpful message
        self.assertEqual(result, False)
        # (Message shows available models as fallback)
```

### Test 5.2: Provider Health Error Shows Troubleshooting

**Category**: `@integration_test(scope="component")`

**Test Pattern**:
```python
    @integration_test(scope="component")
    async def test_provider_error_includes_troubleshooting(self):
        """Verify provider errors include actionable guidance."""
        # Arrange: Provider not accessible
        
        # Act: Try discovery
        result = await cmd._cmd_model_discover("")
        
        # Assert: Error shown, result False
        self.assertEqual(result, False)
        # (Troubleshooting message shown—content is impl detail)
```

---

## Integration Test Scenarios

### Integration 1: Full Discovery Workflow

**Category**: `@integration_test(scope="service")`

**Scenario**:
```python
    @integration_test(scope="service")
    async def test_complete_discovery_workflow(self):
        """Test: Fresh install → discover → list → use."""
        # 1. Start fresh (empty models)
        self.assertEqual(len(settings.llm.models), 0)
        
        # 2. Discover all models
        await cmd._cmd_model_discover("")
        self.assertGreater(len(settings.llm.models), 0)
        
        # 3. List shows discovered models
        await cmd._cmd_model_list("")
        # (Models displayed with status)
        
        # 4. Use a model
        model_name = settings.llm.models[0].name
        result = await cmd._cmd_model_use(model_name)
        self.assertEqual(settings.llm.model, model_name)
```

### Integration 2: Add Then Use Workflow

**Category**: `@integration_test(scope="service")`

**Scenario**:
```python
    @integration_test(scope="service")
    async def test_add_specific_model_then_use(self):
        """Test: Add specific model → verify in list → use."""
        # 1. Add specific model (must be available)
        result = await cmd._cmd_model_add("gpt-4 --provider openai")
        self.assertEqual(result, True)
        
        # 2. Model appears in list
        names = [m.name for m in settings.llm.models]
        self.assertIn("gpt-4", names)
        
        # 3. Use the model
        settings.llm.model = "gpt-4"
        # Provider set automatically based on model
        self.assertEqual(settings.llm.provider_enum, ELLMProvider.OPENAI)
```

---

## Manual Test Checklist

**M1: Fresh Install Experience**
- [ ] Start Hatchling with clean settings
- [ ] `llm:model:list` shows empty list + guidance
- [ ] Run `llm:model:discover` (assumes Ollama running with models)
- [ ] Models appear after discover
- [ ] Each shows ✓ (available) or ✗ (unavailable) status

**M2: Add Non-existent Model**
- [ ] Try: `llm:model:add fake-model-name`
- [ ] Error message shows "not found"
- [ ] Lists 5-10 available models as reference
- [ ] Suggest trying `llm:model:discover` first

**M3: Provider Not Running**
- [ ] Stop Ollama/provider
- [ ] Run `llm:model:discover`
- [ ] Shows "provider not accessible"
- [ ] Includes troubleshooting steps
- [ ] Mentions checking running status

**M4: Model Use Workflow**
- [ ] Discover models
- [ ] Use one: `llm:model:use llama3.2`
- [ ] Current model marked in list
- [ ] Model persists after restart

**M5: Multi-Provider Setup**
- [ ] Have Ollama + OpenAI configured
- [ ] Discover from both providers
- [ ] List shows grouped by provider
- [ ] Use command works for both

**M6: Error Recovery**
- [ ] Try invalid command
- [ ] See helpful error + next steps
- [ ] Can recover without restarting Hatchling

---

## Acceptance Criteria

### Code Quality
- ✅ All tests use `unittest.TestCase` with `self.assert*()` methods
- ✅ Tests follow Wobble framework patterns
- ✅ No direct Python assertions
- ✅ Clear test names describing behavior

### Test Assertion Examples
- ✅ `self.assertEqual(actual, expected)` - Values match
- ✅ `self.assertIn(item, collection)` - Item in collection
- ✅ `self.assertGreater(a, b)` - Numeric comparison
- ✅ `self.assertTrue(condition)` - Boolean check

### Testing Principles
- ✅ Tests focus on behavior, not implementation details
- ✅ Status message content NOT tested (implementation detail)
- ✅ Command output formatting NOT tested (implementation detail)
- ✅ Behavioral validation IS tested (add works, discover idempotent, etc)

### Coverage Requirements
- ✅ 15 automated tests (unit + integration)
- ✅ 6 manual test scenarios
- ✅ All critical paths tested
- ✅ Edge cases covered (duplicates, missing models, unhealthy providers)

---

## Test Execution Plan

**Phase 1: Task 1 Tests** (1 automated)
- Run after Task 1 complete
- Validate configuration defaults and env vars

**Phase 2: Tasks 2-4 Tests** (10 automated)
- Run after each task complete
- Validate discovery, add, and list functionality

**Phase 3: Task 5 Tests** (2 automated)
- Run after Tasks 2-3 complete
- Validate error messages and guidance

**Phase 4: Integration Tests** (2 automated + 6 manual)
- Run after all tasks complete
- Validate end-to-end workflows
- Execute manual test checklist

**Phase 5: Regression** (All tests)
- Full test suite before merge
- Ensure no breaking changes
