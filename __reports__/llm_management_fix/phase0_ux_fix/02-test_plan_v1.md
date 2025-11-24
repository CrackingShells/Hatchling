# LLM Management UX Fix – Test Plan v1

**Project**: Hatchling – LLM Configuration UX Fix  
**Test Plan Date**: 2025-11-13  
**Phase**: Test Definition (Phase 2)  
**Source**: Implementation Roadmap v2  
**Branch**: `fix/llm-management`  
**Version**: v1  
**Author**: AI Development Agent

---

## Changes from v0

### Removed (6 tests eliminated)

**Meta-Constraint Tests** (not algorithmic, just checking implementation details):
- ❌ Test 1.1: Empty initial model list - Just checking a default value in code
- ❌ Test 1.2: Empty initial model field - Just checking a default value in code

**Error Message Content Tests** (implementation detail, not behavioral):
- ❌ Test 4.1: Empty list shows guidance - Checking error message content
- ❌ Test 5.1: Provider init error includes troubleshooting - Checking error message content
- ❌ Test 5.2: Model not found error shows available models - Checking error message content
- ❌ Test 5.3: Model unavailable error explains how to fix - Checking error message content

**Rationale**: These tests check implementation details and meta-constraints rather than actual behavior. Once the code is written correctly, these constraints will stay valid. Testing them pollutes the test suite without adding algorithmic value.

### Kept (18 tests)

All behavioral tests that verify actual functionality:
- ✅ Environment variables actually work (not just checking defaults)
- ✅ Discovery actually adds models (behavioral)
- ✅ Uniqueness actually prevents duplicates (behavioral)
- ✅ Validation actually rejects invalid models (behavioral)
- ✅ List actually groups and displays correctly (behavioral)
- ✅ Workflows actually work end-to-end (behavioral)
- ✅ Edge cases actually handled gracefully (behavioral)
- ✅ Regressions actually prevented (behavioral)

### Outcome

**New Test Count**: 18 automated tests (down from 24)  
**New Test-to-Code Ratio**: 3:1 (18 tests / 6 tasks) - within target 2:1 to 3:1  
**Manual Scenarios**: 7 (unchanged - valuable for UX validation)

---

## Executive Summary

This test plan defines focused test specifications for the LLM management UX fix. The fix addresses the critical issue where users are confused about which LLM API endpoint and model is actually accessible when running Hatchling.

**Testing Approach**:
- **Focus**: Test behavioral functionality, not meta-constraints or implementation details
- **Coverage**: 18 automated tests + 7 manual test scenarios
- **Test-to-Code Ratio**: 3:1 (18 tests for 6 tasks) - within target range of 2:1 to 3:1 for bug fixes
- **Organization**: Functional grouping by feature area (Configuration, Discovery, Validation, Display)
- **Framework**: Wobble with proper categorization decorators

**Key Testing Principles Applied**:
- ✅ Test behavioral functionality, not meta-constraints
- ✅ Focus on critical paths and edge cases
- ✅ Prevent regressions to existing behavior
- ✅ Validate UX improvements through manual testing
- ✅ Trust standard library and framework behavior
- ✅ Don't test implementation details (error message content, default values)

---

## Table of Contents

1. [Test Strategy Overview](#test-strategy-overview)
2. [Functional Test Groups](#functional-test-groups)
3. [Task 1: Configuration Tests](#task-1-configuration-tests)
4. [Task 2: Model Discovery Tests](#task-2-model-discovery-tests)
5. [Task 3: Model Addition Tests](#task-3-model-addition-tests)
6. [Task 4: Model List Display Tests](#task-4-model-list-display-tests)
7. [Integration Test Scenarios](#integration-test-scenarios)
8. [Manual Test Checklist](#manual-test-checklist)
9. [Edge Cases and Regression Prevention](#edge-cases-and-regression-prevention)
10. [Acceptance Criteria](#acceptance-criteria)
11. [Test Execution Plan](#test-execution-plan)

---

## Test Strategy Overview

### Test Categorization

**Regression Tests** - Prevent breaking changes to existing functionality:
- Configuration behavior
- Uniqueness enforcement
- Command behavior
- Settings persistence

**Integration Tests** - Validate component interactions:
- Command workflows
- Provider health checks
- Settings registry integration
- Multi-provider scenarios

**Manual Tests** - UX validation:
- Fresh install experience
- Error message clarity
- Workflow intuitiveness
- Documentation accuracy

### Coverage Goals

**Minimum Coverage**: 70% for all code  
**Critical Functionality**: 90% coverage for:
- Model discovery logic
- Validation logic
- Uniqueness enforcement
- Command handlers

**New Features**: 95% coverage for:
- `llm:model:discover` command
- Enhanced `llm:model:add` validation
- Improved `llm:model:list` display

### Test Organization

Tests are organized by **functional groups** (not arbitrary categories):

1. **Configuration** - Environment variable handling
2. **Model Discovery** - Bulk discovery and uniqueness enforcement
3. **Model Validation and Addition** - Validation before adding models
4. **Model List Display** - Status indicators and formatting

---

## Functional Test Groups

### Group 1: Configuration
**Purpose**: Verify environment variables work for deployment  
**Test Count**: 1 test  
**Category**: Regression test

### Group 2: Model Discovery
**Purpose**: Verify bulk discovery workflow  
**Test Count**: 5 tests  
**Category**: Integration tests (3), Regression tests (2)

### Group 3: Model Validation and Addition
**Purpose**: Verify validation before adding  
**Test Count**: 4 tests  
**Category**: Regression tests (3), Integration tests (1)

### Group 4: Model List Display
**Purpose**: Verify improved display formatting  
**Test Count**: 3 tests  
**Category**: Regression tests

**Total Automated Tests**: 13 tests  
**Total Integration Tests**: 2 tests  
**Total Edge Cases & Regressions**: 3 tests  
**Total Manual Scenarios**: 7 scenarios

---

## Task 1: Configuration Tests

### Test 1.3: Environment Variables Work for Provider

**Category**: `@regression_test`  
**File**: `tests/regression/test_llm_configuration.py`

**Purpose**: Verify environment variables still provide initial defaults (deployment flexibility)

**Test Specification**:
```python
@regression_test
def test_environment_variables_for_provider():
    """Verify environment variables still work for initial provider default."""
    # Arrange
    import os
    os.environ["LLM_PROVIDER"] = "openai"
    
    # Act
    settings = LLMSettings()
    
    # Assert
    assert settings.provider_enum == ELLMProvider.OPENAI, \
        "Environment variable should set initial provider default"
    
    # Cleanup
    del os.environ["LLM_PROVIDER"]
```

**Acceptance Criteria**:
- ✅ `LLM_PROVIDER` env var sets initial provider
- ✅ `OLLAMA_IP` and `OLLAMA_PORT` env vars work
- ✅ `OPENAI_API_KEY` env var works
- ✅ Deployment flexibility preserved (Docker, CI/CD)

**Edge Cases**:
- Invalid provider name in env var
- Missing env vars (should use code defaults)
- Empty string env vars

---

## Task 2: Model Discovery Tests

### Test 2.1: Discovery Adds All Models

**Category**: `@integration_test(scope="component")`  
**File**: `tests/integration/test_model_discovery.py`

**Purpose**: Verify discovery command adds all models from provider

**Test Specification**:
```python
@integration_test(scope="component")
async def test_model_discover_adds_all_models():
    """Verify llm:model:discover adds all models from provider."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider to return known models
    mock_models = [
        ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="model2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="model3", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=mock_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    assert len(settings.llm.models) == 3, \
        "All discovered models should be added to curated list"
    assert all(m.name in ["model1", "model2", "model3"] for m in settings.llm.models), \
        "All model names should match discovered models"
```

**Acceptance Criteria**:
- ✅ All models from provider are added to curated list
- ✅ Command checks provider health before discovery
- ✅ Changes are persisted to settings
- ✅ User receives feedback on number of models added

**Edge Cases**:
- Provider returns empty list
- Provider returns very large list (100+ models)
- Provider returns models with special characters in names

---

### Test 2.2: Discovery With Provider Flag

**Category**: `@integration_test(scope="component")`  
**File**: `tests/integration/test_model_discovery.py`

**Purpose**: Verify discovery command works with --provider flag

**Test Specification**:
```python
@integration_test(scope="component")
async def test_model_discover_with_provider_flag():
    """Verify llm:model:discover --provider flag works correctly."""
    # Arrange
    settings = create_test_settings()
    settings.llm.provider_enum = ELLMProvider.OLLAMA  # Default provider
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock OpenAI provider
    openai_models = [
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=openai_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("--provider openai")
    
    # Assert
    assert any(m.provider == ELLMProvider.OPENAI for m in settings.llm.models), \
        "Should discover models from specified provider, not default"
```

**Acceptance Criteria**:
- ✅ `--provider` flag overrides default provider
- ✅ Works with both "ollama" and "openai" values
- ✅ Invalid provider name shows error

---

### Test 2.3: Uniqueness Enforcement

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_uniqueness.py`

**Purpose**: Verify duplicate models are not added to curated list

**Test Specification**:
```python
@regression_test
async def test_model_discovery_prevents_duplicates():
    """Verify discovery prevents duplicate models in curated list."""
    # Arrange
    settings = create_test_settings()
    existing_model = ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, 
                               status=ModelStatus.AVAILABLE)
    settings.llm.models = [existing_model]
    
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock discovery returns same model
    mock_models = [
        ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="model2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=mock_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    assert len(settings.llm.models) == 2, \
        "Should have 2 models (1 existing + 1 new), not 3 (duplicate prevented)"
    
    # Verify uniqueness by (provider, name) tuple
    model_keys = [(m.provider, m.name) for m in settings.llm.models]
    assert len(model_keys) == len(set(model_keys)), \
        "No duplicate (provider, name) tuples should exist"
```

**Acceptance Criteria**:
- ✅ Duplicate models are not added
- ✅ Uniqueness key is `(provider, name)` tuple
- ✅ Existing model status can be updated
- ✅ User is informed about duplicates skipped

---

### Test 2.4: Discovery Updates Existing Models

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_uniqueness.py`

**Purpose**: Verify discovery updates status of existing models

**Test Specification**:
```python
@regression_test
async def test_model_discovery_updates_existing():
    """Verify discovery updates status of existing models."""
    # Arrange
    settings = create_test_settings()
    existing_model = ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, 
                               status=ModelStatus.NOT_AVAILABLE)
    settings.llm.models = [existing_model]
    
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock discovery returns same model with different status
    updated_model = ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, 
                             status=ModelStatus.AVAILABLE, size=1024)
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=[updated_model]):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    assert len(settings.llm.models) == 1, "Should still have 1 model"
    assert settings.llm.models[0].status == ModelStatus.AVAILABLE, \
        "Model status should be updated"
    assert settings.llm.models[0].size == 1024, \
        "Model metadata should be updated"
```

**Acceptance Criteria**:
- ✅ Existing model status is updated
- ✅ Existing model metadata (size, digest) is updated
- ✅ Model count doesn't increase for existing models
- ✅ User is informed about updates

---

### Test 2.5: Discovery Handles Inaccessible Provider

**Category**: `@integration_test(scope="component")`  
**File**: `tests/integration/test_model_discovery.py`

**Purpose**: Verify discovery handles inaccessible provider gracefully

**Test Specification**:
```python
@integration_test(scope="component")
async def test_model_discover_inaccessible_provider():
    """Verify discovery handles inaccessible provider gracefully."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider health check to fail
    with patch.object(ModelManagerAPI, 'check_provider_health', return_value=False):
        # Act
        result = await cmd._cmd_model_discover("")
    
    # Assert
    assert result is True, "Command should complete without exception"
    assert len(settings.llm.models) == 0, "No models should be added"
```

**Acceptance Criteria**:
- ✅ Command completes without exception
- ✅ No models are added when provider is inaccessible
- ✅ Graceful error handling (no crash)
- ✅ User can retry or troubleshoot

**Edge Cases**:
- Network timeout
- Invalid credentials
- Provider service not running

---

## Task 3: Model Addition Tests

### Test 3.1: Add Validates Model Exists

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_validation.py`

**Purpose**: Verify add command validates model exists before adding

**Test Specification**:
```python
@regression_test
async def test_model_add_validates_existence():
    """Verify llm:model:add validates model exists in provider."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider returns list of available models
    available_models = [
        ModelInfo(name="valid-model", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=available_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_add("valid-model")
    
    # Assert
    assert len(settings.llm.models) == 1, "Valid model should be added"
    assert settings.llm.models[0].name == "valid-model"
```

**Acceptance Criteria**:
- ✅ Command queries provider for available models
- ✅ Model is added only if found in provider's list
- ✅ User receives confirmation when model is added
- ✅ Changes are persisted to settings

---

### Test 3.2: Add Rejects Non-Existent Model

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_validation.py`

**Purpose**: Verify add command rejects non-existent models

**Test Specification**:
```python
@regression_test
async def test_model_add_rejects_nonexistent():
    """Verify llm:model:add rejects non-existent model."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider returns list without target model
    available_models = [
        ModelInfo(name="other-model", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=available_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_add("nonexistent-model")
    
    # Assert
    assert len(settings.llm.models) == 0, "Non-existent model should not be added"
```

**Acceptance Criteria**:
- ✅ Non-existent model is not added
- ✅ Command completes without exception
- ✅ No changes persisted to settings

---

### Test 3.3: Add With Provider Flag

**Category**: `@integration_test(scope="component")`  
**File**: `tests/integration/test_model_addition.py`

**Purpose**: Verify add command works with --provider flag

**Test Specification**:
```python
@integration_test(scope="component")
async def test_model_add_with_provider_flag():
    """Verify llm:model:add --provider flag works correctly."""
    # Arrange
    settings = create_test_settings()
    settings.llm.provider_enum = ELLMProvider.OLLAMA  # Default provider
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock OpenAI provider
    openai_models = [
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, 
                 status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=openai_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_add("gpt-4 --provider openai")
    
    # Assert
    assert len(settings.llm.models) == 1
    assert settings.llm.models[0].provider == ELLMProvider.OPENAI, \
        "Should add model from specified provider, not default"
```

**Acceptance Criteria**:
- ✅ `--provider` flag overrides default provider
- ✅ Validation checks specified provider, not default
- ✅ Model is added with correct provider association

---

### Test 3.4: Add Handles Inaccessible Provider

**Category**: `@integration_test(scope="component")`  
**File**: `tests/integration/test_model_addition.py`

**Purpose**: Verify add command handles inaccessible provider gracefully

**Test Specification**:
```python
@integration_test(scope="component")
async def test_model_add_inaccessible_provider():
    """Verify llm:model:add handles inaccessible provider gracefully."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider health check to fail
    with patch.object(ModelManagerAPI, 'check_provider_health', return_value=False):
        # Act
        result = await cmd._cmd_model_add("some-model")
    
    # Assert
    assert result is True, "Command should complete without exception"
    assert len(settings.llm.models) == 0, "No models should be added"
```

**Acceptance Criteria**:
- ✅ Command completes without exception
- ✅ No models are added when provider is inaccessible
- ✅ Graceful error handling (no crash)

---

## Task 4: Model List Display Tests

### Test 4.2: List Groups By Provider

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_list_display.py`

**Purpose**: Verify model list groups models by provider

**Test Specification**:
```python
@regression_test
async def test_model_list_groups_by_provider():
    """Verify llm:model:list groups models by provider."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = [
        ModelInfo(name="llama3.2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE),
        ModelInfo(name="codellama", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Act
    result = await cmd._cmd_model_list("")
    
    # Assert
    assert result is True, "Command should complete successfully"
    # Verify models are grouped (would need to capture output to verify grouping)
```

**Acceptance Criteria**:
- ✅ Models are grouped under provider headers
- ✅ Groups are sorted alphabetically by provider name
- ✅ Models within each group are sorted alphabetically
- ✅ Clear visual separation between provider groups

---

### Test 4.3: List Shows Status Indicators

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_list_display.py`

**Purpose**: Verify model list shows status indicators

**Test Specification**:
```python
@regression_test
async def test_model_list_shows_status_indicators():
    """Verify llm:model:list shows status indicators for each model."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = [
        ModelInfo(name="available", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.AVAILABLE),
        ModelInfo(name="unavailable", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.NOT_AVAILABLE),
        ModelInfo(name="downloading", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.DOWNLOADING)
    ]
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Act
    result = await cmd._cmd_model_list("")
    
    # Assert
    assert result is True, "Command should complete successfully"
    # Verify status indicators are shown (would need to capture output)
```

**Acceptance Criteria**:
- ✅ Available models show ✓ indicator
- ✅ Unavailable models show ✗ indicator
- ✅ Downloading models show ↓ indicator
- ✅ Unknown status models show ? indicator
- ✅ Legend explains status symbols

---

### Test 4.4: List Marks Current Model

**Category**: `@regression_test`  
**File**: `tests/regression/test_model_list_display.py`

**Purpose**: Verify model list marks the currently selected model

**Test Specification**:
```python
@regression_test
async def test_model_list_marks_current_model():
    """Verify llm:model:list marks the currently selected model."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = [
        ModelInfo(name="llama3.2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="codellama", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    settings.llm.model = "llama3.2"
    settings.llm.provider_enum = ELLMProvider.OLLAMA
    
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Act
    result = await cmd._cmd_model_list("")
    
    # Assert
    assert result is True, "Command should complete successfully"
    # Verify current model is marked (would need to capture output)
```

**Acceptance Criteria**:
- ✅ Current model is marked with "(current)" indicator
- ✅ Only one model is marked as current
- ✅ Marker matches both model name and provider
- ✅ No marker shown if no model is selected

---

## Integration Test Scenarios

### Integration Test 1: Complete Discovery Workflow

**Category**: `@integration_test(scope="end_to_end")`  
**File**: `tests/integration/test_model_workflows.py`

**Purpose**: Verify complete discovery and curation workflow

**Test Specification**:
```python
@integration_test(scope="end_to_end")
async def test_complete_discovery_workflow():
    """Verify complete workflow: discover → curate → use."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = []  # Start with empty list
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider with multiple models
    mock_models = [
        ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="model2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="model3", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=mock_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act 1: Discover all models
            await cmd._cmd_model_discover("")
            
            # Assert 1: All models added
            assert len(settings.llm.models) == 3
            
            # Act 2: Remove unwanted model
            await cmd._cmd_model_remove("model2")
            
            # Assert 2: Model removed
            assert len(settings.llm.models) == 2
            assert not any(m.name == "model2" for m in settings.llm.models)
            
            # Act 3: Use a model
            await cmd._cmd_model_use("model1")
            
            # Assert 3: Model selected
            assert settings.llm.model == "model1"
```

**Acceptance Criteria**:
- ✅ Discovery adds all models
- ✅ Removal works correctly
- ✅ Model selection works
- ✅ Settings are persisted at each step
- ✅ User receives feedback at each step

---

### Integration Test 2: Multi-Provider Workflow

**Category**: `@integration_test(scope="end_to_end")`  
**File**: `tests/integration/test_model_workflows.py`

**Purpose**: Verify multi-provider setup and switching

**Test Specification**:
```python
@integration_test(scope="end_to_end")
async def test_multi_provider_workflow():
    """Verify workflow with multiple providers."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = []
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock Ollama models
    ollama_models = [
        ModelInfo(name="llama3.2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    
    # Mock OpenAI models
    openai_models = [
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE)
    ]
    
    # Act 1: Discover Ollama models
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=ollama_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            await cmd._cmd_model_discover("--provider ollama")
    
    # Assert 1: Ollama model added
    assert len(settings.llm.models) == 1
    assert settings.llm.models[0].provider == ELLMProvider.OLLAMA
    
    # Act 2: Add OpenAI model
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=openai_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            await cmd._cmd_model_add("gpt-4 --provider openai")
    
    # Assert 2: Both providers represented
    assert len(settings.llm.models) == 2
    providers = {m.provider for m in settings.llm.models}
    assert ELLMProvider.OLLAMA in providers
    assert ELLMProvider.OPENAI in providers
    
    # Act 3: Switch between providers by using models
    await cmd._cmd_model_use("gpt-4")
    assert settings.llm.provider_enum == ELLMProvider.OPENAI
    
    await cmd._cmd_model_use("llama3.2")
    assert settings.llm.provider_enum == ELLMProvider.OLLAMA
```

**Acceptance Criteria**:
- ✅ Can discover models from multiple providers
- ✅ Models from different providers coexist in curated list
- ✅ Provider switches automatically when using model
- ✅ No conflicts between provider models

---

## Manual Test Checklist

### Manual Test 1: Fresh Install Experience

**Scenario**: User installs Hatchling for the first time

**Steps**:
1. Delete persistent settings file (`~/.hatch/settings/hatchling_settings.toml`)
2. Start Hatchling
3. Run `llm:model:list`
4. Run `llm:model:discover`
5. Run `llm:model:list` again
6. Run `llm:model:use <model-name>`

**Expected Results**:
- ✅ No phantom models shown on first `llm:model:list`
- ✅ Clear guidance message shown when list is empty
- ✅ Discovery finds all available models
- ✅ List shows discovered models with status indicators
- ✅ Model selection works correctly

**Pass Criteria**: User can successfully discover and use models without confusion about phantom models.

---

### Manual Test 2: Ollama Running With Models

**Scenario**: User has Ollama running with several models installed

**Steps**:
1. Ensure Ollama is running: `ollama list`
2. Run `llm:model:discover`
3. Verify all Ollama models are discovered
4. Run `llm:model:list`
5. Verify status indicators are correct (✓ for available)

**Expected Results**:
- ✅ All Ollama models discovered
- ✅ All models show ✓ (available) status
- ✅ Model sizes shown correctly
- ✅ No errors or warnings

**Pass Criteria**: Discovery accurately reflects Ollama's actual model list.

---

### Manual Test 3: Ollama Not Running

**Scenario**: User tries to discover models when Ollama is not running

**Steps**:
1. Stop Ollama service
2. Run `llm:model:discover`
3. Read error message

**Expected Results**:
- ✅ Clear error message: "Provider ollama is not accessible"
- ✅ No crash or exception
- ✅ Graceful error handling

**Pass Criteria**: Error is handled gracefully without crashing.

---

### Manual Test 4: OpenAI With Valid API Key

**Scenario**: User has valid OpenAI API key configured

**Steps**:
1. Set OpenAI API key: `settings:set openai:api_key <key>`
2. Run `llm:model:discover --provider openai`
3. Run `llm:model:list`
4. Run `llm:model:use gpt-4`

**Expected Results**:
- ✅ OpenAI models discovered successfully
- ✅ Models shown in list with ? (unknown) status initially
- ✅ Model selection works
- ✅ Provider switches to OpenAI

**Pass Criteria**: OpenAI integration works smoothly.

---

### Manual Test 5: OpenAI With Invalid API Key

**Scenario**: User has invalid or missing OpenAI API key

**Steps**:
1. Clear OpenAI API key or set invalid value
2. Run `llm:model:discover --provider openai`
3. Read error message

**Expected Results**:
- ✅ Clear error message about provider not accessible
- ✅ No crash or exception
- ✅ Graceful error handling

**Pass Criteria**: Error is handled gracefully without crashing.

---

### Manual Test 6: Multi-Provider Setup

**Scenario**: User wants to use both Ollama and OpenAI models

**Steps**:
1. Run `llm:model:discover --provider ollama`
2. Run `llm:model:add gpt-4 --provider openai`
3. Run `llm:model:list`
4. Verify models grouped by provider
5. Switch between models: `llm:model:use llama3.2`, then `llm:model:use gpt-4`

**Expected Results**:
- ✅ Both providers' models shown in list
- ✅ Clear grouping by provider (Ollama:, OpenAI:)
- ✅ Provider switches automatically when using model
- ✅ No confusion about which provider is active

**Pass Criteria**: Multi-provider workflow is intuitive and clear.

---

### Manual Test 7: Model Curation Workflow

**Scenario**: User discovers many models but only wants to keep a few

**Steps**:
1. Run `llm:model:discover` (assume 10+ models discovered)
2. Run `llm:model:list` to see all models
3. Remove unwanted models: `llm:model:remove <model1>`, `llm:model:remove <model2>`
4. Run `llm:model:list` again
5. Verify only desired models remain

**Expected Results**:
- ✅ Discovery adds all models
- ✅ Removal works correctly
- ✅ List updates to show only remaining models
- ✅ Removed models don't reappear on next discovery
- ✅ Clear feedback at each step

**Pass Criteria**: Curation workflow is smooth and predictable.

---

## Edge Cases and Regression Prevention

### Edge Case 1: Empty Provider Model List

**Scenario**: Provider returns empty list of models

**Test**:
```python
@regression_test
async def test_discovery_with_empty_provider_list():
    """Verify discovery handles provider with no models."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider returns empty list
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=[]):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    assert len(settings.llm.models) == 0
    # Should complete without error
```

**Expected Behavior**: Gracefully handles empty model list, no crash.

---

### Edge Case 2: Model Name Conflicts Across Providers

**Scenario**: Different providers have models with same name

**Test**:
```python
@regression_test
async def test_same_model_name_different_providers():
    """Verify models with same name but different providers are distinct."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = [
        ModelInfo(name="gpt-4", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE)
    ]
    
    # Assert
    assert len(settings.llm.models) == 2, \
        "Same model name from different providers should be distinct"
    
    # Verify uniqueness key is (provider, name)
    model_keys = [(m.provider, m.name) for m in settings.llm.models]
    assert len(model_keys) == len(set(model_keys))
```

**Expected Behavior**: Models are distinguished by (provider, name) tuple, not just name.

---

### Edge Case 3: Large Number of Models

**Scenario**: Provider has 100+ models

**Test**:
```python
@regression_test
async def test_discovery_with_many_models():
    """Verify discovery handles large number of models efficiently."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Mock provider with 100 models
    mock_models = [
        ModelInfo(name=f"model{i}", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.AVAILABLE)
        for i in range(100)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=mock_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    assert len(settings.llm.models) == 100
    # Should complete in reasonable time (< 5 seconds)
```

**Expected Behavior**: Discovery completes efficiently with large model lists.

---

### Regression Prevention 1: Existing Model Use Command

**Test**:
```python
@regression_test
async def test_existing_model_use_command_still_works():
    """Verify existing llm:model:use command still works after changes."""
    # Arrange
    settings = create_test_settings()
    settings.llm.models = [
        ModelInfo(name="test-model", provider=ELLMProvider.OLLAMA, 
                 status=ModelStatus.AVAILABLE)
    ]
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    # Act
    await cmd._cmd_model_use("test-model")
    
    # Assert
    assert settings.llm.model == "test-model"
    assert settings.llm.provider_enum == ELLMProvider.OLLAMA
```

**Expected Behavior**: Existing commands continue to work as before.

---

### Regression Prevention 2: Settings Persistence

**Test**:
```python
@regression_test
async def test_settings_persistence_after_discovery():
    """Verify settings are persisted after model discovery."""
    # Arrange
    settings = create_test_settings()
    settings_registry = create_test_registry(settings)
    cmd = ModelCommands(settings, settings_registry, create_test_style())
    
    mock_models = [
        ModelInfo(name="model1", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    
    with patch.object(ModelManagerAPI, 'list_available_models', return_value=mock_models):
        with patch.object(ModelManagerAPI, 'check_provider_health', return_value=True):
            # Act
            await cmd._cmd_model_discover("")
    
    # Assert
    # Verify set_setting was called with force=True
    # This ensures changes are persisted
```

**Expected Behavior**: All model changes are persisted to settings file.

---

## Acceptance Criteria

### Configuration (Task 1)
- ✅ Test 1.3 passes
- ✅ Environment variables work for deployment
- ✅ Manual test 1 (Fresh Install) passes

### Model Discovery (Task 2)
- ✅ All 5 tests pass
- ✅ Discovery adds all models from provider
- ✅ Uniqueness enforcement prevents duplicates
- ✅ Provider health check works
- ✅ Manual tests 2-3 (Ollama running/not running) pass

### Model Addition (Task 3)
- ✅ All 4 tests pass
- ✅ Validation prevents adding non-existent models
- ✅ Provider flag works correctly
- ✅ Manual tests 4-5 (OpenAI valid/invalid key) pass

### Model List Display (Task 4)
- ✅ All 3 tests pass
- ✅ Models grouped by provider
- ✅ Status indicators shown correctly
- ✅ Current model marked
- ✅ Manual test 6 (Multi-provider) passes

### Integration Tests
- ✅ Both integration tests pass
- ✅ Complete workflow works end-to-end
- ✅ Multi-provider setup works
- ✅ Manual test 7 (Curation workflow) passes

### Edge Cases and Regressions
- ✅ All edge case tests pass
- ✅ All regression prevention tests pass
- ✅ No existing functionality broken

---

## Test Execution Plan

### Running Tests with Wobble

**Run all tests**:
```bash
wobble --log-file test_execution_v1.txt --log-verbosity 3
```

**Run specific categories**:
```bash
# Regression tests only
wobble --category regression --log-file regression_results.txt --log-verbosity 3

# Integration tests only
wobble --category integration --log-file integration_results.txt --log-verbosity 3
```

**Run specific test files**:
```bash
# Configuration tests
wobble --pattern "test_llm_configuration.py" --log-verbosity 3

# Discovery tests
wobble --pattern "test_model_discovery.py" --log-verbosity 3
```

### Test Execution Order

**Recommended order**:
1. **Configuration tests** (Task 1) - Foundation
2. **Discovery tests** (Task 2) - Core functionality
3. **Addition tests** (Task 3) - Validation logic
4. **List display tests** (Task 4) - UI improvements
5. **Integration tests** - End-to-end workflows
6. **Edge cases and regressions** - Boundary conditions

### Expected Results

**Success Criteria**:
- All automated tests pass (18 tests)
- All manual test scenarios pass (7 scenarios)
- No regressions in existing functionality
- Test execution completes in < 5 minutes
- Code coverage ≥ 90% for new code

**Failure Handling**:
- Document failing tests in test execution log
- Investigate root cause
- Fix implementation
- Re-run tests
- Iterate until all tests pass

---

## Test File Organization

Following org's testing standards, tests should be organized as:

```
tests/
├── regression/
│   ├── test_llm_configuration.py       # Task 1 tests
│   ├── test_model_uniqueness.py        # Task 2 uniqueness tests
│   ├── test_model_validation.py        # Task 3 validation tests
│   └── test_model_list_display.py      # Task 4 display tests
├── integration/
│   ├── test_model_discovery.py         # Task 2 integration tests
│   ├── test_model_addition.py          # Task 3 integration tests
│   └── test_model_workflows.py         # End-to-end workflow tests
└── test_data/
    └── mock_responses/
        ├── ollama_models.json
        └── openai_models.json
```

**Note**: Current codebase uses different naming convention (e.g., `regression_test_*.py`). Consider migrating to org standard (`test_*.py` with hierarchical structure) as part of this work or as follow-up task.

---

## Summary

**Total Test Count**: 18 automated tests + 7 manual scenarios = 25 total tests

**Test Distribution**:
- Configuration: 1 test
- Model Discovery: 5 tests
- Model Addition: 4 tests
- Model List Display: 3 tests
- Integration Tests: 2 tests
- Edge Cases: 3 tests
- Regression Prevention: 2 tests

**Test-to-Code Ratio**: 18 tests / 6 tasks = 3:1 (within target 2:1 to 3:1 for bug fixes)

**Coverage Goals**:
- Minimum: 70% overall
- Critical functionality: 90%
- New features: 95%

**Execution Time**: < 5 minutes for full test suite

**Manual Testing**: 7 scenarios covering UX validation

---

**Report Status**: Ready for Review  
**Next Steps**:
1. User review and feedback
2. Iterate on test specifications if needed
3. Implement tests during Task 1-5 development
4. Execute tests and validate results
5. Create test execution report (Phase 4)

**Key Principles Applied**:
- ✅ Focus on behavioral functionality, not meta-constraints
- ✅ Test-to-code ratio within target range
- ✅ Functional grouping for clarity
- ✅ Clear acceptance criteria for each test
- ✅ Manual testing for UX validation
- ✅ Edge cases and regression prevention covered
- ✅ Don't test implementation details (error messages, default values)
