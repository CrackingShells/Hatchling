# LLM Management UX Fix - Implementation Roadmap

**Date**: 2025-11-07  
**Report Type**: Implementation Roadmap  
**Status**: Ready for Implementation  
**Version**: v0  
**Author**: AI Development Agent  
**Scope**: Phase 0 - Quick Wins (UX Fix Only)

---

## Executive Summary

This roadmap provides a detailed, ordered task list for fixing the critical UX issue where users are confused about which LLM API endpoint and model is actually accessible when running Hatchling.

**Scope**: 8 focused tasks addressing configuration timing, model availability, and user feedback  
**Timeline**: 14-22 hours (1.75-2.75 days)  
**Approach**: Quick wins with high user impact, no major architectural changes  
**Risk Level**: Low - all changes are incremental enhancements to existing systems

**Success Criteria**:
- ✅ Users can configure Ollama IP/port at runtime without restart
- ✅ No phantom models shown in default configuration
- ✅ Clear visibility into which models are actually available
- ✅ Actionable error messages when models are unavailable
- ✅ Automatic discovery on startup and provider switch

---

## Table of Contents

1. [Scope and Objectives](#scope-and-objectives)
2. [Task List](#task-list)
3. [Success Criteria](#success-criteria)
4. [Testing Strategy](#testing-strategy)
5. [Risk Assessment](#risk-assessment)

---

## Scope and Objectives

### In Scope

**Core UX Fixes**:
- Configuration timing issues (env var capture at import time)
- Phantom model elimination (hard-coded defaults)
- Automatic model discovery and validation
- Clear status indicators and error messages
- Provider health checking

**Affected Components**:
- `hatchling/config/llm_settings.py`
- `hatchling/config/ollama_settings.py`
- `hatchling/config/openai_settings.py`
- `hatchling/config/settings.py`
- `hatchling/ui/model_commands.py`
- `hatchling/core/llm/model_manager_api.py`

### Out of Scope

**Deferred to Future Phases** (as per strategic roadmap v2):
- ❌ Model management abstraction (LLMModelManager)
- ❌ User-first configuration system (SQLite storage)
- ❌ Security encryption (keyring + Fernet)
- ❌ Command standardization across providers
- ❌ Major architectural refactoring

### Objectives

1. **Eliminate Configuration Confusion**: Runtime configuration changes work immediately
2. **Remove Phantom Models**: Only show models that are actually available
3. **Improve Visibility**: Clear status indicators for model availability
4. **Better Error Messages**: Actionable guidance when things go wrong
5. **Automatic Discovery**: Reduce manual steps for users

---

## Task List

### Task 1: Fix Configuration Timing Issue

**Goal**: Enable runtime configuration changes by removing import-time environment variable capture

**Effort**: 2-4 hours  
**Priority**: P0 - Critical  
**Pre-conditions**: None

**Files to Modify**:
- `hatchling/config/llm_settings.py`
- `hatchling/config/ollama_settings.py`
- `hatchling/config/openai_settings.py`
- `hatchling/config/settings.py`

**Implementation Steps**:

1. **Remove `default_factory` lambdas** in all settings classes:
   ```python
   # BEFORE (llm_settings.py):
   provider_enum: ELLMProvider = Field(
       default_factory=lambda: LLMSettings.to_provider_enum(os.environ.get("LLM_PROVIDER", "ollama"))
   )
   
   # AFTER:
   provider_enum: ELLMProvider = Field(
       default=ELLMProvider.OLLAMA,
       description="LLM provider to use ('ollama' or 'openai').",
       json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
   )
   ```

2. **Add runtime environment override** in `AppSettings.__init__()`:
   ```python
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._apply_environment_overrides()
   
   def _apply_environment_overrides(self):
       """Apply environment variable overrides at runtime."""
       if provider := os.environ.get("LLM_PROVIDER"):
           self.llm.provider_enum = LLMSettings.to_provider_enum(provider)
       if model := os.environ.get("LLM_MODEL"):
           self.llm.model = model
       if ollama_ip := os.environ.get("OLLAMA_IP"):
           self.ollama.ip = ollama_ip
       if ollama_port := os.environ.get("OLLAMA_PORT"):
           self.ollama.port = int(ollama_port)
       # ... continue for all env-configurable settings
   ```

3. **Update all settings classes** (ollama_settings.py, openai_settings.py):
   - Replace all `default_factory=lambda: os.environ.get(...)` with simple defaults
   - Move environment variable logic to `AppSettings._apply_environment_overrides()`

**Success Gates**:
- ✅ All `default_factory` lambdas removed from settings classes
- ✅ `AppSettings._apply_environment_overrides()` implemented
- ✅ Environment variables applied at runtime, not import time
- ✅ Configuration changes work without application restart
- ✅ Existing tests pass
- ✅ Manual test: Change `OLLAMA_PORT` env var, verify it takes effect immediately

**Testing**:
```python
# Test: Runtime environment override
def test_runtime_env_override():
    os.environ["OLLAMA_PORT"] = "11435"
    settings = AppSettings()
    assert settings.ollama.port == 11435
    
    os.environ["OLLAMA_PORT"] = "11436"
    settings._apply_environment_overrides()
    assert settings.ollama.port == 11436
```

---

### Task 2: Remove Hard-coded Default Models

**Goal**: Eliminate phantom models by starting with empty model list

**Effort**: 1-2 hours  
**Priority**: P0 - Critical  
**Pre-conditions**: Task 1 complete

**Files to Modify**:
- `hatchling/config/llm_settings.py`

**Implementation Steps**:

1. **Replace hard-coded default** with empty list:
   ```python
   # BEFORE:
   models: List[ModelInfo] = Field(
       default_factory=lambda: [
           ModelInfo(name=model[1], provider=model[0], status=ModelStatus.AVAILABLE)
           for model in LLMSettings.extract_provider_model_list(
               os.environ.get("LLM_MODELS", "") if os.environ.get("LLM_MODELS") 
               else "[(ollama, llama3.2), (openai, gpt-4.1-nano)]"
           )
       ],
       description="List of LLMs the user can choose from.",
       json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
   )
   
   # AFTER:
   models: List[ModelInfo] = Field(
       default_factory=list,
       description="List of LLMs the user can choose from. Populated via discovery.",
       json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
   )
   ```

2. **Update default model** to be empty or None:
   ```python
   # BEFORE:
   model: str = Field(
       default_factory=lambda: os.environ.get("LLM_MODEL", "llama3.2"),
       description="Default LLM to use for the selected provider.",
       json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
   )
   
   # AFTER:
   model: Optional[str] = Field(
       default=None,
       description="Default LLM to use for the selected provider. Set via discovery or manually.",
       json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
   )
   ```

3. **Handle None model** in code that uses `settings.llm.model`:
   - Add validation before using model
   - Provide clear error message if no model selected

**Success Gates**:
- ✅ Hard-coded default models removed
- ✅ `models` field starts with empty list
- ✅ `model` field starts with None
- ✅ Code handles None model gracefully
- ✅ Clear error message when no model selected
- ✅ Existing tests updated to handle empty initial state

**Testing**:
```python
# Test: Empty initial state
def test_empty_initial_models():
    settings = LLMSettings()
    assert settings.models == []
    assert settings.model is None
```

---

### Task 3: Add Provider Health Check on Startup

**Goal**: Verify provider accessibility on application startup

**Effort**: 1-2 hours  
**Priority**: P0 - Critical  
**Pre-conditions**: Tasks 1-2 complete

**Files to Modify**:
- `hatchling/config/settings.py`
- `hatchling/core/llm/model_manager_api.py`

**Implementation Steps**:

1. **Add health check** in `AppSettings.__init__()`:
   ```python
   async def _check_provider_health(self):
       """Check health of configured provider on startup."""
       try:
           is_healthy = await ModelManagerAPI.check_provider_health(
               self.llm.provider_enum, self
           )
           if not is_healthy:
               logger.warning(
                   f"Provider {self.llm.provider_enum.value} is not accessible. "
                   f"Please check your configuration."
               )
           return is_healthy
       except Exception as e:
           logger.error(f"Provider health check failed: {e}")
           return False
   ```

2. **Call health check** during initialization:
   ```python
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._apply_environment_overrides()
       
       # Check provider health (async, don't block initialization)
       import asyncio
       try:
           asyncio.create_task(self._check_provider_health())
       except RuntimeError:
           # No event loop, skip health check
           pass
   ```

3. **Improve health check** in `ModelManagerAPI`:
   - Add timeout to prevent hanging
   - Return detailed status (accessible, timeout, error)
   - Cache result for 60 seconds

**Success Gates**:
- ✅ Health check runs on startup
- ✅ Warning logged if provider inaccessible
- ✅ Health check doesn't block initialization
- ✅ Timeout prevents hanging
- ✅ Result cached to avoid repeated checks

**Testing**:
```python
# Test: Provider health check
async def test_provider_health_check():
    settings = AppSettings()
    is_healthy = await settings._check_provider_health()
    assert isinstance(is_healthy, bool)
```

---

### Task 4: Add Model Validation on Startup

**Goal**: Validate configured models against actual provider availability on startup

**Effort**: 1-2 hours
**Priority**: P1 - Important
**Pre-conditions**: Task 3 complete

**Files to Modify**:
- `hatchling/config/settings.py`
- `hatchling/config/llm_settings.py`

**Implementation Steps**:

1. **Add validation method** in `AppSettings`:
   ```python
   async def _validate_configured_models(self):
       """Validate configured models against actual availability."""
       if not self.llm.models:
           logger.info("No models configured, skipping validation")
           return

       try:
           available_models = await ModelManagerAPI.list_available_models(
               self.llm.provider_enum, self
           )
           available_names = {m.name for m in available_models}

           for model in self.llm.models:
               if model.name not in available_names:
                   model.status = ModelStatus.NOT_AVAILABLE
                   logger.warning(
                       f"Model {model.name} is configured but not available "
                       f"from {model.provider.value}"
                   )
               else:
                   model.status = ModelStatus.AVAILABLE
       except Exception as e:
           logger.error(f"Model validation failed: {e}")
   ```

2. **Call validation** after health check:
   ```python
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self._apply_environment_overrides()

       import asyncio
       try:
           asyncio.create_task(self._startup_checks())
       except RuntimeError:
           pass

   async def _startup_checks(self):
       """Run startup health and validation checks."""
       await self._check_provider_health()
       await self._validate_configured_models()
   ```

**Success Gates**:
- ✅ Model validation runs on startup
- ✅ Model status updated based on actual availability
- ✅ Warnings logged for unavailable models
- ✅ Validation doesn't block initialization
- ✅ Handles empty model list gracefully

**Testing**:
```python
# Test: Model validation
async def test_model_validation():
    settings = AppSettings()
    settings.llm.models = [
        ModelInfo(name="nonexistent", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE)
    ]
    await settings._validate_configured_models()
    assert settings.llm.models[0].status == ModelStatus.NOT_AVAILABLE
```

---

### Task 5: Implement Model Discovery Command

**Goal**: Add `llm:model:discover` command for manual model discovery

**Effort**: 4-6 hours
**Priority**: P0 - Critical
**Pre-conditions**: Tasks 1-4 complete

**Files to Modify**:
- `hatchling/ui/model_commands.py`
- `hatchling/config/settings_registry.py`

**Implementation Steps**:

1. **Add command definition** in `ModelCommands.__init__()`:
   ```python
   'llm:model:discover': {
       'handler': self._cmd_model_discover,
       'description': translate('commands.llm.model_discover_description'),
       'is_async': True,
       'args': {
           'provider-name': {
               'positional': False,
               'completer_type': 'suggestions',
               'values': self.settings.llm.provider_names,
               'description': translate('commands.llm.provider_name_arg_description'),
               'required': False
           }
       }
   }
   ```

2. **Implement command handler**:
   ```python
   async def _cmd_model_discover(self, args: str) -> bool:
       """Discover models from provider and update configuration.

       Args:
           args (str): Optional provider name argument.

       Returns:
           bool: True to continue the chat session.
       """
       try:
           args_def = self.commands['llm:model:discover']['args']
           parsed_args = self._parse_args(args, args_def)

           provider_name = parsed_args.get('provider-name', self.settings.llm.provider_enum.value)
           provider = LLMSettings.to_provider_enum(provider_name)

           print(f"Discovering models from {provider.value}...")

           # Check provider health first
           is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)
           if not is_healthy:
               print(f"Error: Provider {provider.value} is not accessible.")
               print(f"Please check your configuration and ensure the provider is running.")
               return True

           # Discover models
           discovered_models = await ModelManagerAPI.list_available_models(provider, self.settings)

           if not discovered_models:
               print(f"No models found for provider {provider.value}")
               return True

           # Update settings with discovered models
           # Merge with existing models from other providers
           existing_other_provider = [
               m for m in self.settings.llm.models
               if m.provider != provider
           ]
           self.settings.llm.models = existing_other_provider + discovered_models

           # Persist to storage
           self.settings_registry.set_setting(
               "llm", "models", self.settings.llm.models, force=True
           )

           print(f"\nDiscovered {len(discovered_models)} models from {provider.value}:")
           for model in discovered_models:
               print(f"  ✓ {model.name}")

           # Update command completions
           self._update_model_completions()

       except Exception as e:
           self.logger.error(f"Error in model discover command: {e}")
           print(f"Error: Model discovery failed - {e}")

       return True

   def _update_model_completions(self):
       """Update model name completions for commands."""
       model_names = [model.name for model in self.settings.llm.models]
       self.commands['llm:model:use']['args']['model-name']['values'] = model_names
       self.commands['llm:model:remove']['args']['model-name']['values'] = model_names
   ```

3. **Add translation strings** (if using i18n):
   ```python
   # In translation files
   "commands.llm.model_discover_description": "Discover available models from provider"
   ```

**Success Gates**:
- ✅ `llm:model:discover` command implemented
- ✅ Command checks provider health before discovery
- ✅ Discovered models merged with existing models from other providers
- ✅ Models persisted to storage
- ✅ Clear user feedback during discovery
- ✅ Command completions updated after discovery
- ✅ Error handling for inaccessible providers

**Testing**:
```python
# Test: Model discovery command
async def test_model_discover_command():
    cmd = ModelCommands(settings, settings_registry, style)
    result = await cmd._cmd_model_discover("")
    assert result is True
    assert len(settings.llm.models) > 0
```

---

### Task 6: Add Automatic Discovery on Provider Switch

**Goal**: Automatically discover models when user switches provider

**Effort**: 2-3 hours
**Priority**: P1 - Important
**Pre-conditions**: Task 5 complete

**Files to Modify**:
- `hatchling/config/settings_registry.py`
- `hatchling/ui/model_commands.py`

**Implementation Steps**:

1. **Add callback** in `SettingsRegistry.set_setting()`:
   ```python
   def set_setting(self, category: str, field: str, value: Any, force: bool = False):
       """Set a setting value with validation and callbacks."""
       # ... existing validation code ...

       # Set the value
       setattr(category_obj, field, value)

       # Trigger callbacks
       if category == "llm" and field == "provider_enum":
           self._on_provider_changed(value)

       # ... existing persistence code ...

   async def _on_provider_changed(self, new_provider: ELLMProvider):
       """Handle provider change by discovering models."""
       logger.info(f"Provider changed to {new_provider.value}, discovering models...")

       try:
           # Check health
           is_healthy = await ModelManagerAPI.check_provider_health(new_provider, self.settings)
           if not is_healthy:
               logger.warning(f"New provider {new_provider.value} is not accessible")
               return

           # Discover models
           discovered_models = await ModelManagerAPI.list_available_models(new_provider, self.settings)

           # Update models for this provider
           existing_other_provider = [
               m for m in self.settings.llm.models
               if m.provider != new_provider
           ]
           self.settings.llm.models = existing_other_provider + discovered_models

           logger.info(f"Discovered {len(discovered_models)} models from {new_provider.value}")

       except Exception as e:
           logger.error(f"Auto-discovery on provider change failed: {e}")
   ```

2. **Update provider switch command** to trigger callback:
   ```python
   # In model_commands.py, ensure provider changes go through settings_registry
   def _cmd_provider_use(self, args: str) -> bool:
       """Switch to a different provider."""
       # ... parse args ...

       # Use settings_registry to trigger callbacks
       self.settings_registry.set_setting(
           "llm", "provider_enum", new_provider, force=True
       )

       print(f"Switched to provider: {new_provider.value}")
       print("Discovering available models...")
   ```

**Success Gates**:
- ✅ Provider change triggers automatic model discovery
- ✅ Discovery runs asynchronously without blocking
- ✅ Models updated for new provider
- ✅ User notified of discovery progress
- ✅ Handles discovery failures gracefully

**Testing**:
```python
# Test: Auto-discovery on provider switch
async def test_auto_discovery_on_provider_switch():
    registry = SettingsRegistry(settings)
    registry.set_setting("llm", "provider_enum", ELLMProvider.OPENAI, force=True)
    await asyncio.sleep(0.1)  # Allow async discovery to run
    assert any(m.provider == ELLMProvider.OPENAI for m in settings.llm.models)
```

---

### Task 7: Improve Error Messages and Status Indicators

**Goal**: Provide clear status indicators and actionable error messages

**Effort**: 2-3 hours
**Priority**: P1 - Important
**Pre-conditions**: Tasks 1-6 complete

**Files to Modify**:
- `hatchling/ui/model_commands.py`
- `hatchling/core/llm/providers/base.py`

**Implementation Steps**:

1. **Enhance `llm:model:list`** with status indicators:
   ```python
   async def _cmd_model_list(self, args: str) -> bool:
       """List all available models with status indicators."""

       if not self.settings.llm.models:
           print("No models configured.")
           print(f"Run 'llm:model:discover' to discover models from {self.settings.llm.provider_enum.value}")
           return True

       print(f"\nConfigured Models (Provider: {self.settings.llm.provider_enum.value}):\n")

       for model_info in self.settings.llm.models:
           # Status indicator
           if model_info.status == ModelStatus.AVAILABLE:
               status_icon = "✓"
               status_color = "green"
           elif model_info.status == ModelStatus.NOT_AVAILABLE:
               status_icon = "✗"
               status_color = "red"
           elif model_info.status == ModelStatus.DOWNLOADING:
               status_icon = "↓"
               status_color = "yellow"
           else:
               status_icon = "?"
               status_color = "gray"

           # Current model indicator
           current = " (current)" if model_info.name == self.settings.llm.model else ""

           print(f"  {status_icon} {model_info.provider.value}/{model_info.name}{current}")

       print("\nLegend: ✓ Available | ✗ Unavailable | ↓ Downloading | ? Unknown")
       return True
   ```

2. **Improve error messages** when model not available:
   ```python
   # In LLMProvider base class or chat initialization
   def validate_model_available(self, model_name: str, settings: AppSettings):
       """Validate model is available before use."""
       model_info = next(
           (m for m in settings.llm.models if m.name == model_name),
           None
       )

       if model_info is None:
           raise ValueError(
               f"Model '{model_name}' is not configured.\n"
               f"Available models: {[m.name for m in settings.llm.models]}\n"
               f"Run 'llm:model:discover' to discover more models."
           )

       if model_info.status != ModelStatus.AVAILABLE:
           raise ValueError(
               f"Model '{model_name}' is not available (status: {model_info.status.value}).\n"
               f"For Ollama models, run: llm:model:add {model_name}\n"
               f"For OpenAI models, check your API key and model name."
           )
   ```

3. **Add helpful hints** in command outputs:
   ```python
   # When provider is inaccessible
   print(f"Error: Cannot connect to {provider.value}")
   print(f"\nTroubleshooting:")
   if provider == ELLMProvider.OLLAMA:
       print(f"  1. Check if Ollama is running: ollama list")
       print(f"  2. Verify connection: curl {settings.ollama.api_base}/api/tags")
       print(f"  3. Check OLLAMA_IP and OLLAMA_PORT environment variables")
   elif provider == ELLMProvider.OPENAI:
       print(f"  1. Verify your OPENAI_API_KEY is set")
       print(f"  2. Check your internet connection")
       print(f"  3. Verify API base URL: {settings.openai.api_base}")
   ```

**Success Gates**:
- ✅ Model list shows status indicators (✓ ✗ ↓ ?)
- ✅ Current model clearly marked
- ✅ Empty model list shows helpful guidance
- ✅ Error messages include troubleshooting steps
- ✅ Provider-specific guidance provided
- ✅ Actionable next steps in all error messages

**Testing**:
```python
# Test: Status indicators in model list
async def test_model_list_status_indicators():
    settings.llm.models = [
        ModelInfo(name="available", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="unavailable", provider=ELLMProvider.OLLAMA, status=ModelStatus.NOT_AVAILABLE)
    ]
    cmd = ModelCommands(settings, settings_registry, style)
    result = await cmd._cmd_model_list("")
    # Verify output contains status indicators
```

---

### Task 8: Update Documentation

**Goal**: Document new commands and workflows for users

**Effort**: 1 hour
**Priority**: P2 - Nice to have
**Pre-conditions**: Tasks 1-7 complete

**Files to Modify**:
- `docs/user-guide/model-management.md` (or equivalent)
- `README.md` (if applicable)
- Command help strings

**Implementation Steps**:

1. **Document model discovery workflow**:
   ```markdown
   ## Model Management

   ### Discovering Available Models

   Hatchling automatically discovers models when you start the application or switch providers.
   To manually discover models:

   ```bash
   llm:model:discover
   ```

   ### Listing Models

   View all configured models with their availability status:

   ```bash
   llm:model:list
   ```

   Status indicators:
   - ✓ Available - Model is ready to use
   - ✗ Unavailable - Model is configured but not accessible
   - ↓ Downloading - Model is being downloaded (Ollama only)
   - ? Unknown - Model status not yet checked
   ```

2. **Document configuration**:
   ```markdown
   ## Configuration

   ### Runtime Configuration

   You can configure Hatchling using environment variables:

   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_IP=localhost
   export OLLAMA_PORT=11434
   ```

   Changes take effect immediately without restarting the application.
   ```

3. **Document troubleshooting**:
   ```markdown
   ## Troubleshooting

   ### No Models Available

   If you see "No models configured":
   1. Run `llm:model:discover` to discover models from your provider
   2. For Ollama, ensure Ollama is running: `ollama list`
   3. For OpenAI, verify your API key is set: `echo $OPENAI_API_KEY`

   ### Model Not Available

   If a model shows ✗ (unavailable):
   - For Ollama: Download the model with `llm:model:add <model-name>`
   - For OpenAI: Verify the model name and your API access
   ```

**Success Gates**:
- ✅ User guide updated with model management section
- ✅ Configuration documented
- ✅ Troubleshooting guide added
- ✅ Command help strings updated
- ✅ Examples provided for common workflows

---

## Success Criteria

### Functional Requirements

- ✅ **Configuration Timing**: Runtime configuration changes work without restart
- ✅ **No Phantom Models**: Default configuration starts with empty model list
- ✅ **Automatic Discovery**: Models discovered on startup and provider switch
- ✅ **Manual Discovery**: `llm:model:discover` command works correctly
- ✅ **Status Indicators**: Model list shows clear availability status
- ✅ **Error Messages**: Actionable guidance when models unavailable
- ✅ **Provider Health**: Health check runs on startup
- ✅ **Model Validation**: Configured models validated against actual availability

### Quality Requirements

- ✅ **No Regressions**: All existing tests pass
- ✅ **Test Coverage**: New functionality has test coverage
- ✅ **Performance**: No noticeable performance degradation
- ✅ **User Experience**: Clear, helpful feedback at every step
- ✅ **Documentation**: User guide updated with new workflows

### User Experience Goals

- ✅ **First Run**: Clear guidance when no models configured
- ✅ **Configuration**: Easy to understand what's configured vs available
- ✅ **Errors**: Actionable troubleshooting steps in error messages
- ✅ **Discovery**: Automatic discovery reduces manual steps
- ✅ **Visibility**: Always clear which provider and model is active

---

## Testing Strategy

### Unit Tests

**Configuration Tests**:
- Test runtime environment override
- Test empty initial model list
- Test environment variable precedence

**Discovery Tests**:
- Test model discovery for each provider
- Test discovery with inaccessible provider
- Test model merging from multiple providers

**Validation Tests**:
- Test model status validation
- Test health check functionality
- Test error message generation

### Integration Tests

**Workflow Tests**:
- Test complete discovery workflow
- Test provider switch with auto-discovery
- Test model selection after discovery

**Command Tests**:
- Test `llm:model:discover` command
- Test `llm:model:list` with status indicators
- Test error handling in commands

### Manual Testing

**Scenarios**:
1. Fresh install with no configuration
2. Ollama running with models
3. Ollama not running
4. OpenAI with valid API key
5. OpenAI with invalid API key
6. Provider switch during session
7. Runtime configuration change

---

## Risk Assessment

### Technical Risks

**R1: Breaking Changes to Configuration**
- **Probability**: Low
- **Impact**: High (existing users affected)
- **Mitigation**: Maintain backward compatibility, provide migration guide
- **Contingency**: Rollback mechanism, support for old format

**R2: Performance Impact from Health Checks**
- **Probability**: Low
- **Impact**: Medium (slower startup)
- **Mitigation**: Async checks, caching, timeouts
- **Contingency**: Make health checks optional

**R3: Discovery Failures**
- **Probability**: Medium
- **Impact**: Medium (users can't discover models)
- **Mitigation**: Comprehensive error handling, fallback to manual configuration
- **Contingency**: Document manual model configuration

### User Experience Risks

**R4: Empty Model List Confusion**
- **Probability**: Medium
- **Impact**: Medium (users don't know what to do)
- **Mitigation**: Clear guidance messages, automatic discovery on startup
- **Contingency**: Provide quick-start guide

**R5: Migration Friction**
- **Probability**: Low
- **Impact**: Low (existing users need to re-discover)
- **Mitigation**: One-time migration, clear communication
- **Contingency**: Support old configuration format temporarily

---

## Implementation Order

**Recommended Sequence**:
1. Task 1 (Configuration Timing) - Foundation for everything else
2. Task 2 (Remove Defaults) - Clean slate for discovery
3. Task 5 (Discovery Command) - Core functionality
4. Task 3 (Health Check) - Validation infrastructure
5. Task 4 (Model Validation) - Builds on health check
6. Task 7 (Error Messages) - Improve user feedback
7. Task 6 (Auto-discovery) - Convenience feature
8. Task 8 (Documentation) - Final polish

**Parallel Opportunities**:
- Tasks 3 and 5 can be developed in parallel
- Task 7 can be developed alongside Tasks 3-6
- Task 8 can be written while testing Tasks 1-7

---

## Next Steps

1. **Review and Approve**: Stakeholder review of this roadmap
2. **Create Branch**: `git checkout -b fix/llm-management-ux`
3. **Begin Implementation**: Start with Task 1
4. **Iterative Testing**: Test after each task completion
5. **User Validation**: Get feedback after Tasks 1-5 complete
6. **Final Review**: Complete testing and documentation
7. **Merge and Release**: Merge to main, communicate changes to users

---

**Report Status**: Ready for Implementation
**Estimated Timeline**: 14-22 hours (1.75-2.75 days)
**Risk Level**: Low
**User Impact**: High (significantly improves UX)


