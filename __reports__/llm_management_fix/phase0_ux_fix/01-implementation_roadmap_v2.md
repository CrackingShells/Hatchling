# LLM Management UX Fix – Implementation Roadmap v2

**Project**: Hatchling – LLM Configuration UX Fix  
**Roadmap Date**: 2025-11-07  
**Phase**: Implementation  
**Source**: Adequation Assessment v2 + User Feedback  
**Branch**: `fix/llm-management`  
**Target**: Bug fix release (patch version bump)  
**Timeline**: 10-15 hours (1.25-2 days)  
**Approach**: Incremental fixes with testing at each step

---

## Executive Summary

This roadmap provides actionable implementation tasks for fixing the critical UX issue where users are confused about which LLM API endpoint and model is actually accessible when running Hatchling.

**Core Issue**: Hard-coded phantom models (`llama3.2`, `gpt-4.1-nano`) shown by default but don't exist on user's system.

**Solution Approach**: Remove hard-coded models, preserve environment variable support for deployment flexibility, implement discovery workflow with manual curation.

**Key Changes from v1**:
- ✅ **Environment Variables**: Keep for deployment flexibility (Docker, CI/CD), not remove
- ✅ **Discovery Workflow**: Bulk add all models, then user curates by removing unwanted
- ✅ **Uniqueness**: Enforce in add logic (not data structure change)
- ✅ **Command Specs**: Precise specifications with validation behavior

**Maintained from v1**:
- ✅ **6 focused tasks**: Clean defaults, discovery command, enhanced add, list display, errors, docs
- ✅ **Quick wins scope**: 10-15 hours total
- ✅ **Leverage existing infrastructure**: Health checks, validation, settings commands already exist

**Architectural Decision**:
Preserve environment variable support for deployment scenarios while fixing the core UX issue (phantom models). Configuration precedence: **Persistent Settings > Environment Variables > Code Defaults**. This avoids breaking Docker/CI/CD deployments while eliminating user confusion.

---

## Git Workflow

**Branch Strategy** (Simplified for fix):

```
main (production)
  └── fix/llm-management (fix branch)
      ├── task/1-clean-defaults
      ├── task/2-discovery-command
      ├── task/3-enhance-add
      ├── task/4-list-display
      ├── task/5-error-messages
      └── task/6-documentation
```

**Workflow Rules**:

1. **All work from `fix/llm-management` branch**
   - Created from `main` (or current development branch)
   - Will be merged back to `main` after all tasks complete

2. **Task branches from fix branch**
   - Branch naming: `task/<task-number>-<short-description>`
   - Example: `task/1-clean-defaults`
   - Created when task work begins
   - Deleted after merge back to fix branch

3. **Merge Hierarchy**:
   - Task branches → `fix/llm-management` (when task complete)
   - `fix/llm-management` → `main` (when ALL tasks complete and tested)

4. **Merge Criteria**:
   - **Task → Fix branch**: Task success gates met, task tests pass
   - **Fix branch → main**: All tasks complete, all tests pass (unit, integration, manual), no regressions

5. **Conventional Commits**:
   - `fix: <description>` for bug fixes
   - `docs: <description>` for documentation
   - `test: <description>` for test additions
   - See `git-workflow.md` for commit message standards

---

## Task Overview

**6 focused tasks, 10-15 hours total**:

| Task | Description | Effort | Pre-conditions |
|------|-------------|--------|----------------|
| 1 | Clean Up Default Configuration | 1-2h | None |
| 2 | Implement Model Discovery Command | 4-6h | Task 1 |
| 3 | Enhance Model Add Command | 2-3h | Task 2 |
| 4 | Improve Model List Display | 2-3h | Task 1 |
| 5 | Better Error Messages | 1-2h | Tasks 2, 3 |
| 6 | Update Documentation | 1h | Tasks 1-5 |

**Parallel Opportunities**:
- Tasks 2 and 4 can be developed in parallel after Task 1
- Task 5 can be developed alongside Tasks 2-3
- Task 6 can be written while testing Tasks 1-5

---

## Task 1: Clean Up Default Configuration

**Branch**: `task/1-clean-defaults`  
**Effort**: 1-2 hours  
**Pre-conditions**: None

### Goal

Remove hard-coded phantom models while preserving environment variable support for deployment flexibility.

### Files to Modify

1. `hatchling/config/llm_settings.py` - Remove hard-coded model list
2. `hatchling/config/ollama_settings.py` - Update field descriptions
3. `hatchling/config/openai_settings.py` - Update field descriptions
4. `hatchling/config/languages/en.toml` - Update setting descriptions

### Implementation Steps

**Step 1.1: Remove hard-coded model list** (`llm_settings.py`)

```python
# BEFORE (lines 87-96):
models: List[ModelInfo] = Field(
    default_factory=lambda: [
        ModelInfo(name=model[1], provider=model[0], status=ModelStatus.AVAILABLE)
        for model in LLMSettings.extract_provider_model_list(
            os.environ.get("LLM_MODELS", "") if os.environ.get("LLM_MODELS") 
            else "[(ollama, llama3.2), (openai, gpt-4.1-nano)]"  # ← Remove this
        )
    ],
    description="List of LLMs the user can choose from.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
)

# AFTER:
models: List[ModelInfo] = Field(
    default_factory=list,  # ← Empty list, no phantom models
    description="Curated list of models. Populate via llm:model:discover or llm:model:add. "
                "Persistent settings override environment variables.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
)
```

**Step 1.2: Update provider field description** (`llm_settings.py`)

```python
# Update line 58:
provider_enum: ELLMProvider = Field(
    default_factory=lambda: LLMSettings.to_provider_enum(os.environ.get("LLM_PROVIDER", "ollama")),
    description="LLM provider to use ('ollama' or 'openai'). "
                "Set via LLM_PROVIDER env var or settings:set command. "
                "Persistent settings override environment variables.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
)
```

**Step 1.3: Update model field** (`llm_settings.py`)

```python
# Update lines 62-66:
model: Optional[str] = Field(  # ← Make Optional
    default=None,  # ← Change to None instead of env var default
    description="Default LLM to use for the selected provider. "
                "Set via settings:set command or llm:model:use command. "
                "Persistent settings override environment variables.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
)
```

**Step 1.4: Update Ollama field descriptions** (`ollama_settings.py`)

```python
# Update lines 19-23:
ip: str = Field(
    default_factory=lambda: os.environ.get("OLLAMA_IP", "localhost"),
    description="IP address for the Ollama API endpoint. "
                "Set via OLLAMA_IP env var or settings:set command. "
                "Persistent settings override environment variables.",
    json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
)

# Update lines 25-29:
port: int = Field(
    default_factory=lambda: int(os.environ.get("OLLAMA_PORT", 11434)),
    description="Port for the Ollama API endpoint. "
                "Set via OLLAMA_PORT env var or settings:set command. "
                "Persistent settings override environment variables.",
    json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
)
```

**Step 1.5: Update OpenAI field descriptions** (`openai_settings.py`)

Update descriptions for `api_key`, `api_base`, `timeout`, etc. to document precedence.

**Step 1.6: Update translation strings** (`config/languages/en.toml`)

```toml
# Update line 34-35:
[settings.llm.models]
name = "LLM Models"
description = "Curated list of models. Populate via llm:model:discover or llm:model:add commands."
hint = "Use llm:model:discover to discover available models from your provider."
```

### Success Gates

- ✅ Hard-coded model list `"[(ollama, llama3.2), (openai, gpt-4.1-nano)]"` removed
- ✅ Default `models` field is empty list
- ✅ Default `model` field is `None`
- ✅ All field descriptions document configuration precedence
- ✅ Environment variable support preserved (lambdas kept)
- ✅ Existing tests pass
- ✅ Manual test: Fresh install shows empty model list, no phantom models

### Testing

```python
# Test: Empty initial state
def test_empty_initial_models():
    settings = LLMSettings()
    assert settings.models == []
    assert settings.model is None

# Test: Env vars still work for initial defaults
def test_env_var_defaults():
    os.environ["LLM_PROVIDER"] = "openai"
    settings = LLMSettings()
    assert settings.provider_enum == ELLMProvider.OPENAI
```

---

## Task 2: Implement Model Discovery Command

**Branch**: `task/2-discovery-command`
**Effort**: 4-6 hours
**Pre-conditions**: Task 1 complete

### Goal

Add `llm:model:discover` command that discovers ALL available models from a provider and adds them to the curated list with uniqueness checking.

### Files to Modify

1. `hatchling/ui/model_commands.py` - Add discovery command
2. `hatchling/config/languages/en.toml` - Add translation strings

### Implementation Steps

**Step 2.1: Add command definition** (`model_commands.py`)

Add to `_register_commands()` method around line 105:

```python
'llm:model:discover': {
    'handler': self._cmd_model_discover,
    'description': translate('commands.llm.model_discover_description'),
    'is_async': True,
    'args': {
        '--provider': {
            'positional': False,
            'completer_type': 'suggestions',
            'values': self.settings.llm.provider_names,
            'description': translate('commands.llm.provider_name_arg_description'),
            'required': False
        }
    }
}
```

**Step 2.2: Implement helper methods** (`model_commands.py`)

Add after existing methods:

```python
def _add_model_to_curated_list(self, new_model: ModelInfo) -> Tuple[bool, bool]:
    """Add model to curated list with uniqueness check.

    Args:
        new_model: Model to add

    Returns:
        Tuple[bool, bool]: (was_added, was_updated)
            - was_added: True if new model added
            - was_updated: True if existing model updated
    """
    # Check if model already exists (by provider + name)
    existing = next(
        (m for m in self.settings.llm.models
         if m.provider == new_model.provider and m.name == new_model.name),
        None
    )

    if existing:
        # Update status if different
        if existing.status != new_model.status:
            existing.status = new_model.status
            existing.size = new_model.size
            existing.modified_at = new_model.modified_at
            existing.digest = new_model.digest
            return (False, True)  # Not added, but updated
        return (False, False)  # Already exists, no change

    # Add new model
    self.settings.llm.models.append(new_model)
    return (True, False)  # Added, not updated

def _model_exists_in_curated_list(self, model: ModelInfo) -> bool:
    """Check if model exists in curated list.

    Args:
        model: Model to check

    Returns:
        bool: True if model exists
    """
    return any(
        m.provider == model.provider and m.name == model.name
        for m in self.settings.llm.models
    )

def _update_model_completions(self):
    """Update model name completions for commands."""
    model_names = [model.name for model in self.settings.llm.models]

    # Update completions for commands that use model names
    if 'llm:model:use' in self.commands:
        self.commands['llm:model:use']['args']['model-name']['values'] = model_names
    if 'llm:model:remove' in self.commands:
        self.commands['llm:model:remove']['args']['model-name']['values'] = model_names

def _show_provider_troubleshooting(self, provider: ELLMProvider):
    """Show provider-specific troubleshooting steps.

    Args:
        provider: Provider that is not accessible
    """
    print(f"\nTroubleshooting:")
    if provider == ELLMProvider.OLLAMA:
        print(f"  1. Check if Ollama is running: ollama list")
        print(f"  2. Verify connection: curl {self.settings.ollama.api_base}/api/tags")
        print(f"  3. Check OLLAMA_IP and OLLAMA_PORT settings")
        print(f"     Current: {self.settings.ollama.ip}:{self.settings.ollama.port}")
    elif provider == ELLMProvider.OPENAI:
        print(f"  1. Verify your OPENAI_API_KEY is set")
        print(f"  2. Check your internet connection")
        print(f"  3. Verify API base URL: {self.settings.openai.api_base}")
```

**Step 2.3: Implement command handler** (`model_commands.py`)

Add after existing command handlers:

```python
async def _cmd_model_discover(self, args: str) -> bool:
    """Discover all available models from provider and add to curated list.

    Args:
        args (str): Optional --provider flag to specify provider.

    Returns:
        bool: True to continue the chat session.
    """
    try:
        # Parse args
        args_def = self.commands['llm:model:discover']['args']
        parsed_args = self._parse_args(args, args_def)

        # Determine provider (from flag or current setting)
        provider_name = parsed_args.get('--provider', self.settings.llm.provider_enum.value)
        provider = LLMSettings.to_provider_enum(provider_name)

        print(f"Discovering models from {provider.value}...")

        # Check provider health first
        is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)
        if not is_healthy:
            print(f"✗ Provider {provider.value} is not accessible.")
            self._show_provider_troubleshooting(provider)
            return True

        # Discover models
        discovered_models = await ModelManagerAPI.list_available_models(provider, self.settings)

        if not discovered_models:
            print(f"No models found for provider {provider.value}")
            print(f"This may indicate a configuration issue.")
            self._show_provider_troubleshooting(provider)
            return True

        # Add to curated list (with uniqueness check)
        added_count = 0
        updated_count = 0

        for model in discovered_models:
            was_added, was_updated = self._add_model_to_curated_list(model)
            if was_added:
                added_count += 1
            elif was_updated:
                updated_count += 1

        # Persist to settings
        self.settings_registry.set_setting("llm", "models", self.settings.llm.models, force=True)

        # Display results
        print(f"\nDiscovered {len(discovered_models)} models:")
        for model in discovered_models[:10]:  # Show first 10
            print(f"  ✓ {model.name}")
        if len(discovered_models) > 10:
            print(f"  ... ({len(discovered_models) - 10} more)")

        print(f"\nAdded {added_count} new models to your curated list.")
        if updated_count > 0:
            print(f"Updated {updated_count} existing models.")

        if added_count == 0 and updated_count == 0:
            print("All discovered models were already in your curated list.")

        # Update command completions
        self._update_model_completions()

    except Exception as e:
        self.logger.error(f"Error in model discover command: {e}")
        print(f"✗ Model discovery failed: {e}")

    return True
```

**Step 2.4: Add translation strings** (`config/languages/en.toml`)

Add to commands section:

```toml
[commands.llm.model_discover_description]
value = "Discover all available models from provider and add to curated list"
```

### Success Gates

- ✅ `llm:model:discover` command registered and callable
- ✅ Command checks provider health before discovery
- ✅ Discovers all models from specified provider
- ✅ Adds models to curated list with uniqueness check (no duplicates)
- ✅ Updates existing models (status, size, etc.)
- ✅ Persists changes to settings file
- ✅ Clear user feedback (counts of added/updated models)
- ✅ Error handling for inaccessible provider with troubleshooting steps
- ✅ Command completions updated after discovery
- ✅ Works with both Ollama and OpenAI providers
- ✅ `--provider` flag works correctly
- ✅ Defaults to current provider when flag not specified

### Testing

```python
# Test: Discovery adds all models
async def test_model_discover_adds_all():
    cmd = ModelCommands(settings, settings_registry, style)
    initial_count = len(settings.llm.models)

    await cmd._cmd_model_discover("")

    assert len(settings.llm.models) > initial_count
    # Verify no duplicates
    model_keys = [(m.provider, m.name) for m in settings.llm.models]
    assert len(model_keys) == len(set(model_keys))

# Test: Discovery with specific provider
async def test_model_discover_with_provider():
    cmd = ModelCommands(settings, settings_registry, style)

    await cmd._cmd_model_discover("--provider openai")

    # Should have OpenAI models
    assert any(m.provider == ELLMProvider.OPENAI for m in settings.llm.models)

# Test: Discovery handles inaccessible provider
async def test_model_discover_inaccessible_provider():
    # Mock provider health check to return False
    with patch.object(ModelManagerAPI, 'check_provider_health', return_value=False):
        cmd = ModelCommands(settings, settings_registry, style)
        result = await cmd._cmd_model_discover("")
        assert result is True  # Command completes without error
```

---

## Task 3: Enhance Model Add Command

**Branch**: `task/3-enhance-add`
**Effort**: 2-3 hours
**Pre-conditions**: Task 2 complete (uses helper methods from Task 2)

### Goal

Update `llm:model:add` command to validate model exists in provider's available list before adding to curated list.

### Files to Modify

1. `hatchling/ui/model_commands.py` - Update `_cmd_model_add` method

### Implementation Steps

**Step 3.1: Update command definition** (`model_commands.py`)

Update around line 70 to add `--provider` flag:

```python
'llm:model:add': {
    'handler': self._cmd_model_add,
    'description': translate('commands.llm.model_add_description'),
    'is_async': True,  # ← Change to async
    'args': {
        'model-name': {
            'positional': True,
            'completer_type': 'suggestions',
            'values': [],  # Will be populated dynamically
            'description': translate('commands.llm.model_name_arg_description'),
            'required': True
        },
        '--provider': {
            'positional': False,
            'completer_type': 'suggestions',
            'values': self.settings.llm.provider_names,
            'description': translate('commands.llm.provider_name_arg_description'),
            'required': False
        }
    }
}
```

**Step 3.2: Rewrite command handler** (`model_commands.py`)

Replace existing `_cmd_model_add` method (around lines 203-233):

```python
async def _cmd_model_add(self, args: str) -> bool:
    """Add a specific model to curated list after validation.

    Validates that the model exists in the provider's available list before adding.
    For Ollama, this may trigger a download if the model is not local.

    Args:
        args (str): Model name and optional --provider flag.

    Returns:
        bool: True to continue the chat session.
    """
    try:
        # Parse args
        args_def = self.commands['llm:model:add']['args']
        parsed_args = self._parse_args(args, args_def)

        model_name = parsed_args.get('model-name')
        provider_name = parsed_args.get('--provider', self.settings.llm.provider_enum.value)
        provider = LLMSettings.to_provider_enum(provider_name)

        print(f"Checking availability of '{model_name}' in {provider.value}...")

        # Check provider health
        is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)
        if not is_healthy:
            print(f"✗ Provider {provider.value} is not accessible.")
            self._show_provider_troubleshooting(provider)
            return True

        # Get available models from provider
        available_models = await ModelManagerAPI.list_available_models(provider, self.settings)

        if not available_models:
            print(f"✗ No models found for provider {provider.value}")
            self._show_provider_troubleshooting(provider)
            return True

        # Search for target model
        target_model = next(
            (m for m in available_models if m.name == model_name),
            None
        )

        if not target_model:
            print(f"✗ Model '{model_name}' not found in {provider.value}")
            print(f"\nAvailable models:")
            for model in available_models[:10]:
                print(f"  - {model.name}")
            if len(available_models) > 10:
                print(f"  ... ({len(available_models) - 10} more)")
            print(f"\nTip: Run 'llm:model:discover --provider {provider.value}' to see all models.")
            return True

        # Check if already in curated list
        if self._model_exists_in_curated_list(target_model):
            print(f"Model '{model_name}' is already in your curated list.")
            return True

        # For Ollama, optionally trigger download if not local
        if provider == ELLMProvider.OLLAMA and target_model.status != ModelStatus.AVAILABLE:
            print(f"Model '{model_name}' is not downloaded locally.")
            print(f"Downloading... (this may take a while)")
            success = await ModelManagerAPI.pull_model(model_name, provider, self.settings)
            if not success:
                print(f"✗ Failed to download model '{model_name}'")
                return True
            print(f"✓ Model downloaded successfully")
            target_model.status = ModelStatus.AVAILABLE

        # Add to curated list
        self.settings.llm.models.append(target_model)

        # Persist
        self.settings_registry.set_setting("llm", "models", self.settings.llm.models, force=True)

        print(f"✓ Model found")
        print(f"✓ Added to your curated list")
        print(f"\nUse this model with: llm:model:use {model_name}")

        # Update completions
        self._update_model_completions()

    except Exception as e:
        self.logger.error(f"Error in model add command: {e}")
        print(f"✗ Failed to add model: {e}")

    return True
```

### Success Gates

- ✅ Command validates model exists in provider's available list
- ✅ Uniqueness check prevents duplicates
- ✅ Clear feedback for success/failure
- ✅ Helpful suggestions when model not found (shows available models)
- ✅ For Ollama: Triggers download if model not local
- ✅ For OpenAI: Validates model name against API
- ✅ Changes persisted to settings
- ✅ Command completions updated
- ✅ `--provider` flag works correctly
- ✅ Error handling for inaccessible provider

### Testing

```python
# Test: Add existing model
async def test_model_add_existing():
    cmd = ModelCommands(settings, settings_registry, style)

    # Discover first to get available models
    await cmd._cmd_model_discover("")
    available_model = settings.llm.models[0].name

    # Try to add again
    await cmd._cmd_model_add(available_model)
    # Should report already exists

# Test: Add non-existent model
async def test_model_add_nonexistent():
    cmd = ModelCommands(settings, settings_registry, style)

    result = await cmd._cmd_model_add("nonexistent-model-xyz")
    # Should report not found and show available models

# Test: Add with specific provider
async def test_model_add_with_provider():
    cmd = ModelCommands(settings, settings_registry, style)

    await cmd._cmd_model_add("gpt-4 --provider openai")
    # Should validate against OpenAI's model list
```

---

## Task 4: Improve Model List Display

**Branch**: `task/4-list-display`
**Effort**: 2-3 hours
**Pre-conditions**: Task 1 complete

### Goal

Add status indicators and better formatting to `llm:model:list` command.

### Files to Modify

1. `hatchling/ui/model_commands.py` - Update `_cmd_model_list` method

### Implementation Steps

**Step 4.1: Rewrite command handler** (`model_commands.py`)

Replace existing `_cmd_model_list` method (around lines 185-201):

```python
async def _cmd_model_list(self, args: str) -> bool:
    """List all curated models with status indicators.

    Shows models grouped by provider with availability status.

    Args:
        args (str): Optional filter (not implemented yet).

    Returns:
        bool: True to continue the chat session.
    """
    if not self.settings.llm.models:
        print("No models configured.")
        print(f"\nRun 'llm:model:discover' to discover models from {self.settings.llm.provider_enum.value}")
        print(f"Or run 'llm:model:add <model-name>' to add a specific model")
        return True

    print(f"\nYour Curated Models:\n")

    # Group models by provider
    from collections import defaultdict
    models_by_provider = defaultdict(list)
    for model in self.settings.llm.models:
        models_by_provider[model.provider].append(model)

    # Display each provider's models
    for provider, models in sorted(models_by_provider.items(), key=lambda x: x[0].value):
        print(f"{provider.value.capitalize()}:")

        for model_info in sorted(models, key=lambda m: m.name):
            # Status indicator
            if model_info.status == ModelStatus.AVAILABLE:
                status_icon = "✓"
            elif model_info.status == ModelStatus.NOT_AVAILABLE:
                status_icon = "✗"
            elif model_info.status == ModelStatus.DOWNLOADING:
                status_icon = "↓"
            else:
                status_icon = "?"

            # Current model indicator
            is_current = (
                model_info.name == self.settings.llm.model and
                model_info.provider == self.settings.llm.provider_enum
            )
            current_marker = " (current)" if is_current else ""

            # Size info (if available)
            size_info = ""
            if model_info.size:
                size_gb = model_info.size / (1024**3)
                size_info = f" [{size_gb:.1f}GB]"

            print(f"  {status_icon} {model_info.name}{size_info}{current_marker}")

        print()  # Blank line between providers

    # Legend
    print("Legend:")
    print("  ✓ Available    - Model is ready to use")
    print("  ✗ Unavailable  - Model is configured but not accessible")
    print("  ↓ Downloading  - Model is being downloaded")
    print("  ? Unknown      - Model status not yet validated")

    return True
```

### Success Gates

- ✅ Empty model list shows helpful guidance
- ✅ Models grouped by provider
- ✅ Status indicators displayed (✓ ✗ ↓ ?)
- ✅ Current model clearly marked
- ✅ Model size shown (if available)
- ✅ Legend explains status symbols
- ✅ Sorted alphabetically within each provider
- ✅ Clear, readable formatting

### Testing

```python
# Test: Empty list shows guidance
async def test_model_list_empty():
    settings.llm.models = []
    cmd = ModelCommands(settings, settings_registry, style)

    result = await cmd._cmd_model_list("")
    # Should show guidance message

# Test: List shows status indicators
async def test_model_list_with_models():
    settings.llm.models = [
        ModelInfo(name="llama3.2", provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ModelInfo(name="gpt-4", provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE)
    ]
    settings.llm.model = "llama3.2"
    settings.llm.provider_enum = ELLMProvider.OLLAMA

    cmd = ModelCommands(settings, settings_registry, style)
    result = await cmd._cmd_model_list("")
    # Should show both models with status indicators and current marker
```

---

## Task 5: Better Error Messages

**Branch**: `task/5-error-messages`
**Effort**: 1-2 hours
**Pre-conditions**: Tasks 2, 3 complete

### Goal

Improve error messages throughout model management commands with actionable guidance.

### Files to Modify

1. `hatchling/ui/model_commands.py` - Enhance error messages
2. `hatchling/ui/cli_chat.py` - Improve provider initialization errors

### Implementation Steps

**Step 5.1: Enhance provider initialization errors** (`cli_chat.py`)

Update around lines 80-107:

```python
try:
    ProviderRegistry.get_provider(self.settings_registry.settings.llm.provider_enum)
except Exception as e:
    msg = f"Failed to initialize {self.settings_registry.settings.llm.provider_enum.value} LLM provider: {e}"
    msg += "\n\nTroubleshooting:"

    provider = self.settings_registry.settings.llm.provider_enum
    if provider == ELLMProvider.OLLAMA:
        msg += f"\n  1. Check if Ollama is running: ollama list"
        msg += f"\n  2. Verify connection: curl {self.settings_registry.settings.ollama.api_base}/api/tags"
        msg += f"\n  3. Check your Ollama settings:"
        msg += f"\n     IP: {self.settings_registry.settings.ollama.ip}"
        msg += f"\n     Port: {self.settings_registry.settings.ollama.port}"
        msg += f"\n  4. Update settings: settings:set ollama:ip <ip>"
    elif provider == ELLMProvider.OPENAI:
        msg += f"\n  1. Verify your OPENAI_API_KEY is set"
        msg += f"\n  2. Check your internet connection"
        msg += f"\n  3. Verify API base URL: {self.settings_registry.settings.openai.api_base}"
        msg += f"\n  4. Update API key: settings:set openai:api_key <key>"

    msg += f"\n\nYou can list supported providers with: llm:provider:supported"
    msg += f"\nYou can check provider status with: llm:provider:status"

    self.logger.warning(msg)
```

**Step 5.2: Add error context to model commands** (`model_commands.py`)

The error messages are already improved in Tasks 2 and 3 via `_show_provider_troubleshooting()`.
Verify all error paths use this helper method.

**Step 5.3: Improve model:use error messages** (`model_commands.py`)

Update `_cmd_model_use` method to provide better guidance when model not found:

```python
async def _cmd_model_use(self, args: str) -> bool:
    """Set the default model to use for the current session."""
    try:
        args_def = self.commands['llm:model:use']['args']
        parsed_args = self._parse_args(args, args_def)

        model_name = parsed_args.get('model-name')

        # Find model in curated list
        model_info = next(
            (m for m in self.settings.llm.models if m.name == model_name),
            None
        )

        if not model_info:
            print(f"✗ Model '{model_name}' not found in your curated list.")
            print(f"\nYour curated models:")
            for m in self.settings.llm.models:
                print(f"  - {m.provider.value}/{m.name}")
            print(f"\nTo add this model:")
            print(f"  1. Run 'llm:model:discover' to discover all available models")
            print(f"  2. Or run 'llm:model:add {model_name}' to add this specific model")
            return True

        # Check if model is available
        if model_info.status != ModelStatus.AVAILABLE:
            print(f"⚠ Model '{model_name}' is not currently available (status: {model_info.status.value})")
            if model_info.provider == ELLMProvider.OLLAMA:
                print(f"\nTo download this model:")
                print(f"  llm:model:add {model_name}")
            else:
                print(f"\nPlease check your {model_info.provider.value} configuration.")
            return True

        # Set model and provider
        self.settings_registry.set_setting("llm", "model", model_info.name, force=True)
        self.settings_registry.set_setting("llm", "provider_enum", model_info.provider, force=True)

        print(f"✓ Switched to model: {model_info.provider.value}/{model_info.name}")

    except Exception as e:
        self.logger.error(f"Error in model use command: {e}")
        print(f"✗ Failed to switch model: {e}")

    return True
```

### Success Gates

- ✅ Provider initialization errors include troubleshooting steps
- ✅ Model not found errors show available models
- ✅ Model unavailable errors explain how to fix
- ✅ All error messages include actionable next steps
- ✅ Provider-specific guidance (Ollama vs OpenAI)
- ✅ Clear formatting with symbols (✓ ✗ ⚠)

### Testing

Manual testing of error scenarios:
- Ollama not running
- Invalid API key
- Model not in curated list
- Model not available

---

## Task 6: Update Documentation

**Branch**: `task/6-documentation`
**Effort**: 1 hour
**Pre-conditions**: Tasks 1-5 complete

### Goal

Document the new workflow, configuration precedence, and commands.

### Files to Create/Modify

1. Create `docs/user-guide/model-management.md` - New user guide
2. Update `README.md` - Add quick start for model management (if applicable)

### Implementation Steps

**Step 6.1: Create model management user guide**

Create `docs/user-guide/model-management.md`:

```markdown
# Model Management Guide

This guide explains how to manage LLM models in Hatchling.

## Configuration Precedence

Hatchling uses a three-tier configuration system:

1. **Persistent Settings** (highest priority)
   - Saved in `~/.hatch/settings/hatchling_settings.toml`
   - Modified via `settings:set` command
   - Always overrides other sources

2. **Environment Variables** (medium priority)
   - Used as initial defaults when no persistent settings exist
   - Useful for Docker, CI/CD, multi-environment setups
   - Examples: `LLM_PROVIDER`, `OLLAMA_IP`, `OLLAMA_PORT`, `OPENAI_API_KEY`

3. **Code Defaults** (lowest priority)
   - Fallback values when no other source provides configuration
   - Example: `LLM_PROVIDER=ollama`, `OLLAMA_IP=localhost`

## Model Management Workflow

### 1. Configure Provider Endpoint

```bash
# For Ollama (if not using defaults)
settings:set ollama:ip 192.168.1.100
settings:set ollama:port 11434

# For OpenAI
settings:set openai:api_key your-api-key-here
```

### 2. Discover Available Models

Discover all models from your provider:

```bash
llm:model:discover
```

Or discover from a specific provider:

```bash
llm:model:discover --provider ollama
llm:model:discover --provider openai
```

This adds ALL discovered models to your curated list.

### 3. Curate Your Model List

Remove models you don't want:

```bash
llm:model:remove unwanted-model
```

Or add specific models without bulk discovery:

```bash
llm:model:add llama3.2
llm:model:add gpt-4 --provider openai
```

### 4. List Your Models

View your curated models with status indicators:

```bash
llm:model:list
```

Output example:
```
Your Curated Models:

Ollama:
  ✓ llama3.2 [4.7GB] (current)
  ✓ codellama [3.8GB]
  ✗ mistral (not available)

OpenAI:
  ? gpt-4
  ? gpt-4-turbo

Legend:
  ✓ Available    - Model is ready to use
  ✗ Unavailable  - Model is configured but not accessible
  ↓ Downloading  - Model is being downloaded
  ? Unknown      - Model status not yet validated
```

### 5. Use a Model

Switch to a model from your curated list:

```bash
llm:model:use llama3.2
```

The provider is set automatically based on the model.

## Command Reference

### Discovery Commands

- `llm:model:discover [--provider <name>]` - Discover all models from provider
- `llm:model:add <model-name> [--provider <name>]` - Add specific model after validation
- `llm:model:list` - List curated models with status indicators

### Management Commands

- `llm:model:use <model-name>` - Switch to a model
- `llm:model:remove <model-name>` - Remove model from curated list

### Provider Commands

- `llm:provider:supported` - List supported providers
- `llm:provider:status [provider-name]` - Check provider health

## Troubleshooting

### No Models Available

If you see "No models configured":

1. Run `llm:model:discover` to discover models from your provider
2. For Ollama, ensure Ollama is running: `ollama list`
3. For OpenAI, verify your API key is set: `settings:get openai:api_key`

### Model Not Available

If a model shows ✗ (unavailable):

- **For Ollama**: Download the model with `llm:model:add <model-name>`
- **For OpenAI**: Verify the model name and your API access

### Provider Not Accessible

If provider health check fails:

**For Ollama**:
1. Check if Ollama is running: `ollama list`
2. Verify connection: `curl http://localhost:11434/api/tags`
3. Check settings: `settings:get ollama:ip` and `settings:get ollama:port`

**For OpenAI**:
1. Verify API key: `settings:get openai:api_key`
2. Check internet connection
3. Verify API base URL: `settings:get openai:api_base`

## Environment Variables

For Docker/CI/CD deployments, you can use environment variables:

```bash
# Provider selection
export LLM_PROVIDER=ollama

# Ollama configuration
export OLLAMA_IP=localhost
export OLLAMA_PORT=11434

# OpenAI configuration
export OPENAI_API_KEY=your-key-here
export OPENAI_API_URL=https://api.openai.com/v1
```

Note: Persistent settings always override environment variables.

## Examples

### Example 1: Fresh Install with Ollama

```bash
# 1. Discover all Ollama models
llm:model:discover

# 2. Remove unwanted models
llm:model:remove phi
llm:model:remove gemma

# 3. Use a model
llm:model:use llama3.2
```

### Example 2: Multi-Provider Setup

```bash
# 1. Discover Ollama models
llm:model:discover --provider ollama

# 2. Add specific OpenAI models
llm:model:add gpt-4 --provider openai
llm:model:add gpt-4-turbo --provider openai

# 3. List all models
llm:model:list

# 4. Switch between providers by using models
llm:model:use llama3.2  # Uses Ollama
llm:model:use gpt-4     # Uses OpenAI
```

### Example 3: Targeted Model Addition

```bash
# Add specific model without discovering all
llm:model:add llama3.2 --provider ollama

# Use the model
llm:model:use llama3.2
```
```

**Step 6.2: Update README (if applicable)**

Add a "Model Management" section to the main README with a link to the detailed guide.

### Success Gates

- ✅ Model management user guide created
- ✅ Configuration precedence documented
- ✅ Workflow documented with examples
- ✅ All commands documented
- ✅ Troubleshooting guide included
- ✅ Environment variable usage documented
- ✅ README updated (if applicable)

---

## Success Criteria

### Functional Requirements

- ✅ No hard-coded phantom models in default configuration
- ✅ Empty model list on fresh install
- ✅ `llm:model:discover` command discovers all models from provider
- ✅ `llm:model:add` command validates before adding
- ✅ `llm:model:remove` command removes from curated list
- ✅ `llm:model:list` command shows status indicators
- ✅ Uniqueness enforced (no duplicate models)
- ✅ Environment variables work for deployment scenarios
- ✅ Persistent settings override environment variables
- ✅ Clear error messages with troubleshooting steps

### Quality Requirements

- ✅ All existing tests pass
- ✅ New functionality has test coverage
- ✅ No performance degradation
- ✅ Clear, helpful user feedback at every step
- ✅ Documentation complete and accurate

### User Experience Goals

- ✅ First run: Clear guidance when no models configured
- ✅ Configuration: Easy to understand what's configured vs available
- ✅ Errors: Actionable troubleshooting steps in error messages
- ✅ Discovery: Intuitive workflow (discover → curate → use)
- ✅ Visibility: Always clear which provider and model is active

---

## Deferred Features

These features are out of scope for this fix but may be considered for future releases:

**Automatic Model Validation on Startup**:
- Rationale: Adds startup time, user prefers manual control
- Timeline: Future enhancement if user feedback indicates need
- Benefit: Automatic status updates for configured models

**Model Management Abstraction (LLMModelManager)**:
- Rationale: Over-engineering for current needs
- Timeline: Phase 2 (if architectural refactoring needed)
- Benefit: Cleaner separation of concerns

**User-First Configuration System (SQLite storage)**:
- Rationale: Current TOML-based system works well
- Timeline: Phase 2-3 (if proven necessary)
- Benefit: More flexible storage, better querying

---

## Testing Strategy

### Unit Tests

**Configuration Tests**:
- Test empty initial model list
- Test environment variable defaults
- Test persistent settings override env vars

**Discovery Tests**:
- Test model discovery for each provider
- Test uniqueness enforcement
- Test model merging from multiple providers

**Validation Tests**:
- Test model add with validation
- Test model add with non-existent model
- Test duplicate prevention

### Integration Tests

**Workflow Tests**:
- Test complete discovery workflow
- Test multi-provider workflow
- Test model selection after discovery

**Command Tests**:
- Test all model commands
- Test error handling in commands
- Test command completions update

### Manual Testing

**Scenarios**:
1. Fresh install with no configuration
2. Ollama running with models
3. Ollama not running
4. OpenAI with valid API key
5. OpenAI with invalid API key
6. Multi-provider setup
7. Model curation workflow

---

## Topological Ordering

**Critical Path** (must complete in order):
1. Task 1 (Clean Defaults) - Foundation for all other tasks
2. Task 2 (Discovery Command) - Core functionality
3. Task 3 (Enhance Add) - Depends on Task 2 helper methods
4. Task 5 (Error Messages) - Depends on Tasks 2, 3
5. Task 6 (Documentation) - Depends on all tasks

**Parallel Opportunities**:
- Task 4 (List Display) can start after Task 1 (parallel with Task 2)
- Task 5 (Error Messages) can start alongside Tasks 2-3
- Task 6 (Documentation) can be written while testing Tasks 1-5

---

**Report Version**: v2
**Status**: Ready for Implementation
**Next Steps**:
1. Create `fix/llm-management` branch from `main`
2. Create task branches for each task
3. Begin implementation with Task 1 (Clean Up Default Configuration)
4. Test after each task completion
5. Merge all tasks to fix branch
6. Final testing and merge to main


