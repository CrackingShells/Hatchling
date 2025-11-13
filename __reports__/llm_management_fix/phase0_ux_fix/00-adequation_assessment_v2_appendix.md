# Adequation Assessment v2 - Appendix

## Command Specifications

### llm:model:discover

**Syntax**:
```bash
llm:model:discover [--provider <provider_name>]
```

**Arguments**:
- `--provider` (optional): Provider name (ollama, openai). Defaults to current provider.

**Behavior**:
1. Determine target provider (from flag or current setting)
2. Call `ModelManagerAPI.list_available_models(provider)`
3. For each discovered model:
   - Check if already in curated list (by provider + name)
   - Add if not present
   - Update status if already present
4. Persist updated model list to settings
5. Display summary to user

**Output**:
```
Discovering models from ollama...
Discovered 10 models:
  ✓ llama3.2
  ✓ codellama
  ✓ mistral
  ✓ phi
  ✓ gemma
  ... (5 more)

Added 8 new models to your curated list.
Updated 2 existing models.
```

**Error Handling**:
- Provider not accessible: Show troubleshooting steps
- No models found: Inform user, suggest checking provider configuration
- Network error: Show error message, suggest retry

---

### llm:model:add

**Syntax**:
```bash
llm:model:add <model_name> [--provider <provider_name>]
```

**Arguments**:
- `<model_name>` (required): Name of model to add
- `--provider` (optional): Provider name. Defaults to current provider.

**Behavior**:
1. Determine target provider (from flag or current setting)
2. Call `ModelManagerAPI.list_available_models(provider)`
3. Search for `<model_name>` in available models
4. If found:
   - Check if already in curated list
   - Add if not present (with uniqueness check)
   - Inform user if already present
   - For Ollama: Optionally trigger download if not local
5. If not found:
   - Report model not found
   - Show list of available models (or suggest `llm:model:discover`)
6. Persist updated model list to settings

**Output (Success)**:
```
Checking availability of 'llama3.2' in ollama...
✓ Model found
✓ Added to your curated list

Use this model with: llm:model:use llama3.2
```

**Output (Already Exists)**:
```
Model 'llama3.2' is already in your curated list.
```

**Output (Not Found)**:
```
✗ Model 'nonexistent' not found in ollama

Available models:
  - llama3.2
  - codellama
  - mistral
  ... (more)

Tip: Run 'llm:model:discover' to see all available models.
```

---

### llm:model:remove

**Syntax**:
```bash
llm:model:remove <model_name>
```

**Arguments**:
- `<model_name>` (required): Name of model to remove

**Behavior**:
1. Search for `<model_name>` in curated list
2. If found:
   - Remove from list
   - Persist updated list to settings
   - Inform user
3. If not found:
   - Report model not in curated list
   - Show current curated list

**Output (Success)**:
```
✓ Removed 'phi' from your curated list
```

**Output (Not Found)**:
```
✗ Model 'nonexistent' not found in your curated list

Your curated models:
  - ollama/llama3.2
  - ollama/codellama
  - openai/gpt-4
```

---

### llm:model:list

**Enhanced with Status Indicators**

**Syntax**:
```bash
llm:model:list
```

**Output**:
```
Your Curated Models:

Ollama:
  ✓ llama3.2 (current)
  ✓ codellama
  ✗ mistral (not available)

OpenAI:
  ? gpt-4 (not validated)
  ? gpt-4-turbo

Legend:
  ✓ Available    - Model is ready to use
  ✗ Unavailable  - Model is configured but not accessible
  ? Unknown      - Model status not yet validated
```

---

## Refined Task List

### Task 1: Clean Up Default Configuration (1-2 hours)

**Goal**: Remove hard-coded phantom models while preserving env var support

**Changes**:

**1. Remove hard-coded model list**:
```python
# hatchling/config/llm_settings.py

# BEFORE:
models: List[ModelInfo] = Field(
    default_factory=lambda: [
        ModelInfo(name=model[1], provider=model[0], status=ModelStatus.AVAILABLE)
        for model in LLMSettings.extract_provider_model_list(
            os.environ.get("LLM_MODELS", "") if os.environ.get("LLM_MODELS") 
            else "[(ollama, llama3.2), (openai, gpt-4.1-nano)]"  # ← Remove this
        )
    ]
)

# AFTER:
models: List[ModelInfo] = Field(
    default_factory=list,  # ← Empty list
    description="Curated list of models. Populate via llm:model:discover or llm:model:add.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
)
```

**2. Keep env var support, update descriptions**:
```python
provider_enum: ELLMProvider = Field(
    default_factory=lambda: LLMSettings.to_provider_enum(
        os.environ.get("LLM_PROVIDER", "ollama")
    ),
    description="LLM provider. Set via LLM_PROVIDER env var or settings:set command. "
                "Persistent settings override env vars.",
    json_schema_extra={"access_level": SettingAccessLevel.NORMAL}
)
```

**3. Update all env var field descriptions** to document precedence

**Success Gates**:
- ✅ Hard-coded model list removed
- ✅ Default model list is empty
- ✅ Env var support preserved for deployment
- ✅ Field descriptions document precedence
- ✅ Existing tests pass

---

### Task 2: Implement Model Discovery Command (4-6 hours)

**Goal**: Add `llm:model:discover` command with bulk add functionality

See main implementation roadmap document for detailed implementation.

**Key Features**:
- Discovers all models from provider
- Adds all to curated list with uniqueness check
- Updates existing models (status)
- Persists to settings
- Clear user feedback

**Success Gates**:
- ✅ Command discovers all models from provider
- ✅ Models added to curated list with uniqueness check
- ✅ Existing models updated (status)
- ✅ Changes persisted to settings
- ✅ Clear user feedback
- ✅ Error handling for inaccessible provider

---

### Task 3: Enhance Model Add Command (2-3 hours)

**Goal**: Update `llm:model:add` to validate before adding

**Enhanced Behavior**:
- Check if model exists in provider's available list
- Add to curated list if found (with uniqueness check)
- Report if not found
- For Ollama: Optionally trigger download

**Success Gates**:
- ✅ Validates model exists before adding
- ✅ Uniqueness check prevents duplicates
- ✅ Clear feedback for success/failure
- ✅ Helpful suggestions when model not found
- ✅ Changes persisted to settings

---

### Task 4: Improve Model List Display (2-3 hours)

**Goal**: Add status indicators and better formatting

**Features**:
- Status indicators (✓ ✗ ?)
- Group by provider
- Indicate current model
- Legend for symbols

---

### Task 5: Better Error Messages (1-2 hours)

**Goal**: Actionable guidance when things go wrong

**Features**:
- Provider-specific troubleshooting
- Suggested next steps
- Clear error descriptions

---

### Task 6: Update Documentation (1 hour)

**Goal**: Document workflow, precedence, and commands

**Sections**:
- Configuration precedence
- Model management workflow
- Command reference
- Troubleshooting guide

---

**Total Effort**: 10-15 hours (1.25-2 days)

