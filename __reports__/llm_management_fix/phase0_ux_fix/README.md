# LLM Management UX Fix - Phase 0

This directory contains analysis and implementation planning for fixing the critical UX issue where users are confused about which LLM API endpoint and model is actually accessible when running Hatchling.

---

## Documents

### Analysis Reports

- **[00-adequation_assessment_v2.md](./00-adequation_assessment_v2.md)** ⭐ **CURRENT** - Second revision after user feedback
  - Keeps environment variables for deployment flexibility
  - Clarifies bulk discovery workflow (add all, then curate)
  - Specifies uniqueness enforcement (in logic, not data structure)
  - Precise command specifications with validation
  - **Status**: Ready for Review
  - **Appendix**: [00-adequation_assessment_v2_appendix.md](./00-adequation_assessment_v2_appendix.md)

- **[00-adequation_assessment_v1.md](./00-adequation_assessment_v1.md)** 📦 **ARCHIVED** - First revision (superseded)
  - Suggested removing all env vars (incorrect)
  - Misunderstood discovery workflow

- **[00-adequation_assessment_v0.md](./00-adequation_assessment_v0.md)** 📦 **ARCHIVED** - Initial assessment (superseded)
  - Contains errors: claimed gaps that already exist
  - Misunderstood configuration issue
  - Over-engineered solution (8 tasks)

### Implementation Planning

- **[01-implementation_roadmap_v3.md](./01-implementation_roadmap_v3.md)** ⭐ **CURRENT** - Implementation roadmap v3 (Critical fixes)
  - Addresses test style, status indicators, discovery behavior, and docs workflow
  - 5 focused tasks (Task 6 documentation deferred for stakeholder interaction)
  - Estimated effort: 10-15 hours (1.25-2 days)
  - Uses unittest.TestCase assertions (Wobble style)
  - Removes DOWNLOADING and UNKNOWN statuses
  - Clarifies that discover/add only work with already-available models
  - **Status**: Ready for Implementation

- **[01-implementation_roadmap_v2.md](./01-implementation_roadmap_v2.md)** 📦 **ARCHIVED** - Implementation roadmap v2 (superseded)
  - Based on v2 assessment
  - Contains issues fixed in v3 (test style, status indicators)
  - 6 tasks with some task 6 autonomous docs

- **[01-implementation_roadmap_v0.md](./01-implementation_roadmap_v0.md)** 📦 **ARCHIVED** - Initial roadmap (superseded)
  - Based on v0 assessment with errors
  - 8 tasks including unnecessary ones

### Test Planning

- **[02-test_plan_v2.md](./02-test_plan_v2.md)** ⭐ **CURRENT** - Test plan v2 (Corrected assertions and statuses)
  - 15 automated tests + 6 manual scenarios
  - All tests use `unittest.TestCase` with `self.assert*()` methods
  - Removes DOWNLOADING and UNKNOWN status tests
  - Clarifies that tests validate behavior, not implementation details
  - Focus on Ollama/OpenAI actual workflows
  - **Status**: Ready for Test Implementation

- **[02-test_plan_v1.md](./02-test_plan_v1.md)** 📦 **ARCHIVED** - Test plan v1 (superseded)
  - Contains issues fixed in v2 (test assertions, status indicators)
  - 18 automated tests with some meta-constraint tests
  - **Status**: Ready for Review
  - **Summary**: [02-test_plan_summary_v1.md](./02-test_plan_summary_v1.md)

- **[02-test_plan_v0.md](./02-test_plan_v0.md)** 📦 **ARCHIVED** - Initial test plan (superseded)
  - 24 automated tests (over-tested meta-constraints)
  - Included error message content tests (implementation details)
  - Refined in v1 to focus on behavioral tests

### Supporting Documents

- **[WORK_SESSION_SUMMARY.md](./WORK_SESSION_SUMMARY.md)** - Session overview and key decisions

---

## Quick Summary

### The UX Issue

Users are confused about which LLM API endpoint and model is actually accessible when running Hatchling:
- Configuration changes don't take effect without restart
- Phantom models shown that aren't actually available
- Unclear error messages when models fail
- No visibility into actual provider/model availability

### Root Causes

1. **Configuration Timing**: Environment variables captured at import time via `default_factory` lambdas
2. **Phantom Models**: Hard-coded default models that may not exist: `"[(ollama, llama3.2), (openai, gpt-4.1-nano)]"`
3. **No Validation**: Models pre-registered as AVAILABLE without checking actual provider state
4. **Poor Feedback**: No status indicators or actionable error messages

### Solution Approach

**Phase 0 - Quick Wins** (1.75-2.75 days):
1. ✅ Fix configuration timing (remove lambda captures)
2. ✅ Remove hard-coded default models
3. ✅ Add provider health check on startup
4. ✅ Add model validation on startup
5. ✅ Implement model discovery command
6. ✅ Add automatic discovery on provider switch
7. ✅ Improve error messages and status indicators
8. ✅ Update documentation

**Deferred to Future**:
- ❌ Model management abstraction (LLMModelManager)
- ❌ User-first configuration system (SQLite storage)
- ❌ Security encryption (keyring + Fernet)
- ❌ Major architectural refactoring

### Key Improvements Over Original Plan

The original Phase 0 (from strategic_implementation_roadmap_v2.md) had 3 tasks:
1. Configuration timing fix
2. Default model cleanup
3. Model discovery command

**Refined Phase 0 (v2) has 6 tasks**:
1. Clean up default configuration (remove hard-coded models, KEEP env vars)
2. Implement model discovery command (bulk add all models)
3. Enhance model add command (validate before adding)
4. Improve model list display (status indicators)
5. Better error messages
6. Documentation updates

**Why the refinement (v2)**:
- ✅ Keep env vars for deployment flexibility (Docker, CI/CD)
- ✅ Remove hard-coded phantom models (core issue)
- ✅ Bulk discovery workflow: add all, then curate by removing
- ✅ Uniqueness enforcement in logic (not data structure)
- ✅ Clear precedence: Persistent > Env > Code defaults
- ✅ Leverages existing infrastructure (health checks, validation already exist)

---

## Implementation Status

- ⏳ **Phase 0**: Not Started
  - ✅ Phase 1: Analysis - Complete
  - ✅ Phase 2: Test Definition - Complete (Ready for Review)
  - ⏳ Phase 3: Implementation - Not Started
    - Task 1: Clean Up Default Configuration - Not Started
    - Task 2: Implement Model Discovery Command - Not Started
    - Task 3: Enhance Model Add Command - Not Started
    - Task 4: Improve Model List Display - Not Started
    - Task 5: Better Error Messages - Not Started
  - ⏳ Phase 4: Test Execution - Not Started
  - ⏳ Phase 5: Stakeholder Documentation Review - Not Started (Task 6 deferred)

---

## Success Criteria

### Functional Requirements
- ✅ Runtime configuration changes work without restart
- ✅ No phantom models in default configuration
- ✅ Automatic model discovery on startup and provider switch
- ✅ Clear status indicators for model availability
- ✅ Actionable error messages with troubleshooting steps

### Quality Requirements
- ✅ No regressions in existing functionality
- ✅ Test coverage for all new functionality
- ✅ No noticeable performance degradation
- ✅ Clear, helpful user feedback at every step

### User Experience Goals
- ✅ First-run experience provides clear guidance
- ✅ Always clear what's configured vs actually available
- ✅ Errors include actionable troubleshooting steps
- ✅ Automatic discovery reduces manual steps
- ✅ Visibility into active provider and model

---

## Related Documents

### Source Analysis
- `__reports__/llm_management_fix/phase1_analysis/architectural_analysis_v1.md` - Comprehensive architectural analysis
- `__reports__/llm_management_fix/phase1_analysis/strategic_implementation_roadmap_v2.md` - Original strategic roadmap

### Affected Components
- `hatchling/config/llm_settings.py` - LLM configuration
- `hatchling/config/ollama_settings.py` - Ollama configuration
- `hatchling/config/openai_settings.py` - OpenAI configuration
- `hatchling/config/settings.py` - Application settings
- `hatchling/ui/model_commands.py` - Model management commands
- `hatchling/core/llm/model_manager_api.py` - Model management API

---

## Next Steps

1. **Review Reports**: Stakeholder review of adequation assessment and implementation roadmap
2. **Approve Scope**: Confirm enhanced Phase 0 scope (8 tasks vs original 3)
3. **Create Branch**: `git checkout -b fix/llm-management-ux`
4. **Begin Implementation**: Start with Task 1 (Configuration Timing Fix)
5. **Iterative Testing**: Test after each task completion
6. **User Validation**: Get feedback after core tasks (1-5) complete
7. **Final Review**: Complete testing and documentation
8. **Merge and Release**: Merge to main, communicate changes to users

---

## Risk Assessment

**Overall Risk Level**: Low

**Key Risks**:
- R1: Breaking changes to configuration (Low probability, High impact) - Mitigated by backward compatibility
- R2: Performance impact from health checks (Low probability, Medium impact) - Mitigated by async checks and caching
- R3: Discovery failures (Medium probability, Medium impact) - Mitigated by comprehensive error handling
- R4: Empty model list confusion (Medium probability, Medium impact) - Mitigated by clear guidance messages
- R5: Migration friction (Low probability, Low impact) - Mitigated by one-time migration and clear communication

---

## Timeline

**Estimated Effort**: 10-15 hours (1.25-2 days)

**Task Breakdown**:
- Task 1: Simplify Configuration - 1-2 hours
- Task 2: Remove Defaults - 1 hour
- Task 3: Discovery Command - 4-6 hours
- Task 4: Model List Display - 2-3 hours
- Task 5: Error Messages - 1-2 hours
- Task 6: Documentation - 1 hour

**Parallel Opportunities**:
- Tasks 4 and 5 can be developed in parallel
- Task 6 can be written while testing Tasks 1-5

---

**Last Updated**: 2025-11-07
**Status**: Ready for Implementation
**Next Action**: Begin implementation with Task 1 (Clean Up Default Configuration)

---

## Quick Start for Developers

### Setup

```bash
# Create fix branch
git checkout -b fix/llm-management

# Create first task branch
git checkout -b task/1-clean-defaults
```

### Implementation Order

1. **Task 1**: Clean Up Default Configuration (1-2h)
   - Remove hard-coded models
   - Update field descriptions
   - Test: Empty initial state

2. **Task 2**: Implement Model Discovery Command (4-6h)
   - Add `llm:model:discover` command
   - Implement uniqueness checking
   - Test: Discovery workflow

3. **Task 3**: Enhance Model Add Command (2-3h)
   - Update `llm:model:add` with validation
   - Test: Add existing/non-existent models

4. **Task 4**: Improve Model List Display (2-3h)
   - Add status indicators
   - Group by provider
   - Test: Display formatting

5. **Task 5**: Better Error Messages (1-2h)
   - Enhance error messages
   - Add troubleshooting steps
   - Test: Error scenarios

6. **Task 6**: Update Documentation (1h)
   - Create user guide
   - Document workflow
   - Update README

### Testing

After each task:
```bash
# Run unit tests
pytest tests/

# Manual testing
python -m hatchling
# Test the implemented commands
```

### Merge

After all tasks complete:
```bash
# Merge to fix branch
git checkout fix/llm-management
git merge task/6-documentation

# Final testing
pytest tests/
# Manual regression testing

# Merge to main
git checkout main
git merge fix/llm-management
```

