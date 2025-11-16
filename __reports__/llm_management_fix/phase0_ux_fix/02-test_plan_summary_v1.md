# Test Plan Summary - LLM Management UX Fix v1

**Date**: 2025-11-13  
**Phase**: Test Definition (Phase 2)  
**Status**: Ready for Review  
**Full Test Plan**: [02-test_plan_v1.md](./02-test_plan_v1.md)

---

## Changes from v0

### Removed (6 tests eliminated)

**Meta-Constraint Tests** (not algorithmic):
- ❌ Empty initial model list - Just checking a default value
- ❌ Empty initial model field - Just checking a default value

**Error Message Content Tests** (implementation detail):
- ❌ Empty list shows guidance - Checking error message content
- ❌ Provider init error includes troubleshooting - Checking error message content
- ❌ Model not found error shows available models - Checking error message content
- ❌ Model unavailable error explains how to fix - Checking error message content

**Rationale**: These tests check implementation details and meta-constraints rather than actual behavior. Once the code is written correctly, these constraints will stay valid. Testing them pollutes the test suite without adding algorithmic value.

### Kept (18 tests)

All behavioral tests that verify actual functionality work correctly.

### Outcome

**New Test Count**: 18 automated tests (down from 24)  
**New Test-to-Code Ratio**: 3:1 (18 tests / 6 tasks) - within target 2:1 to 3:1  
**Manual Scenarios**: 7 (unchanged - valuable for UX validation)

---

## Executive Summary

Comprehensive test plan with **18 automated tests** and **7 manual test scenarios** focused on behavioral functionality.

**Key Metrics**:
- **Test-to-Code Ratio**: 3:1 (18 tests / 6 tasks) - within target 2:1 to 3:1 for bug fixes
- **Coverage Goals**: 70% minimum, 90% for critical functionality, 95% for new features
- **Execution Time**: < 5 minutes for full test suite
- **Test Framework**: Wobble with proper categorization decorators

---

## Test Organization

Tests are organized by **functional groups**:

### 1. Configuration (1 test)
- Environment variables work for deployment

### 2. Model Discovery (5 tests)
- Discovery adds all models from provider
- Discovery with --provider flag
- Uniqueness enforcement (no duplicates)
- Discovery updates existing models
- Discovery handles inaccessible provider

### 3. Model Validation and Addition (4 tests)
- Add validates model exists before adding
- Add rejects non-existent model
- Add with --provider flag
- Add handles inaccessible provider

### 4. Model List Display (3 tests)
- List groups models by provider
- List shows status indicators (✓ ✗ ↓ ?)
- List marks current model

### 5. Integration Tests (2 tests)
- Complete discovery workflow (discover → curate → use)
- Multi-provider workflow (Ollama + OpenAI)

### 6. Edge Cases and Regressions (3 tests)
- Empty provider model list
- Model name conflicts across providers
- Large number of models (100+)
- Existing model use command still works
- Settings persistence after discovery

---

## Manual Test Scenarios

**7 UX validation scenarios**:
1. Fresh install experience (no phantom models)
2. Ollama running with models (discovery works)
3. Ollama not running (graceful error handling)
4. OpenAI with valid API key (integration works)
5. OpenAI with invalid API key (graceful error handling)
6. Multi-provider setup (both providers work together)
7. Model curation workflow (discover → remove → use)

---

## Test Distribution by Task

| Task | Description | Test Count | Categories |
|------|-------------|------------|------------|
| 1 | Clean Up Default Configuration | 1 | Regression |
| 2 | Implement Model Discovery | 5 | Integration (3), Regression (2) |
| 3 | Enhance Model Add | 4 | Regression (3), Integration (1) |
| 4 | Improve Model List Display | 3 | Regression |
| - | Integration Workflows | 2 | Integration |
| - | Edge Cases & Regressions | 3 | Regression |

**Total**: 18 automated tests + 7 manual scenarios

---

## Key Testing Principles Applied

✅ **Focus on behavioral functionality**, not meta-constraints  
✅ **Test-to-code ratio** within target range (2:1 to 3:1 for bug fixes)  
✅ **Functional grouping** for clarity (not arbitrary categories)  
✅ **Clear acceptance criteria** for each test  
✅ **Manual testing** for UX validation  
✅ **Edge cases and regression prevention** covered  
✅ **Trust boundaries** respected (don't test Pydantic, Python stdlib)  
✅ **Don't test implementation details** (error messages, default values)

---

## Acceptance Criteria Summary

### Configuration (Task 1)
- ✅ Environment variables work for deployment
- ✅ Fresh install shows empty model list (manual test)

### Model Discovery (Task 2)
- ✅ Discovery adds all models from provider
- ✅ Uniqueness enforcement prevents duplicates
- ✅ Provider health check works
- ✅ Graceful error handling when provider inaccessible

### Model Addition (Task 3)
- ✅ Validation prevents adding non-existent models
- ✅ Provider flag works correctly
- ✅ Graceful error handling when provider inaccessible

### Model List Display (Task 4)
- ✅ Models grouped by provider
- ✅ Status indicators shown correctly (✓ ✗ ↓ ?)
- ✅ Current model marked

### Integration Tests
- ✅ Complete workflow works end-to-end
- ✅ Multi-provider setup works seamlessly

### Edge Cases & Regressions
- ✅ Empty model lists handled gracefully
- ✅ Model name conflicts handled correctly
- ✅ Large model lists handled efficiently
- ✅ Existing functionality not broken

---

## Test Execution Plan

### Running Tests with Wobble

```bash
# Run all tests
wobble --log-file test_execution_v1.txt --log-verbosity 3

# Run specific categories
wobble --category regression --log-file regression_results.txt --log-verbosity 3
wobble --category integration --log-file integration_results.txt --log-verbosity 3

# Run specific test files
wobble --pattern "test_llm_configuration.py" --log-verbosity 3
wobble --pattern "test_model_discovery.py" --log-verbosity 3
```

### Recommended Execution Order

1. Configuration tests (Task 1) - Foundation
2. Discovery tests (Task 2) - Core functionality
3. Addition tests (Task 3) - Validation logic
4. List display tests (Task 4) - UI improvements
5. Integration tests - End-to-end workflows
6. Edge cases and regressions - Boundary conditions

---

## Next Steps

1. **User Review**: Review test plan and provide feedback
2. **Iteration**: Refine test specifications based on feedback (if needed)
3. **Implementation**: Implement tests during Task 1-6 development
4. **Execution**: Run tests and validate results
5. **Reporting**: Create test execution report (Phase 4)

---

## Files Created

- **[02-test_plan_v1.md](./02-test_plan_v1.md)** - Complete test plan with detailed specifications
- **[02-test_plan_summary_v1.md](./02-test_plan_summary_v1.md)** - This summary document

---

**Status**: ✅ Test Plan v1 Complete - Ready for Review  
**Next Phase**: Implementation (Phase 3) - pending user approval
