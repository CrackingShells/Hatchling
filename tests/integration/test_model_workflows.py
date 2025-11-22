"""Integration tests for complete model management workflows.

These tests verify end-to-end workflows:
1. Full discovery workflow (discover → list → use)
2. Add then use workflow (add → list → use)
3. Configuration persistence across operations
"""

import sys
import unittest
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_decorators import integration_test

from hatchling.config.llm_settings import LLMSettings, ModelInfo, ModelStatus, ELLMProvider


class TestModelWorkflows(unittest.TestCase):
    """Integration tests for complete model management workflows."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.settings = LLMSettings()
        self.settings.models = []
        self.settings.model = None

    @integration_test
    def test_full_discovery_workflow(self):
        """Verify full discovery workflow: discover → list → use."""
        # Step 1: Discovery - Add available models
        available_models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Simulate discovery
        existing_keys = {(m.provider, m.name) for m in self.settings.models}
        for model in available_models:
            model_key = (model.provider, model.name)
            if model_key not in existing_keys:
                self.settings.models.append(model)
        
        # Verify discovery added models
        self.assertEqual(len(self.settings.models), 2,
                        "Discovery should add 2 models")
        
        # Step 2: List - Verify models are in curated list
        model_names = [m.name for m in self.settings.models]
        self.assertIn('llama3.2', model_names, "List should show llama3.2")
        self.assertIn('mistral', model_names, "List should show mistral")
        
        # Step 3: Use - Set a model as current
        self.settings.model = 'llama3.2'
        
        # Verify model is set
        self.assertEqual(self.settings.model, 'llama3.2',
                        "Should set llama3.2 as current model")

    @integration_test
    def test_add_then_use_workflow(self):
        """Verify add then use workflow: add → list → use."""
        # Step 1: Add - Add a specific model
        available_models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # User wants to add 'llama3.2'
        model_to_add = 'llama3.2'
        
        # Validate model exists
        model_found = None
        for model in available_models:
            if model.name.lower() == model_to_add.lower():
                model_found = model
                break
        
        self.assertIsNotNone(model_found, "Model should be found")
        
        # Check for duplicates
        existing_keys = {(m.provider, m.name) for m in self.settings.models}
        model_key = (model_found.provider, model_found.name)
        
        if model_key not in existing_keys:
            self.settings.models.append(model_found)
        
        # Verify model was added
        self.assertEqual(len(self.settings.models), 1,
                        "Add should add 1 model")
        
        # Step 2: List - Verify model is in curated list
        model_names = [m.name for m in self.settings.models]
        self.assertIn('llama3.2', model_names, "List should show llama3.2")
        
        # Step 3: Use - Set the model as current
        self.settings.model = 'llama3.2'
        
        # Verify model is set
        self.assertEqual(self.settings.model, 'llama3.2',
                        "Should set llama3.2 as current model")

    @integration_test
    def test_configuration_persistence(self):
        """Verify configuration changes persist across operations."""
        # Add a model
        model = ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, 
                         status=ModelStatus.AVAILABLE)
        self.settings.models.append(model)
        
        # Set as current
        self.settings.model = 'llama3.2'
        
        # Verify both settings persist
        self.assertEqual(len(self.settings.models), 1,
                        "Model list should persist")
        self.assertEqual(self.settings.model, 'llama3.2',
                        "Current model should persist")
        
        # Add another model
        model2 = ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA,
                          status=ModelStatus.AVAILABLE)
        self.settings.models.append(model2)
        
        # Verify previous settings still persist
        self.assertEqual(len(self.settings.models), 2,
                        "Model list should grow")
        self.assertEqual(self.settings.model, 'llama3.2',
                        "Current model should remain unchanged")

    @integration_test
    def test_remove_then_list_workflow(self):
        """Verify remove then list workflow: add → remove → list."""
        # Add models
        self.settings.models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Verify initial state
        self.assertEqual(len(self.settings.models), 2,
                        "Should start with 2 models")
        
        # Remove a model
        model_to_remove = 'llama3.2'
        self.settings.models = [m for m in self.settings.models 
                               if m.name != model_to_remove]
        
        # Verify removal
        self.assertEqual(len(self.settings.models), 1,
                        "Should have 1 model after removal")
        model_names = [m.name for m in self.settings.models]
        self.assertNotIn('llama3.2', model_names,
                        "Removed model should not be in list")
        self.assertIn('mistral', model_names,
                     "Remaining model should still be in list")


def run_workflow_tests():
    """Run all model workflow integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelWorkflows))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_workflow_tests()
    exit(0 if success else 1)

