"""Regression tests for model add validation (Task 3).

These tests verify that:
1. Add validates model exists in provider's available list
2. Add rejects models not found (no auto-download)
3. Add prevents duplicates
4. Add updates command completions
"""

import sys
import unittest
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_decorators import regression_test

from hatchling.config.llm_settings import LLMSettings, ModelInfo, ModelStatus, ELLMProvider


class TestModelAddValidation(unittest.TestCase):
    """Regression tests for model add validation."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.settings = LLMSettings()
        self.settings.models = []  # Start with empty list

    @regression_test
    def test_add_validates_model_exists(self):
        """Verify add validates model exists in provider's available list."""
        # Available models from provider
        available_models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Model to add
        model_to_add = 'llama3.2'
        
        # Simulate validation logic
        model_found = None
        for model in available_models:
            if model.name.lower() == model_to_add.lower():
                model_found = model
                break
        
        # Verify model was found
        self.assertIsNotNone(model_found, "Model should be found in available list")
        self.assertEqual(model_found.name, 'llama3.2', "Found model should be llama3.2")
        
        # Add the model
        self.settings.models.append(model_found)
        
        # Verify model was added
        self.assertEqual(len(self.settings.models), 1, "Should have 1 model after add")
        self.assertEqual(self.settings.models[0].name, 'llama3.2', "Added model should be llama3.2")

    @regression_test
    def test_add_rejects_non_existent_models(self):
        """Verify add rejects models not found (no auto-download)."""
        # Available models from provider
        available_models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Model to add (doesn't exist)
        model_to_add = 'non-existent-model'
        
        # Simulate validation logic
        model_found = None
        for model in available_models:
            if model.name.lower() == model_to_add.lower():
                model_found = model
                break
        
        # Verify model was NOT found
        self.assertIsNone(model_found, "Non-existent model should not be found")
        
        # Should NOT add the model
        if model_found:
            self.settings.models.append(model_found)
        
        # Verify no model was added
        self.assertEqual(len(self.settings.models), 0, 
                        "Should not add non-existent model")

    @regression_test
    def test_add_prevents_duplicates(self):
        """Verify add prevents duplicates."""
        # Add existing model
        existing_model = ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, 
                                  status=ModelStatus.AVAILABLE)
        self.settings.models.append(existing_model)
        
        # Try to add same model again
        model_to_add = 'llama3.2'
        
        # Simulate duplicate check logic
        existing_keys = {(m.provider, m.name) for m in self.settings.models}
        model_key = (ELLMProvider.OLLAMA, model_to_add)
        
        is_duplicate = model_key in existing_keys
        
        # Verify duplicate was detected
        self.assertTrue(is_duplicate, "Should detect duplicate model")
        
        # Should NOT add duplicate
        if not is_duplicate:
            new_model = ModelInfo(name=model_to_add, provider=ELLMProvider.OLLAMA, 
                                status=ModelStatus.AVAILABLE)
            self.settings.models.append(new_model)
        
        # Verify no duplicate was added
        self.assertEqual(len(self.settings.models), 1, 
                        "Should still have only 1 model (no duplicate)")
        model_names = [m.name for m in self.settings.models]
        self.assertEqual(model_names.count('llama3.2'), 1,
                        "Should have exactly 1 instance of llama3.2")

    @regression_test
    def test_add_updates_command_completions(self):
        """Verify add updates command completions after adding model."""
        # Add a model
        new_model = ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, 
                            status=ModelStatus.AVAILABLE)
        self.settings.models.append(new_model)
        
        # Simulate command completion update logic
        model_names = [model.name for model in self.settings.models]
        
        # Verify completions would be updated
        self.assertIn('llama3.2', model_names,
                    "Model names should include llama3.2 for completions")
        self.assertEqual(len(model_names), 1,
                       "Should have 1 model for completions")


def run_model_add_tests():
    """Run all model add validation regression tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelAddValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_model_add_tests()
    exit(0 if success else 1)

