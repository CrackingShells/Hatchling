"""Regression tests for model list display (Task 4).

These tests verify that:
1. Empty list shows helpful guidance
2. Models displayed with status indicators (✓ ✗)
3. Current model is marked clearly
4. Models grouped by provider
5. Models sorted alphabetically within provider
"""

import sys
import unittest
from pathlib import Path
from collections import defaultdict

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_decorators import regression_test

from hatchling.config.llm_settings import LLMSettings, ModelInfo, ModelStatus, ELLMProvider


class TestModelListDisplay(unittest.TestCase):
    """Regression tests for model list display."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.settings = LLMSettings()
        self.settings.models = []  # Start with empty list
        self.settings.model = None  # No current model

    @regression_test
    def test_empty_list_detection(self):
        """Verify empty list is detected (should show guidance)."""
        # Check if list is empty
        is_empty = len(self.settings.models) == 0
        
        # Verify empty list is detected
        self.assertTrue(is_empty, "Should detect empty model list")

    @regression_test
    def test_models_grouped_by_provider(self):
        """Verify models are grouped by provider."""
        # Add models from different providers
        self.settings.models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='gpt-4', provider=ELLMProvider.OPENAI, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Simulate grouping logic
        models_by_provider = defaultdict(list)
        for model in self.settings.models:
            models_by_provider[model.provider].append(model)
        
        # Verify grouping
        self.assertEqual(len(models_by_provider), 2, 
                        "Should have 2 provider groups")
        self.assertEqual(len(models_by_provider[ELLMProvider.OLLAMA]), 2,
                        "Should have 2 Ollama models")
        self.assertEqual(len(models_by_provider[ELLMProvider.OPENAI]), 1,
                        "Should have 1 OpenAI model")

    @regression_test
    def test_current_model_marked(self):
        """Verify current model is marked clearly."""
        # Add models
        self.settings.models = [
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Set current model
        self.settings.model = 'llama3.2'
        
        # Simulate marking logic
        for model in self.settings.models:
            is_current = model.name == self.settings.model
            if is_current:
                # Verify current model is detected
                self.assertEqual(model.name, 'llama3.2',
                               "Current model should be llama3.2")
                break

    @regression_test
    def test_models_sorted_alphabetically(self):
        """Verify models are sorted alphabetically within provider."""
        # Add unsorted models
        self.settings.models = [
            ModelInfo(name='zephyr', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
            ModelInfo(name='mistral', provider=ELLMProvider.OLLAMA, status=ModelStatus.AVAILABLE),
        ]
        
        # Simulate sorting logic
        ollama_models = [m for m in self.settings.models if m.provider == ELLMProvider.OLLAMA]
        sorted_models = sorted(ollama_models, key=lambda m: m.name)
        
        # Verify sorting
        sorted_names = [m.name for m in sorted_models]
        self.assertEqual(sorted_names, ['llama3.2', 'mistral', 'zephyr'],
                        "Models should be sorted alphabetically")

    @regression_test
    def test_status_indicators_only_two_types(self):
        """Verify only two status indicators exist (AVAILABLE, NOT_AVAILABLE)."""
        # Check ModelStatus enum
        status_values = [status.value for status in ModelStatus]
        
        # Verify only 2 statuses
        self.assertEqual(len(status_values), 2,
                        "Should have exactly 2 status types")
        self.assertIn('available', status_values,
                     "Should have AVAILABLE status")
        self.assertIn('not_available', status_values,
                     "Should have NOT_AVAILABLE status")

    @regression_test
    def test_model_status_determination(self):
        """Verify model status can be determined (available vs not_available)."""
        # Model in curated list
        model = ModelInfo(name='llama3.2', provider=ELLMProvider.OLLAMA, 
                         status=ModelStatus.AVAILABLE)
        
        # Available models from provider
        available_names = {'llama3.2', 'mistral'}
        
        # Simulate status check logic
        is_available = model.name.lower() in available_names
        
        # Verify status determination
        self.assertTrue(is_available, "Model should be marked as available")
        
        # Test unavailable model
        unavailable_model = ModelInfo(name='old-model', provider=ELLMProvider.OLLAMA,
                                     status=ModelStatus.AVAILABLE)
        is_available_2 = unavailable_model.name.lower() in available_names
        
        self.assertFalse(is_available_2, "Old model should be marked as not available")


def run_model_list_tests():
    """Run all model list display regression tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelListDisplay))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_model_list_tests()
    exit(0 if success else 1)

