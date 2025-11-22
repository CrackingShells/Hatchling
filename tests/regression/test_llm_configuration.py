"""Regression tests for LLM configuration cleanup (Task 1).

These tests verify that:
1. Hard-coded phantom models are removed
2. Default models list is empty
3. Default model is None
4. Environment variables still work for deployment
5. ModelStatus enum is simplified to AVAILABLE/NOT_AVAILABLE only
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_decorators import regression_test

from hatchling.config.llm_settings import LLMSettings, ModelStatus, ELLMProvider
from hatchling.config.ollama_settings import OllamaSettings
from hatchling.config.openai_settings import OpenAISettings


class TestLLMConfigurationCleanup(unittest.TestCase):
    """Regression tests for LLM configuration cleanup."""

    def setUp(self):
        """Save original environment variables before each test."""
        self._original_env = dict(os.environ)

    def tearDown(self):
        """Restore environment variables after each test."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @regression_test
    def test_default_models_list_is_empty(self):
        """Verify that default models list is empty (no phantom models)."""
        # Clear any LLM_MODELS env var
        os.environ.pop('LLM_MODELS', None)
        
        settings = LLMSettings()
        
        self.assertEqual(len(settings.models), 0,
                        "Default models list should be empty (no phantom models)")

    @regression_test
    def test_default_model_is_none(self):
        """Verify that default model is None (must be explicitly selected)."""
        # Clear any LLM_MODEL env var
        os.environ.pop('LLM_MODEL', None)
        
        settings = LLMSettings()
        
        self.assertIsNone(settings.model,
                         "Default model should be None (must be explicitly selected)")

    @regression_test
    def test_model_status_enum_simplified(self):
        """Verify that ModelStatus enum only has AVAILABLE and NOT_AVAILABLE."""
        # Check that only expected statuses exist
        status_values = [status.value for status in ModelStatus]
        
        self.assertIn('available', status_values,
                     "ModelStatus should have AVAILABLE")
        self.assertIn('not_available', status_values,
                     "ModelStatus should have NOT_AVAILABLE")
        self.assertEqual(len(status_values), 2,
                        "ModelStatus should only have 2 statuses (AVAILABLE, NOT_AVAILABLE)")

    @regression_test
    def test_environment_variable_llm_provider_works(self):
        """Verify LLM_PROVIDER env var sets initial provider."""
        os.environ['LLM_PROVIDER'] = 'openai'
        
        settings = LLMSettings()
        
        self.assertEqual(settings.provider_enum, ELLMProvider.OPENAI,
                        "LLM_PROVIDER env var should set initial provider")

    @regression_test
    def test_environment_variable_llm_models_works(self):
        """Verify LLM_MODELS env var provides initial models for deployment."""
        os.environ['LLM_MODELS'] = '[(ollama, llama3.2), (openai, gpt-4)]'
        
        settings = LLMSettings()
        
        self.assertEqual(len(settings.models), 2,
                        "LLM_MODELS env var should provide initial models")
        model_names = [m.name for m in settings.models]
        self.assertIn('llama3.2', model_names,
                     "LLM_MODELS should include llama3.2")
        self.assertIn('gpt-4', model_names,
                     "LLM_MODELS should include gpt-4")

    @regression_test
    def test_ollama_env_vars_set_endpoint(self):
        """Verify OLLAMA_IP and OLLAMA_PORT env vars work."""
        os.environ['OLLAMA_IP'] = '192.168.1.100'
        os.environ['OLLAMA_PORT'] = '11435'
        
        settings = OllamaSettings()
        
        self.assertEqual(settings.ip, '192.168.1.100',
                        "OLLAMA_IP env var should set IP address")
        self.assertEqual(settings.port, 11435,
                        "OLLAMA_PORT env var should set port")

    @regression_test
    def test_openai_api_key_env_var_works(self):
        """Verify OPENAI_API_KEY env var works."""
        os.environ['OPENAI_API_KEY'] = 'test-api-key-12345'
        
        settings = OpenAISettings()
        
        self.assertEqual(settings.api_key, 'test-api-key-12345',
                        "OPENAI_API_KEY env var should set API key")

    @regression_test
    def test_no_hard_coded_phantom_models(self):
        """Verify no hard-coded phantom models like llama3.2 or gpt-4.1-nano."""
        # Clear env vars to test code defaults only
        os.environ.pop('LLM_MODELS', None)
        os.environ.pop('LLM_MODEL', None)
        
        settings = LLMSettings()
        
        # Should have no models by default
        self.assertEqual(len(settings.models), 0,
                        "Should have no hard-coded phantom models")
        
        # Should have no default model
        self.assertIsNone(settings.model,
                         "Should have no hard-coded default model")


def run_llm_configuration_tests():
    """Run all LLM configuration regression tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLLMConfigurationCleanup))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_llm_configuration_tests()
    exit(0 if success else 1)

