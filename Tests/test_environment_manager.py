#!/usr/bin/env python3
import sys
import unittest
import logging
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from Hatch.package_environments import HatchEnvironmentManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestEnvironmentManager(unittest.TestCase):
    """Test suite for the HatchEnvironmentManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for environments
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create environments directory
        self.envs_dir = self.test_dir / "envs"
        self.envs_dir.mkdir()
        
        # Mock registry path
        self.registry_path = parent_dir / "Hatch-Registry" / "hatch_packages_registry.json"
        
        # Development packages directory
        self.dev_dir = parent_dir / "Hatch-Dev"
        
        # Copy relevant files to test directory to avoid modifying the real ones
        # We need to modify the environment manager to accept a custom environments_dir for testing
        
        # Initialize environment manager with test directory
        self.env_manager = self._create_test_env_manager()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_env_manager(self):
        """Create an environment manager for testing purposes.
        
        This requires modifying the HatchEnvironmentManager class to accept
        a custom environments directory, or patching it for testing.
        """
        # For now, we'll use the standard environment manager, but in a real test
        # we would need to modify it to use our test directory
        return HatchEnvironmentManager(registry_path=self.registry_path)
    
    def test_environment_creation(self):
        """Test creating a new environment."""
        # Create a test environment
        result = self.env_manager.create_environment("test_env", "Test environment")
        self.assertTrue(result, "Failed to create environment")
        
        # Check if the environment exists
        self.assertTrue(self.env_manager.environment_exists("test_env"), 
                       "Environment was not created or cannot be found")
        
        # Check if the environment is in the list
        env_list = self.env_manager.list_environments()
        env_names = [env["name"] for env in env_list]
        self.assertIn("test_env", env_names, "Environment not found in list")
        
    def test_set_current_environment(self):
        """Test setting the current environment."""
        # Create a test environment
        self.env_manager.create_environment("test_env_2", "Another test environment")
        
        # Set as current
        result = self.env_manager.set_current_environment("test_env_2")
        self.assertTrue(result, "Failed to set current environment")
        
        # Check if it's the current environment
        current_env = self.env_manager.get_current_environment()
        self.assertEqual(current_env, "test_env_2", "Current environment was not set correctly")
    
    def test_environment_removal(self):
        """Test removing an environment."""
        # Create a test environment
        self.env_manager.create_environment("temp_env", "Temporary environment")
        
        # Verify it exists
        self.assertTrue(self.env_manager.environment_exists("temp_env"), 
                       "Environment was not created")
        
        # Remove it
        result = self.env_manager.remove_environment("temp_env")
        self.assertTrue(result, "Failed to remove environment")
        
        # Verify it no longer exists
        self.assertFalse(self.env_manager.environment_exists("temp_env"), 
                        "Environment was not removed")
    
    def test_package_installation(self):
        """Test installing a package to an environment."""
        # Create a test environment
        self.env_manager.create_environment("pkg_test_env", "Package testing environment")
        self.env_manager.set_current_environment("pkg_test_env")
        
        # Install a base package
        base_pkg_path = self.dev_dir / "base_pkg_1"
        result = self.env_manager.add_package_to_environment(str(base_pkg_path))
        
        # This test might fail in the current implementation if it has dependencies
        # on the actual filesystem structure. In a real test, we'd need to mock
        # the file operations or modify the environment manager to be more testable.
        
        # Check if the package is installed
        packages = self.env_manager.list_packages_in_environment("pkg_test_env")
        package_names = [pkg["name"] for pkg in packages]
        self.assertIn("base_pkg_1", package_names, "Package was not installed correctly")
    
    def test_dependency_installation(self):
        """Test installing a package with dependencies."""
        # Create a test environment
        self.env_manager.create_environment("dep_test_env", "Dependency testing environment")
        self.env_manager.set_current_environment("dep_test_env")
        
        # First install the base package
        base_pkg_path = self.dev_dir / "base_pkg_1"
        self.env_manager.add_package_to_environment(str(base_pkg_path))
        
        # Now install a package that depends on the base package
        simple_dep_path = self.dev_dir / "simple_dep_pkg"
        result = self.env_manager.add_package_to_environment(str(simple_dep_path))
        
        # Check if both packages are installed
        packages = self.env_manager.list_packages_in_environment("dep_test_env")
        package_names = [pkg["name"] for pkg in packages]
        self.assertIn("base_pkg_1", package_names, "Base package was not installed")
        self.assertIn("simple_dep_pkg", package_names, "Dependent package was not installed")

if __name__ == "__main__":
    unittest.main()