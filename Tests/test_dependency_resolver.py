#!/usr/bin/env python3
import sys
import unittest
import logging
import tempfile
import json
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "Hatch-Validator"))
sys.path.insert(0, str(parent_dir))

from dependency_resolver import DependencyResolver

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestDependencyResolver(unittest.TestCase):
    """Test suite for the DependencyResolver class."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry_path = parent_dir / "Hatch-Registry" / "hatch_packages_registry.json"
        self.resolver = DependencyResolver(registry_path=self.registry_path)
        self.dev_dir = parent_dir / "Hatch-Dev"
        
    def test_version_compatibility(self):
        """Test version compatibility checking."""
        # Test exact version match
        self.assertTrue(
            self.resolver.is_version_compatible("1.0.0", "==", "1.0.0"),
            "Exact version match failed"
        )
        
        # Test greater than or equal
        self.assertTrue(
            self.resolver.is_version_compatible("1.1.0", ">=", "1.0.0"),
            "Greater than or equal failed"
        )
        
        # Test less than or equal
        self.assertTrue(
            self.resolver.is_version_compatible("0.9.0", "<=", "1.0.0"),
            "Less than or equal failed"
        )
        
        # Test not equal
        self.assertTrue(
            self.resolver.is_version_compatible("1.1.0", "!=", "1.0.0"),
            "Not equal version check failed"
        )
        
        # Test version mismatch
        self.assertFalse(
            self.resolver.is_version_compatible("0.9.0", "==", "1.0.0"),
            "Version mismatch should be detected"
        )
        
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # First, we need to ensure the circular dependency packages are in the registry
        # For testing purposes, we'll create a temporary registry with our test packages
        temp_registry = self._create_temp_registry_with_circular_deps()
        
        # Create a resolver with the temporary registry
        resolver = DependencyResolver(registry_path=temp_registry)
        
        # Test circular dependency detection
        has_circular, cycle = resolver.check_circular_dependencies("circular_dep_pkg_1", "1.0.0")
        
        self.assertTrue(has_circular, "Circular dependency was not detected")
        self.assertIn("circular_dep_pkg_1", cycle, "Circular dependency cycle incorrect")
        self.assertIn("circular_dep_pkg_2", cycle, "Circular dependency cycle incorrect")
        
    def test_dependency_resolution(self):
        """Test resolving all dependencies for a package."""
        # Create a temporary registry with all test packages
        temp_registry = self._create_temp_registry_with_all_packages()
        
        # Create a resolver with the temporary registry
        resolver = DependencyResolver(registry_path=temp_registry)
        
        # Resolve dependencies for complex_dep_pkg which depends on base_pkg_1 and base_pkg_2
        result = resolver._resolve_registry_dependencies("complex_dep_pkg", "1.0.0")
        
        # Check that all expected packages are resolved
        packages = {pkg["name"] for pkg in result.get("resolved_packages", [])}
        self.assertIn("complex_dep_pkg", packages, "Main package not in resolved packages")
        self.assertIn("base_pkg_1", packages, "Dependency base_pkg_1 not resolved")
        self.assertIn("base_pkg_2", packages, "Dependency base_pkg_2 not resolved")

    def _create_temp_registry_with_circular_deps(self):
        """Create a temporary registry file with circular dependencies for testing."""
        registry_data = {
            "registry_schema_version": "1.0.0",
            "last_updated": "2025-05-04T12:00:00Z",
            "artifact_base_url": "https://artifacts.crackingshells.org/packages",
            "repositories": [
                {
                    "name": "test-repo",
                    "url": "https://example.com/test-repo",
                    "packages": [
                        {
                            "name": "circular_dep_pkg_1",
                            "description": "Package with circular dependency for testing",
                            "category": "testing",
                            "tags": ["test", "dependency", "circular"],
                            "versions": [
                                {
                                    "version": "1.0.0",
                                    "path": "circular_dep_pkg_1",
                                    "metadata_path": "hatch_metadata.json",
                                    "base_version": None,
                                    "dependencies_added": [
                                        {"name": "circular_dep_pkg_2", "version_constraint": ">=1.0.0"}
                                    ],
                                    "artifacts": [],
                                    "added_date": "2025-05-04T12:00:00Z"
                                }
                            ],
                            "latest_version": "1.0.0"
                        },
                        {
                            "name": "circular_dep_pkg_2",
                            "description": "Package with circular dependency for testing",
                            "category": "testing",
                            "tags": ["test", "dependency", "circular"],
                            "versions": [
                                {
                                    "version": "1.0.0",
                                    "path": "circular_dep_pkg_2",
                                    "metadata_path": "hatch_metadata.json",
                                    "base_version": None,
                                    "dependencies_added": [
                                        {"name": "circular_dep_pkg_1", "version_constraint": ">=1.0.0"}
                                    ],
                                    "artifacts": [],
                                    "added_date": "2025-05-04T12:00:00Z"
                                }
                            ],
                            "latest_version": "1.0.0"
                        }
                    ],
                    "last_indexed": "2025-05-04T12:00:00Z"
                }
            ],
            "stats": {
                "total_packages": 2,
                "total_versions": 2,
                "total_artifacts": 0
            }
        }
        
        # Create a temporary file and write the registry data to it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(temp_file.name, 'w') as f:
            json.dump(registry_data, f)
        
        return temp_file.name
        
    def _create_temp_registry_with_all_packages(self):
        """Create a temporary registry file with all test packages for testing."""
        registry_data = {
            "registry_schema_version": "1.0.0",
            "last_updated": "2025-05-04T12:00:00Z",
            "artifact_base_url": "https://artifacts.crackingshells.org/packages",
            "repositories": [
                {
                    "name": "test-repo",
                    "url": "https://example.com/test-repo",
                    "packages": [
                        {
                            "name": "base_pkg_1",
                            "description": "Base package with no dependencies for testing",
                            "category": "testing",
                            "tags": ["test", "base"],
                            "versions": [
                                {
                                    "version": "1.0.0",
                                    "path": "base_pkg_1",
                                    "metadata_path": "hatch_metadata.json",
                                    "base_version": None,
                                    "dependencies_added": [],
                                    "artifacts": [],
                                    "added_date": "2025-05-04T12:00:00Z"
                                }
                            ],
                            "latest_version": "1.0.0"
                        },
                        {
                            "name": "base_pkg_2",
                            "description": "Second base package with no dependencies for testing",
                            "category": "testing",
                            "tags": ["test", "base"],
                            "versions": [
                                {
                                    "version": "1.0.0",
                                    "path": "base_pkg_2",
                                    "metadata_path": "hatch_metadata.json",
                                    "base_version": None,
                                    "dependencies_added": [],
                                    "artifacts": [],
                                    "added_date": "2025-05-04T12:00:00Z"
                                }
                            ],
                            "latest_version": "1.0.0"
                        },
                        {
                            "name": "complex_dep_pkg",
                            "description": "Package with multiple dependencies for testing",
                            "category": "testing",
                            "tags": ["test", "dependency", "complex"],
                            "versions": [
                                {
                                    "version": "1.0.0",
                                    "path": "complex_dep_pkg",
                                    "metadata_path": "hatch_metadata.json",
                                    "base_version": None,
                                    "dependencies_added": [
                                        {"name": "base_pkg_1", "version_constraint": ">=1.0.0"},
                                        {"name": "base_pkg_2", "version_constraint": ">=1.0.0"}
                                    ],
                                    "artifacts": [],
                                    "added_date": "2025-05-04T12:00:00Z"
                                }
                            ],
                            "latest_version": "1.0.0"
                        }
                    ],
                    "last_indexed": "2025-05-04T12:00:00Z"
                }
            ],
            "stats": {
                "total_packages": 3,
                "total_versions": 3,
                "total_artifacts": 0
            }
        }
        
        # Create a temporary file and write the registry data to it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(temp_file.name, 'w') as f:
            json.dump(registry_data, f)
        
        return temp_file.name

if __name__ == "__main__":
    unittest.main()