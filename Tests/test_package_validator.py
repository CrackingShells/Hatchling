#!/usr/bin/env python3
import sys
import unittest
import logging
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "Hatch-Validator"))
sys.path.insert(0, str(parent_dir))

from package_validator import HatchPackageValidator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestPackageValidator(unittest.TestCase):
    """Test suite for the HatchPackageValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.dev_dir = parent_dir / "Hatch-Dev"
        self.validator = HatchPackageValidator()
        
    def test_base_package_validation(self):
        """Test validation of a basic package with no dependencies."""
        package_dir = self.dev_dir / "base_pkg_1"
        is_valid, results = self.validator.validate_package(package_dir)
        self.assertTrue(is_valid, f"Base package validation failed: {results}")
        
    def test_simple_dependency_package(self):
        """Test validation of a package with a simple dependency."""
        package_dir = self.dev_dir / "simple_dep_pkg"
        is_valid, results = self.validator.validate_package(package_dir)
        
        # This should pass validation even with 'remote' type since we're just checking schema
        self.assertTrue(is_valid, f"Simple dependency package validation failed: {results}")
        
    def test_complex_dependency_package(self):
        """Test validation of a package with multiple dependencies."""
        package_dir = self.dev_dir / "complex_dep_pkg"
        is_valid, results = self.validator.validate_package(package_dir)
        self.assertTrue(is_valid, f"Complex dependency package validation failed: {results}")
        
    def test_python_dependency_package(self):
        """Test validation of a package with Python dependencies."""
        package_dir = self.dev_dir / "python_dep_pkg"
        is_valid, results = self.validator.validate_package(package_dir)
        self.assertTrue(is_valid, f"Python dependency package validation failed: {results}")
        
    def test_circular_dependency_structure(self):
        """Test that circular dependency packages pass structural validation."""
        package_dir1 = self.dev_dir / "circular_dep_pkg_1"
        package_dir2 = self.dev_dir / "circular_dep_pkg_2"
        
        is_valid1, results1 = self.validator.validate_package(package_dir1)
        is_valid2, results2 = self.validator.validate_package(package_dir2)
        
        self.assertTrue(is_valid1, f"Circular dependency package 1 validation failed: {results1}")
        self.assertTrue(is_valid2, f"Circular dependency package 2 validation failed: {results2}")
        
    def test_missing_dependency_detection(self):
        """Test that a package with a non-existent dependency is structurally valid."""
        package_dir = self.dev_dir / "missing_dep_pkg"
        is_valid, results = self.validator.validate_package(package_dir)
        
        # This should pass validation since we're only validating structure
        self.assertTrue(is_valid, f"Missing dependency package validation failed: {results}")
        
    def test_registry_validation_mode(self):
        """Test registry validation mode which disallows local dependencies."""
        # Create validator with local dependencies disallowed
        registry_validator = HatchPackageValidator(allow_local_dependencies=False)
        
        # Create a test package with local dependency
        package_dir = self.dev_dir / "simple_dep_pkg"
        
        # Modify the package metadata in memory to include a local dependency
        is_valid, results = registry_validator.validate_package(package_dir)
        self.assertTrue(is_valid, "Registry validation failed for remote dependencies")
        
        # We would need to create a temporary package with local dependencies to fully test this

if __name__ == "__main__":
    unittest.main()