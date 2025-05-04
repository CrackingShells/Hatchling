#!/usr/bin/env python3
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

if __name__ == "__main__":
    print("Running all Hatch package management system tests...")
    
    # Discover and run all tests in current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Set exit code based on test results
    sys.exit(0 if result.wasSuccessful() else 1)