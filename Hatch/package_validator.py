import json
import logging
import jsonschema
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class PackageValidationError(Exception):
    """Exception raised for package validation errors."""
    pass

class HatchPackageValidator:
    def __init__(self):
        """Initialize the Hatch package validator."""
        self.logger = logging.getLogger("hatch.package_validator")
        self.logger.setLevel(logging.INFO)
        self.schema_path = Path(__file__).parent.parent / "Hatch-Schemas" / "package" / "latest" / "hatch_pkg_metadata_schema.json"
    
    def validate_metadata_schema(self, metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Validate the metadata against the JSON schema.
        
        Args:
            metadata: The metadata to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of validation errors)
        """
        
        # Load schema
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load schema: {e}")
            return False, [f"Failed to load schema: {e}"]
        
        # Validate against schema
        errors = []
        try:
            jsonschema.validate(instance=metadata, schema=schema)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            return False, errors
    
    def validate_entry_point_exists(self, package_dir: Path, entry_point: str) -> Tuple[bool, List[str]]:
        """
        Validate that the entry point file exists.
        
        Args:
            package_dir: Path to the package directory
            entry_point: Name of the entry point file
            
        Returns:
            Tuple[bool, List[str]]: (exists, list of validation errors)
        """
        entry_path = package_dir / entry_point
        if not entry_path.exists():
            return False, [f"Entry point file '{entry_point}' does not exist"]
        if not entry_path.is_file():
            return False, [f"Entry point '{entry_point}' is not a file"]
        return True, []
    
    def validate_tools_exist(self, package_dir: Path, entry_point: str, tools: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate that the tools declared in metadata exist in the entry point file.
        
        Args:
            package_dir: Path to the package directory
            entry_point: Name of the entry point file
            tools: List of tool definitions from metadata
            
        Returns:
            Tuple[bool, List[str]]: (all_exist, list of validation errors)
        """
        if not tools:
            return True, []
            
        errors = []
        all_exist = True
        
        # Import the module
        try:
            module_path = package_dir / entry_point
            spec = importlib.util.spec_from_file_location("module.name", module_path)
            if spec is None or spec.loader is None:
                return False, [f"Could not load entry point module: {entry_point}"]
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for each tool
            for tool in tools:
                tool_name = tool.get('name')
                if not tool_name:
                    errors.append(f"Tool missing name in metadata")
                    all_exist = False
                    continue
                    
                if not hasattr(module, tool_name):
                    errors.append(f"Tool '{tool_name}' not found in entry point")
                    all_exist = False
                    continue
                    
                # Ensure it's a callable function
                tool_obj = getattr(module, tool_name)
                if not callable(tool_obj):
                    errors.append(f"Tool '{tool_name}' exists but is not a callable function")
                    all_exist = False
                    
        except Exception as e:
            return False, [f"Error validating tools: {str(e)}"]
            
        return all_exist, errors
    
    def validate_package(self, package_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a Hatch package in the specified directory.
        
        Args:
            package_dir: Path to the package directory
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation results)
        """
        results = {
            'valid': True,
            'metadata_schema': {'valid': False, 'errors': []},
            'entry_point': {'valid': False, 'errors': []},
            'tools': {'valid': False, 'errors': []},
            'metadata': None
        }
        
        # Check if package directory exists
        if not package_dir.exists() or not package_dir.is_dir():
            results['valid'] = False
            results['metadata_schema']['errors'].append(f"Package directory does not exist: {package_dir}")
            return False, results
        
        # Check for metadata file
        metadata_path = package_dir / "hatch_metadata.json"
        if not metadata_path.exists():
            results['valid'] = False
            results['metadata_schema']['errors'].append("hatch_metadata.json not found")
            return False, results
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                results['metadata'] = metadata
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            results['valid'] = False
            results['metadata_schema']['errors'].append(f"Failed to parse metadata: {e}")
            return False, results
        
        # Validate metadata schema
        schema_valid, schema_errors = self.validate_metadata_schema(metadata)
        results['metadata_schema']['valid'] = schema_valid
        results['metadata_schema']['errors'] = schema_errors
        
        # If schema validation failed, don't continue
        if not schema_valid:
            results['valid'] = False
            return False, results
        
        # Get entry point from metadata
        entry_point = metadata.get('entry_point')
        if not entry_point:
            results['valid'] = False
            results['entry_point']['errors'].append("No entry_point specified in metadata")
            return False, results
        
        # Validate entry point
        entry_valid, entry_errors = self.validate_entry_point_exists(package_dir, entry_point)
        results['entry_point']['valid'] = entry_valid
        results['entry_point']['errors'] = entry_errors
        
        if not entry_valid:
            results['valid'] = False
        
        # Validate tools
        tools = metadata.get('tools', [])
        if tools:
            tools_valid, tools_errors = self.validate_tools_exist(package_dir, entry_point, tools)
            results['tools']['valid'] = tools_valid
            results['tools']['errors'] = tools_errors
            
            if not tools_valid:
                results['valid'] = False
        else:
            results['tools']['valid'] = True
        
        return results['valid'], results