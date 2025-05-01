import logging
import json
from pathlib import Path

import template_generator as tg
from package_validator import HatchPackageValidator
from package_environments import HatchEnvironmentManager

class HatchPackageManager:
    def __init__(self):
        """Initialize the Hatch package manager."""
        self.logger = logging.getLogger("hatch.package_manager")
        self.validator = HatchPackageValidator()
        self.env_manager = HatchEnvironmentManager()
        
        self.logger.setLevel(logging.INFO)

    def create_package_template(self,
                                target_dir: Path, package_name: str, category: str = "", description: str = ""):
        """
        Create a template package with the required structure in the specified directory.
        
        Args:
            target_dir: Directory where the package will be created
            package_name: Name of the package
            category: Package category
            description: Package description
            
        Returns:
            Path: Path to the created package directory
        """
        package_dir = target_dir / package_name
        
        # Check if directory already exists
        if package_dir.exists():
            self.logger.warning(f"Package directory already exists: {package_dir}")
            return package_dir
        
        # Create package directory
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create init file
        with open(package_dir / "__init__.py", 'w') as f:
            f.write(tg.generate_init_py())
        
        # Create server file
        with open(package_dir / "server.py", 'w') as f:
            f.write(tg.generate_server_py(package_name))
        
        # Create metadata file
        metadata = tg.generate_metadata_json(package_name, category, description)
        with open(package_dir / "hatch_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        with open(package_dir / "README.md", 'w') as f:
            f.write(tg.generate_readme_md(package_name, description))
        
        self.logger.info(f"Created package template at: {package_dir}")
        return package_dir
        
    def validate_package(self, package_dir: Path) -> bool:
        """
        Validate a Hatch package in the specified directory.
        
        Args:
            package_dir: Path to the package directory
            
        Returns:
            bool: True if package is valid, False otherwise
        """
        self.logger.info(f"Validating package: {package_dir}")
        
        # Run validation
        is_valid, results = self.validator.validate_package(package_dir)
        
        # Log validation results
        if is_valid:
            self.logger.info(f"Package validation successful: {package_dir}")
        else:
            self.logger.error(f"Package validation failed: {package_dir}")
            
            if not results['metadata_schema']['valid']:
                for error in results['metadata_schema']['errors']:
                    self.logger.error(f"Metadata schema error: {error}")
                    
            if not results['entry_point']['valid']:
                for error in results['entry_point']['errors']:
                    self.logger.error(f"Entry point error: {error}")
                    
            if not results['tools']['valid']:
                for error in results['tools']['errors']:
                    self.logger.error(f"Tool validation error: {error}")
        
        return is_valid
        
    # Environment management functions
    
    def create_environment(self, name: str, description: str = "") -> bool:
        """
        Create a new environment.
        
        Args:
            name: Name of the environment
            description: Description of the environment
            
        Returns:
            bool: True if created successfully, False if environment already exists
        """
        return self.env_manager.create_environment(name, description)
    
    def remove_environment(self, name: str) -> bool:
        """
        Remove an environment.
        
        Args:
            name: Name of the environment to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        return self.env_manager.remove_environment(name)
    
    def list_environments(self) -> list:
        """
        List all available environments.
        
        Returns:
            List[Dict]: List of environment information dictionaries
        """
        return self.env_manager.list_environments()
    
    def get_current_environment(self) -> str:
        """
        Get the name of the current environment.
        
        Returns:
            str: Name of the current environment
        """
        return self.env_manager.get_current_environment()
    
    def set_current_environment(self, name: str) -> bool:
        """
        Set the current environment.
        
        Args:
            name: Name of the environment to set as current
            
        Returns:
            bool: True if successful, False if environment doesn't exist
        """
        return self.env_manager.set_current_environment(name)
    
    def environment_exists(self, name: str) -> bool:
        """
        Check if an environment exists.
        
        Args:
            name: Name of the environment to check
            
        Returns:
            bool: True if environment exists, False otherwise
        """
        return self.env_manager.environment_exists(name)