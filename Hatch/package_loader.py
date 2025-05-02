import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from package_validator import HatchPackageValidator
from package_environments import HatchEnvironmentManager


class PackageLoaderError(Exception):
    """Exception raised for package loading errors."""
    pass

class HatchPackageLoader:
    def __init__(self, pkg_manager, env_manager: HatchEnvironmentManager, validator: HatchPackageValidator):
        """Initialize the Hatch package loader.

        Args:
            pkg_manager: Package manager instance
            env_manager: Environment manager instance
            validator: Package validator instance
        """
        
        self.logger = logging.getLogger("hatch.package_loader")
        self.logger.setLevel(logging.INFO)
        self.pkg_manager = pkg_manager
        self.env_manager = env_manager
        self.validator = validator
    
    def _resolve_package_path(self, package_dir: str) -> Path:
        """
        Resolve package directory path, handling both absolute and relative paths.
        
        Args:
            package_dir: Directory path as string (absolute or relative)
            
        Returns:
            Path: Resolved absolute path to the package directory
        """
        path = Path(package_dir)
        
        # If not absolute, make it absolute from current directory
        if not path.is_absolute():
            path = Path.cwd() / path
            
        # Normalize path
        path = path.resolve()
        
        return path
    
    def _get_package_metadata(self, package_dir: Path) -> Dict[str, Any]:
        """
        Get package metadata from the package directory.
        
        Args:
            package_dir: Path to the package directory
            
        Returns:
            Dict[str, Any]: Package metadata
            
        Raises:
            PackageLoaderError: If metadata file doesn't exist or is invalid
        """
        metadata_path = package_dir / "hatch_metadata.json"
        
        if not metadata_path.exists():
            raise PackageLoaderError(f"Metadata file not found in package directory: {package_dir}")
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise PackageLoaderError(f"Failed to parse package metadata: {e}")
    
    def add_package_to_environment(self, package_dir: str) -> bool:
        """
        Add a package to the current environment.
        
        Args:
            package_dir: Directory path of the package (absolute or relative)
            
        Returns:
            bool: True if package was added successfully
            
        Raises:
            PackageLoaderError: If package validation fails or other errors occur
        """
        # Resolve package path
        path = self._resolve_package_path(package_dir)
        
        # Validate package
        if self.pkg_manager.validate_package(path):
        
            # Get package metadata
            metadata = self._get_package_metadata(path)
                
            # Get current environment
            current_env = self.env_manager.get_current_environment()
            
            # Load environments data
            environments = self.env_manager._load_environments()
            if current_env not in environments:
                raise PackageLoaderError(f"Current environment not found: {current_env}")
            
            # Check if package with same name is already in environment            
            for pkg in environments[current_env]['packages']:
                if pkg.get('name') == metadata["name"]:
                    # If same path, just update metadata
                    if Path(pkg.get('path', '')).resolve() == path:
                        pkg['metadata'] = metadata
                        self.logger.info(f"Updated package in environment: {metadata["name"]}")
                        self.env_manager._save_environments(environments)
                        return True
                    else:
                        raise PackageLoaderError(f"Package with name '{metadata["name"]}' already exists in environment with different path")
            
            # Add package to environment
            package_info = {
                'name': metadata["name"],
                'version': metadata["version"],
                'source':{
                    'type': 'local',
                    'path': str(path),
                    'uri': "file://"+str(path),
                },
                'hatch_compliant': True,
            }
            
            environments[current_env]['packages'].append(package_info)
            
            # Save environments
            self.env_manager._save_environments(environments)
            
            self.logger.info(f"Added package to environment '{current_env}': {metadata["name"]} ({path})")
            return True
        else:
            raise PackageLoaderError(f"Package validation failed for: {package_dir}")
        
        return False
    
    def remove_package_from_environment(self, package_name: str) -> bool:
        """
        Remove a package from the current environment.
        
        Args:
            package_name: Name of the package to remove
            
        Returns:
            bool: True if package was removed successfully, False if package not found
        """
        # Get current environment
        current_env = self.env_manager.get_current_environment()
        
        # Load environments data
        environments = self.env_manager._load_environments()
        if current_env not in environments:
            raise PackageLoaderError(f"Current environment not found: {current_env}")
        
        # Find and remove package
        packages = environments[current_env]['packages']
        for i, pkg in enumerate(packages):
            if pkg.get('name') == package_name:
                del packages[i]
                self.env_manager._save_environments(environments)
                self.logger.info(f"Removed package from environment '{current_env}': {package_name}")
                return True
        
        self.logger.warning(f"Package not found in environment '{current_env}': {package_name}")
        return False
    
    def list_packages_in_environment(self, env_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all packages in an environment.
        
        Args:
            env_name: Name of the environment (default: current environment)
            
        Returns:
            List[Dict[str, Any]]: List of package information dictionaries
        """
        # Get environment name
        if env_name is None:
            env_name = self.env_manager.get_current_environment()
        
        # Load environments data
        environments = self.env_manager._load_environments()
        if env_name not in environments:
            raise PackageLoaderError(f"Environment not found: {env_name}")
        
        # Get packages
        packages = environments[env_name]['packages']
        
        return [pkg.copy() for pkg in packages]