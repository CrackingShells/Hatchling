import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Fix import path to include the full module path
from Hatch.package_loader import HatchPackageLoader

class HatchEnvironmentError(Exception):
    """Exception raised for environment management errors."""
    pass

class HatchEnvironmentManager:
    def __init__(self, registry_path=None):
        """Initialize the Hatch environment manager."""
        self.logger = logging.getLogger("hatch.environment_manager")
        self.logger.setLevel(logging.INFO)
        self.environments_file = Path(__file__).parent / "envs" / "environments.json"
        self.current_env_file = Path(__file__).parent / "envs" / "current_env"
        
        # Ensure the environments directory exists
        self.environments_dir = Path(__file__).parent / "envs"
        self.environments_dir.mkdir(exist_ok=True)
        
        # Initialize package loader
        self.package_loader = HatchPackageLoader()
        
        # Set registry path
        if registry_path is None:
            registry_path = Path(__file__).parent.parent / "Hatch-Registry" / "hatch_packages_registry.json"
            
        self.registry_path = registry_path
        
        # Lazily initialize dependency resolver when needed
        self._dependency_resolver = None
        
        # Initialize environments file if it doesn't exist
        if not self.environments_file.exists():
            self._initialize_environments_file()
        
        # Initialize current environment file if it doesn't exist
        if not self.current_env_file.exists():
            self._initialize_current_env_file()
        
        # Load environments into cache
        self._environments = self._load_environments_from_disk()
        self._current_env_name = self._load_current_env_name_from_disk()
    
    @property
    def dependency_resolver(self):
        """Lazy initialization of the dependency resolver to avoid circular imports."""
        if self._dependency_resolver is None:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path
            
            # Add Validator to path if needed
            validator_path = str(Path(__file__).parent.parent / "Hatch-Validator")
            if validator_path not in sys.path:
                sys.path.insert(0, validator_path)
                
            from dependency_resolver import DependencyResolver
            self._dependency_resolver = DependencyResolver(env_manager=self, registry_path=self.registry_path)
        
        return self._dependency_resolver
    
    def _initialize_environments_file(self):
        """Create the initial environments file with default environment."""
        default_environments = {
            "default": {
                "name": "default",
                "description": "Default environment",
                "created_at": "",
                "packages": []
            }
        }
        
        with open(self.environments_file, 'w') as f:
            json.dump(default_environments, f, indent=2)
        
        self.logger.info("Initialized environments file with default environment")
    
    def _initialize_current_env_file(self):
        """Create the current environment file pointing to the default environment."""
        with open(self.current_env_file, 'w') as f:
            f.write("default")
        
        self.logger.info("Initialized current environment to default")
    
    def _load_environments(self) -> Dict:
        """Load environments from the environments file."""
        try:
            with open(self.environments_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load environments: {e}")
            self._initialize_environments_file()
            return {"default": {"name": "default", "description": "Default environment", "created_at": "", "packages": []}}
    
    def _load_environments_from_disk(self) -> Dict:
        """Load environments from the environments file."""
        # This is an alias for _load_environments for backward compatibility
        return self._load_environments()
    
    def _load_current_env_name_from_disk(self) -> str:
        """Load current environment name from disk."""
        try:
            with open(self.current_env_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            self._initialize_current_env_file()
            return "default"
    
    def get_environments(self) -> Dict:
        """Get environments from cache."""
        return self._environments
    
    def reload_environments(self):
        """Reload environments from disk."""
        self._environments = self._load_environments_from_disk()
        self._current_env_name = self._load_current_env_name_from_disk()
        self.logger.info("Reloaded environments from disk")
    
    def _save_environments(self, environments: Dict):
        """Save environments to the environments file."""
        try:
            with open(self.environments_file, 'w') as f:
                json.dump(environments, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save environments: {e}")
            raise HatchEnvironmentError(f"Failed to save environments: {e}")
    
    def get_current_environment(self) -> str:
        """Get the name of the current environment from cache."""
        return self._current_env_name
    
    def get_current_environment_data(self) -> Dict:
        """Get the data for the current environment."""
        return self._environments[self._current_env_name]
    
    def set_current_environment(self, env_name: str) -> bool:
        """
        Set the current environment.
        
        Args:
            env_name: Name of the environment to set as current
            
        Returns:
            bool: True if successful, False if environment doesn't exist
        """
        # Check if environment exists
        if env_name not in self._environments:
            self.logger.error(f"Environment does not exist: {env_name}")
            return False
        
        # Set current environment
        try:
            with open(self.current_env_file, 'w') as f:
                f.write(env_name)
            
            # Update cache
            self._current_env_name = env_name
            
            self.logger.info(f"Current environment set to: {env_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set current environment: {e}")
            return False
    
    def list_environments(self) -> List[Dict]:
        """
        List all available environments.
        
        Returns:
            List[Dict]: List of environment information dictionaries
        """
        result = []
        for name, env_data in self._environments.items():
            env_info = env_data.copy()
            env_info["is_current"] = (name == self._current_env_name)
            result.append(env_info)
        
        return result
    
    def create_environment(self, name: str, description: str = "") -> bool:
        """
        Create a new environment.
        
        Args:
            name: Name of the environment
            description: Description of the environment
            
        Returns:
            bool: True if created successfully, False if environment already exists
        """
        # Allow alphanumeric characters and underscores
        if not name or not all(c.isalnum() or c == '_' for c in name):
            self.logger.error("Environment name must be alphanumeric or underscore")
            return False
        
        # Check if environment already exists
        if name in self._environments:
            self.logger.warning(f"Environment already exists: {name}")
            return False
        
        # Create new environment
        self._environments[name] = {
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "packages": []
        }
        
        # Save environments and update cache
        self._save_environments(self._environments)
        self.logger.info(f"Created environment: {name}")
        return True
    
    def remove_environment(self, name: str) -> bool:
        """
        Remove an environment.
        
        Args:
            name: Name of the environment to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        # Cannot remove default environment
        if name == "default":
            self.logger.error("Cannot remove default environment")
            return False
        
        # Check if environment exists
        if name not in self._environments:
            self.logger.warning(f"Environment does not exist: {name}")
            return False
        
        # Check if it's the current environment
        current_env = self._current_env_name
        if name == current_env:
            # Reset to default environment
            self.set_current_environment("default")
        
        # Remove environment
        del self._environments[name]
        
        # Save environments and update cache
        self._save_environments(self._environments)
        self.logger.info(f"Removed environment: {name}")
        return True
    
    def environment_exists(self, name: str) -> bool:
        """
        Check if an environment exists.
        
        Args:
            name: Name of the environment to check
            
        Returns:
            bool: True if environment exists, False otherwise
        """
        return name in self._environments
    
    def add_package_to_environment(self, package_path_or_name: str, env_name: Optional[str] = None, 
                                  version: Optional[str] = None) -> bool:
        """
        Add a package to an environment.
        
        Args:
            package_path_or_name: Path to local package or name of registry package
            env_name: Target environment name (default: current environment)
            version: Version to install (required for registry packages)
            
        Returns:
            bool: True if package was added successfully
        """
        # Determine environment
        if env_name is None:
            env_name = self._current_env_name
            
        if not self.environment_exists(env_name):
            self.logger.error(f"Environment does not exist: {env_name}")
            return False
            
        env_data = self._environments[env_name]
        
        # Determine if we're dealing with a local or registry package
        is_local = Path(package_path_or_name).exists()
        
        try:
            # Handle local package
            if is_local:
                package_path = Path(package_path_or_name)
                
                # Load package metadata
                metadata_path = package_path / "hatch_metadata.json"
                if not metadata_path.exists():
                    self.logger.error(f"Package metadata not found: {metadata_path}")
                    return False
                    
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                package_name = metadata.get('name')
                package_version = metadata.get('version')
                
                if not package_name or not package_version:
                    self.logger.error("Invalid package metadata: missing name or version")
                    return False
                
                # Check for circular dependencies
                has_circular, cycle = self.dependency_resolver.check_circular_dependencies(package_name, package_version)
                if has_circular:
                    self.logger.error(f"Circular dependency detected: {' -> '.join(cycle)}")
                    return False
                
                # Validate dependencies
                missing_deps = self.dependency_resolver.get_missing_hatch_dependencies(
                    metadata.get('hatch_dependencies', [])
                )
                
                # Install missing dependencies
                for dep in missing_deps:
                    dep_name = dep.get('name')
                    self.logger.info(f"Installing missing dependency: {dep_name}")
                    
                    # Check if it's a local dependency
                    if dep.get('type') == 'local':
                        uri = dep.get('uri')
                        if not uri:
                            self.logger.error(f"Missing URI for local dependency: {dep_name}")
                            return False
                            
                        # Local path from file:// URI
                        if uri.startswith('file://'):
                            local_path = Path(uri[7:])
                            if not self.add_package_to_environment(str(local_path), env_name):
                                return False
                    else:
                        # Remote dependency - we need to get version and install
                        dep_version = self.dependency_resolver._find_latest_version(
                            dep_name, dep.get('version_constraint', '')
                        )
                        
                        if not dep_version:
                            self.logger.error(f"Could not find suitable version for {dep_name}")
                            return False
                            
                        if not self.add_package_to_environment(dep_name, env_name, dep_version):
                            return False
                
                # Install the package itself
                target_dir = self.environments_dir / env_name
                target_dir.mkdir(exist_ok=True)
                
                self.package_loader.install_local_package(package_path, target_dir, package_name)
                
                # Update environment data
                package_entry = {
                    'name': package_name,
                    'version': package_version,
                    'added_date': datetime.datetime.now().isoformat(),
                    'path': str(target_dir / package_name)
                }
                
                # Add to environment packages or update if exists
                exists = False
                for i, pkg in enumerate(env_data['packages']):
                    if pkg['name'] == package_name:
                        env_data['packages'][i] = package_entry
                        exists = True
                        break
                        
                if not exists:
                    env_data['packages'].append(package_entry)
                
                # Save changes
                self._save_environments(self._environments)
                
                self.logger.info(f"Added package {package_name} to environment {env_name}")
                return True
                
            else:
                # Handle registry package
                package_name = package_path_or_name
                
                if not version:
                    self.logger.error("Version required for registry packages")
                    return False
                
                # Check for circular dependencies
                has_circular, cycle = self.dependency_resolver.check_circular_dependencies(package_name, version)
                if has_circular:
                    self.logger.error(f"Circular dependency detected: {' -> '.join(cycle)}")
                    return False
                    
                # Resolve dependencies
                resolved_deps = self.dependency_resolver.resolve_dependencies(package_name, version=version)
                
                # Install all required packages in order
                target_dir = self.environments_dir / env_name
                target_dir.mkdir(exist_ok=True)
                
                # Get the package URL from registry
                repo_info, pkg_info = None, None
                for repo in self.dependency_resolver.registry_data.get('repositories', []):
                    for pkg in repo.get('packages', []):
                        if pkg['name'] == package_name:
                            repo_info, pkg_info = repo, pkg
                            break
                    if pkg_info:
                        break
                        
                if not pkg_info:
                    self.logger.error(f"Package {package_name} not found in registry")
                    return False
                
                # Find the specific version
                version_info = None
                for ver in pkg_info.get('versions', []):
                    if ver['version'] == version:
                        version_info = ver
                        break
                        
                if not version_info:
                    self.logger.error(f"Version {version} not found for package {package_name}")
                    return False
                
                # Construct package URL
                # TODO: Implement proper URL construction based on registry format
                package_url = f"{repo_info.get('url', '')}/{pkg_info['name']}/{version_info['version']}"
                
                # Install the package
                self.package_loader.install_remote_package(package_url, package_name, version, target_dir)
                
                # Update environment data
                package_entry = {
                    'name': package_name,
                    'version': version,
                    'added_date': datetime.datetime.now().isoformat(),
                    'path': str(target_dir / package_name)
                }
                
                # Add to environment packages or update if exists
                exists = False
                for i, pkg in enumerate(env_data['packages']):
                    if pkg['name'] == package_name:
                        env_data['packages'][i] = package_entry
                        exists = True
                        break
                        
                if not exists:
                    env_data['packages'].append(package_entry)
                
                # Save changes
                self._save_environments(self._environments)
                
                self.logger.info(f"Added package {package_name}@{version} to environment {env_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add package: {e}")
            return False
    
    def remove_package_from_environment(self, package_name: str, env_name: Optional[str] = None) -> bool:
        """
        Remove a package from an environment.
        
        Args:
            package_name: Name of the package to remove
            env_name: Environment name (default: current environment)
            
        Returns:
            bool: True if package was removed successfully
        """
        # Determine environment
        if env_name is None:
            env_name = self._current_env_name
            
        if not self.environment_exists(env_name):
            self.logger.error(f"Environment does not exist: {env_name}")
            return False
            
        env_data = self._environments[env_name]
        
        # Check if package exists in environment
        package_index = None
        package_path = None
        
        for i, pkg in enumerate(env_data['packages']):
            if pkg['name'] == package_name:
                package_index = i
                package_path = pkg.get('path')
                break
                
        if package_index is None:
            self.logger.error(f"Package {package_name} not found in environment {env_name}")
            return False
            
        # Remove package directory if it exists
        if package_path:
            try:
                package_dir = Path(package_path)
                if package_dir.exists() and package_dir.is_dir():
                    import shutil
                    shutil.rmtree(package_dir)
            except Exception as e:
                self.logger.error(f"Failed to remove package directory: {e}")
                # Continue anyway to remove from environment data
        
        # Remove from environment data
        env_data['packages'].pop(package_index)
        
        # Save changes
        self._save_environments(self._environments)
        
        self.logger.info(f"Removed package {package_name} from environment {env_name}")
        return True
    
    def list_packages_in_environment(self, env_name: Optional[str] = None) -> List[Dict]:
        """
        List all packages in an environment.
        
        Args:
            env_name: Environment name (default: current environment)
            
        Returns:
            List[Dict]: List of package information dictionaries
        """
        # Determine environment
        if env_name is None:
            env_name = self._current_env_name
            
        if not self.environment_exists(env_name):
            self.logger.error(f"Environment does not exist: {env_name}")
            return []
            
        env_data = self._environments[env_name]
        return env_data['packages']
    
    def get_environment_path(self, env_name: Optional[str] = None) -> Path:
        """
        Get the path to an environment directory.
        
        Args:
            env_name: Environment name (default: current environment)
            
        Returns:
            Path: Path to the environment directory
        """
        if env_name is None:
            env_name = self._current_env_name
            
        if not self.environment_exists(env_name):
            raise HatchEnvironmentError(f"Environment does not exist: {env_name}")
            
        return self.environments_dir / env_name