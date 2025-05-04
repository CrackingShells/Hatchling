import logging
import json
import sys
from pathlib import Path

import template_generator as tg
from package_environments import HatchEnvironmentManager
from registry_retriever import RegistryRetriever

# Add Validator to path
parent_dir = str(Path(__file__).parent.parent/"Hatch-Validator")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from package_validator import HatchPackageValidator

class HatchPackageManager:
    def __init__(self, registry_path=None):
        """
        Initialize the Hatch package manager.
        
        Args:
            registry_path: Path to the registry JSON file. If None, will use the default path.
        """
        self.logger = logging.getLogger("hatch.package_manager")
        self.logger.setLevel(logging.INFO)
        
        # Set default registry path if not provided
        if registry_path is None:
            registry_path = Path(__file__).parent.parent / "Hatch-Registry" / "hatch_packages_registry.json"
        
        # Initialize the registry retriever
        self.registry_retriever = RegistryRetriever(local_registry_path=registry_path)
        
        # Initialize components in the right order
        self.validator = HatchPackageValidator()
        self.env_manager = HatchEnvironmentManager(registry_path=registry_path)
        
        # Access the package loader and dependency resolver through the environment manager
        # This avoids circular dependencies
        self.logger.info("Hatch Package Manager initialized")

    # Getter methods to access components through the environment manager
    @property
    def package_loader(self):
        """Get the package loader instance from environment manager."""
        return self.env_manager.package_loader
    
    @property
    def dependency_resolver(self):
        """Get the dependency resolver instance from environment manager."""
        return self.env_manager.dependency_resolver

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
    
    # Package loading functions
    
    def add_package(self, package_dir: str) -> bool:
        """
        Add a package to the current environment.
        
        Args:
            package_dir: Path to the package directory (absolute or relative)
            
        Returns:
            bool: True if package was added successfully
        """
        try:
            return self.package_loader.add_package_to_environment(package_dir)
        except Exception as e:
            self.logger.error(f"Failed to add package: {e}")
            return False
    
    def remove_package(self, package_name: str) -> bool:
        """
        Remove a package from the current environment.
        
        Args:
            package_name: Name of the package to remove
            
        Returns:
            bool: True if package was removed successfully
        """
        try:
            return self.package_loader.remove_package_from_environment(package_name)
        except Exception as e:
            self.logger.error(f"Failed to remove package: {e}")
            return False
    
    def list_packages(self, env_name=None) -> list:
        """
        List all packages in an environment.
        
        Args:
            env_name: Name of the environment (default: current environment)
            
        Returns:
            List[Dict]: List of package information dictionaries
        """
        try:
            return self.package_loader.list_packages_in_environment(env_name)
        except Exception as e:
            self.logger.error(f"Failed to list packages: {e}")
            return []
            
    # Dependency resolution methods
    
    def resolve_package_dependencies(self, package_name: str, version: str) -> dict:
        """
        Resolve all dependencies for a specific package version.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            dict: Resolved dependency information
        """
        try:
            return self.dependency_resolver.resolve_dependencies(package_name, version)
        except Exception as e:
            self.logger.error(f"Failed to resolve dependencies for {package_name}@{version}: {e}")
            return {"resolved_packages": [], "python_dependencies": []}
    
    def get_full_package_dependencies(self, package_name: str, version: str) -> dict:
        """
        Get the full dependency information for a specific package version by
        reconstructing it from differential storage.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            dict: Full dependency information
        """
        try:
            return self.dependency_resolver.get_full_package_dependencies(package_name, version)
        except Exception as e:
            self.logger.error(f"Failed to get dependencies for {package_name}@{version}: {e}")
            return {"dependencies": [], "python_dependencies": [], "compatibility": {}}
    
    def check_circular_dependencies(self, package_name: str, version: str) -> tuple:
        """
        Check if a package has circular dependencies.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            tuple: (has_circular_deps, cycle_path)
        """
        try:
            return self.dependency_resolver.check_circular_dependencies(package_name, version)
        except Exception as e:
            self.logger.error(f"Failed to check circular dependencies for {package_name}@{version}: {e}")
            return False, []
    
    def update_registry(self) -> bool:
        """
        Reload the registry data from the file.
        
        Returns:
            bool: True if successful
        """
        try:
            return self.dependency_resolver.update_registry()
        except Exception as e:
            self.logger.error(f"Failed to update registry: {e}")
            return False
            
    def install_with_dependencies(self, package_name: str, version: str = None, resolve_transitive: bool = True) -> bool:
        """
        Install a package along with all its dependencies.
        
        Args:
            package_name: Name of the package
            version: Specific version to install (if None, uses latest)
            resolve_transitive: Whether to install transitive dependencies
            
        Returns:
            bool: True if installation was successful
        """
        self.logger.info(f"Installing package {package_name} with dependencies")
        
        try:
            # Get the latest version if not specified
            if version is None:
                for repo in self.dependency_resolver.registry_data.get("repositories", []):
                    for pkg in repo.get("packages", []):
                        if pkg["name"] == package_name:
                            version = pkg.get("latest_version")
                            break
                    if version:
                        break
            
            if not version:
                self.logger.error(f"Package {package_name} not found in registry")
                return False
                
            # Check for circular dependencies
            has_circular, cycle = self.check_circular_dependencies(package_name, version)
            if has_circular:
                self.logger.error(f"Circular dependency detected: {' -> '.join(cycle)}")
                return False
                
            # Resolve dependencies
            if resolve_transitive:
                resolved = self.resolve_package_dependencies(package_name, version)
                packages_to_install = resolved["resolved_packages"]
                python_deps = resolved["python_dependencies"]
            else:
                dep_info = self.get_full_package_dependencies(package_name, version)
                packages_to_install = [{"name": package_name, "version": version}]
                python_deps = dep_info["python_dependencies"]
            
            # Install Python dependencies first
            for py_dep in python_deps:
                name = py_dep["name"]
                constraint = py_dep.get("version_constraint", "")
                pkg_manager = py_dep.get("package_manager", "pip")
                
                # TODO: Integrate with proper Python package installation
                self.logger.info(f"Would install Python dependency: {name}{constraint} using {pkg_manager}")
            
            # Install Hatch packages in dependency order (reverse the list since it's in dependency-first order)
            for pkg in reversed(packages_to_install):
                pkg_name = pkg["name"]
                pkg_version = pkg["version"]
                
                if pkg_name != package_name or pkg_version != version:
                    self.logger.info(f"Installing dependency: {pkg_name}@{pkg_version}")
                    # TODO: Implement actual package installation
                
            # Install the requested package
            # TODO: Implement actual package installation logic here
            self.logger.info(f"Installing main package: {package_name}@{version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install {package_name} with dependencies: {e}")
            return False