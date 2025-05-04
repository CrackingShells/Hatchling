import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import subprocess
import sys
import pkg_resources
from packaging import version, specifiers

class DependencyResolutionError(Exception):
    """Exception raised for dependency resolution errors."""
    pass

class DependencyResolver:
    """
    Unified dependency resolver that handles both local package dependencies 
    and registry-based dependency resolution.
    """
    
    def __init__(self, env_manager=None, registry_path=None):
        """Initialize the Dependency resolver.
        
        Args:
            env_manager: Optional environment manager instance for local package resolution
            registry_path: Optional path to the registry JSON file for registry-based resolution
        """
        self.logger = logging.getLogger("hatch.dependency_resolver")
        self.logger.setLevel(logging.INFO)
        self.env_manager = env_manager
        self.registry_path = registry_path
        
        # Load registry data if path is provided
        self.registry_data = None
        if registry_path:
            self.registry_data = self._load_registry()
    
    def _load_registry(self) -> dict:
        """Load the package registry data from file."""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            return {"repositories": []}
    
    def parse_hatch_dependency(self, dep: Dict[str, str]) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse a Hatch dependency into name, operator, version.
        
        Args:
            dep: Dict format ({"name": "package_name", "version_constraint": ">=1.0.0"})
            
        Returns:
            Tuple[str, Optional[str], Optional[str]]: (name, operator, version)
        """
            
        pkg_name = dep.get("name")
        if not pkg_name:
            raise DependencyResolutionError(f"Invalid dependency object, missing name: {dep}")
            
        version_spec = dep.get("version_constraint", "")
        
        # Parse version constraint if provided
        if version_spec:
            match = re.match(r'^([<>=!~]+)(\d+(?:\.\d+)*)$', version_spec)
            if not match:
                raise DependencyResolutionError(f"Invalid version constraint: {version_spec}")
            operator, version_req = match.groups()
        else:
            operator, version_req = None, None
            
        return pkg_name, operator, version_req
    
    def parse_python_dependency(self, dep: Dict[str, str]) -> Tuple[str, Optional[str], Optional[str], str]:
        """Parse a Python dependency into name, operator, version and package manager.
        
        Args:
            dep: Dependency in either string format ('package_name>=1.0.0') 
                 or dict format ({"name": "package_name", "version_constraint": ">=1.0.0", 
                                 "package_manager": "pip"})
            
        Returns:
            Tuple[str, Optional[str], Optional[str], str]: (name, operator, version, package_manager)
        """
        pkg_name = dep.get("name")
        if not pkg_name:
            raise DependencyResolutionError(f"Invalid dependency object, missing name: {dep}")
            
        version_spec = dep.get("version_constraint", "")
        
        # Parse version constraint if provided
        if version_spec:
            match = re.match(r'^([<>=!~]+)(\d+(?:\.\d+)*)$', version_spec)
            if not match:
                raise DependencyResolutionError(f"Invalid version constraint: {version_spec}")
            operator, version_req = match.groups()
        else:
            operator, version_req = None, None
            
        # Get package manager
        pm = dep.get("package_manager", "pip")
        if pm not in ["pip", "conda"]:
            raise DependencyResolutionError(f"Unsupported package manager: {pm}")
            
        return pkg_name, operator, version_req, pm
    
    def get_available_packages(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available packages in the current environment.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of packages by name
        """
        if not self.env_manager:
            raise DependencyResolutionError("Environment manager is required for this operation")
            
        current_env = self.env_manager.get_current_environment()
        environments = self.env_manager._load_environments()
        
        if current_env not in environments:
            raise DependencyResolutionError(f"Current environment not found: {current_env}")
        
        return {
            pkg.get('name'): pkg for pkg in environments[current_env]['packages']
        }
    
    def is_version_compatible(self, installed_version: str, requirement_operator: str, 
                             requirement_version: str) -> bool:
        """
        Check if an installed version is compatible with a requirement.
        
        Args:
            installed_version: The installed version
            requirement_operator: The requirement operator (e.g. '>=', '==')
            requirement_version: The required version
            
        Returns:
            bool: True if compatible, False otherwise
        """
        if not requirement_operator or not requirement_version:
            return True
            
        try:
            pkg_version = version.parse(installed_version)
            req_spec = specifiers.SpecifierSet(f"{requirement_operator}{requirement_version}")
            return req_spec.contains(str(pkg_version))
        except Exception as e:
            self.logger.error(f"Error checking version compatibility: {e}")
            return False
    
    def get_missing_hatch_dependencies(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Check which Hatch dependencies are missing or don't satisfy version requirements.
        
        Args:
            dependencies: List of dependency objects in format [{"name": "pkg_name", "version_constraint": ">=1.0.0"}, ...]
            
        Returns:
            List[Dict[str, Any]]: List of missing/incompatible dependencies with details
        """
        available_packages = self.get_available_packages()
        missing_deps = []
        
        for dep in dependencies:
            try:
                pkg_name, operator, version_req = self.parse_hatch_dependency(dep)
                
                # Check if package exists
                if pkg_name not in available_packages:
                    missing_deps.append({
                        'reason': 'not_found',
                        'name': pkg_name,
                        'operator': operator,
                        'required_version': version_req
                    })
                    continue
                
                # Check version if specified
                if operator and version_req:
                    installed_version = available_packages[pkg_name].get('version', '0.0.0')
                    if not self.is_version_compatible(installed_version, operator, version_req):
                        missing_deps.append({
                            'reason': 'version_mismatch',
                            'name': pkg_name,
                            'installed_version': installed_version,
                            'operator': operator,
                            'required_version': version_req
                        })
            except Exception as e:
                self.logger.error(f"Error checking dependency {dep}: {e}")
                missing_deps.append({
                    'dependency': dep,
                    'reason': 'parse_error',
                    'error': str(e)
                })
                
        return missing_deps
    
    def get_missing_python_dependencies(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Check which Python dependencies are missing or don't satisfy version requirements.
        
        Args:
            dependencies: List of dependency objects in format [{"name": "pkg_name", "version_constraint": ">=1.0.0", "package_manager": "pip"}, ...]
            
        Returns:
            List[Dict[str, Any]]: List of missing/incompatible dependencies with details
        """
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing_deps = []
        
        for dep in dependencies:
            try:
                pkg_name, operator, version_req, pm = self.parse_python_dependency(dep)
                
                # Check if package is installed
                if pkg_name not in installed_packages:
                    missing_deps.append({
                        'reason': 'not_installed',
                        'name': pkg_name,
                        'package_manager': pm,
                        'operator': operator,
                        'required_version': version_req
                    })
                    
                # Check version if specified
                else:
                    if operator and version_req:
                        installed_version = installed_packages[pkg_name]
                        if not self.is_version_compatible(installed_version, operator, version_req):
                            missing_deps.append({
                                'reason': 'version_mismatch',
                                'name': pkg_name,
                                'installed_version': installed_version,
                                'operator': operator,
                                'required_version': version_req,
                                'package_manager': pm
                            })
            except Exception as e:
                self.logger.error(f"Error checking Python dependency {json.dumps(dep)}: {e}")
                missing_deps.append({
                    'dependency': dep,
                    'reason': 'parse_error',
                    'error': str(e)
                })
                
        return missing_deps
    
    def install_python_dependency(self, pkg_name: str, pm: str = "pip", operator: Optional[str] = None, 
                                 version_req: Optional[str] = None, dry_run: bool = False) -> bool:
        """
        Install a Python dependency.
        
        Args:
            pkg_name: Name of the package to install
            operator: Version operator (e.g. '>=', '==')
            version_req: Required version
            pm: Package manager ('pip' or 'conda')
            dry_run: If True, only print commands without executing
            
        Returns:
            bool: True if installation was successful, False otherwise
        """
        if pm == 'pip':
            return self._install_with_pip(pkg_name, operator, version_req, dry_run)
        elif pm == 'conda':
            return self._install_with_conda(pkg_name, operator, version_req, dry_run)
        else:
            self.logger.error(f"Unsupported package manager: {pm}")
            return False
    
    def _install_with_pip(self, pkg_name:str, operator: Optional[str], version_req: Optional[str],
                           dry_run: bool) -> bool:
        """Install a dependency using pip."""

        # Construct pip command
        if operator and version_req:
            pkg_name = f"{pkg_name}{operator}{version_req}"
        cmd = [sys.executable, '-m', 'pip', 'install', pkg_name]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        if dry_run:
            self.logger.info(f"Would run: {' '.join(cmd)}")
            return True
            
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Successfully installed dependency: {pkg_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing dependency: {e.stderr}")
            return False
    
    def _install_with_conda(self, pkg_name: str, operator: Optional[str], version_req: Optional[str],
                            dry_run: bool) -> bool:
        """Install a dependency using conda."""
        
        # Construct conda command
        if operator and version_req:
            pkg_name = f"{pkg_name}{operator}{version_req}"
        cmd = ['conda', 'install', '-y', pkg_name]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        if dry_run:
            self.logger.info(f"Would run: {' '.join(cmd)}")
            return True
            
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Successfully installed dependency: {pkg_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error installing dependency: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("Conda not found in PATH")
            return False
    
    def install_hatch_dependency(self, pkg_name: str, version_req: Optional[str] = None,
                               operator: Optional[str] = None, pkg_registry: Dict[str, str] = None,
                               dry_run: bool = False) -> bool:
        """
        Install a Hatch package dependency.
        
        Args:
            pkg_name: Name of the package to install
            version_req: Version of the package to install
            operator: Version operator (e.g. '>=', '==')
            pkg_registry: Dictionary mapping package names to their repository locations
            dry_run: If True, only print commands without executing
            
        Returns:
            bool: True if installation was successful, False otherwise
        """
        try:            
            if not pkg_registry:
                self.logger.error("Package registry is required to install Hatch dependencies.")
                return False
            
            if pkg_name not in pkg_registry:
                self.logger.error(f"Package not found in registry: {pkg_name}")
                return False
                
            repo_url = pkg_registry[pkg_name]
            
            if dry_run:
                self.logger.info(f"Would install Hatch package {pkg_name} from {repo_url}")
                return True
            
            # TODO: Implement actual package installation from registry
            # This is a placeholder for the real implementation
            # this can be handled by the package loader
            # e.g. self.pkg_loader.add_from_registry(pkg_name, repo_url)
            
            self.logger.info(f"Installing Hatch package {pkg_name} from {repo_url}")

            return True
                
        except Exception as e:
            self.logger.error(f"Error installing Hatch dependency {pkg_name}: {e}")
            return False
    
    def resolve_dependencies(self, package_path_or_name, install: bool = False, 
                            dry_run: bool = False, version: str = None) -> bool:
        """
        Resolve all dependencies for a package - either a local package or a registry package.
        
        Args:
            package_path_or_name: Either a Path object to a local package directory or a package name string
            install: If True, attempt to install missing dependencies
            dry_run: If True, don't actually install dependencies
            version: Version string (required if package_path_or_name is a name and not a path)
            
        Returns:
            bool: True if all dependencies are resolved, False otherwise
        """
        if isinstance(package_path_or_name, Path) or (isinstance(package_path_or_name, str) and "/" in package_path_or_name):
            # Local package path provided
            return self._resolve_local_dependencies(Path(package_path_or_name), install, dry_run)
        else:
            # Package name provided
            if not version:
                raise DependencyResolutionError("Version is required when resolving dependencies by package name")
            return self._resolve_registry_dependencies(package_path_or_name, version)
    
    def _resolve_local_dependencies(self, package_path: Path, install: bool = False, 
                                  dry_run: bool = False) -> bool:
        """
        Resolve all dependencies for a local package.
        
        Args:
            package_path: Path to the local package directory
            install: If True, attempt to install missing dependencies
            dry_run: If True, don't actually install dependencies
            
        Returns:
            bool: True if all dependencies are resolved, False otherwise
        """
        if not self.env_manager:
            raise DependencyResolutionError("Environment manager is required for local dependency resolution")
            
        # Load package metadata
        metadata_path = package_path / "hatch_metadata.json"
        if not metadata_path.exists():
            raise DependencyResolutionError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise DependencyResolutionError(f"Invalid metadata JSON: {e}")
        
        results = {
            'hatch_dependencies': {
                'required': metadata.get('dependencies', []),
                'missing': []
            },
            'python_dependencies': {
                'required': metadata.get('python_dependencies', []),
                'missing': []
            }
        }
        
        success = True

        # Check Hatch dependencies
        if results['hatch_dependencies']['required']:
            results['hatch_dependencies']['missing'] = self.get_missing_hatch_dependencies(results['hatch_dependencies']['required'])
            
            # Install missing dependencies if requested
            if install:
                for pkg in results['hatch_dependencies']['missing']:
                    _success = self.install_hatch_dependency(pkg['name'],
                                                             version_req=pkg.get('version'),
                                                             operator=pkg.get('operator'),
                                                             pkg_registry=None, dry_run=dry_run)
                    
                    success = success and _success
        
        # Check Python dependencies
        if results['python_dependencies']['required']:
            results['python_dependencies']['missing'] = self.get_missing_python_dependencies(results['python_dependencies']['required'])
            
            # Install missing Python dependencies if requested
            if install:
                for pkg in results['python_dependencies']['missing']:
                    _success = self.install_python_dependency(pkg['name'], pkg.get('package_manager', 'pip'), 
                                                            pkg.get('operator'), pkg.get('required_version'), dry_run)
                    
                    success = success and _success
        
        return success
    
    def _resolve_registry_dependencies(self, package_name: str, version: str) -> Dict:
        """
        Resolve all direct and transitive dependencies for a registry package.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            Dict with resolved dependencies information:
                - resolved_packages: List of {name, version} for all Hatch packages
                - python_dependencies: List of {name, version_constraint, package_manager}
        """
        if not self.registry_data:
            raise DependencyResolutionError("Registry data is required for registry dependency resolution")
            
        visited = set()
        resolved_deps = []
        all_python_deps = {}
        
        self._resolve_dep_tree(package_name, version, visited, resolved_deps, all_python_deps)
        
        # Convert python dependencies dict back to list
        python_deps_list = [
            {
                "name": name, 
                "version_constraint": info["version_constraint"],
                "package_manager": info["package_manager"]
            } 
            for name, info in all_python_deps.items()
        ]
        
        return {
            "resolved_packages": resolved_deps,
            "python_dependencies": python_deps_list
        }
    
    def get_full_package_dependencies(self, package_name: str, version: str) -> Dict:
        """
        Get the full dependency information for a specific package version by
        reconstructing it from differential storage in the registry.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            Dict containing:
                - dependencies: List of Hatch package dependencies with version constraints
                - python_dependencies: List of Python package dependencies
                - compatibility: Dict with hatchling and python version requirements
        """
        if not self.registry_data:
            raise DependencyResolutionError("Registry data is required for this operation")
            
        # Find the package in the registry
        package_data = None
        version_data = None
        
        for repo in self.registry_data.get("repositories", []):
            for pkg in repo.get("packages", []):
                if pkg["name"] == package_name:
                    package_data = pkg
                    for ver in pkg.get("versions", []):
                        if ver["version"] == version:
                            version_data = ver
                            break
                    break
            if package_data:
                break
                
        if not package_data or not version_data:
            self.logger.error(f"Package {package_name} version {version} not found in registry")
            return {"dependencies": [], "python_dependencies": [], "compatibility": {}}
        
        # Now we need to rebuild the full dependency data by applying differential changes
        return self._reconstruct_dependencies(package_data, version_data)
    
    def _reconstruct_dependencies(self, package_data: Dict, version_data: Dict) -> Dict:
        """
        Reconstruct full dependency information by walking back through version chain
        and applying all differential changes.
        
        Args:
            package_data: Package data from registry
            version_data: Specific version data from registry
            
        Returns:
            Dict with full dependency information
        """
        # Initialize with empty collections
        dependencies = {}  # name -> constraint
        python_dependencies = {}  # name -> {constraint, package_manager}
        compatibility = {}
        
        # Get the version chain by following base_version references
        version_chain = self._get_version_chain(package_data, version_data)
        
        # Apply all changes in the version chain from oldest to newest
        for ver in version_chain:
            # Process Hatch dependencies
            for dep in ver.get("dependencies_added", []):
                dependencies[dep["name"]] = dep.get("version_constraint", "")
                
            for dep_name in ver.get("dependencies_removed", []):
                if dep_name in dependencies:
                    del dependencies[dep_name]
                    
            for dep in ver.get("dependencies_modified", []):
                if dep["name"] in dependencies:
                    dependencies[dep["name"]] = dep.get("version_constraint", "")
            
            # Process Python dependencies
            for dep in ver.get("python_dependencies_added", []):
                python_dependencies[dep["name"]] = {
                    "version_constraint": dep.get("version_constraint", ""),
                    "package_manager": dep.get("package_manager", "pip")
                }
                
            for dep_name in ver.get("python_dependencies_removed", []):
                if dep_name in python_dependencies:
                    del python_dependencies[dep_name]
                    
            for dep in ver.get("python_dependencies_modified", []):
                if dep["name"] in python_dependencies:
                    python_dependencies[dep["name"]] = {
                        "version_constraint": dep.get("version_constraint", ""),
                        "package_manager": dep.get("package_manager", python_dependencies[dep["name"]].get("package_manager", "pip"))
                    }
            
            # Process compatibility changes
            if "compatibility_changes" in ver:
                compat_changes = ver["compatibility_changes"]
                if "hatchling" in compat_changes:
                    compatibility["hatchling"] = compat_changes["hatchling"]
                if "python" in compat_changes:
                    compatibility["python"] = compat_changes["python"]
        
        # Convert dictionaries back to lists for the expected return format
        deps_list = [{"name": name, "version_constraint": constraint} 
                    for name, constraint in dependencies.items()]
        
        python_deps_list = [
            {
                "name": name, 
                "version_constraint": info["version_constraint"],
                "package_manager": info["package_manager"]
            } 
            for name, info in python_dependencies.items()
        ]
        
        return {
            "dependencies": deps_list,
            "python_dependencies": python_deps_list,
            "compatibility": compatibility
        }
    
    def _get_version_chain(self, package_data: Dict, version_data: Dict) -> List[Dict]:
        """
        Build the chain of versions from the first version to the requested version.
        This is used to apply differential changes in sequence.
        
        Args:
            package_data: Package data from registry
            version_data: Starting version data
            
        Returns:
            List of version data objects ordered from oldest to newest
        """
        chain = [version_data]
        current = version_data
        
        # Keep looking up base versions until we reach a version with no base (the first version)
        while "base_version" in current and current["base_version"]:
            base_version = current["base_version"]
            found = False
            
            for ver in package_data.get("versions", []):
                if ver["version"] == base_version:
                    chain.append(ver)
                    current = ver
                    found = True
                    break
            
            if not found:
                self.logger.error(f"Base version {base_version} not found in package {package_data['name']}")
                break
        
        # Reverse to get oldest first
        chain.reverse()
        return chain
    
    def _resolve_dep_tree(
        self, 
        package_name: str, 
        version: str, 
        visited: Set[str], 
        resolved_deps: List[Dict], 
        all_python_deps: Dict
    ) -> None:
        """
        Recursively resolve the dependencies for a package and its dependencies.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            visited: Set of visited package identifiers to detect cycles
            resolved_deps: List to collect resolved dependencies
            all_python_deps: Dict to collect all Python dependencies
        """
        # Create a unique identifier for this package+version combination
        pkg_id = f"{package_name}@{version}"
        
        # Skip if already visited to prevent infinite recursion
        if pkg_id in visited:
            return
        
        visited.add(pkg_id)
        
        # Get full dependency information for this package version
        dep_info = self.get_full_package_dependencies(package_name, version)
        
        # Add to resolved list
        resolved_deps.append({"name": package_name, "version": version})
        
        # Collect all Python dependencies
        for py_dep in dep_info.get("python_dependencies", []):
            name = py_dep["name"]
            if name not in all_python_deps:
                all_python_deps[name] = {
                    "version_constraint": py_dep.get("version_constraint", ""),
                    "package_manager": py_dep.get("package_manager", "pip")
                }
            else:
                # TODO: Handle version constraint conflicts
                pass
        
        # Recursively resolve Hatch dependencies
        # TODO: Handle version constraint selection
        for dep in dep_info.get("dependencies", []):
            dep_name = dep["name"]
            # Find the latest version matching the constraint
            # For now, just use the latest version available
            dep_version = self._find_latest_version(dep_name, dep.get("version_constraint", ""))
            if dep_version:
                self._resolve_dep_tree(dep_name, dep_version, visited, resolved_deps, all_python_deps)
            else:
                self.logger.error(f"Could not find a suitable version for dependency {dep_name}")
    
    def _find_latest_version(self, package_name: str, version_constraint: str) -> Optional[str]:
        """
        Find the latest version of a package that satisfies the version constraint.
        
        Args:
            package_name: Name of the package
            version_constraint: Version constraint string (e.g., '>=1.0.0')
            
        Returns:
            The version string, or None if no suitable version found
        """
        # Parse the constraint if provided
        constraint = None
        if version_constraint:
            try:
                constraint = specifiers.SpecifierSet(version_constraint)
            except:
                self.logger.error(f"Invalid version constraint: {version_constraint}")
                return None
        
        available_versions = []
        
        # Find all versions of the package across repositories
        for repo in self.registry_data.get("repositories", []):
            for pkg in repo.get("packages", []):
                if pkg["name"] == package_name:
                    # Add all versions that satisfy the constraint
                    for ver_data in pkg.get("versions", []):
                        ver = ver_data["version"]
                        if not constraint or constraint.contains(ver):
                            available_versions.append(ver)
                    
                    # If we found at least one version, return the latest
                    if available_versions:
                        # Sort versions semantically
                        available_versions.sort(key=lambda v: version.parse(v), reverse=True)
                        return available_versions[0]
        
        return None
    
    def check_circular_dependencies(self, package_name: str, version: str) -> Tuple[bool, List[str]]:
        """
        Check if a package has circular dependencies.
        
        Args:
            package_name: Name of the package
            version: Version of the package
            
        Returns:
            Tuple of (has_circular_deps, cycle_path)
                - has_circular_deps: True if circular dependencies exist
                - cycle_path: List of package names in the circular dependency path
        """
        if not self.registry_data:
            raise DependencyResolutionError("Registry data is required for this operation")
            
        visited = set()
        path = []
        cycle_path = []
        
        def dfs(curr_pkg, curr_ver):
            pkg_id = f"{curr_pkg}@{curr_ver}"
            
            # If we encounter a package already in our current path, we have a cycle
            if pkg_id in path:
                # Extract the cycle
                cycle_start_idx = path.index(pkg_id)
                detected_cycle = path[cycle_start_idx:] + [pkg_id]
                nonlocal cycle_path
                cycle_path = [p.split('@')[0] for p in detected_cycle]
                return True
            
            # If we've already visited this node via another path, it's safe
            if pkg_id in visited:
                return False
                
            visited.add(pkg_id)
            path.append(pkg_id)
            
            # Get full dependency information for this package version
            dep_info = self.get_full_package_dependencies(curr_pkg, curr_ver)
            
            # Check each dependency for cycles
            for dep in dep_info.get("dependencies", []):
                dep_name = dep["name"]
                # Find best matching version
                dep_version = self._find_latest_version(dep_name, dep.get("version_constraint", ""))
                
                if dep_version:
                    if dfs(dep_name, dep_version):
                        return True
                else:
                    self.logger.warning(f"Could not find a suitable version for dependency {dep_name}")
            
            # Remove from current path when backtracking
            path.pop()
            return False
        
        has_circular = dfs(package_name, version)
        return has_circular, cycle_path
    
    def update_registry(self) -> bool:
        """
        Reload registry data from file.
        
        Returns:
            bool: True if registry was successfully updated
        """
        if not self.registry_path:
            self.logger.error("Registry path not set")
            return False
            
        try:
            self.registry_data = self._load_registry()
            return True
        except Exception as e:
            self.logger.error(f"Failed to update registry: {e}")
            return False