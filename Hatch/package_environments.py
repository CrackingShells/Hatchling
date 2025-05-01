import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

class HatchEnvironmentError(Exception):
    """Exception raised for environment management errors."""
    pass

class HatchEnvironmentManager:
    def __init__(self):
        """Initialize the Hatch environment manager."""
        self.logger = logging.getLogger("hatch.environment_manager")
        self.logger.setLevel(logging.INFO)
        self.environments_file = Path(__file__).parent / "envs" / "environments.json"
        self.current_env_file = Path(__file__).parent / "envs" / "current_env"
        
        # Ensure the environments directory exists
        self.environments_dir = Path(__file__).parent / "envs"
        self.environments_dir.mkdir(exist_ok=True)
        
        # Initialize environments file if it doesn't exist
        if not self.environments_file.exists():
            self._initialize_environments_file()
        
        # Initialize current environment file if it doesn't exist
        if not self.current_env_file.exists():
            self._initialize_current_env_file()
    
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
    
    def _save_environments(self, environments: Dict):
        """Save environments to the environments file."""
        try:
            with open(self.environments_file, 'w') as f:
                json.dump(environments, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save environments: {e}")
            raise HatchEnvironmentError(f"Failed to save environments: {e}")
    
    def get_current_environment(self) -> str:
        """Get the name of the current environment."""
        try:
            with open(self.current_env_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            self._initialize_current_env_file()
            return "default"
    
    def set_current_environment(self, env_name: str) -> bool:
        """
        Set the current environment.
        
        Args:
            env_name: Name of the environment to set as current
            
        Returns:
            bool: True if successful, False if environment doesn't exist
        """
        # Check if environment exists
        environments = self._load_environments()
        if env_name not in environments:
            self.logger.error(f"Environment does not exist: {env_name}")
            return False
        
        # Set current environment
        try:
            with open(self.current_env_file, 'w') as f:
                f.write(env_name)
            
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
        environments = self._load_environments()
        current_env = self.get_current_environment()
        
        result = []
        for name, env_data in environments.items():
            env_info = env_data.copy()
            env_info["is_current"] = (name == current_env)
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
        if not name or not name.isalnum():
            self.logger.error("Environment name must be alphanumeric")
            return False
        
        environments = self._load_environments()
        
        # Check if environment already exists
        if name in environments:
            self.logger.warning(f"Environment already exists: {name}")
            return False
        
        # Create new environment
        import datetime
        environments[name] = {
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "packages": []
        }
        
        # Save environments
        self._save_environments(environments)
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
        
        environments = self._load_environments()
        
        # Check if environment exists
        if name not in environments:
            self.logger.warning(f"Environment does not exist: {name}")
            return False
        
        # Check if it's the current environment
        current_env = self.get_current_environment()
        if name == current_env:
            # Reset to default environment
            self.set_current_environment("default")
        
        # Remove environment
        del environments[name]
        
        # Save environments
        self._save_environments(environments)
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
        environments = self._load_environments()
        return name in environments