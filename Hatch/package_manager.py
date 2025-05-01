import logging
import template_generator as tg
import json
from pathlib import Path

class HatchPackageManager:
    def __init__(self):
        """Initialize the Hatch package manager."""
        self.logger = logging.getLogger("hatch.package_manager")
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