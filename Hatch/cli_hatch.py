import argparse
import logging
import sys
from pathlib import Path

from package_manager import HatchPackageManager

def main():
    """Main entry point for Hatch CLI"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Hatch package manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create template command
    create_parser = subparsers.add_parser("create", help="Create a new package template")
    create_parser.add_argument("name", help="Package name")
    create_parser.add_argument("--dir", "-d", default=".", help="Target directory (default: current directory)")
    create_parser.add_argument("--category", "-c", default="", help="Package category")
    create_parser.add_argument("--description", "-D", default="", help="Package description")
    
    # Validate package command
    validate_parser = subparsers.add_parser("validate", help="Validate a package")
    validate_parser.add_argument("package_dir", help="Path to package directory")
 
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize package manager
    manager = HatchPackageManager()
    
    # Execute commands
    if args.command == "create":
        target_dir = Path(args.dir).resolve()
        package_dir = manager.create_package_template(
            target_dir=target_dir,
            package_name=args.name,
            category=args.category,
            description=args.description
        )
        print(f"Package template created at: {package_dir}")

    elif args.command == "validate":
        package_path = Path(args.package_dir).resolve()
        
        is_valid = manager.validate_package(package_path)
        
        if is_valid:
            print(f"Package validation SUCCESSFUL: {package_path}")
            return 0
        else:
            print(f"Package validation FAILED: {package_path}")
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())