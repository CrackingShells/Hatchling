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
    
    # Environment management commands
    env_subparsers = subparsers.add_parser("env", help="Environment management commands").add_subparsers(
        dest="env_command", help="Environment command to execute"
    )
    
    # Create environment command
    env_create_parser = env_subparsers.add_parser("create", help="Create a new environment")
    env_create_parser.add_argument("name", help="Environment name")
    env_create_parser.add_argument("--description", "-D", default="", help="Environment description")
    
    # Remove environment command
    env_remove_parser = env_subparsers.add_parser("remove", help="Remove an environment")
    env_remove_parser.add_argument("name", help="Environment name")
    
    # List environments command
    env_subparsers.add_parser("list", help="List all available environments")
    
    # Set current environment command
    env_use_parser = env_subparsers.add_parser("use", help="Set the current environment")
    env_use_parser.add_argument("name", help="Environment name")
    
    # Show current environment command
    env_subparsers.add_parser("current", help="Show the current environment")
    
    # Package management commands
    pkg_subparsers = subparsers.add_parser("package", help="Package management commands").add_subparsers(
        dest="pkg_command", help="Package command to execute"
    )
    
    # Add package command
    pkg_add_parser = pkg_subparsers.add_parser("add", help="Add a package to the current environment")
    pkg_add_parser.add_argument("package_dir", help="Path to package directory (absolute or relative)")
    
    # Remove package command
    pkg_remove_parser = pkg_subparsers.add_parser("remove", help="Remove a package from the current environment")
    pkg_remove_parser.add_argument("package_name", help="Name of the package to remove")
    
    # List packages command
    pkg_list_parser = pkg_subparsers.add_parser("list", help="List packages in an environment")
    pkg_list_parser.add_argument("--env", "-e", help="Environment name (default: current environment)")
 
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
    
    elif args.command == "env":
        if args.env_command == "create":
            if manager.create_environment(args.name, args.description):
                print(f"Environment created: {args.name}")
                return 0
            else:
                print(f"Failed to create environment: {args.name}")
                return 1
                
        elif args.env_command == "remove":
            if manager.remove_environment(args.name):
                print(f"Environment removed: {args.name}")
                return 0
            else:
                print(f"Failed to remove environment: {args.name}")
                return 1
                
        elif args.env_command == "list":
            environments = manager.list_environments()
            print("Available environments:")
            for env in environments:
                current_marker = "* " if env.get("is_current") else "  "
                description = f" - {env.get('description')}" if env.get("description") else ""
                print(f"{current_marker}{env.get('name')}{description}")
            return 0
            
        elif args.env_command == "use":
            if manager.set_current_environment(args.name):
                print(f"Current environment set to: {args.name}")
                return 0
            else:
                print(f"Failed to set environment: {args.name}")
                return 1
                
        elif args.env_command == "current":
            current_env = manager.get_current_environment()
            print(f"Current environment: {current_env}")
            return 0
            
        else:
            parser.print_help()
            return 1
            
    elif args.command == "package":
        if args.pkg_command == "add":
            if manager.add_package(args.package_dir):
                print(f"Package added successfully from: {args.package_dir}")
                return 0
            else:
                print(f"Failed to add package from: {args.package_dir}")
                return 1
                
        elif args.pkg_command == "remove":
            if manager.remove_package(args.package_name):
                print(f"Package removed: {args.package_name}")
                return 0
            else:
                print(f"Failed to remove package: {args.package_name}")
                return 1
                
        elif args.pkg_command == "list":
            env_name = args.env if args.env else manager.get_current_environment()
            packages = manager.list_packages(env_name)
            
            if not packages:
                print(f"No packages found in environment: {env_name}")
                return 0
                
            print(f"Packages in environment '{env_name}':")
            for pkg in packages:
                print(f"{pkg['name']} ({pkg['version']})\tHatch compliant: {pkg['hatch_compliant']}\tsource: {pkg['source']['uri']}\tlocation: {pkg['source']['path']}")
            return 0
            
        else:
            parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())