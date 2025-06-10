"""Hatch package manager commands module for the chat interface.

This module provides commands for interacting with the Hatch package manager,
including environment management, package operations, and template creation.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

from hatchling.core.logging.session_debug_log import SessionDebugLog
from hatchling.config.settings import ChatSettings
from hatchling.core.chat.abstract_commands import AbstractCommands

# Import Hatch components - assumes Hatch is installed or available in the Python path
from hatch import HatchEnvironmentManager
from hatch import create_package_template


class HatchCommands(AbstractCommands):
    """Handles Hatch package manager commands in the chat interface."""

    def _register_commands(self) -> None:
        """Register all available Hatch package manager commands."""
        # New standardized command registration format
        self.commands = {
            # Environment commands
            'hatch:env:list': {
                'handler': self._cmd_env_list,
                'description': "List all available Hatch environments",
                'is_async': False,
                'args': {}
            },
            'hatch:env:create': {
                'handler': self._cmd_env_create,
                'description': "Create a new Hatch environment",
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': "Name for the new environment",
                        'required': True
                    },
                    'description': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': "Description for the environment",
                        'aliases': ['D'],
                        'default': '',
                        'required': False
                    }
                }
            },
            'hatch:env:remove': {
                'handler': self._cmd_env_remove,
                'description': "Remove a Hatch environment",
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'environment',
                        'description': "Name of the environment to remove",
                        'required': True
                    }
                }
            },
            'hatch:env:current': {
                'handler': self._cmd_env_current,
                'description': "Show the current Hatch environment",
                'is_async': False,
                'args': {}
            },
            'hatch:env:use': {
                'handler': self._cmd_env_use,
                'description': "Set the current Hatch environment",
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'environment',
                        'description': "Name of the environment to use",
                        'required': True
                    }
                }
            },
            # Package commands
            'hatch:pkg:add': {
                'handler': self._cmd_pkg_add,
                'description': "Add a package to an environment",
                'is_async': False,
                'args': {
                    'package_path_or_name': {
                        'positional': True,
                        'completer_type': 'local_package',
                        'description': "Path or name of the package to add",
                        'required': True
                    },
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': "Environment to add the package to",
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    },
                    'version': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': "Version of the package to add",
                        'aliases': ['v'],
                        'default': None,
                        'required': False
                    },
                    'force-download': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': "Force download even if already available",
                        'aliases': ['f'],
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'refresh-registry': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': "Refresh the registry before installing",
                        'aliases': ['r'],
                        'default': False,
                        'is_flag': True,
                        'required': False
                    }
                }
            },
            'hatch:pkg:remove': {
                'handler': self._cmd_pkg_remove,
                'description': "Remove a package from an environment",
                'is_async': False,
                'args': {
                    'package_name': {
                        'positional': True,
                        'completer_type': 'package',
                        'description': "Name of the package to remove",
                        'required': True
                    },
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': "Environment to remove the package from",
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    }
                }
            },
            'hatch:pkg:list': {
                'handler': self._cmd_pkg_list,
                'description': "List packages in an environment",
                'is_async': False,
                'args': {
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': "Environment to list packages from",
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    }
                }
            },
            # Package creation command
            'hatch:create': {
                'handler': self._cmd_create_package,
                'description': "Create a new package template",
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': "Name of the package to create",
                        'required': True
                    },
                    'dir': {
                        'positional': False,
                        'completer_type': 'path',
                        'description': "Directory to create the package in",
                        'aliases': ['d'],
                        'default': '.',
                        'required': False
                    },
                    'description': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': "Description of the package",
                        'aliases': ['D'],
                        'default': '',
                        'required': False
                    }
                }
            },
            # Package validation command
            'hatch:validate': {
                'handler': self._cmd_validate_package,
                'description': "Validate a package",
                'is_async': False,
                'args': {
                    'package_dir': {
                        'positional': True,
                        'completer_type': 'path',
                        'description': "Directory of the package to validate",
                        'required': True
                    }
                }
            }
        }
    
    def print_commands_help(self) -> None:
        """Print help for all available chat commands."""
        print_formatted_text(FormattedText([
            ('class:header', "\n=== Hatch Chat Commands ===\n")
        ]), style=self.style)
        
        super().print_commands_help()

    def format_command(self, cmd_name: str, cmd_info: Dict[str, Any], group: str = 'hatch') -> list:
        """Format Hatch commands with custom styling."""
        return [
            (f'class:command.name.{group}', f"{cmd_name}"),
            ('', ' - '),
            ('class:command.description', f"{cmd_info['description']}")
        ]
    
    def _cmd_env_list(self, _: str) -> bool:
        """List all available Hatch environments.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            environments = self.env_manager.list_environments()
            
            if not environments:
                print("No Hatch environments found.")
                return True
            
            print("Available Hatch environments:")
            for env in environments:
                current_marker = "* " if env.get("is_current") else "  "
                description = f" - {env.get('description')}" if env.get("description") else ""
                print(f"{current_marker}{env.get('name')}{description}")
                
        except Exception as e:
            self.logger.error(f"Error listing environments: {e}")
            
        return True
    
    def _cmd_env_create(self, args: str) -> bool:
        """Create a new Hatch environment.
        
        Args:
            args (str): Environment name and optional description.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'name': {'positional': True},
            'description': {'aliases': ['D'], 'default': ''}
        }
        
        parsed_args = self._parse_args(args, arg_defs)

        if 'name' not in parsed_args or not parsed_args['name']:
            self.logger.error("Environment name is required.")
            self._print_command_help('hatch:env:create')
            return True
        
        try:
            name = parsed_args['name']
            description = parsed_args.get('description', '')
            
            if self.env_manager.create_environment(name, description):                
                self.logger.info(f"Environment created: {name}")
            else:
                self.logger.error(f"Failed to create environment: {name}")
                
        except Exception as e:
            self.logger.error(f"Error creating environment: {e}")
            
        return True
    
    def _cmd_env_remove(self, args: str) -> bool:
        """Remove a Hatch environment.
        
        Args:
            args (str): Environment name.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'name': {'positional': True}
        }
        
        parsed_args = self._parse_args(args, arg_defs)

        if 'name' not in parsed_args or not parsed_args['name']:
            self.logger.error("Environment name is required.")
            self._print_command_help('hatch:env:remove')
            return True
        
        try:
            name = parsed_args['name']

            if self.env_manager.remove_environment(name):
                self.logger.info(f"Environment removed: {name}")
            else:
                self.logger.error(f"Failed to remove environment: {name}")
                
        except Exception as e:
            self.logger.error(f"Error removing environment: {e}")
            
        return True
    
    def _cmd_env_current(self, _: str) -> bool:
        """Show the current Hatch environment.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            current_env = self.env_manager.get_current_environment()
            if current_env:
                self.logger.info(f"Current environment: {current_env}")
            else:
                self.logger.info("No current environment set.")
                
        except Exception as e:
            self.logger.error(f"Error getting current environment: {e}")
            
        return True
    
    def _cmd_env_use(self, args: str) -> bool:
        """Set the current Hatch environment.
        
        Args:
            args (str): Environment name.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'name': {'positional': True}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        if 'name' not in parsed_args or not parsed_args['name']:
            self.logger.error(f"Environment name is required.")
            self._print_command_help('hatch:env:use')
            return True
        
        try:
            name = parsed_args['name']

            if self.env_manager.set_current_environment(name):
                self.logger.info(f"Current environment set to: {name}")

                # When changing the current environment, we must handle
                # disconnecting from the previous environment's tools if any,
                # and connecting to the new environment's tools.
                if self.chat_session.tool_executor.tools_enabled:
                    
                    # Disconnection
                    self.chat_session.tool_executor.disconnect_tools()
                    self.logger.info("Disconnected from previous environment's tools.")

                    # Get the new environment's entry points for the MCP servers
                    mcp_servers_url = self.env_manager.get_servers_entry_points(name)

                    if mcp_servers_url:
                        # Reconnect to the new environment's tools
                        connected = self.chat_session.initialize_mcp(mcp_servers_url)
                        if not connected:
                            self.logger.error("Failed to connect to new environment's MCP servers. Tools not enabled.")
                        else:
                            self.logger.info("Connected to new environment's MCP servers successfully!")

            else:
                self.logger.error(f"Failed to set environment: {name}")
                
        except Exception as e:
            self.logger.error(f"Error setting current environment: {e}")
            
        return True
    
    def _cmd_pkg_add(self, args: str) -> bool:
        """Add a package to an environment.
        
        Args:
            args (str): Package path or name and options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'package_path_or_name': {'positional': True},
            'env': {'aliases': ['e'], 'default': None},
            'version': {'aliases': ['v'], 'default': None},
            'force-download': {'aliases': ['f'], 'default': False, 'action': 'store_true'},
            'refresh-registry': {'aliases': ['r'], 'default': False, 'action': 'store_true'}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        if 'package_path_or_name' not in parsed_args or not parsed_args['package_path_or_name']:
            self.logger.error("Package path or name is required.")
            self._print_command_help('hatch:pkg:add')
            return True
        
        try:
            package = parsed_args['package_path_or_name']
            env = parsed_args.get('env')
            version = parsed_args.get('version')
            force_download = parsed_args.get('force-download', False)
            refresh_registry = parsed_args.get('refresh-registry', False)

            if self.env_manager.add_package_to_environment(package, env, version, force_download, refresh_registry):
                self.logger.info(f"Successfully added package: {package}")
            else:
                self.logger.error(f"Failed to add package: {package}")
                
        except Exception as e:
            self.logger.error(f"Error adding package: {e}")

        return True
    
    def _cmd_pkg_remove(self, args: str) -> bool:
        """Remove a package from an environment.
        
        Args:
            args (str): Package name and options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'package_name': {'positional': True},
            'env': {'aliases': ['e'], 'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        if 'package_name' not in parsed_args or not parsed_args['package_name']:
            self.logger.error("Package name is required.")
            self._print_command_help('hatch:pkg:remove')
            return True
        
        try:
            package_name = parsed_args['package_name']
            env = parsed_args.get('env')

            if self.env_manager.remove_package(package_name, env):
                self.logger.info(f"Successfully removed package: {package_name}")
            else:
                self.logger.error(f"Failed to remove package: {package_name}")
                
        except Exception as e:
            self.logger.error(f"Error removing package: {e}")

        return True

    def _cmd_pkg_list(self, args: str) -> bool:
        """List packages in an environment.
        
        Args:
            args (str): Environment options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'env': {'aliases': ['e'], 'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        env = parsed_args.get('env')
        
        try:
            packages = self.env_manager.list_packages(env)
            if not packages:
                env_name = env if env else "current environment"
                self.logger.info(f"No packages found in {env_name}.")
                return True
            
            env_name = env if env else "current environment"
            self.logger.info(f"Listing {len(packages)} packages in {env_name}")
            print(f"Packages in {env_name}:")
            for pkg in packages:
                print(f"{pkg['name']} ({pkg['version']})  Hatch compliant: {pkg['hatch_compliant']} Source: {pkg['source']['uri']}  Location: {pkg['source']['path']}")
                
        except Exception as e:
            self.logger.error(f"Error listing packages: {e}")
            
        return True
    
    def _cmd_create_package(self, args: str) -> bool:
        """Create a new package template.
        
        Args:
            args (str): Package name and options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'name': {'positional': True},
            'dir': {'aliases': ['d'], 'default': '.'},
            'description': {'aliases': ['D'], 'default': ''}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        if 'name' not in parsed_args or not parsed_args['name']:
            self.logger.error("Package name is required.")
            self._print_command_help('hatch:create')
            return True
        
        try:
            name = parsed_args['name']
            target_dir = Path(parsed_args.get('dir', '.')).resolve()
            description = parsed_args.get('description', '')
            
            package_dir = create_package_template(
                target_dir=target_dir,
                package_name=name,
                description=description
            )
            
            self.logger.info(f"Package template created at: {package_dir}")
                
        except Exception as e:
            self.logger.error(f"Error creating package template: {e}")
            
        return True
    
    def _cmd_validate_package(self, args: str) -> bool:
        """Validate a package.
        
        Args:
            args (str): Package directory.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'package_dir': {'positional': True}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        if 'package_dir' not in parsed_args or not parsed_args['package_dir']:
            self.logger.error("Package directory is required.")
            self._print_command_help('hatch:validate')
            return True
        
        try:
            package_path = Path(parsed_args['package_dir']).resolve()
            
            # Use the validator from environment manager
            is_valid, validation_results = self.env_manager.package_validator.validate_package(package_path)
            
            if is_valid:
                self.logger.info(f"Package validation SUCCESSFUL: {package_path}")
            else:
                self.logger.warning(f"Package validation FAILED: {package_path}")
                if validation_results and isinstance(validation_results, dict):
                    for key, issues in validation_results.items():
                        self.logger.warning(f"\n{key} issues:")
                        for issue in issues:
                            self.logger.warning(f"- {issue}")

        except Exception as e:
            self.logger.error(f"Error validating package: {e}")

        return True