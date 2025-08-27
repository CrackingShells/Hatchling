"""Hatch package manager commands module for the chat interface.

This module provides commands for interacting with the Hatch package manager,
including environment management, package operations, and template creation.
"""

from typing import Dict, Any
from pathlib import Path

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

# Import Hatch components - assumes Hatch is installed or available in the Python path
# from hatch import HatchEnvironmentManager
from hatch import create_package_template

from hatchling.config.i18n import translate
from hatchling.mcp_utils.manager import mcp_manager
from hatchling.ui.abstract_commands import AbstractCommands

class HatchCommands(AbstractCommands):
    """Handles Hatch package manager commands in the chat interface."""

    def _register_commands(self) -> None:
        """Register all available Hatch package manager commands."""
        
        self.commands = {
            # Environment commands
            'hatch:env:list': {
                'handler': self._cmd_env_list,
                'description': translate('commands.hatch.env_list_description'),
                'is_async': False,
                'args': {}
            },
            'hatch:env:create': {
                'handler': self._cmd_env_create,
                'description': translate('commands.hatch.env_create_description'),
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': translate('commands.args.env_name_description'),
                        'required': True
                    },
                    'description': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.env_description_description'),
                        'aliases': ['D'],
                        'default': '',
                        'required': False
                    },
                    'python-version': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.python_version_description'),
                        'default': None,
                        'required': False
                    },
                    'no-python': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.no_python_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'no-hatch-mcp-server': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.no_hatch_mcp_server_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'hatch_mcp_server_tag': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.hatch_mcp_server_tag_description'),
                        'default': None,
                        'required': False
                    }
                }
            },
            'hatch:env:remove': {
                'handler': self._cmd_env_remove,
                'description': translate('commands.hatch.env_remove_description'),
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'environment',
                        'description': translate('commands.args.env_remove_name_description'),
                        'required': True
                    }
                }
            },
            'hatch:env:current': {
                'handler': self._cmd_env_current,
                'description': translate('commands.hatch.env_current_description'),
                'is_async': False,
                'args': {}
            },
            'hatch:env:use': {
                'handler': self._cmd_env_use,
                'description': translate('commands.hatch.env_use_description'),
                'is_async': True,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'environment',
                        'description': translate('commands.args.env_use_name_description'),
                        'required': True
                    }
                }
            },
            # Package commands
            'hatch:pkg:add': {
                'handler': self._cmd_pkg_add,
                'description': translate('commands.hatch.pkg_add_description'),
                'is_async': False,
                'args': {
                    'package_path_or_name': {
                        'positional': True,
                        'completer_type': 'local_package',
                        'description': translate('commands.args.package_path_or_name_description'),
                        'required': True
                    },
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.env_target_description'),
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    },
                    'version': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.package_version_description'),
                        'aliases': ['v'],
                        'default': None,
                        'required': False
                    },
                    'force-download': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.force_download_description'),
                        'aliases': ['f'],
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'refresh-registry': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.refresh_registry_description'),
                        'aliases': ['r'],
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'auto-approve': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.auto_approve_description'),
                        'aliases': ['y'],
                        'default': False,
                        'is_flag': True,
                        'required': False
                    }
                }
            },
            'hatch:pkg:remove': {
                'handler': self._cmd_pkg_remove,
                'description': translate('commands.hatch.pkg_remove_description'),
                'is_async': False,
                'args': {
                    'package_name': {
                        'positional': True,
                        'completer_type': 'package',
                        'description': translate('commands.args.package_name_description'),
                        'required': True
                    },
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.env_remove_package_description'),
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    }
                }
            },
            'hatch:pkg:list': {
                'handler': self._cmd_pkg_list,
                'description': translate('commands.hatch.pkg_list_description'),
                'is_async': False,
                'args': {
                    'env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.env_list_packages_description'),
                        'aliases': ['e'],
                        'default': None,
                        'required': False
                    }
                }
            },
            # Package creation command
            'hatch:pkg:create': {
                'handler': self._cmd_create_package,
                'description': translate('commands.hatch.pkg_create_description'),
                'is_async': False,
                'args': {
                    'name': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': translate('commands.args.package_create_name_description'),
                        'required': True
                    },
                    'dir': {
                        'positional': False,
                        'completer_type': 'path',
                        'description': translate('commands.args.dir_description'),
                        'aliases': ['d'],
                        'default': '.',
                        'required': False
                    },
                    'description': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.package_description_description'),
                        'aliases': ['D'],
                        'default': '',
                        'required': False
                    }
                }
            },
            # Package validation command
            'hatch:pkg:validate': {
                'handler': self._cmd_validate_package,
                'description': translate('commands.hatch.pkg_validate_description'),
                'is_async': False,
                'args': {
                    'package_dir': {
                        'positional': True,
                        'completer_type': 'path',
                        'description': translate('commands.args.package_dir_description'),
                        'required': True
                    }
                }
            },
            # Python environment management commands
            'hatch:env:python:init': {
                'handler': self._cmd_env_python_init,
                'description': translate('commands.hatch.env_python_init_description'),
                'is_async': False,
                'args': {
                    'hatch_env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.hatch_env_description'),
                        'default': None,
                        'required': False
                    },
                    'python-version': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.python_version_description'),
                        'default': None,
                        'required': False
                    },
                    'force': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.force_recreation_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'no-hatch-mcp-server': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.no_hatch_mcp_server_wrapper_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    },
                    'hatch_mcp_server_tag': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.hatch_mcp_server_tag_description'),
                        'default': None,
                        'required': False
                    }
                }
            },
            'hatch:env:python:info': {
                'handler': self._cmd_env_python_info,
                'description': translate('commands.hatch.env_python_info_description'),
                'is_async': False,
                'args': {
                    'hatch_env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.hatch_env_description'),
                        'default': None,
                        'required': False
                    },
                    'detailed': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.detailed_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    }
                }
            },
            'hatch:env:python:remove': {
                'handler': self._cmd_env_python_remove,
                'description': translate('commands.hatch.env_python_remove_description'),
                'is_async': False,
                'args': {
                    'hatch_env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.hatch_env_description'),
                        'default': None,
                        'required': False
                    },
                    'force': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.force_removal_description'),
                        'default': False,
                        'is_flag': True,
                        'required': False
                    }
                }
            },
            'hatch:env:python:shell': {
                'handler': self._cmd_env_python_shell,
                'description': translate('commands.hatch.env_python_shell_description'),
                'is_async': False,
                'args': {
                    'hatch_env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.hatch_env_description'),
                        'default': None,
                        'required': False
                    },
                    'cmd': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.cmd_description'),
                        'default': None,
                        'required': False
                    }
                }
            },
            'hatch:env:python:add-hatch-mcp': {
                'handler': self._cmd_env_python_add_hatch_mcp,
                'description': translate('commands.hatch.env_python_add_hatch_mcp_description'),
                'is_async': False,
                'args': {
                    'hatch_env': {
                        'positional': False,
                        'completer_type': 'environment',
                        'description': translate('commands.args.hatch_env_special_description'),
                        'default': None,
                        'required': False
                    },
                    'tag': {
                        'positional': False,
                        'completer_type': 'none',
                        'description': translate('commands.args.tag_description'),
                        'default': None,
                        'required': False
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
            environments = mcp_manager.hatch_env_manager.list_environments()
            
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
        
        Creates a new Hatch environment with optional Python environment support and
        automatic hatch_mcp_server installation. The MCP server installation can be
        controlled via command flags.
        
        Args:
            args (str): Environment name and optional parameters including:
                      - description: Environment description
                      - python-version: Specific Python version
                      - no-python: Skip Python environment creation
                      - no-hatch-mcp-server: Skip MCP server installation
                      - hatch_mcp_server_tag: Git tag/branch for MCP server
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'name': {'positional': True},
            'description': {'aliases': ['D'], 'default': ''},
            'python-version': {'default': None},
            'no-python': {'default': False, 'action': 'store_true'},
            'no-hatch-mcp-server': {'default': False, 'action': 'store_true'},
            'hatch_mcp_server_tag': {'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)

        if 'name' not in parsed_args or not parsed_args['name']:
            self.logger.error("Environment name is required.")
            self._print_command_help('hatch:env:create')
            return True
        
        try:
            name = parsed_args['name']
            description = parsed_args.get('description', '')
            python_version = parsed_args.get('python-version')
            create_python_env = not parsed_args.get('no-python', False)
            no_hatch_mcp_server = parsed_args.get('no-hatch-mcp-server', False)
            hatch_mcp_server_tag = parsed_args.get('hatch_mcp_server_tag')
            
            if mcp_manager.hatch_env_manager.create_environment(
                name, 
                description, 
                python_version=python_version, 
                create_python_env=create_python_env,
                no_hatch_mcp_server=no_hatch_mcp_server,
                hatch_mcp_server_tag=hatch_mcp_server_tag
            ):                
                self.logger.info(f"Environment created: {name}")
                if create_python_env and python_version:
                    self.logger.info(f"Python environment initialized with version: {python_version}")
                elif create_python_env:
                    self.logger.info("Python environment initialized with default version")
                else:
                    self.logger.info("Python environment creation skipped (--no-python)")
                
                # Provide feedback about MCP server installation
                if create_python_env and not no_hatch_mcp_server:
                    if hatch_mcp_server_tag:
                        self.logger.info(f"hatch_mcp_server installed with tag/branch: {hatch_mcp_server_tag}")
                    else:
                        self.logger.info("hatch_mcp_server installed with default branch")
                elif no_hatch_mcp_server:
                    self.logger.info("hatch_mcp_server installation skipped (--no-hatch-mcp-server)")
                elif not create_python_env:
                    self.logger.info("hatch_mcp_server installation skipped (no Python environment)")
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

            if mcp_manager.hatch_env_manager.remove_environment(name):
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
            current_env = mcp_manager.hatch_env_manager.get_current_environment()
            if current_env:
                self.logger.info(f"Current environment: {current_env}")
            else:
                self.logger.info("No current environment set.")
                
        except Exception as e:
            self.logger.error(f"Error getting current environment: {e}")
            
        return True
    
    async def _cmd_env_use(self, args: str) -> bool:
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

            if mcp_manager.hatch_env_manager.set_current_environment(name):
                self.logger.info(f"Current environment set to: {name}")

                # When changing the current environment, we must handle
                # disconnecting from the previous environment's tools if any,
                # and connecting to the new environment's tools.
                if mcp_manager.is_connected:
                    
                    # Disconnection
                    await mcp_manager.disconnect_all()
                    self.logger.info("Disconnected from previous environment's tools.")

                    # Get the new environment's entry points for the MCP servers
                    mcp_servers_url = mcp_manager.hatch_env_manager.get_servers_entry_points(name)

                    if mcp_servers_url:
                        # Reconnect to the new environment's tools
                        connected = await mcp_manager.connect_to_servers(mcp_servers_url)
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
            'refresh-registry': {'aliases': ['r'], 'default': False, 'action': 'store_true'},
            'auto-approve': {'aliases': ['y'], 'default': False, 'action': 'store_true'}
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
            auto_approve = parsed_args.get('auto-approve', False)

            if mcp_manager.hatch_env_manager.add_package_to_environment(package, env, version, force_download, refresh_registry, auto_approve):
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

            if mcp_manager.hatch_env_manager.remove_package(package_name, env):
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
            packages = mcp_manager.hatch_env_manager.list_packages(env)
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
            
            # Create validator with registry data from environment manager
            from hatch_validator import HatchPackageValidator
            validator = HatchPackageValidator(
                version="latest",
                allow_local_dependencies=True,
                registry_data=mcp_manager.hatch_env_manager.registry_data
            )
            
            # Validate the package
            is_valid, validation_results = validator.validate_package(package_path)
            
            if is_valid:
                self.logger.info(f"Package validation SUCCESSFUL: {package_path}")
            else:
                self.logger.warning(f"Package validation FAILED: {package_path}")
                
                # Print detailed validation results if available
                if validation_results and isinstance(validation_results, dict):
                    for category, result in validation_results.items():
                        if category != 'valid' and category != 'metadata' and isinstance(result, dict):
                            if not result.get('valid', True) and result.get('errors'):
                                self.logger.warning(f"\n{category.replace('_', ' ').title()} errors:")
                                for error in result['errors']:
                                    self.logger.warning(f"  - {error}")

        except Exception as e:
            self.logger.error(f"Error validating package: {e}")

        return True
    
    def _cmd_env_python_init(self, args: str) -> bool:
        """Initialize Python environment for a Hatch environment.
        
        Creates a Python environment using conda/mamba for the specified Hatch environment,
        with optional hatch_mcp_server wrapper installation. The MCP server installation
        can be controlled via command flags.
        
        Args:
            args (str): Environment options including:
                      - hatch_env: Hatch environment name (optional)
                      - python-version: Python version (optional)
                      - force: Force recreation if exists
                      - no-hatch-mcp-server: Skip MCP server installation
                      - hatch_mcp_server_tag: Git tag/branch for MCP server
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'hatch_env': {'positional': False, 'default': None},
            'python-version': {'default': None},
            'force': {'default': False, 'is_flag': True},
            'no-hatch-mcp-server': {'default': False, 'action': 'store_true'},
            'hatch_mcp_server_tag': {'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        hatch_env = parsed_args.get('hatch_env')
        python_version = parsed_args.get('python-version')
        force = parsed_args.get('force', False)
        no_hatch_mcp_server = parsed_args.get('no-hatch-mcp-server', False)
        hatch_mcp_server_tag = parsed_args.get('hatch_mcp_server_tag')

        try:
            if mcp_manager.hatch_env_manager.create_python_environment_only(
                hatch_env, 
                python_version, 
                force,
                no_hatch_mcp_server=no_hatch_mcp_server,
                hatch_mcp_server_tag=hatch_mcp_server_tag
            ):
                env_name = hatch_env if hatch_env else "current environment"
                self.logger.info(f"Python environment initialized for: {env_name}")
                
                if python_version:
                    self.logger.info(f"Python version: {python_version}")
                    
                # Show Python environment info
                python_info = mcp_manager.hatch_env_manager.get_python_environment_info(hatch_env)
                if python_info:
                    self.logger.info(f"  Python executable: {python_info['python_executable']}")
                    self.logger.info(f"  Python version: {python_info.get('python_version', 'Unknown')}")
                    self.logger.info(f"  Conda environment: {python_info.get('conda_env_name', 'N/A')}")
                
                # Provide feedback about MCP server installation
                if not no_hatch_mcp_server:
                    if hatch_mcp_server_tag:
                        self.logger.info(f"hatch_mcp_server installed with tag/branch: {hatch_mcp_server_tag}")
                    else:
                        self.logger.info("hatch_mcp_server installed with default branch")
                else:
                    self.logger.info("hatch_mcp_server installation skipped (--no-hatch-mcp-server)")
            else:
                env_name = hatch_env if hatch_env else "current environment"
                self.logger.error(f"Failed to initialize Python environment for: {env_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing Python environment: {e}")
            
        return True
    
    def _cmd_env_python_info(self, args: str) -> bool:
        """Show Python environment information.
        
        Args:
            args (str): Environment options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'hatch_env': {'positional': False, 'default': None},
            'detailed': {'default': False, 'is_flag': True}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        hatch_env = parsed_args.get('hatch_env')
        detailed = parsed_args.get('detailed', False)

        try:
            env_name = hatch_env if hatch_env else "current environment"
            
            if detailed:
                # Get detailed diagnostics
                python_info = mcp_manager.hatch_env_manager.get_python_environment_diagnostics(hatch_env)
                if python_info:
                    self.logger.info(f"Detailed Python environment diagnostics for {env_name}:")
                    for key, value in python_info.items():
                        if isinstance(value, dict):
                            self.logger.info(f"  {key}:")
                            for sub_key, sub_value in value.items():
                                self.logger.info(f"    {sub_key}: {sub_value}")
                        else:
                            self.logger.info(f"  {key}: {value}")
                else:
                    self.logger.info(f"No detailed Python environment diagnostics found for {env_name}.")
            else:
                # Get basic info
                python_info = mcp_manager.hatch_env_manager.get_python_environment_info(hatch_env)
                if python_info:
                    self.logger.info(f"Python environment information for {env_name}:")
                    for key, value in python_info.items():
                        if key == "packages":
                            continue
                        self.logger.info(f"  {key}: {value}")
                    # List packages separately
                    self.logger.info("  Packages:")
                    for _pkg in python_info.get("packages", []):
                        self.logger.info(f"    - {_pkg['name']} ({_pkg['version']})") 
                else:
                    self.logger.info(f"No Python environment information found for {env_name}.")
                
        except Exception as e:
            self.logger.error(f"Error getting Python environment information: {e}")
            
        return True
    
    def _cmd_env_python_remove(self, args: str) -> bool:
        """Remove Python environment.
        
        Args:
            args (str): Environment options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'hatch_env': {'positional': False, 'default': None},
            'force': {'default': False, 'is_flag': True}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        hatch_env = parsed_args.get('hatch_env')
        force = parsed_args.get('force', False)

        try:
            env_name = hatch_env if hatch_env else "current environment"
            
            if not force:
                # Ask for confirmation if not forced
                self.logger.warning(f"This will remove the Python environment for {env_name}. Use --force to skip confirmation.")
                return True
            
            if mcp_manager.hatch_env_manager.remove_python_environment_only(hatch_env):
                self.logger.info(f"Python environment removed for Hatch environment: {env_name}")
            else:
                self.logger.error(f"Failed to remove Python environment for Hatch environment: {env_name}")
                
        except Exception as e:
            self.logger.error(f"Error removing Python environment: {e}")
            
        return True
    
    def _cmd_env_python_shell(self, args: str) -> bool:
        """Launch Python shell in environment.
        
        Args:
            args (str): Environment options.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'hatch_env': {'positional': False, 'default': None},
            'cmd': {'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        hatch_env = parsed_args.get('hatch_env')
        cmd = parsed_args.get('cmd')

        try:
            env_name = hatch_env if hatch_env else "current environment"
            
            if mcp_manager.hatch_env_manager.launch_python_shell(hatch_env, cmd):
                if cmd:
                    self.logger.info(f"Executing command '{cmd}' in Python environment for {env_name}")
                else:
                    self.logger.info(f"Python shell launched in Hatch environment: {env_name}")
            else:
                self.logger.error(f"Failed to launch Python shell in Hatch environment: {env_name}")
                
        except Exception as e:
            self.logger.error(f"Error launching Python shell: {e}")
            
        return True
    
    def _cmd_env_python_add_hatch_mcp(self, args: str) -> bool:
        """Add hatch_mcp_server wrapper to the environment.
        
        Installs the hatch_mcp_server wrapper in an existing Python environment
        within a Hatch environment. The environment must have a valid Python
        environment already initialized.
        
        Args:
            args (str): Installation options including:
                      - hatch_env: Hatch environment name (optional)
                      - tag: Git tag/branch reference for installation
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'hatch_env': {'positional': False, 'default': None},
            'tag': {'default': None}
        }
        
        parsed_args = self._parse_args(args, arg_defs)
        hatch_env = parsed_args.get('hatch_env')
        tag = parsed_args.get('tag')

        try:
            env_name = hatch_env if hatch_env else mcp_manager.hatch_env_manager.get_current_environment()
            
            if mcp_manager.hatch_env_manager.install_mcp_server(hatch_env, tag):
                self.logger.info(f"hatch_mcp_server wrapper installed successfully in environment: {env_name}")
                if tag:
                    self.logger.info(f"Using tag/branch: {tag}")
                else:
                    self.logger.info("Using default branch")
            else:
                self.logger.error(f"Failed to install hatch_mcp_server wrapper in environment: {env_name}")
                
        except Exception as e:
            self.logger.error(f"Error installing hatch_mcp_server wrapper: {e}")
            
        return True