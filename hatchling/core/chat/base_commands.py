"""Base chat commands module for handling core chat interface commands.

Contains the BaseChatCommands class which provides basic command handling functionality
for the chat interface, including help, exit, log control and tool management.
"""

import logging
from typing import Tuple

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

from hatchling.core.logging.session_debug_log import SessionDebugLog
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.mcp_utils.manager import mcp_manager
from hatchling.config.settings import ChatSettings
from hatchling.core.chat.abstract_commands import AbstractCommands

from hatch import HatchEnvironmentManager


class BaseChatCommands(AbstractCommands):
    """Handles processing of command inputs in the chat interface."""

    def _register_commands(self) -> None:
        """Register all available chat commands with their handlers."""
        # New standardized command registration format
        self.commands = {
            'help': {
                'handler': self._cmd_help,
                'description': "Display help for available commands",
                'is_async': False,
                'args': {}
            },
            'exit': {
                'handler': self._cmd_exit,
                'description': "End the chat session",
                'is_async': False,
                'args': {}
            },
            'quit': {
                'handler': self._cmd_exit,
                'description': "End the chat session (alias for exit)",
                'is_async': False,
                'args': {}
            },
            'clear': {
                'handler': self._cmd_clear,
                'description': "Clear the chat history",
                'is_async': False,
                'args': {}
            },
            'show_logs': {
                'handler': self._cmd_show_logs,
                'description': "Display session logs",
                'is_async': False,
                'args': {
                    'count': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': ['10', '20', '50', '100'],
                        'description': 'Number of log entries to show',
                        'required': False
                    }
                }
            },
            'set_log_level': {
                'handler': self._cmd_set_log_level,
                'description': "Change log level",
                'is_async': False,
                'args': {
                    'level': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': ['debug', 'info', 'warning', 'error', 'critical'],
                        'description': 'Log level name',
                        'required': True
                    }
                }
            },
            'set_max_tool_call_iterations': {
                'handler': self._cmd_set_max_iterations,
                'description': "Set max tool call iterations",
                'is_async': False,
                'args': {
                    'iterations': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': 'Number of iterations (positive integer)',
                        'required': True
                    }
                }
            },
            'set_max_working_time': {
                'handler': self._cmd_set_max_working_time,
                'description': "Set max working time in seconds",
                'is_async': False,
                'args': {
                    'seconds': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': 'Time in seconds (positive number)',
                        'required': True
                    }
                }
            },
            'enable_tools': {
                'handler': self._cmd_enable_tools,
                'description': "Enable MCP tools",
                'is_async': True,
                'args': {}
            },
            'disable_tools': {
                'handler': self._cmd_disable_tools,
                'description': "Disable MCP tools",
                'is_async': True,
                'args': {}
            }        }

        # Keep old format for backward compatibility
        self.sync_commands = {}
        self.async_commands = {}
        
        for cmd_name, cmd_info in self.commands.items():
            if cmd_info['is_async']:
                self.async_commands[cmd_name] = (cmd_info['handler'], cmd_info['description'])
            else:
                self.sync_commands[cmd_name] = (cmd_info['handler'], cmd_info['description'])
    
    def print_commands_help(self) -> None:
        """Print help for all available chat commands."""
        print_formatted_text(FormattedText([
            ('class:header', "\n=== Base Chat Commands ===\n")
        ]), style=self.style)

        # Call parent class method to print formatted commands
        super().print_commands_help()

    def format_command(self, cmd_name: str, cmd_info: dict, group: str = 'base') -> list:
        """Format base commands with custom styling."""
        return [
            (f'class:command.name.{group}', f"{cmd_name}"),
            ('', ' - '),
            ('class:command.description', f"{cmd_info['description']}")
        ]

    def _cmd_help(self, _: str) -> bool:
        """
        This is the only command that is picked up by the ChatCommandHandler
        and not here.
        """
        pass
    
    def _cmd_exit(self, _: str) -> bool:
        """Exit the chat session.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: False to end the chat session.
        """
        print("Ending chat session...")
        return False
    
    def _cmd_clear(self, _: str) -> bool:
        """Clear chat history.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        self.chat_session.history.clear()
        print("Chat history cleared!")
        return True
    
    def _cmd_show_logs(self, args: str) -> bool:
        """Display session logs.
        
        Args:
            args (str): Optional number of log entries to show.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            logs_to_show = int(args) if args.strip() else None
            print(self.chat_session.debug_log.get_logs(logs_to_show))
        except ValueError:
            print(f"Invalid number: {args}")
            print("Usage: show_logs [n]")
        return True
    
    def _cmd_set_log_level(self, args: str) -> bool:
        """Set the log level.
        
        Args:
            args (str): Log level name (debug, info, warning, error, critical).
            
        Returns:
            bool: True to continue the chat session.
        """
        level_name = args.strip().lower()
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        
        if level_name in level_map:
            logging_manager.set_log_level(level_map[level_name])
            self.logger.info(f"Log level set to {level_name}")
            if logging_manager.log_level > logging.INFO:
                # the only place where use a print given the change of log level might disable the logger
                print(f"Log level set to {level_name}")
        else:
            self.logger.error(f"Unknown log level: {level_name}. Available levels are: debug, info, warning, error, critical")
        return True
    
    async def _cmd_enable_tools(self, _: str) -> bool:
        """Enable MCP tools.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """

        # If tools are already enabled, do nothing
        if self.chat_session.tool_executor.tools_enabled:
            self.logger.warning("MCP tools are already enabled.")
            return True

        # Get the name of the current environment
        name = self.env_manager.get_current_environment()

        # Retrieve the new environment's entry points for the MCP servers
        mcp_servers_url = self.env_manager.get_servers_entry_points(name)
        if mcp_servers_url:
            # Reconnect to the new environment's tools
            connected = await self.chat_session.initialize_mcp(mcp_servers_url)
            if not connected:
                self.logger.error("Failed to connect to new environment's MCP servers. Tools not enabled.")
            else:
                self.logger.info("Connected to new environment's MCP servers successfully!")
        else:
            self.logger.error("No MCP servers found for the current environment. Tools cannot be enabled.")
            return False
        return True

    async def _cmd_disable_tools(self, _: str) -> bool:
        """Disable MCP tools.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        if self.chat_session.tool_executor.tools_enabled:
            await mcp_manager.disconnect_all()
            self.chat_session.tool_executor.tools_enabled = False
            self.logger.info("MCP tools disabled successfully!")
        else:
            self.logger.warning("MCP tools are already disabled.")
        return True
    
    def _cmd_set_max_iterations(self, args: str) -> bool:
        """Set maximum tool call iterations.
        
        Args:
            args (str): Number of iterations.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            iterations = int(args.strip())
            if iterations > 0:
                self.settings.max_tool_call_iteration = iterations
                self.logger.info(f"Maximum tool call iterations set to {iterations}")
            else:
                self.logger.error("Maximum iterations must be greater than 0")
        except ValueError:
            self.logger.error("Invalid value for maximum iterations. Usage: set_max_tool_call_iterations <positive integer>")
        return True
    
    def _cmd_set_max_working_time(self, args: str) -> bool:
        """Set maximum working time for tool operations.
        
        Args:
            args (str): Time in seconds.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            seconds = float(args.strip())
            if seconds > 0:
                self.settings.max_working_time = seconds
                self.logger.info(f"Maximum working time set to {seconds} seconds")
            else:
                self.logger.error("Maximum working time must be greater than 0")
        except ValueError:
            self.logger.error("Invalid value for maximum working time. Usage: set_max_working_time <positive number>")
        return True