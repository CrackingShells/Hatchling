"""Base chat commands module for handling core chat interface commands.

Contains the BaseChatCommands class which provides basic command handling functionality
for the chat interface, including help, exit, log control and tool management.
"""

import logging

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

from hatchling import __version__
from hatchling.config.i18n import translate
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.ui.abstract_commands import AbstractCommands


class BaseChatCommands(AbstractCommands):
    """Handles processing of command inputs in the chat interface."""

    def __init__(self, chat_session, settings_registry, style, cli_chat):
        super().__init__(chat_session, settings_registry, style)
        self.cli_chat = cli_chat

    def _register_commands(self) -> None:
        """Register all available chat commands with their handlers."""
        # New standardized command registration format with i18n support
        self.commands = {
            'help': {
                'handler': self._cmd_help,
                'description': translate("commands.base.help_description"),
                'is_async': False,
                'args': {}
            },
            'exit': {
                'handler': self._cmd_exit,
                'description': translate("commands.base.exit_description"),
                'is_async': False,
                'args': {}
            },
            'quit': {
                'handler': self._cmd_exit,
                'description': translate("commands.base.quit_description"),
                'is_async': False,
                'args': {}
            },
            'clear': {
                'handler': self._cmd_clear,
                'description': translate("commands.base.clear_description"),
                'is_async': False,
                'args': {}
            },
            'show_logs': {
                'handler': self._cmd_show_logs,
                'description': translate("commands.base.show_logs_description"),
                'is_async': False,
                'args': {
                    'count': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': ['10', '20', '50', '100'],
                        'description': translate('commands.args.value_description'),
                        'required': False
                    }
                }
            },
            'set_log_level': {
                'handler': self._cmd_set_log_level,
                'description': translate("commands.base.set_log_level_description"),
                'is_async': False,
                'args': {
                    'level': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': ['debug', 'info', 'warning', 'error', 'critical'],
                        'description': translate('commands.args.value_description'),
                        'required': True
                    }
                }
            },
            'version': {
                'handler': self._cmd_version,
                'description': translate("commands.base.version_description"),
                'is_async': False,
                'args': {}
            },
            ':load': {
                'handler': self._cmd_load,
                'description': translate("commands.load.description"),
                'is_async': True,
                'args_description': translate("commands.load.args_description"),
                'example': ":load my_script.txt"
            }
        }

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
        This command is picked up by the ChatCommandHandler and not here.
        That's because it concerns all commands, not just base commands.
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
    
    def _cmd_version(self, _: str) -> bool:
        """Display the current version of Hatchling.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        self.logger.info(f"Hatchling version: {__version__}")
        return True

    async def _cmd_load(self, filename: str) -> bool:
        """Handles the :load command to execute commands from a file."""
        if not filename:
            self.logger.error("Error: Please provide a filename for the :load command.")
            return True # Continue interactive session
        
        await self.cli_chat._process_commands_from_file(filename)
        return True # Continue interactive session