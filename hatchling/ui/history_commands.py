"""History commands module for the chat interface.

This module provides commands for managing the chat message history,
including deleting messages and clearing the history.
"""

from typing import Dict, Any, Optional

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from hatchling.config.i18n import translate
from hatchling.ui.abstract_commands import AbstractCommands
from hatchling.core.chat.message_history_registry import MessageHistoryRegistry
from hatchling.core.logging.logging_manager import logging_manager


class HistoryCommands(AbstractCommands):
    """Handles history management commands in the chat interface."""

    def __init__(self, chat_session, settings_registry, style: Optional[Style] = None):
        super().__init__(chat_session, settings_registry, style)
        self.current_chat_uid = chat_session.session_id
        self.logger = logging_manager.get_session("HistoryCommands")
        self._register_commands()

    def _register_commands(self) -> None:
        """Register all available history commands."""
        self.commands = {
            'history:delete': {
                'handler': self._cmd_history_delete,
                'description': translate('commands.history.delete_description'),
                'is_async': False,
                'args': {
                    'count': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': translate('commands.args.history_delete_count_description'),
                        'required': False,
                        'default': '1' # Default to deleting the last message
                    }
                }
            },
            'history:keep': {
                'handler': self._cmd_history_keep,
                'description': translate('commands.history.keep_description'),
                'is_async': False,
                'args': {
                    'count': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': translate('commands.args.history_keep_count_description'),
                        'required': True
                    }
                }
            },
            'history:show': {
                'handler': self._cmd_history_show,
                'description': translate('commands.history.show_description'),
                'is_async': False,
                'args': {
                    'count': {
                        'positional': True,
                        'completer_type': 'none',
                        'description': translate('commands.args.history_show_count_description'),
                        'required': False
                    }
                }
            },
            'history:export': {
                'handler': self._cmd_history_export,
                'description': translate('commands.history.export_description'),
                'is_async': False,
                'args': {
                    'file_path': {
                        'positional': True,
                        'completer_type': 'path',
                        'description': translate('commands.args.history_export_file_path_description'),
                        'required': True
                    }
                }
            },
            'history:save': {
                'handler': self._cmd_history_save,
                'description': translate('commands.history.save_description'),
                'is_async': False,
                'args': {
                    'file_path': {
                        'positional': True,
                        'completer_type': 'path',
                        'description': translate('commands.args.history_file_path_description'),
                        'required': True
                    }
                }
            },
            'history:load': {
                'handler': self._cmd_history_load,
                'description': translate('commands.history.load_description'),
                'is_async': False,
                'args': {
                    'file_path': {
                        'positional': True,
                        'completer_type': 'path',
                        'description': translate('commands.args.history_file_path_description'),
                        'required': True
                    }
                }
            },
            'history:clear': {
                'handler': self._cmd_history_clear,
                'description': translate('commands.history.clear_description'),
                'is_async': False,
                'args': {}
            }
        }

    def print_commands_help(self) -> None:
        """Print help for all available history commands."""
        print_formatted_text(FormattedText([
            ('class:header', "\n=== History Chat Commands ===\n")
        ]), style=self.style)
        
        super().print_commands_help()

    def format_command(self, cmd_name: str, cmd_info: Dict[str, Any], group: str = 'history') -> list:
        """Format history commands with custom styling."""
        return [
            (f'class:command.name.{group}', f"{cmd_name}"),
            ('', ' - '),
            ('class:command.description', f"{cmd_info['description']}")
        ]

    def _get_current_history(self):
        """Helper to get the current chat's message history."""
        history = MessageHistoryRegistry.get_or_create_history(self.current_chat_uid)
        # MessageHistoryRegistry.get_or_create_history is guaranteed to return a MessageHistory instance,
        # so 'history' will never be None. The previous check 'if not history:' was misleading
        # because an empty MessageHistory instance (due to its __len__ method returning 0)
        # evaluates to False, triggering an unnecessary error log.
        return history

    def _cmd_history_delete(self, args: str) -> bool:
        """Delete the last N messages from the history.
        
        Args:
            args (str): Number of messages to delete.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'count': {'positional': True, 'default': '1'}
        }
        parsed_args = self._parse_args(args, arg_defs)
        
        history = self._get_current_history()
        if not history:
            return True

        count_str = parsed_args.get('count', '1')
        try:
            count = int(count_str)
            if count <= 0:
                self.logger.error("Count must be a positive integer.")
                return True
        except ValueError:
            self.logger.error(f"Invalid count '{count_str}'. Please provide a positive integer.")
            return True

        history.delete_last_n_messages(count)
        self.logger.info(f"Deleted last {count} messages from history.")
        return True

    def _cmd_history_keep(self, args: str) -> bool:
        """Keep only the last N messages in the history.
        
        Args:
            args (str): Number of messages to keep.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'count': {'positional': True}
        }
        parsed_args = self._parse_args(args, arg_defs)
        
        history = self._get_current_history()
        if not history:
            return True

        count_str = parsed_args.get('count')
        if not count_str:
            self.logger.error("Count is required for /history keep.")
            return True

        try:
            count = int(count_str)
            if count < 0:
                self.logger.error("Count must be a non-negative integer.")
                return True
        except ValueError:
            self.logger.error(f"Invalid count '{count_str}'. Please provide a non-negative integer.")
            return True

        history.keep_last_n_messages(count)
        self.logger.info(f"Kept last {count} messages in history.")
        return True

    def _cmd_history_show(self, args: str) -> bool:
        """Display the formatted chat history.
        
        Args:
            args (str): Optional number of latest messages to show.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'count': {'positional': True, 'default': None}
        }
        parsed_args = self._parse_args(args, arg_defs)

        history = self._get_current_history()
        if not history:
            return True
        
        count = None
        count_str = parsed_args.get('count')
        if count_str:
            try:
                count = int(count_str)
                if count <= 0:
                    self.logger.error("Count must be a positive integer for /history show.")
                    return True
            except ValueError:
                self.logger.error(f"Invalid count '{count_str}'. Please provide a positive integer or omit for full history.")
                return True

        formatted_history = history.get_formatted_history(n=count)
        print_formatted_text(FormattedText([
            ('class:header', "\n=== Chat History ===\n"),
            ('', formatted_history)
        ]), style=self.style)
        
        return True

    def _cmd_history_export(self, args: str) -> bool:
        """Export the formatted chat history to a file.
        
        Args:
            args (str): The file path to export the history to.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'file_path': {'positional': True}
        }
        parsed_args = self._parse_args(args, arg_defs)

        file_path = parsed_args.get('file_path')
        if not file_path:
            self.logger.error("File path is required for /history export.")
            return True
        
        history = self._get_current_history()
        if not history:
            return True
        
        try:
            formatted_history = history.get_formatted_history()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_history)
            self.logger.info(f"History exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export history to {file_path}: {e}")
        
        return True

    def _cmd_history_save(self, args: str) -> bool:
        """Save the current chat history to a file.
        
        Args:
            args (str): The file path to save the history to.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'file_path': {'positional': True}
        }
        parsed_args = self._parse_args(args, arg_defs)

        file_path = parsed_args.get('file_path')
        if not file_path:
            self.logger.error("File path is required for /history save.")
            return True
        
        history = self._get_current_history()
        if not history:
            return True
        
        try:
            history.save_history_to_file(file_path)
            self.logger.info(f"History saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save history to {file_path}: {e}")
        
        return True

    def _cmd_history_load(self, args: str) -> bool:
        """Load chat history from a file.
        
        Args:
            args (str): The file path to load the history from.
            
        Returns:
            bool: True to continue the chat session.
        """
        arg_defs = {
            'file_path': {'positional': True}
        }
        parsed_args = self._parse_args(args, arg_defs)

        file_path = parsed_args.get('file_path')
        if not file_path:
            self.logger.error("File path is required for /history load.")
            return True
        
        # Ensure a history instance exists for the current chat UID
        history = MessageHistoryRegistry.get_or_create_history(self.current_chat_uid)
        
        try:
            history.load_history_from_file(file_path)
            self.logger.info(f"History loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load history from {file_path}: {e}")
        
        return True

    def _cmd_history_clear(self, _: str) -> bool:
        """Clear the entire history for the current chat.
        
        Args:
            _ (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        history = self._get_current_history()
        if not history:
            return True
        
        history.clear()
        self.logger.info("Cleared entire chat history.")
        return True
