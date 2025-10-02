"""Chat command handler module for processing user commands in the chat interface.

This module provides a central handler for all chat commands by combining
base commands, Hatch-specific commands, and settings commands into a unified interface.
"""

from typing import Tuple, Optional

from prompt_toolkit.completion import FuzzyCompleter
from prompt_toolkit.styles import Style

from hatchling.core.logging.logging_manager import logging_manager

from hatchling.config.i18n import translate
from hatchling.config.settings_registry import SettingsRegistry
from hatchling.mcp_utils.manager import mcp_manager
from hatchling.ui.base_commands import BaseChatCommands
from hatchling.ui.command_completion import CommandCompleter
from hatchling.ui.command_lexer import ChatCommandLexer
from hatchling.ui.hatch_commands import HatchCommands
from hatchling.ui.mcp_commands import MCPCommands
from hatchling.ui.model_commands import ModelCommands
from hatchling.ui.settings_commands import SettingsCommands

class ChatCommandHandler:
    """Handles processing of command inputs in the chat interface."""    
    def __init__(self, chat_session, settings_registry: SettingsRegistry, style: Optional[Style] = None, cli_chat=None):
        """Initialize the command handler.
        
        Args:
            chat_session: The chat session this handler is associated with.
            settings_registry (SettingsRegistry): The settings registry containing configuration.
            style (Optional[Style]): Style for formatting command output.
            cli_chat: The CLIChat instance for accessing its methods.
        """


        self.settings_registry = settings_registry
        self.cli_chat = cli_chat
        self.base_commands = BaseChatCommands(chat_session, settings_registry, style, cli_chat)
        self.hatch_commands = HatchCommands(chat_session, settings_registry, style)
        self.settings_commands = SettingsCommands(chat_session, settings_registry, style)
        self.mcp_commands = MCPCommands(chat_session, settings_registry, style)
        self.model_commands = ModelCommands(chat_session, settings_registry, style)

        self.logger = logging_manager.get_session("hatchling.core.chat.command_handler")

        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register all available chat commands with their handlers."""
        # Combine all commands from all handlers
        self.commands = {}
        self.commands.update(self.base_commands.reload_commands())
        self.commands.update(self.hatch_commands.reload_commands())
        self.commands.update(self.settings_commands.reload_commands())
        self.commands.update(self.mcp_commands.reload_commands())
        self.commands.update(self.model_commands.reload_commands())

        self.command_completer = FuzzyCompleter(CommandCompleter(self.commands, mcp_manager.hatch_env_manager))
        self.command_lexer = ChatCommandLexer(self.commands)
        
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
        print("\n=== Chat Commands ===")
        print("Type 'help' for this help message")
        print()
        
        self.base_commands.print_commands_help()
        self.hatch_commands.print_commands_help()
        self.settings_commands.print_commands_help()
        self.mcp_commands.print_commands_help()
        self.model_commands.print_commands_help()
            
        print("======================\n")

    def set_commands_language(self, language_code: str) -> None:
        """Set the language for all commands.
        
        Args:
            language_code (str): The language code to set.
        """
        if not self.settings_registry:
            self.logger.error(translate("errors.settings_registry_not_available"))

        try:
            success = self.settings_registry.set_language(language_code)
            if success:
                self._register_commands()  # Re-register commands to apply new language
            else:
                self.logger.error(translate("errors.set_language_failed", language=language_code))
        except Exception as e:
            self.logger.error(str(e))
        

    async def process_command(self, user_input: str) -> Tuple[bool, bool]:
        """Process a potential command from user input.
        
        Args:
            user_input (str): The user's input text.
            
        Returns:
            Tuple[bool, bool]: (is_command, should_continue)
              - is_command: True if input was a command
              - should_continue: False if chat session should end
        """
        user_input = user_input.strip()
        
        # Handle empty input
        if not user_input:
            return True, True
            
        # Extract command and arguments
        parts = user_input.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "help":
            self.print_commands_help()
            return True, True
        
        if command == "settings:language:set":
            self.set_commands_language(args.strip())
            return True, True
        
        # Check if the input is a registered command
        if command in self.sync_commands:
            handler_func, _ = self.sync_commands[command]
            return True, handler_func(args)
        elif command in self.async_commands:
            async_handler_func, _ = self.async_commands[command]
            return True, await async_handler_func(args)
            
        # Not a command
        return False, True

    def get_all_command_metadata(self) -> dict:
        """Get all command metadata from both command handlers.
        
        Returns:
            dict: Combined command metadata from base and hatch commands.
        """
        return self.commands