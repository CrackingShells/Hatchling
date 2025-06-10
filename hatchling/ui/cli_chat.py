import aiohttp
import logging
import asyncio
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession, print_formatted_text as print_pt
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from hatchling.core.logging.logging_manager import logging_manager
from hatchling.core.llm.model_manager import ModelManager
from hatchling.core.llm.chat_session import ChatSession
from hatchling.core.chat.chat_command_handler import ChatCommandHandler
from hatchling.core.chat.command_completion import CommandCompleterFactory
from hatchling.core.chat.command_lexer import ChatCommandLexer
from hatchling.config.settings import ChatSettings
from hatchling.mcp_utils.manager import mcp_manager
# Import removed - using centralized logging system

from hatch import HatchEnvironmentManager

class CLIChat:
    """Command-line interface for chat functionality."""    
    
    def __init__(self, settings: ChatSettings):
        """Initialize the CLI chat interface.
        
        Args:
            settings (ChatSettings): The chat settings to use.
        """
        # Store settings first
        self.settings = settings
        
        # Get a logger - styling is already configured at the application level
        self.logger = logging_manager.get_session("CLIChat")
        
        # Initialize prompt toolkit session with history
        history_dir = Path.home() / '.hatch' / 'histories'
        history_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup persistent history with 500 entries limit
        try:
            self.prompt_session = PromptSession(
                history=FileHistory(str(history_dir / '.user_inputs')))
        except (IOError, OSError) as e:
            self.logger.warning(f"Could not create history file: {e}")
            self.logger.warning("Falling back to in-memory history")
            self.prompt_session = PromptSession(history=InMemoryHistory())
        
        # Define command styling for both help display and real-time input highlighting
        self.command_style = Style.from_dict({
            # Help display styles
            'command.name': 'bold #44ff00',          # Green bold for command names
            'command.description': "#ffffff",        # White for descriptions
            'command.args': 'italic #87afff',        # Light blue italic for arguments
            'header': 'bold #ff9d00 underline',      # Orange underline for headers

            # Group specific styles for help
            'command.name.hatch': 'bold #00b7c3',    # Teal for Hatch commands
            'command.name.base': 'bold #44ff00',     # Green for base commands
            'group.default': '',                     # Default group style
            
            # Real-time input highlighting styles
            'command.name': 'bold #44ff00',          # Command names - bright green
            'command.args.base': 'bold #87afff',     # Base command arguments - blue
            'command.args.hatch': 'bold #00b7c3',    # Hatch command arguments - teal
            'command.args.invalid': '#ff6b6b',       # Invalid arguments - red
            'command.value.path': '#ffb347',         # Path values - orange
            'command.value.number': '#98fb98',       # Number values - light green
            'command.value.string': '#dda0dd',       # String values - plum
            'command.value.generic': '#f0f0f0',      # Generic values - light gray
            'text.default': '#ffffff',               # Default text - white
        })
        
        self.env_manager = HatchEnvironmentManager(
            environments_dir = self.settings.hatch_envs_dir,
            cache_ttl = 86400,  # 1 day default
        )
            
        # Create the model manager
        self.model_manager = ModelManager(settings, self.logger)
        
        # Chat session will be initialized during startup
        self.chat_session = None
        self.cmd_handler = None
    
    async def initialize(self) -> bool:
        """Initialize the chat environment.
        
        Returns:
            bool: True if initialization was successful.
        """
        # Check if Ollama service is available
        available, message = await self.model_manager.check_ollama_service()
        if not available:
            self.logger.error(message)
            self.logger.error(f"Please ensure the Ollama service is running at {self.settings.ollama_api_url} before running this script.")
            return False
        
        self.logger.info(message)
        
        # Check if MCP server is available
        self.logger.info("Checking MCP server availability...")
        # Get the name of the current environment
        name = self.env_manager.get_current_environment()
        # Retrieve the environment's entry points for the MCP servers
        mcp_servers_url = self.env_manager.get_servers_entry_points(name)
        mcp_available = await mcp_manager.initialize(mcp_servers_url)
        if mcp_available:
            self.logger.info("MCP server is available! Tool calling is ready to use.")
            self.logger.info("You can enable tools during the chat session by typing 'enable_tools'")
        else:
            self.logger.warning("MCP server is not available. Continuing without MCP tools...")
            
        # Initialize chat session
        self.chat_session = ChatSession(self.settings)
        # Initialize command handler
        self.cmd_handler = ChatCommandHandler(self.chat_session, self.settings, self.env_manager, self.logger, self.command_style)
        
        # Initialize command completer
        self.command_completer = CommandCompleterFactory.create_completer(self.cmd_handler)
        
        # Initialize command lexer for real-time syntax highlighting
        all_commands = self.cmd_handler.get_all_command_metadata()
        self.command_lexer = ChatCommandLexer(all_commands)
        
        return True
    
    async def check_and_pull_model(self, session: aiohttp.ClientSession) -> bool:
        """Check if the model is available and pull it if necessary.
        
        Args:
            session (aiohttp.ClientSession): The session to use for API calls.
            
        Returns:
            bool: True if model is available (either already or after pulling).
        """
        try:
            # Check if model is available
            is_model_available = await self.model_manager.check_availability(session, self.settings.ollama_model)
            
            if is_model_available:
                self.logger.info(f"Model {self.settings.ollama_model} is already pulled.")
                return True
            else:
                await self.model_manager.pull_model(session, self.settings.ollama_model)
                return True
        except Exception as e:
            self.logger.error(f"Error checking/pulling model: {e}")
            return False
    
    async def start_interactive_session(self) -> None:
        """Run an interactive chat session with message history."""
        if not self.chat_session or not self.cmd_handler:
            self.logger.error("Chat session not initialized. Call initialize() first.")
            return
        
        self.logger.info(f"Starting interactive chat with {self.settings.ollama_model}")
        print_pt(FormattedText([('cyan bold', '\n=== Hatchling Chat Interface ===\n')]))
        self.cmd_handler.print_commands_help()
        
        async with aiohttp.ClientSession() as session:
            # Check and pull the model if needed
            if not await self.check_and_pull_model(session):
                self.logger.error("Failed to ensure model availability")
                return
              # Start the interactive chat loop
            while True: 
                try:
                    # Get user input with prompt_toolkit with a styled prompt
                    if self.chat_session.tool_executor.tools_enabled:
                        status_style = ('fg:#5fafff  bold', '[Tools enabled]') #aqua pearl
                    else:
                        status_style = ('fg:#005f5f', '[Tools disabled]') #very dark cyan
                    
                    # Create formatted prompt
                    prompt_message = [
                        status_style,
                        ('', ' You: ')
                    ]
                    # Use patch_stdout to prevent output interference
                    with patch_stdout():
                        user_message = await self.prompt_session.prompt_async(
                            FormattedText(prompt_message),
                            completer=self.command_completer,
                            lexer=self.command_lexer,
                            style=self.command_style
                        )
                    
                    # Process as command if applicable
                    is_command, should_continue = await self.cmd_handler.process_command(user_message)
                    if is_command:
                        if not should_continue:
                            break
                        continue
                    
                    # Handle normal message
                    if not user_message.strip():
                        # Skip empty input
                        continue
                      # Send the query
                    print_pt(FormattedText([('green', '\nAssistant: ')]), end='', flush=True)
                    await self.chat_session.send_message(user_message, session)
                    print_pt('')  # Add an extra newline for readability
                except KeyboardInterrupt:
                    print_pt(FormattedText([('red', '\nInterrupted. Ending chat session...')]))
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    print_pt(FormattedText([('red', f'\nError: {e}')]))
    
    async def initialize_and_run(self) -> None:
        """Initialize the environment and run the interactive chat session."""
        try:
            # Initialize the chat environment
            if not await self.initialize():
                return
            
            # Start the interactive session
            await self.start_interactive_session()
            
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            self.logger.error(error_msg)
            return
        
        finally:
            # Clean up any remaining MCP server processes
            # We disconnect by default after checking MCP availability
            await mcp_manager.disconnect_all()