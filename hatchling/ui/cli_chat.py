import asyncio
import aiohttp
import logging
from typing import Optional

from hatchling.core.logging.logging_manager import logging_manager
from hatchling.core.llm.model_manager import ModelManager
from hatchling.core.llm.chat_session import ChatSession
from hatchling.core.chat.chat_command_handler import ChatCommandHandler
from hatchling.config.settings import ChatSettings
from hatchling.mcp_utils.manager import mcp_manager

from hatch import HatchEnvironmentManager

class CLIChat:
    """Command-line interface for chat functionality."""
    
    def __init__(self, settings: ChatSettings):
        """Initialize the CLI chat interface.
        
        Args:
            settings (ChatSettings): The chat settings to use.
        """
        # Create a debug log if not provided
        self.logger = logging_manager.get_session("CLIChat",
                                formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        self.settings = settings
        
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
        self.cmd_handler = ChatCommandHandler(self.chat_session, self.settings, self.env_manager, self.logger)
        
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
        self.cmd_handler.print_commands_help()
        
        async with aiohttp.ClientSession() as session:
            # Check and pull the model if needed
            if not await self.check_and_pull_model(session):
                self.logger.error("Failed to ensure model availability")
                return
            
            # Start the interactive chat loop
            while True:
                try:
                    # Get user input
                    status = "[Tools enabled]" if self.chat_session.tool_executor.tools_enabled else "[Tools disabled]"
                    user_message = input(f"{status} You: ")
                    
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
                    print("\nAssistant: ", end="", flush=True)
                    await self.chat_session.send_message(user_message, session)
                    print()  # Add an extra newline for readability
                    
                except KeyboardInterrupt:
                    print("\nInterrupted. Ending chat session...")
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    print(f"\nError: {e}")
    
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