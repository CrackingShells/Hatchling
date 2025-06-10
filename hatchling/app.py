import asyncio
import sys
from hatchling.core.logging.logging_config import configure_logging
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings import ChatSettings
from hatchling.ui.cli_chat import CLIChat

# Configure global logging first, before any other imports or logger initializations
# Simply determine if we're in an interactive environment
is_interactive = sys.stdout.isatty()

# Let the centralized logging system handle all the details
configure_logging(enable_styling=is_interactive)

# Get logger with custom formatter - now using the centralized styling system
log = logging_manager.get_session("AppMain")

async def main_async():
    """Main entry point for the application.
    
    Returns:
        int: Exit code - 0 for successful execution.
    
    Raises:
        Exception: Any unhandled exceptions that occur during execution.
    """
    try:
        # Create settings with MCP server path
        settings = ChatSettings()
        
        # Create and run CLI chat interface
        cli_chat = CLIChat(settings)
        
        await cli_chat.initialize_and_run()
        
        return 0
        
    except KeyboardInterrupt:
        log.info("Application interrupted by user")
    except Exception as e:
        log.error(f"Error in main application: {e}")

def main():
    """Entry point function that runs the async main function.
    
    Returns:
        int: Exit code from the async main function.
    """
    return asyncio.run(main_async())

if __name__ == "__main__":
    # Run the application
    sys.exit(main())