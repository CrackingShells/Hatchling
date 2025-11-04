import asyncio
import sys
import argparse
from hatchling.core.logging.logging_config import configure_logging
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings_registry import SettingsRegistry
from hatchling.config.i18n import init_translation_loader
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
    parser = argparse.ArgumentParser(description="Hatchling CLI Chat Interface")
    parser.add_argument("-f", "--input-file", type=str, help="Path to a file containing commands to execute.")
    args = parser.parse_args()

    settings_registry = None
    try:
    
        settings_registry = SettingsRegistry()

        # Initialize translation loader
        init_translation_loader(languages_dir=settings_registry.settings.paths.hatchling_source_dir / "hatchling" / "config" / "languages",
                                 default_language_code=settings_registry.settings.ui.language_code)

        # Create and run CLI chat interface
        cli_chat = CLIChat(settings_registry, input_file_path=args.input_file)
        
        await cli_chat.initialize_and_run()
        
        return 0
        
    except KeyboardInterrupt:
        log.info("Application interrupted by user")
    except Exception as e:
        log.error(f"Error in main application: {e}")
    finally:
        try:
            settings_registry.save_persistent_settings()
            log.info("Persistent settings saved on exit")
        except Exception as e:
            log.error(f"Failed to save persistent settings on exit: {e}")

def main():
    """Entry point function that runs the async main function.
    
    Returns:
        int: Exit code from the async main function.
    """
    return asyncio.run(main_async())

if __name__ == "__main__":
    # Run the application
    sys.exit(main())