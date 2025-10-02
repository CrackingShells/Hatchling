import asyncio

from prompt_toolkit import PromptSession, print_formatted_text as print_pt
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings

from hatchling.core.logging.logging_manager import logging_manager
from hatchling.core.llm.providers.registry import ProviderRegistry
from hatchling.core.llm.chat_session import ChatSession
from hatchling.config.settings_registry import SettingsRegistry
from hatchling.mcp_utils.manager import mcp_manager
from hatchling.ui.chat_command_handler import ChatCommandHandler
from hatchling.ui.cli_event_subscriber import CLIEventSubscriber

class CLIChat:
    """Command-line interface for chat functionality."""

    def __init__(self, settings_registry: SettingsRegistry, input_file_path: str = None):
        """Initialize the CLI chat interface.
        
        Args:
            settings (SettingsRegistry): The settings management instance containing configuration.
            input_file_path (str, optional): Path to a file containing commands to execute. Defaults to None.
        """
        # Store settings first
        self.settings_registry = settings_registry
        self.input_file_path = input_file_path
        
        # Get a logger - styling is already configured at the application level
        self.logger = logging_manager.get_session("CLIChat")
        
        # Initialize prompt toolkit session with history
        history_dir = self.settings_registry.settings.paths.hatchling_cache_dir / 'histories'
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
            
            # Toolbar styles
            'toolbar.default': "#63818d",            # Sky blue for default toolbar
            'toolbar.tool': '#ffa500',               # Orange for tool execution
            'toolbar.error': '#ff6b6b',              # Red for errors
            'toolbar.info': "#49a949",               # Light green for info
            
            # Right prompt styles
            'right-prompt': '#d3d3d3',               # Light gray for right prompt
        })
            
        # Provider will be initialized during startup
        # Initialize the provider
        try:
            ProviderRegistry.get_provider(self.settings_registry.settings.llm.provider_enum)
        
        except Exception as e:
            msg = f"Failed to initialize {self.settings_registry.settings.llm.provider_enum} LLM provider: {e}"
            msg += "\nEnsure the LLM provider name is correct in your settings."
            msg += "\nYou can list providers compatible with Hatchling using `model:provider:list` command."
            msg += "\nEnsure you have switched to a supported provider before trying to use the chat interface."
            self.logger.warning(msg)
        
        finally:
            # Initialize chat session
            self.chat_session = ChatSession()
            
            # Initialize CLI event subscriber for UI state management
            self.cli_event_subscriber = CLIEventSubscriber()
            
            # Register CLI subscriber with chat session (decoupled)
            self.chat_session.register_subscriber(self.cli_event_subscriber)
            mcp_manager.publisher.subscribe(self.cli_event_subscriber)

            # Initialize command handler
            self.cmd_handler = ChatCommandHandler(self.chat_session, self.settings_registry, self.command_style, self)
            
            # Setup key bindings
            self.key_bindings = self._create_key_bindings()
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings for UI control.
        
        Returns:
            KeyBindings: Configured key bindings for the CLI.
        """
        kb = KeyBindings()
        
        @kb.add('f2')
        def _(event):
            """Cycle toolbar view mode."""
            self.cli_event_subscriber.cycle_toolbar_view()
            # Force UI refresh by triggering a redraw
            event.app.invalidate()
        
        @kb.add('f3')
        def _(event):
            """Cycle right prompt view mode."""
            self.cli_event_subscriber.cycle_right_prompt_view()
            # Force UI refresh by triggering a redraw
            event.app.invalidate()
        
        @kb.add('f4')
        def _(event):
            """Clear current error/info messages."""
            self.cli_event_subscriber.current_error = None
            self.cli_event_subscriber.current_info = None
            event.app.invalidate()
        
        @kb.add('f10')
        def _(event):
            """Panic: forcibly reset UI state and allow user input."""
            self.logger.warning("F10 PANIC: Forcibly resetting UI state to allow user input.")
            self.cli_event_subscriber.set_processing_user_message(False)
            event.app.invalidate()
        
        @kb.add('f12')
        def _(event):
            """Show help for key bindings."""
            help_text = (
                "📋 Key Bindings:\n"
                "F2  - Cycle toolbar views\n"
                "F3  - Cycle right prompt views\n"
                "F4  - Clear messages\n"
                "F12 - Show this help\n"
            )
            print_pt(FormattedText([('class:toolbar.info', help_text)]))
        
        return kb
    
    def _get_bottom_toolbar(self) -> FormattedText:
        """Get bottom toolbar text with styling.
        
        Returns:
            FormattedText: Formatted toolbar text.
        """
        toolbar_text = self.cli_event_subscriber.get_toolbar_text()
        
        # Style based on content
        if toolbar_text.startswith('❌'):
            return FormattedText([('class:toolbar.error', toolbar_text)])
        elif toolbar_text.startswith('ℹ️'):
            return FormattedText([('class:toolbar.info', toolbar_text)])
        elif toolbar_text.startswith('🔧'):
            return FormattedText([('class:toolbar.tool', toolbar_text)])
        else:
            return FormattedText([('class:toolbar.default', toolbar_text)])
    
    def _get_right_prompt(self) -> FormattedText:
        """Get right prompt text with styling.
        
        Returns:
            FormattedText: Formatted right prompt text.
        """
        right_prompt_text = self.cli_event_subscriber.get_right_prompt_text()
        return FormattedText([('class:right-prompt', right_prompt_text)])
    
    async def start_interactive_session(self) -> None:
        """Run an interactive chat session with message history or process commands from a file."""

        if self.input_file_path:
            await self._process_commands_from_file(self.input_file_path)
            # After processing the file, continue to interactive mode

        #async with aiohttp.ClientSession() as session:
        while True: 
            try:
                # Create formatted prompt
                prompt_message = [
                    # status_style,
                    ('green', 'You: ')
                ]
                # Use patch_stdout to prevent output interference
                with patch_stdout():
                    user_input = await self.prompt_session.prompt_async(
                        FormattedText(prompt_message),
                        completer=self.cmd_handler.command_completer,
                        lexer=self.cmd_handler.command_lexer,
                        style=self.command_style,
                        key_bindings=self.key_bindings,
                        bottom_toolbar=self._get_bottom_toolbar,
                        rprompt=self._get_right_prompt
                    )
                
                # Process as command if applicable
                is_command, should_continue = await self.cmd_handler.process_command(user_input)
                if is_command:
                    if not should_continue:
                        break
                    continue
                
                # Handle normal message
                if not user_input.strip():
                    # Skip empty input
                    continue

                # Mark that we're starting to process user message
                self.cli_event_subscriber.set_processing_user_message(True)

                # Clear previous content from event subscriber
                #self.cli_event_subscriber.clear_content_buffer()
                
                try:
                    # Send the message (this will trigger streaming events)
                    await self.chat_session.send_message(user_input)
                    
                    # Wait for all processing to complete (tool chains, etc.)
                    await self._monitor_right_to_prompt()

                except Exception as send_error:
                    # Make sure to reset state on error
                    self.cli_event_subscriber.set_processing_user_message(False)
                    raise send_error
                
                print_pt('')  # Add an extra newline for readability

            except KeyboardInterrupt:
                print_pt(FormattedText([('red', '\nInterrupted. Ending chat session...')]))
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
                print_pt(FormattedText([('red', f'\nError: {e}')]))

    async def _process_commands_from_file(self, file_path: str) -> None:
        """Processes commands from a given file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    user_input = line.strip()
                    if not user_input:
                        continue

                    print_pt(FormattedText([('green', f'You: {user_input}')]))

                    is_command, should_continue = await self.cmd_handler.process_command(user_input)
                    if is_command:
                        if not should_continue:
                            break
                        continue

                    self.cli_event_subscriber.set_processing_user_message(True)
                    try:
                        await self.chat_session.send_message(user_input)
                        await self._monitor_right_to_prompt()
                    except Exception as send_error:
                        self.cli_event_subscriber.set_processing_user_message(False)
                        self.logger.error(f"Error processing message from file: {send_error}")
                    print_pt('')
        except FileNotFoundError:
            self.logger.error(f"Error: Input file not found at {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing input file: {e}")
    
    async def initialize_and_run(self) -> None:
        """Initialize the environment and run the interactive chat session."""
        try:
            # Start the interactive session
            await self.start_interactive_session()
            
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            self.logger.error(error_msg)
            return
        
        finally:
            # Clean up any remaining MCP server processes only if tools were enabled
            if self.chat_session and len(mcp_manager.get_enabled_tools()) > 0:
                await mcp_manager.disconnect_all()

    async def _monitor_right_to_prompt(self) -> None:
        """
        Blocks until the all conditions are satisfied to finish a prompt loop
        and go back to async prompt input.
        """
        # Give a small delay to allow events to propagate
        await asyncio.sleep(0.1)
        
        while not self.cli_event_subscriber.is_ready_for_user_input():
            self.logger.debug("Waiting for user input readiness...")
            await asyncio.sleep(0.25)  # Check every 250ms