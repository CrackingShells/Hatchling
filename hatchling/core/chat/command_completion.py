"""Command completion module for the chat interface.

This module provides autocompletion functionality for chat commands using prompt_toolkit.
It implements a three-phase approach:
- Phase 2: Static command completion (command names, subcommands, argument flags)
- Phase 3: Dynamic value completion (environment names, package names, file paths)
"""

from typing import List, Dict, Any, Optional, Iterable
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.completion.filesystem import PathCompleter
from prompt_toolkit.document import Document
from pathlib import Path

from hatch import HatchEnvironmentManager


class CommandCompleter(Completer):
    """Main completer class that provides autocompletion for chat commands."""
    
    def __init__(self, command_metadata: Dict[str, Dict[str, Any]], env_manager: HatchEnvironmentManager):
        """Initialize the command completer.
        
        Args:
            command_metadata: Dictionary containing command metadata from ChatCommandHandler
            env_manager: Hatch environment manager for dynamic completion
        """
        self.command_metadata = command_metadata
        self.env_manager = env_manager
        self.path_completer = PathCompleter()
        
        # Cache for dynamic completions to improve performance
        self._environment_cache = None
        self._package_cache = {}  # env_name -> package_list
        
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Get completions for the current document position.
        
        Args:
            document: The current document
            complete_event: Completion event details
            
        Yields:
            Completion: Available completions
        """
        # Get the current line and cursor position
        text = document.text_before_cursor
        
        # If we're at the beginning or have no text, suggest command names
        if not text or text.isspace():
            yield from self._get_command_completions("")
            return
            
        # Split the input into parts
        parts = text.split()
        
        # If we're still typing the first word, complete command names
        if len(parts) == 1 and not text.endswith(' '):
            yield from self._get_command_completions(parts[0])
            return
            
        # If we have a command, complete its arguments
        if len(parts) >= 1:
            command = parts[0].lower()
            if command in self.command_metadata:
                yield from self._get_argument_completions(command, parts[1:], text)
                return
                
        # Fallback - no completions available
        return []
        
    def _get_command_completions(self, prefix: str) -> Iterable[Completion]:
        """Get completions for command names.
        
        Args:
            prefix: The current command prefix being typed
            
        Yields:
            Completion: Command completions
        """
        for cmd_name, cmd_info in self.command_metadata.items():
            if cmd_name.lower().startswith(prefix.lower()):
                # Calculate the start position for replacement
                start_position = -len(prefix) if prefix else 0
                
                yield Completion(
                    text=cmd_name,
                    start_position=start_position,
                    display=cmd_name,
                    display_meta=cmd_info.get('description', '')
                )
                
    def _get_argument_completions(self, command: str, args: List[str], full_text: str) -> Iterable[Completion]:
        """Get completions for command arguments.
        
        Args:
            command: The command name
            args: List of arguments already typed
            full_text: The full input text
            
        Yields:
            Completion: Argument completions
        """
        cmd_info = self.command_metadata[command]
        arg_defs = cmd_info.get('args', {})
        
        if not arg_defs:
            return []
            
        # Check if we're completing a flag argument (starts with -)
        current_word = args[-1] if args and not full_text.endswith(' ') else ""
        
        if current_word.startswith('-'):
            yield from self._get_flag_completions(arg_defs, current_word)
            return
            
        # Check if the previous argument was a flag that expects a value
        if len(args) >= 2 and args[-2].startswith('-'):
            flag_name = args[-2].lstrip('-')
            yield from self._get_flag_value_completions(arg_defs, flag_name, current_word)
            return
            
        # Complete positional arguments
        yield from self._get_positional_completions(arg_defs, args, current_word)
        
        # Also suggest available flags
        if not current_word.startswith('-'):
            yield from self._get_available_flags(arg_defs, args)
            
    def _get_flag_completions(self, arg_defs: Dict[str, Dict], prefix: str) -> Iterable[Completion]:
        """Get completions for flag arguments.
        
        Args:
            arg_defs: Argument definitions
            prefix: Current flag prefix being typed
            
        Yields:
            Completion: Flag completions
        """
        for arg_name, arg_def in arg_defs.items():
            if arg_def.get('positional', False):
                continue
                
            # Complete long form (--argument)
            long_form = f"--{arg_name}"
            if long_form.startswith(prefix):
                start_position = -len(prefix)
                yield Completion(
                    text=long_form,
                    start_position=start_position,
                    display=long_form,
                    display_meta=arg_def.get('description', '')
                )
                
            # Complete short form aliases (-a)
            aliases = arg_def.get('aliases', [])
            for alias in aliases:
                short_form = f"-{alias}"
                if short_form.startswith(prefix):
                    start_position = -len(prefix)
                    yield Completion(
                        text=short_form,
                        start_position=start_position,
                        display=short_form,
                        display_meta=f"{arg_def.get('description', '')} (alias for --{arg_name})"
                    )
                    
    def _get_flag_value_completions(self, arg_defs: Dict[str, Dict], flag_name: str, current_value: str) -> Iterable[Completion]:
        """Get completions for flag values.
        
        Args:
            arg_defs: Argument definitions
            flag_name: The flag name that expects a value
            current_value: Current value being typed
            
        Yields:
            Completion: Value completions
        """
        # Find the argument definition (could be by name or alias)
        arg_def = None
        for name, definition in arg_defs.items():
            if name == flag_name or flag_name in definition.get('aliases', []):
                arg_def = definition
                break
                
        if not arg_def:
            return []
            
        yield from self._get_value_completions(arg_def, current_value)
        
    def _get_positional_completions(self, arg_defs: Dict[str, Dict], args: List[str], current_word: str) -> Iterable[Completion]:
        """Get completions for positional arguments.
        
        Args:
            arg_defs: Argument definitions
            args: Arguments already provided
            current_word: Current word being typed
            
        Yields:
            Completion: Positional argument completions
        """
        # Find positional arguments in order
        positional_args = [
            (name, definition) for name, definition in arg_defs.items() 
            if definition.get('positional', False)
        ]
        
        # Determine which positional argument we're completing
        # Account for flags that might have consumed some arguments
        positional_index = len([arg for arg in args if not arg.startswith('-')])
        if current_word and not current_word.startswith('-'):
            positional_index -= 1
            
        if 0 <= positional_index < len(positional_args):
            _, arg_def = positional_args[positional_index]
            yield from self._get_value_completions(arg_def, current_word)
            
    def _get_available_flags(self, arg_defs: Dict[str, Dict], args: List[str]) -> Iterable[Completion]:
        """Get available flag suggestions.
        
        Args:
            arg_defs: Argument definitions
            args: Arguments already provided
            
        Yields:
            Completion: Available flag completions
        """
        # Get flags that have already been used
        used_flags = set()
        for arg in args:
            if arg.startswith('--'):
                used_flags.add(arg[2:])
            elif arg.startswith('-') and len(arg) == 2:
                # Find the flag name for this alias
                for name, definition in arg_defs.items():
                    if arg[1] in definition.get('aliases', []):
                        used_flags.add(name)
                        break
                        
        # Suggest unused flags
        for arg_name, arg_def in arg_defs.items():
            if arg_def.get('positional', False) or arg_name in used_flags:
                continue
                
            long_form = f"--{arg_name}"
            yield Completion(
                text=long_form,
                start_position=0,
                display=long_form,
                display_meta=arg_def.get('description', '')
            )
            
    def _get_value_completions(self, arg_def: Dict[str, Any], current_value: str) -> Iterable[Completion]:
        """Get completions for argument values based on completer type.
        
        Args:
            arg_def: Argument definition
            current_value: Current value being typed
            
        Yields:
            Completion: Value completions
        """
        completer_type = arg_def.get('completer_type', 'none')
        start_position = -len(current_value) if current_value else 0
        
        if completer_type == 'suggestions':
            # Static suggestions
            suggestions = arg_def.get('values', [])
            for suggestion in suggestions:
                if suggestion.lower().startswith(current_value.lower()):
                    yield Completion(
                        text=suggestion,
                        start_position=start_position,
                        display=suggestion
                    )
                    
        elif completer_type == 'environment':
            # Dynamic environment completions
            environments = self._get_environments()
            for env_name in environments:
                if env_name.lower().startswith(current_value.lower()):
                    yield Completion(
                        text=env_name,
                        start_position=start_position,
                        display=env_name,
                        display_meta="Hatch environment"
                    )
                    
        elif completer_type == 'package':
            # Dynamic package completions
            packages = self._get_packages()
            for pkg_name in packages:
                if pkg_name.lower().startswith(current_value.lower()):
                    yield Completion(
                        text=pkg_name,
                        start_position=start_position,
                        display=pkg_name,
                        display_meta="Installed package"
                    )
        elif completer_type == 'path':
            # File path completions using prompt_toolkit's PathCompleter
            document = Document(current_value, len(current_value))
            for completion in self.path_completer.get_completions(document, None):
                yield completion
                
        elif completer_type == 'local_package':
            # Path completion with Hatch package detection/styling
            document = Document(current_value, len(current_value))
            
            # Get basic path completions first
            for completion in self.path_completer.get_completions(document, None):
                # Get full path by combining current path with completion
                full_path = self._get_full_path(current_value, completion.text)
                
                # Check if it's a Hatch package
                is_hatch_package = self._is_hatch_package(full_path)
                
                # Apply appropriate styling based on package status
                style = "fg:ansigreen bold" if is_hatch_package else "fg:ansired"
                display_meta = "Hatch Package" if is_hatch_package else "Directory"
                
                # Yield modified completion with styling
                yield Completion(
                    text=completion.text,
                    start_position=completion.start_position,
                    display=completion.display,
                    display_meta=display_meta,
                    style=style
                )
                
        # For 'none' type, no completions are provided
        
    def _get_environments(self) -> List[str]:
        """Get list of available Hatch environments.
        
        Returns:
            List[str]: Environment names
        """
        if self._environment_cache is None:
            try:
                environments = self.env_manager.list_environments()
                self._environment_cache = [env.get('name', '') for env in environments if env.get('name')]
            except Exception:
                self._environment_cache = []
                
        return self._environment_cache
        
    def _get_packages(self, env_name: Optional[str] = None) -> List[str]:
        """Get list of installed packages in an environment.
        
        Args:
            env_name: Environment name (uses current if None)
            
        Returns:
            List[str]: Package names
        """
        cache_key = env_name or 'current'
        
        if cache_key not in self._package_cache:
            try:
                packages = self.env_manager.list_packages(env_name)
                self._package_cache[cache_key] = [pkg.get('name', '') for pkg in packages if pkg.get('name')]
            except Exception:
                self._package_cache[cache_key] = []
        return self._package_cache[cache_key]
        
    def _is_hatch_package(self, path_str: str) -> bool:
        """Check if a directory contains hatch_metadata.json.
        
        Args:
            path_str: Path to the directory to check
            
        Returns:
            bool: True if directory contains hatch_metadata.json
        """
        try:
            path = Path(path_str)
            if path.is_dir():
                metadata_file = path / "hatch_metadata.json"
                return metadata_file.exists()
        except (OSError, PermissionError):
            pass
        return False
        
    def _get_full_path(self, current_input: str, completion_text: str) -> str:
        """Combine current input with completion to get full path.
        
        Args:
            current_input: The text the user has typed so far
            completion_text: The completion suggestion
            
        Returns:
            str: Full path combining input and completion
        """
        try:
            # Handle cases where current_input is empty or just whitespace
            if not current_input.strip():
                return completion_text
                
            # Handle relative paths correctly
            if current_input.endswith('/') or current_input.endswith('\\'):
                return current_input + completion_text
            
            # Get the directory part of the current input
            current_path = Path(current_input)
            if current_path.is_absolute():
                # For absolute paths, combine with parent directory
                return str(current_path.parent / completion_text)
            else:
                # For relative paths, get parent directory
                parent = current_path.parent
                if str(parent) == '.':
                    return completion_text
                return str(parent / completion_text)
        except (OSError, ValueError):
            # Fallback to simple concatenation if path operations fail
            return completion_text
        
    def invalidate_cache(self):
        """Invalidate cached dynamic completions."""
        self._environment_cache = None
        self._package_cache.clear()


class CommandCompleterFactory:
    """Factory class for creating command completers."""
    
    @staticmethod
    def create_completer(command_handler) -> CommandCompleter:
        """Create a CommandCompleter instance from a ChatCommandHandler.
        
        Args:
            command_handler: ChatCommandHandler instance
            
        Returns:
            CommandCompleter: Configured completer instance
        """
        return CommandCompleter(
            command_metadata=command_handler.commands,
            env_manager=command_handler.base_commands.env_manager
        )
