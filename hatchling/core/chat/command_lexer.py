"""Command lexer for real-time syntax highlighting of chat commands.

This module provides a custom lexer that highlights commands, arguments, and values
as the user types them in the chat interface.
"""

import re
from typing import Iterator, List, Tuple, Dict, Any

from prompt_toolkit.lexers import Lexer
from prompt_toolkit.document import Document


class ChatCommandLexer(Lexer):
    """Custom lexer for highlighting chat commands in real-time."""
    
    def __init__(self, command_metadata: Dict[str, Dict[str, Any]]):
        """Initialize the lexer with command metadata.
        
        Args:
            command_metadata: Dictionary containing command information including
                            command names, arguments, and their types.
        """
        self.command_metadata = command_metadata
        
        # Build command patterns
        self.command_names = set(command_metadata.keys())
        
        # Build argument patterns for each command
        self.command_args = {}
        for cmd_name, cmd_info in command_metadata.items():
            if 'args' in cmd_info:
                self.command_args[cmd_name] = cmd_info['args']
    
    def lex_document(self, document: Document) -> callable:
        """Lex the document and return a function that yields style/text tuples.
        
        Args:
            document: The document to lex.
            
        Returns:
            A function that takes a line number and yields (style, text) tuples.
        """
        def get_tokens(line_number: int):
            # Get the line content
            try:
                lines = document.text.split('\n')
                if line_number >= len(lines):
                    return []
                
                line_text = lines[line_number]
                if not line_text.strip():
                    return [('', line_text)]
                    
                # Tokenize the input
                tokens = self._tokenize(line_text)
                
                result = []
                for token_type, token_text in tokens:
                    result.append((self._get_style_for_token(token_type), token_text))
                
                return result
            except Exception:
                # Fall back to plain text if tokenization fails
                return [('', line_text if 'line_text' in locals() else '')]
        
        return get_tokens
    
    def _tokenize(self, text: str) -> List[Tuple[str, str]]:
        """Tokenize the input text into command components.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of (token_type, token_text) tuples.
        """
        tokens = []
        
        # Simple tokenization - split by spaces but respect quotes
        parts = self._split_respecting_quotes(text)
        
        if not parts:
            return [('text', text)]
        
        # First part should be the command
        command = parts[0]
        
        # Check if it's a valid command
        if command in self.command_names:
            # Get command group for styling
            cmd_info = self.command_metadata[command]
            group = 'hatch' if command.startswith('hatch:') else 'base'

            tokens.append((f'command.{group}', command))

            # Process arguments
            if len(parts) > 1:
                tokens.extend(self._tokenize_arguments(command, parts[1:], group))
        else:
            # Not a command, treat as regular text
            tokens.append(('text', text))
        
        return tokens
    
    def _split_respecting_quotes(self, text: str) -> List[str]:
        """Split text by spaces while respecting quoted strings.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text parts.
        """
        parts = []
        current_part = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_part += char
            elif char.isspace() and not in_quotes:
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                # Add whitespace as separate token to preserve formatting
                while i < len(text) and text[i].isspace():
                    current_part += text[i]
                    i += 1
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                i -= 1  # Adjust for the extra increment
            else:
                current_part += char
            
            i += 1
                
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def _tokenize_arguments(self, command: str, arg_parts: List[str], group: str) -> List[Tuple[str, str]]:
        """Tokenize command arguments.
        
        Args:
            command: The command name.
            arg_parts: List of argument parts.
            group: Command group (base/hatch).
            
        Returns:
            List of (token_type, token_text) tuples.
        """
        tokens = []
        command_args = self.command_args.get(command, {})
        
        for i, part in enumerate(arg_parts):
            if part.isspace():
                tokens.append(('whitespace', part))
                continue
                
            # Check if it's a flag argument (starts with - or --)
            if part.startswith('-'):
                arg_name = part.lstrip('-')
                
                # Check if it's a valid argument for this command
                if arg_name in command_args:
                    tokens.append((f'argument.{group}', part))
                else:
                    # Check aliases
                    found_alias = False
                    for name, arg_def in command_args.items():
                        if arg_name in arg_def.get('aliases', []):
                            tokens.append((f'argument.{group}', part))
                            found_alias = True
                            break
                    
                    if not found_alias:
                        tokens.append(('argument.invalid', part))
            else:
                # Could be a positional argument or a value
                # For simplicity, we'll style positional args as values
                # A more sophisticated approach would track argument expectations
                if self._looks_like_path(part):
                    tokens.append(('value.path', part))
                elif self._looks_like_number(part):
                    tokens.append(('value.number', part))
                elif part.startswith('"') or part.startswith("'"):
                    tokens.append(('value.string', part))
                else:
                    tokens.append(('value.generic', part))
        
        return tokens
    
    def _looks_like_path(self, text: str) -> bool:
        """Check if text looks like a file path."""
        return ('/' in text or '\\' in text or 
                text.startswith('./') or text.startswith('../') or
                text.endswith('.py') or text.endswith('.json') or
                text.endswith('.txt'))
    
    def _looks_like_number(self, text: str) -> bool:
        """Check if text looks like a number."""
        try:
            float(text)
            return True
        except ValueError:
            return False
    def _get_style_for_token(self, token_type: str) -> str:
        """Get the CSS style class for a token type.
        
        Args:
            token_type: The token type.
            
        Returns:
            CSS style class name.
        """
        # Map token types to CSS classes
        style_map = {
            'command.base': 'class:command.name.base',
            'command.hatch': 'class:command.name.hatch',
            'command': 'class:command.name',  # Fallback for generic commands
            'argument.base': 'class:command.args.base',
            'argument.hatch': 'class:command.args.hatch',
            'argument.invalid': 'class:command.args.invalid',
            'value.path': 'class:command.value.path',
            'value.number': 'class:command.value.number',
            'value.string': 'class:command.value.string',
            'value.generic': 'class:command.value.generic',
            'whitespace': '',
            'text': 'class:text.default',
        }
        
        return style_map.get(token_type, '')
