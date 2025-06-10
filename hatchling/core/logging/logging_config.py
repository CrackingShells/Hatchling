"""Global logging configuration for the Hatchling application.

This module provides centralized logging setup for the entire application,
maintaining clean separation between UI and backend components while applying
consistent formatting and styling.
"""

import sys
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from hatchling.core.logging.logging_manager import logging_manager

# Import UI styling only when needed, keeping the dependency optional
_has_prompt_toolkit = False
try:
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.patch_stdout import patch_stdout
    _has_prompt_toolkit = True
except ImportError:
    _has_prompt_toolkit = False


class StyledHandler(logging.StreamHandler):
    """A logging handler that can output styled logs if prompt_toolkit is available."""
    
    # Default style mapping for different log levels
    DEFAULT_STYLES = {
        logging.DEBUG: 'fg:cyan',
        logging.INFO: 'fg:white',
        logging.WARNING: 'fg:yellow',
        logging.ERROR: 'fg:red',
        logging.CRITICAL: 'bg:red fg:white bold',
    }
    
    def __init__(self, formatter: Optional[logging.Formatter] = None, 
                styles: Optional[Dict[int, str]] = None,
                force_styling: bool = False):
        """Initialize the styled logging handler.
        
        Args:
            formatter: Formatter to use for log messages
            styles: Mapping of log levels to styles
            force_styling: If True, attempt styled output even in non-interactive environments
        """
        super().__init__()
        self.styles = styles or self.DEFAULT_STYLES
        self.supports_styling = _has_prompt_toolkit and (force_styling or sys.stdout.isatty())
        
        if formatter:
            self.setFormatter(formatter)
        else:
            self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Format and emit a log record with styling when available.
        
        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            
            if self.supports_styling:
                style = self.styles.get(record.levelno, 'fg:white')
                
                # Format the message with prompt_toolkit styling
                formatted_text = FormattedText([(style, msg)])
                
                # Use patch_stdout to avoid interfering with input prompts
                with patch_stdout():
                    print_formatted_text(formatted_text)
            else:
                # Fall back to normal output when styling isn't available
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
                
        except Exception:
            self.handleError(record)


def configure_logging(enable_styling: bool = True, 
                     log_file: Optional[Path] = None,
                     log_level: Optional[int] = None) -> None:
    """Configure global logging for the application.
    
    Should be called at application startup to set up logging for all components.
    
    Args:
        enable_styling: Whether to enable styled console output (if available)
        log_file: Optional path to the log file. If None, will use environment variable or default
        log_level: Default logging level. If None, will use environment variable or default to INFO
    """
    # Determine log level from environment variable if not specified
    if log_level is None:
        log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = log_levels.get(log_level_str, logging.INFO)
    
    # Determine log file from environment variable if not specified
    if log_file is None:
        log_dir = Path(os.environ.get('LOG_DIR', Path.home() / '.hatch' / 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / 'hatchling.log'
    
    # Set the log level in the logging manager
    logging_manager.set_log_level(log_level)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create default formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up console output with optional styling
    console_handler = StyledHandler(
        formatter=formatter,
        force_styling=enable_styling  # Use the provided enable_styling parameter
    )
    root_logger.addHandler(console_handler)
    
    # Add file logging
    try:
        from logging.handlers import RotatingFileHandler
        
        # Ensure the log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to set up file logging: {str(e)}")
    
    # Set the log level on the root logger
    root_logger.setLevel(log_level)
    
    logging.debug("Logging configured successfully")
