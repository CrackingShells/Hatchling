"""Logging management for the Hatchling application.

This module provides centralized logging configuration, management of logging sessions,
and consistent logging across the application.
"""

import logging
import io
import os
from pathlib import Path
from typing import Dict, Optional, List, Union
from logging.handlers import RotatingFileHandler
from hatchling.core.logging.session_debug_log import SessionDebugLog


class LoggingManager:
    """Singleton manager for handling all logging sessions in the application."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure singleton pattern implementation.
        
        Returns:
            LoggingManager: The singleton instance of LoggingManager.
        """
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logging manager if not already initialized."""
        if self._initialized:
            return
            
        # Initialize only once
        self._initialized = True
        
        # Default formatter for root logger (used for sessions)
        self.default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Store all session loggers by name
        self.sessions: Dict[str, SessionDebugLog] = {}
        
        # Default log values (will be overridden by configure_logging)
        self.log_level = logging.INFO
        self.log_file = Path.home() / '.hatch' / 'logs' / 'hatchling.log'
    
    def set_log_level(self, level: int) -> None:
        """Set the log level for all loggers and handlers.
        
        Args:
            level (int): The log level (e.g., logging.DEBUG, logging.INFO).
        """
        self.log_level = level
    
        # CRITICAL: Set root logger first
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Set level for all handlers on root logger
        for handler in root_logger.handlers:
            handler.setLevel(self.log_level)

    def get_session(self, name: str, 
                   formatter: Optional[logging.Formatter] = None) -> SessionDebugLog:
        """Get a session debug log by name, creating it if it doesn't exist.
        
        Args:
            name (str): The name of the session debug log.
            formatter (logging.Formatter, optional): Custom formatter for this session. Defaults to None.
            
        Returns:
            SessionDebugLog: The session debug log instance.
        """
        if name not in self.sessions:
            # Create a new session if it doesn't exist
            if formatter is None:
                self.sessions[name] = SessionDebugLog(name, self.default_formatter)
            else:
                self.sessions[name] = SessionDebugLog(name, formatter)
                
        return self.sessions[name]
    
    def get_all_sessions(self) -> List[str]:
        """Get a list of all session names.
        
        Returns:
            List[str]: List of session names.
        """
        return list(self.sessions.keys())
    
    def clear_session(self, name: str) -> bool:
        """Clear a specific session log.
        
        Args:
            name (str): The name of the session to clear.
            
        Returns:
            bool: True if the session was cleared, False if it doesn't exist.
        """
        if name in self.sessions:
            self.sessions[name].clear_logs()
            return True
        return False
    
    def clear_all_sessions(self) -> None:
        """Clear all session logs."""
        for session in self.sessions.values():
            session.clear_logs()
    
    def create_console_handler(self, formatter: Optional[logging.Formatter] = None) -> logging.StreamHandler:
        """Create a console handler with the current CLI log level.
        
        Args:
            formatter (logging.Formatter, optional): Formatter to use, falls back to default formatter. Defaults to None.
            
        Returns:
            logging.StreamHandler: A configured console handler.
        """
        console = logging.StreamHandler()
        console.setLevel(self.log_level)
        console.setFormatter(formatter or self.default_formatter)
        return console

# Create a global instance that can be imported elsewhere
logging_manager = LoggingManager()