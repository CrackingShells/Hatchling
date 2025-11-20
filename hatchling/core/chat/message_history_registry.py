"""Message history registry for global access to chat message histories.

This module provides a centralized registry for mapping UIDs to MessageHistory
instances, enabling global access and management of multiple chat histories
across sessions, users, or contexts.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings import AppSettings
from .message_history import MessageHistory

logger = logging_manager.get_session("MessageHistoryRegistry")


class MessageHistoryRegistry:
    """Registry for message histories by UID.
    
    This class provides a centralized mapping between UIDs and their
    corresponding MessageHistory instances. It enables global access to
    histories while history management.
    
    The registry follows these principles:
    - Single source of truth for UID-to-history mappings
    - Dynamic registration and lookup with async safety
    - Clear error handling for missing UIDs
    - Extensibility for future history management features
    """

    # Class-level storage for singleton behavior
    _histories: Dict[str, MessageHistory] = {}

    @classmethod
    def register_history(cls, uid: str, history: MessageHistory) -> None:
        """Register a MessageHistory instance for a UID.
        
        Args:
            uid (str): The unique identifier for this history (e.g., session_id, user_id).
            history (MessageHistory): The MessageHistory instance to register.
        
        Raises:
            TypeError: If the history is not a MessageHistory instance.
        """
        if not isinstance(history, MessageHistory):
            raise TypeError(f"history must be a MessageHistory instance, got {type(history)}")
        if uid in cls._histories:
            logger.warning(f"Overriding existing history for UID '{uid}'")
        cls._histories[uid] = history
        logger.debug(f"Registered history for UID '{uid}'")

    @classmethod
    def get_history(cls, uid: str) -> Optional[MessageHistory]:
        """Get a MessageHistory instance for the given UID.
        
        Args:
            uid (str): The UID to get a history for.
            
        Returns:
            MessageHistory: The MessageHistory instance, or None if not found.
        """
        return cls._histories.get(uid)


    @classmethod
    def get_or_create_history(cls, uid: str, settings: AppSettings = None) -> MessageHistory:
        """Get a MessageHistory instance for the given UID, or create one if it doesn't exist.
        
        Args:
            uid (str): The UID to get or create a history for.
            settings (AppSettings, optional): Application settings to pass to MessageHistory constructor.
        
        Returns:
            MessageHistory: The existing or newly created MessageHistory instance.
        """
        history = cls._histories.get(uid)
        if history is not None:
            return history
        history = MessageHistory(settings=settings)
        cls._histories[uid] = history
        logger.debug(f"Created and registered new history for UID '{uid}'")
        return history    

    @classmethod
    def unregister_history(cls, uid: str) -> Optional[MessageHistory]:
        """Unregister a MessageHistory for the given UID.
        
        This method is primarily intended for cleanup and testing purposes.
        
        Args:
            uid (str): The UID to unregister.
        
        Returns:
            MessageHistory: The unregistered MessageHistory instance, or None if not found.
        """
        history = cls._histories.pop(uid, None)
        if history:
            logger.debug(f"Unregistered history for UID '{uid}'")
        return history

    @classmethod
    def is_registered(cls, uid: str) -> bool:
        """Check if a MessageHistory is registered for the given UID.
        
        Args:
            uid (str): The UID to check.
            
        Returns:
            bool: True if a history is registered for the UID, False otherwise.
        """
        return uid in cls._histories

    @classmethod
    def get_registered_uids(cls) -> List[str]:
        """Get a list of all registered UIDs.
        
        Returns:
            List[str]: List of registered UID identifiers.
        """
        return list(cls._histories.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered histories.
        
        This method is primarily intended for testing purposes.
        It removes all entries from the registry.
        """
        cls._histories.clear()
        logger.debug("Cleared all registered histories")

    @classmethod
    def __len__(cls) -> int:
        """Get the number of registered histories.
        
        Returns:
            int: Number of registered histories.
        """
        return len(cls._histories)

    @classmethod
    def __contains__(cls, uid: str) -> bool:
        """Check if a UID is registered.
        
        Args:
            uid (str): The UID to check.
            
        Returns:
            bool: True if the UID is registered, False otherwise.
        """
        return uid in cls._histories

    @classmethod
    def __repr__(cls) -> str:
        """Get a string representation of the registry.
        
        Returns:
            str: String representation showing registered UIDs.
        """
        uids = list(cls._histories.keys())
        return f"MessageHistoryRegistry(uids={uids})"
