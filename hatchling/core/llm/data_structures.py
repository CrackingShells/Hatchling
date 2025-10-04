"""Core data structures for LLM functionality.

This module contains common data structures used across the LLM system
to avoid circular import dependencies.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

# TODO: Standardize of dataclasses usage. Where to put them?
# What to name them? Do we keep them as dataclasses or up to pydantic?
# This "todo" stands for all the dataclasses in this project. This
# must be tackled globally
@dataclass
class ToolCallParsedResult:
    """Normalized representation of a tool call event."""
    tool_call_id: str
    function_name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parse result to a dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments
        }
    

@dataclass
class ToolCallExecutionResult:
    """Data class to hold the result of a tool call execution."""
    tool_call_id: str
    function_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error
        }
