"""Settings for configuring OpenAI LLMs.

Contains configuration options for connecting to and controlling OpenAI LLM generation.
"""

import os
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

from .settings_access_level import SettingAccessLevel

class OpenAIToolChoice(Enum):
    """Enum for OpenAI tool choice options."""
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class OpenAISettings(BaseModel):
    """Settings for configuring OpenAI LLMs.

    Includes connection and generation parameters for OpenAI.
    """

    api_key: Optional[str] = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
        description="The API key used to authenticate with OpenAI services.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )

    api_base: str = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
        description="The base URL for OpenAI API requests.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )

    timeout: int = Field(
        default_factory=lambda: int(os.environ.get("OPENAI_TIMEOUT", 60)),
        description="Timeout in seconds for OpenAI API requests.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    max_completion_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("OPENAI_MAX_COMPLETION_TOKENS", 2048)),
        description="The maximum number of tokens for OpenAI completions. This includes visible and reasoning tokens (when enabled). The higher the value, the more tokens can be generated, but it may increase costs.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    temperature: float = Field(
        default_factory=lambda: float(os.environ.get("OPENAI_TEMPERATURE", 0.7)),
        description="Sampling temperature for OpenAI completions.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    top_p: float = Field(
        default_factory=lambda: float(os.environ.get("OPENAI_TOP_P", 1.0)),
        description="Nucleus sampling parameter for OpenAI completions.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    tool_choice: Optional[OpenAIToolChoice] = Field(
        default_factory=lambda: os.environ.get("OPENAI_TOOL_CHOICE", OpenAIToolChoice.AUTO),
        description="The tool choice for OpenAI API requests.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    class Config:
        extra = "forbid"
