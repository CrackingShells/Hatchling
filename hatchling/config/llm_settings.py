"""Settings for LLM (Large Language Model) configuration."""

import os
import re
from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum

from .settings_access_level import SettingAccessLevel
from hatchling.core.logging.logging_manager import logging_manager

class ELLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class ModelStatus(Enum):
    """Status of a model."""
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    name: str
    provider: ELLMProvider
    status: ModelStatus
    size: Optional[int] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a dictionary representation of the model."""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "status": self.status.value,
            "size": self.size,
            "modified_at": self.modified_at,
            "digest": self.digest,
            "details": self.details,
            "error_message": self.error_message,
        }

logger = logging_manager.get_session("LLMSettings")

class LLMSettings(BaseModel):
    """Settings for LLM (Large Language Model) configuration."""

    # Provider selection
    provider_enum: ELLMProvider = Field(
        default_factory=lambda: LLMSettings.to_provider_enum(os.environ.get("LLM_PROVIDER", "ollama")),
        description="LLM provider to use ('ollama' or 'openai').",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    model: Optional[str] = Field(
        default=None,
        description="Default LLM to use for the selected provider. Must be explicitly selected from discovered models.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    @staticmethod
    def extract_provider_model_list(s: str) -> List[Tuple[ELLMProvider, str]]:
        """
        Extract a list of (provider_name, model_name) tuples from a string using regex.

        Args:
            s (str): Input string containing tuples in the format (provider,model).

        Returns:
            List[tuple[str, str]]: List of extracted tuples.
        """
        pattern = r"\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_.-]+)\s*\)"
        res = [(
            LLMSettings.to_provider_enum(match[0]),
            match[1]
        ) for match in re.findall(pattern, s)]

        return res

    models: List[ModelInfo] = Field(
        default_factory=lambda: [
            ModelInfo(name=model[1], provider=model[0], status=ModelStatus.AVAILABLE)
            for model in LLMSettings.extract_provider_model_list(
                os.environ.get("LLM_MODELS", "")
            )
        ] if os.environ.get("LLM_MODELS") else [],
        description="Curated list of LLM models. Use 'llm:model:discover' to populate from available models. Environment variable LLM_MODELS can provide initial models for deployment.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    @property
    def provider_name(self) -> str:
        """Return the current LLM provider."""
        return self.provider_enum.value
    
    @property
    def provider_names(self) -> List[str]:
        """Return a list of all available LLM providers."""
        return [provider.value for provider in ELLMProvider]
    
    @property
    def provider_enums(self) -> List[ELLMProvider]:
        """Return a list of all available LLM provider enums."""
        return List(ELLMProvider)
    
    @staticmethod
    def to_provider_enum(provider_name: str) -> ELLMProvider:
        """Convert the provider name to its corresponding enum."""
        return ELLMProvider(provider_name)

    class Config:
        extra = "forbid"