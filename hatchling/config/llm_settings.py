"""Settings for LLM (Large Language Model) configuration."""

import os
from typing import Optional
from pydantic import BaseModel, Field

from .settings_access_level import SettingAccessLevel

class LLMSettings(BaseModel):
    """Settings for LLM (Large Language Model) configuration."""

    # Provider selection
    provider: str = Field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai"),
        description="LLM provider to use ('ollama' or 'openai').",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    # Ollama settings
    ollama_api_url: str = Field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST_API", "http://localhost:11434/api"),
        description="URL for the Ollama API endpoint.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )
    ollama_model: str = Field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2"),
        description="Ollama model to use for chat interactions.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    # OpenAI settings
    openai_api_url: str = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
        description="URL for the OpenAI API endpoint.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )
    openai_model: str = Field(
        default_factory=lambda: os.environ.get("CHATGPT_MODEL", "gpt-4o-mini"),
        description="OpenAI model to use for chat interactions.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.environ.get("CHATGPT_API_KEY", ""),
        description="API key for OpenAI services.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )

    def get_active_provider(self) -> str:
        """Return the currently active LLM provider."""
        return self.provider.lower()

    def get_active_model(self) -> str:
        """Return the currently active model name based on provider."""
        provider = self.get_active_provider()
        if provider == "ollama":
            return self.ollama_model
        elif provider == "openai":
            return self.openai_model
        else:
            return self.ollama_model  # fallback

    def get_active_api_url(self) -> str:
        """Return the currently active API URL based on provider."""
        provider = self.get_active_provider()
        if provider == "ollama":
            return self.ollama_api_url
        elif provider == "openai":
            return self.openai_api_url
        else:
            return self.ollama_api_url  # fallback

    def get_active_api_key(self) -> Optional[str]:
        """Return the currently active API key based on provider."""
        provider = self.get_active_provider()
        if provider == "openai":
            return self.openai_api_key
        else:
            return None  # Ollama doesn't need API key

    class Config:
        extra = "forbid"