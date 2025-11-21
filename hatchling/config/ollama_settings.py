"""Settings for LLM (Large Language Model) configuration.

Contains configuration options for connecting to and controlling Ollama LLM generation.

Configuration Precedence:
    Persistent Settings (user.toml) > Environment Variables > Code Defaults

Environment variables provide deployment flexibility (Docker, CI/CD) while
persistent settings allow runtime configuration changes without restart.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field

from .settings_access_level import SettingAccessLevel

class OllamaSettings(BaseModel):
    """Settings for LLM (Large Language Model) configuration.

    Includes connection and generation parameters for Ollama.
    """

    # Ollama Connection settings
    ip: str = Field(
        default_factory=lambda: os.environ.get("OLLAMA_IP", "localhost"),
        description="IP address for the Ollama API endpoint.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )

    port: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_PORT", 11434)),
        description="Port for the Ollama API endpoint.",
        json_schema_extra={"access_level": SettingAccessLevel.PROTECTED},
    )

    # Ollama Model settings

    num_ctx: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_NUM_CTX", 4096)),
        description="Sets the size of the context window used to generate the next token. (Default: 4096)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    repeat_last_n: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_REPEAT_LAST_N", 64)),
        description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    repeat_penalty: float = Field(
        default_factory=lambda: float(os.environ.get("OLLAMA_REPEAT_PENALTY", 1.1)),
        description="Sets how strongly to penalize repetitions. Higher values penalize more strongly. (Default: 1.1)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    temperature: float = Field(
        default_factory=lambda: float(os.environ.get("OLLAMA_TEMPERATURE", 0.8)),
        description="The temperature of the model. Higher values make answers more creative. (Default: 0.8)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    timeout: float = Field(
        default_factory=lambda: float(os.environ.get("OLLAMA_TIMEOUT", 30.0)),
        description="Timeout in seconds for Ollama API requests. (Default: 30.0)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    seed: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_SEED", 0)),
        description="Sets the random number seed to use for generation. (Default: 0)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Sets the stop sequences to use. When encountered, generation stops. Multiple patterns allowed.",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    num_predict: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_NUM_PREDICT", -1)),
        description="Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    top_k: int = Field(
        default_factory=lambda: int(os.environ.get("OLLAMA_TOP_K", 40)),
        description="Reduces the probability of generating nonsense. Higher values give more diverse answers. (Default: 40)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    top_p: float = Field(
        default_factory=lambda: float(os.environ.get("OLLAMA_TOP_P", 0.9)),
        description="Works with top-k. Higher values lead to more diverse text. (Default: 0.9)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )
    min_p: float = Field(
        default_factory=lambda: float(os.environ.get("OLLAMA_MIN_P", 0.0)),
        description="Alternative to top_p, ensures a balance of quality and variety. (Default: 0.0)",
        json_schema_extra={"access_level": SettingAccessLevel.NORMAL},
    )

    @property
    def api_base(self) -> str:
        return f"http://{self.ip}:{self.port}"

    class Config:
        extra = "forbid"