import os
import logging
from pathlib import Path
from hatchling.core.logging.logging_manager import logging_manager

class ChatSettings:
    """Manages chat configuration settings."""
    
    def __init__(self,
                 ollama_api_url: str = os.environ.get("OLLAMA_HOST_API", "http://localhost:11434/api"),
                 ollama_model: str = os.environ.get("OLLAMA_MODEL", "mistral-small3.1"),
                 openai_api_url: str = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1"),
                 openai_model: str = os.environ.get("CHATGPT_MODEL", "gpt-4.1"),
                 openai_api_key: str = os.environ.get("CHATGPT_API_KEY", ""),
                 llm_provider: str = os.environ.get("LLM_PROVIDER", "openai"),
                 hatch_envs_dir: str = os.environ.get("HATCH_ENVS_DIR", Path.home() / ".hatch" / "envs"),
                 max_tool_call_iteration: int = 5,
                 max_working_time: float = 3000.0):
        """Initialize chat settings with configurable parameters.
        
        Args:
            ollama_api_url (str, optional): URL for the Ollama API.
            ollama_model (str, optional): Ollama LLM to use.
            openai_api_url (str, optional): URL for the OpenAI API.
            openai_model (str, optional): OpenAI model to use.
            openai_api_key (str, optional): API key for OpenAI.
            llm_provider (str, optional): Which provider to use ("ollama" or "openai").
            hatch_envs_dir (str, optional): Directory for Hatch environments.
            max_tool_call_iteration (int, optional): Maximum number of tool call iterations.
            max_working_time (float, optional): Maximum time in seconds for tool operations.
        """
        self.logger = logging_manager.get_session("ChatSettings",
                                                  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


        self.ollama_api_url = ollama_api_url
        self.ollama_model = ollama_model
        self.openai_api_url = openai_api_url
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self.logger.info(self.openai_api_key)
        self.llm_provider = llm_provider.lower()

        # If hatch_envs_dir is:
        # - an absolute path, it is used as is
        # - a relative path, it is resolved against the user's home directory
        if os.path.isabs(hatch_envs_dir):
            self.hatch_envs_dir = hatch_envs_dir
        else:
            self.hatch_envs_dir = Path.home() / hatch_envs_dir
        
        # New settings for tool calling control
        self.max_tool_call_iteration = max_tool_call_iteration  # Maximum number of tool call iterations
        self.max_working_time = max_working_time  # Maximum time in seconds for tool operations

        self.logger.info(
            f"ChatSettings initialized with provider: {self.llm_provider}, "
            f"Ollama model: {self.ollama_model}, OpenAI model: {self.openai_model}"
        )
        self.logger.info(
            f"Max tool call iterations: {self.max_tool_call_iteration}, Max working time: {self.max_working_time} seconds"
        )
        self.logger.info(f"Hatch environments directory: {self.hatch_envs_dir}")

    def get_active_model(self):
        """Return the currently active model name based on provider."""
        if self.llm_provider == "ollama":
            return self.ollama_model
        elif self.llm_provider == "openai":
            return self.openai_model
        else:
            return None

    def get_active_provider(self):
        """Return the currently active LLM provider."""
        return self.llm_provider