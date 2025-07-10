"""Model management module for interfacing with Ollama models.

This module provides functionality for checking model availability, pulling models,
and verifying the Ollama service status.
"""

import json
from typing import List, Tuple
import aiohttp
import os
from hatchling.mcp_utils.ollama_adapter import OllamaMCPAdapter
from hatchling.core.logging.session_debug_log import SessionDebugLog
from hatchling.config.settings import ChatSettings

class ModelManager:
    """Manages model availability and downloading."""

    def __init__(self, settings: ChatSettings, debug_log: SessionDebugLog = None):
        """Initialize the model manager.

        Args:
            settings (ChatSettings): Settings for chat configuration.
            debug_log (SessionDebugLog, optional): Logger for debugging messages. Defaults to None.
        """
        self.settings = settings
        self.logger = debug_log

    async def check_availability(self, session: aiohttp.ClientSession, model_name: str) -> bool:
        """Check if a model is available in Ollama.

        Args:
            session (aiohttp.ClientSession): HTTP session to use for the request to Ollama/tags.
            model_name (str): Name of the model to check.

        Returns:
            bool: True if the model is available, False otherwise.

        Raises:
            Exception: If there is an error checking for model availability.
        """
        try:
            async with session.get(f"{self.settings.ollama_api_url}/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["name"] for model in data.get("models", [])]
                    self.logger.info(f"Available models: {available_models}")
                    return model_name in available_models
                else:
                    text = await response.text()
                    raise Exception(f"Failed to list models: {response.status}, {text}")
        except Exception as e:
            raise Exception(f"Error checking available models: {e}")

    async def pull_model(self, session: aiohttp.ClientSession, model_name: str) -> None:
        """Pull a model from Ollama.

        Args:
            session (aiohttp.ClientSession): HTTP session to use for the request to Ollama/pull.
            model_name (str): Name of the model to pull.

        Raises:
            Exception: If there is an error pulling the model.
        """
        self.logger.info(f"Pulling model: {model_name}")

        try:
            async with session.post(
                f"{self.settings.ollama_api_url}/pull",
                json={"name": model_name},
                timeout=None  # No timeout for model downloading
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    error_msg = f"Failed to pull model: {response.status}, {text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                self.logger.info(f"Started downloading {model_name}. This may take a while...")
                async for line in response.content.iter_any():
                    if line:
                        try:
                            line_text = line.decode('utf-8').strip()
                            if not line_text:
                                continue

                            # Split by newlines to handle multiple JSON objects in one line
                            json_objects = line_text.split('\n')
                            for json_obj in json_objects:
                                if not json_obj.strip():
                                    continue

                                try:
                                    data = json.loads(json_obj)
                                    if "status" in data:
                                        print(f"Status: {data['status']}", flush=True)
                                    if "completed" in data and "total" in data:
                                        percentage = (data["completed"] / data["total"]) * 100
                                        print(f"Progress: {percentage:.2f}%", flush=True)
                                except json.JSONDecodeError:
                                    self.logger.error(f"Received invalid JSON during model pull: {json_obj}")
                                except Exception as e:
                                    self.logger.error(f"Error processing model pull response: {e}\nData: {json_obj}")
                        except Exception as e:
                            self.logger.error(f"Error processing line: {e}")

                self.logger.info(f"Download of {model_name} completed.")
        except Exception as e:
            error_msg = f"Error pulling model: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def check_ollama_service(self) -> Tuple[bool, str]:
        """Asynchronously check if Ollama service is available.

        Effectively tries to get the version of the service with Ollama/version
        with a timeout of 5 seconds.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the service is available
                - str: A descriptive message
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.settings.ollama_api_url}/version", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return True, f"Ollama service is running, version: {data.get('version')}"
                    else:
                        return False, f"Ollama service returned status code: {response.status}"
        except Exception as e:
            return False, f"Ollama service is not available: {e}"

    async def check_openai_service(self) -> Tuple[bool, str]:
        """Check if OpenAI API key is set and the model is available (via a test API call)."""
        import aiohttp
        if not self.settings.openai_api_key:
            return False, "OpenAI API key is missing. Please set CHATGPT_API_KEY in your environment."
        try:
            headers = {
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.settings.openai_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.settings.openai_api_url}/chat/completions", json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        return True, f"OpenAI service is available and model '{self.settings.openai_model}' is accessible."
                    else:
                        text = await response.text()
                        return False, f"OpenAI service returned status {response.status}: {text}"
        except Exception as e:
            return False, f"OpenAI service is not available or model is invalid: {e}"