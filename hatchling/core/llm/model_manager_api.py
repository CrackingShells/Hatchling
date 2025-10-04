"""Model Management Utility API.

This module provides a clean, static utility API for model manage
different LLM providers. It offers model discovery, health checking, and
metadata operations without requiring instance management.
"""

from typing import List, Tuple, Optional
from tqdm import tqdm

from ollama import AsyncClient, ListResponse
from openai import AsyncOpenAI

from hatchling.core.llm.providers.registry import ProviderRegistry
from hatchling.config.llm_settings import ELLMProvider
from hatchling.config.settings import AppSettings
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.llm_settings import ModelInfo, ModelStatus

logger = logging_manager.get_session("ModelManagerAPI")

class ModelManagerAPI:
    """Static utility API for model management across LLM providers.
    
    This class provides unified model management operations:
    - Model discovery and listing
    - Health checking and availability  
    - Model pulling (where supported)
    - Provider service validation
    """
    
    @staticmethod
    async def check_provider_health(provider: ELLMProvider, settings: AppSettings = None) -> Tuple[bool, str]:
        """Check if an LLM provider service is healthy and accessible.
        
        Args:
            provider (ELLMProvider): The provider to check.
            settings (AppSettings, optional): Application settings containing API keys and URLs.
                                            If None, uses the singleton instance.
            
        Returns:
            Tuple[bool, str]: Success flag and descriptive message.
        """
        settings = settings or AppSettings.get_instance()
        is_healthy = True

        try:
            if provider == ELLMProvider.OLLAMA:
                ollama_models = await ModelManagerAPI._list_ollama_models(settings)
                is_healthy &= ollama_models is not None and len(ollama_models) > 0
            elif provider == ELLMProvider.OPENAI:
                openai_models = await ModelManagerAPI._list_openai_models(settings)
                is_healthy &= openai_models is not None and len(openai_models) > 0

        except Exception as e:
            return False

        return is_healthy

    @staticmethod
    def list_providers() -> List[ELLMProvider]:
        """List all available LLM providers.
        
        Returns:
            List[ELLMProvider]: List of supported LLM providers.
        """
        return ProviderRegistry.list_providers()

    @staticmethod
    async def list_available_models(provider: Optional[ELLMProvider] = None,
                                    settings: Optional[AppSettings] = None) -> List[ModelInfo]:
        """List all available models, optionally filtered by provider.
        
        Args:
            provider (ELLMProvider, optional): Filter by provider. If None, returns all models.
            settings (AppSettings, optional): Application settings.
                                            If None, uses the singleton instance.
        
        Returns:
            List[ModelInfo]: List of model information.
        """
        try:
            settings = settings or AppSettings.get_instance()
            all_models : List[ModelInfo] = []

            if provider is None or provider == ELLMProvider.OLLAMA:
                all_models += await ModelManagerAPI._list_ollama_models(settings)

            if provider is None or provider == ELLMProvider.OPENAI:
                all_models += await ModelManagerAPI._list_openai_models(settings)

            logger.debug(f"Available models: {all_models}")
        
            return all_models
        
        except Exception as e:
            raise e

    @staticmethod
    async def is_model_available(model_name: str, provider: ELLMProvider, settings: Optional[AppSettings] = None) -> ModelInfo:
        """Check if a specific model is available for the given provider.
        
        Args:
            model_name (str): Name of the model to check.
            provider (ELLMProvider): The provider to check against.
            settings (AppSettings, optional): Application settings.
                                            If None, uses the singleton instance.
        
        Returns:
            ModelInfo: Information about the whole model. Check model.status for availability.
            If model is not found, returns a ModelInfo with status ModelStatus.NOT_AVAILABLE
            and error_message.
        """
        try:
            models = await ModelManagerAPI.list_available_models(provider, settings)

            for model in models:
                if model.name.lower() == model_name.lower():
                    logger.info(f"Model '{model_name}' found for provider {provider.value}")
                    return model
            
            return ModelInfo(name=model_name, provider=provider,
                            status=ModelStatus.NOT_AVAILABLE, error_message="Model not found")
        
        except Exception as e:
            return ModelInfo(name=model_name, provider=provider,
                             status=ModelStatus.NOT_AVAILABLE, error_message=str(e))


    @staticmethod
    async def pull_model(model_name: str, provider: ELLMProvider, settings: AppSettings = None) -> bool:
        """Pull/download a model.

        For Ollama, this uses the official client to pull models, then we
        add the model to the user's available models list.
        For other providers like OpenAI, we operate a check for availability against the
        official model list. If the model exists, there is no download operation,
        but we do add the model to the user's available models list.
        
        Args:
            model_name (str): Name of the model to pull.
            provider (ELLMProvider): The provider to use.
            settings (AppSettings, optional): Application settings.
                                            If None, uses the singleton instance.
        Returns:
            bool: True if model was pulled successfully.
        """
        logger = logging_manager.get_session("ModelManagerAPI")
        
        settings = settings or AppSettings.get_instance()
        successful = False
        if provider == ELLMProvider.OLLAMA:
            successful = await ModelManagerAPI._pull_ollama_model(model_name, settings)
        elif provider == ELLMProvider.OPENAI:
            successful = await ModelManagerAPI._pull_openai_model(model_name, settings)
        else:
            logger.error(f"Unsupported provider: {provider}")

        if successful:
            new_model_info = ModelInfo(
                name=model_name,
                provider=provider,
                status=ModelStatus.AVAILABLE,
            )
            if not new_model_info in settings.llm.models:
                logger.info(f"Adding model {model_name} to available models for provider {provider.value}")
                settings.llm.models.append(new_model_info)
            else:
                logger.info(f"Model {model_name} is already in the available models for provider {provider.value}. No action taken.")

        return successful


    @staticmethod
    async def _list_ollama_models(settings: AppSettings) -> List[ModelInfo]:
        """List available Ollama models using the official client.
        
        Args:
            settings (AppSettings): Application settings.
            
        Returns:
            List[ModelInfo]: List of available models.
        """
        try:
            client = AsyncClient(host=settings.ollama.api_base)

            # Use the official client to list models
            models_response: ListResponse = await client.list()
            models = []
            
            for model_data in models_response.models:
                
                logger.debug(f"Model data: {model_data}")


                models.append(ModelInfo(
                    name=model_data.model,
                    provider=ELLMProvider.OLLAMA,
                    status=ModelStatus.AVAILABLE,
                    size=model_data.size,
                    modified_at=model_data.modified_at,
                    digest=model_data.digest,
                    details=model_data.details
                ))

            return models
            
        except Exception as e:
            # Fallback to log error and return empty list
            logger.error(f"Error listing Ollama models: {e}")
            raise e  # Re-raise to be caught by the outer try-except
    
    @staticmethod
    async def _list_openai_models(settings: AppSettings) -> List[ModelInfo]:
        """List available OpenAI models.
        
        Args:
            settings (AppSettings): Application settings.
            
        Returns:
            List[ModelInfo]: List of available models.
        """
        # For OpenAI, we return commonly available models since the API
        # model listing requires different permissions and pricing
        
        try:
            client = AsyncOpenAI(api_key=settings.openai.api_key, base_url=settings.openai.api_base)

            models_response = await client.models.list()
            models = []
            for model in models_response.data:
                models.append(ModelInfo(
                        name=model.id,
                        provider=ELLMProvider.OPENAI,
                        status=ModelStatus.AVAILABLE,
                        details={"type": "remote"}))
                
            return models
                
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return []

    @staticmethod
    async def _pull_openai_model(model_name: str, settings: AppSettings) -> bool:
        """Pull an OpenAI model by checking its availability.
        
        Args:
            model_name (str): Name of the model to pull.
            settings (AppSettings): Application settings.
            
        Returns:
            bool: True if model was pulled successfully.
        """
        logger = logging_manager.get_session("ModelManagerAPI")
        
        try:            
            # Check if the model exists
            models = await ModelManagerAPI._list_openai_models(settings)
            for model in models:
                if model.name.lower() == model_name.lower():
                    logger.info(f"Model '{model_name}' is available on OpenAI.")
                    return True
            
            logger.warning(f"Model '{model_name}' not found on OpenAI.")
            return False
            
        except Exception as e:
            logger.error(f"Error pulling OpenAI model {model_name}: {e}")
            return False

    @staticmethod
    async def _pull_ollama_model(model_name: str, settings: AppSettings) -> bool:
        """Pull an Ollama model using the official client.
        
        Args:
            model_name (str): Name of the model to pull.
            settings (AppSettings): Application settings.
            
        Returns:
            bool: True if model was pulled successfully.
        """
        logger = logging_manager.get_session("ModelManagerAPI")
        
        try:
            # Build host URL using settings
            client = AsyncClient(host=settings.ollama.api_base)
            
            logger.info(f"Starting to pull model: {model_name}")
            
            # Use the official client's pull method with streaming
            async for progress in await client.pull(model_name, stream=True):
                status = progress.get("status", "")
                
                #use tqdm for progress bar
                if status == "downloading":
                    total = progress.get("total", 0)
                    completed = progress.get("completed", 0)
                    percentage = (completed / total) * 100 if total > 0 else 0
                    
                    # Log progress with tqdm
                    tqdm.write(f"Downloading {model_name}: {percentage:.2f}% ({completed}/{total})")
                    
                elif status == "verifying sha256 digest":
                    tqdm.write(f"Verifying SHA256 digest for {model_name}")
                    
                elif status == "writing manifest":
                    tqdm.write(f"Writing manifest for {model_name}")
                    
                elif status == "success":
                    tqdm.write(f"Successfully pulled model: {model_name}")
                
                elif status == "error":
                    error_message = progress.get("error", "Unknown error")
                    logger.error(f"Error pulling model {model_name}: {error_message}")
                    return False

                
                # Log important status updates
                if status in ["downloading", "verifying sha256 digest", "writing manifest", "success"]:
                    logger.info(f"Model {model_name}: {status}")
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False


# Create singleton instance for easy access
model_api = ModelManagerAPI()
