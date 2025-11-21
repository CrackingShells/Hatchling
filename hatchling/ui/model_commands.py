"""LLM Model Management Commands.

This module provides CLI commands for managing LLM models and providers.
Commands follow the format 'llm:target:action' for clarity and consistency.
"""

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

from hatchling.ui.abstract_commands import AbstractCommands
from hatchling.core.llm.model_manager_api import ModelManagerAPI
from hatchling.config.llm_settings import LLMSettings

class ModelCommands(AbstractCommands):
    """CLI commands for LLM model and provider management."""
    
    def _register_commands(self) -> None:
        """Register all model-related commands."""
        
        from hatchling.config.i18n import translate
        self.commands = {
            # Provider management commands
            'llm:provider:supported': {
                'handler': self._cmd_provider_supported,
                'description': translate('commands.llm.provider_supported_description'),
                'is_async': False,
                'args': {}
            },
            'llm:provider:status': {
                'handler': self._cmd_provider_status,
                'description': translate('commands.llm.provider_status_description'),
                'is_async': True,
                'args': {
                    'provider-name': {
                        'positional': False,
                        'completer_type': 'suggestions',
                        'values': self.settings.llm.provider_names,
                        'description': translate('commands.llm.provider_name_arg_description'),
                        'required': False
                    }
                }
            },
            # Model management commands
            'llm:model:list': {
                'handler': self._cmd_model_list,
                'description': translate('commands.llm.model_list_description'),
                'is_async': True,
                'args': {}
            },
            'llm:model:discover': {
                'handler': self._cmd_model_discover,
                'description': translate('commands.llm.model_discover_description'),
                'is_async': True,
                'args': {
                    'provider-name': {
                        'positional': False,
                        'completer_type': 'suggestions',
                        'values': self.settings.llm.provider_names,
                        'description': translate('commands.llm.provider_name_arg_description'),
                        'required': False
                    }
                }
            },
            'llm:model:add': {
                'handler': self._cmd_model_add,
                'description': translate('commands.llm.model_add_description'),
                'is_async': True,
                'args': {
                    'provider-name': {
                        'positional': False,
                        'completer_type': 'suggestions',
                        'values': self.settings.llm.provider_names,
                        'description': translate('commands.llm.provider_name_arg_description'),
                        'required': False
                    },
                    'model-name': {
                        'positional': True,
                        'description': translate('commands.llm.model_name_arg_description'),
                        'required': True
                    }
                }
            },
            'llm:model:use': {
                'handler': self._cmd_model_use,
                'description': translate('commands.llm.model_use_description'),
                'is_async': False,
                'args': {
                    'model-name': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': [model.name for model in self.settings.llm.models],
                        'description': translate('commands.llm.model_name_arg_description'),
                        'required': True
                    },
                    'force-confirmed':{
                        'positional': False,
                        'completer_type': 'boolean',
                        'description': translate('commands.llm.force_confirmed_arg_description'),
                        'required': False,
                        'default': False,
                        'is_flag': True
                    }
                }
            },
            'llm:model:remove': {
                'handler': self._cmd_model_remove,
                'description': translate('commands.llm.model_remove_description'),
                'is_async': False,
                'args': {
                    'model-name': {
                        'positional': True,
                        'completer_type': 'suggestions',
                        'values': [model.name for model in self.settings.llm.models],
                        'description': translate('commands.llm.model_name_arg_description'),
                        'required': True
                    }
                }
            }
        }

    def print_commands_help(self) -> None:
        """Print help for all available chat commands."""
        print_formatted_text(FormattedText([
            ('class:header', "\n=== Model Commands ===\n")
        ]), style=self.style)

        # Call parent class method to print formatted commands
        super().print_commands_help()
    
    # =============================================================================
    # Provider Management Commands
    # =============================================================================

    def _cmd_provider_supported(self, args: str) -> bool:
        """List all supported LLM providers.
        This command retrieves and displays all LLM providers supported by the system.

        Args:
            args (str): Unused arguments.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            providers = ModelManagerAPI.list_providers()
            
            if not providers:
                print("No LLM providers found")
                return True
            
            print("Available LLM Providers:")
            for provider in providers:
                print(f"  {provider}")
                
        except Exception as e:
            print(f"Error listing providers: {e}")
            self.logger.error(f"Error in provider list command: {e}")
            
        return True
    
    async def _cmd_provider_status(self, args: str) -> bool:
        """Check status of a specific provider.

        Args:
            args (str): Provider name argument.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            args_def = self.commands['llm:provider:status']['args']
            parsed_args = self._parse_args(args, args_def)
            provider_name = parsed_args.get('provider-name', '')
            providers = self.settings.llm.provider_enums

            if provider_name:
                # Get provider health
                providers = [self.settings.llm.to_provider_enum(provider_name)] 
            
            for provider in providers:
                is_healthy = await ModelManagerAPI.check_provider_health(provider)
                if is_healthy:
                    print(f"Provider: {provider} - Status: AVAILABLE")
                    models = await ModelManagerAPI.list_available_models(provider)
                    print(f"  - Models: {[model.name for model in models]}")
                else:
                    print(f"Provider: {provider.value} - Status: UNAVAILABLE")

        except Exception as e:
            self.logger.error(f"Error in provider status command: {e}")
        
        finally:
            return True
    
    # =============================================================================
    # Model Management Commands
    # =============================================================================
    
    async def _cmd_model_list(self, args: str) -> bool:
        """List curated models with availability status indicators.

        Shows models grouped by provider with status indicators:
        - ✓ AVAILABLE: Model is accessible and ready to use
        - ✗ UNAVAILABLE: Model is configured but not accessible at provider

        Args:
            args (str): Optional provider name or search query to filter models.

        Returns:
            bool: True to continue the chat session.
        """

        # Check if curated list is empty
        if not self.settings.llm.models:
            print("📋 Your curated model list is empty.")
            print("\nTo add models:")
            print("  1. Discover all available models:")
            print("     llm:model:discover")
            print("  2. Or add a specific model:")
            print("     llm:model:add <model-name>")
            print("\nFor Ollama models, pull them first:")
            print("  ollama pull <model-name>")
            return True

        # Group models by provider
        from collections import defaultdict
        from hatchling.config.llm_settings import ModelStatus

        models_by_provider = defaultdict(list)
        for model in self.settings.llm.models:
            models_by_provider[model.provider].append(model)

        # Display models grouped by provider
        print("📋 Curated LLM Models:\n")

        for provider, models in sorted(models_by_provider.items(), key=lambda x: x[0].value):
            print(f"  {provider.value.upper()}:")

            # Check provider health
            is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)

            if not is_healthy:
                print(f"    ⚠️  Provider not accessible")
                for model in sorted(models, key=lambda m: m.name):
                    current_marker = " (current)" if model.name == self.settings.llm.model else ""
                    print(f"      ✗ {model.name}{current_marker}")
                print()
                continue

            # Fetch available models from provider to check status
            try:
                available_models = await ModelManagerAPI.list_available_models(provider, self.settings)
                available_names = {m.name.lower() for m in available_models}
            except Exception as e:
                self.logger.error(f"Error fetching models from {provider.value}: {e}")
                available_names = set()

            # Display each model with status
            for model in sorted(models, key=lambda m: m.name):
                # Determine status
                is_available = model.name.lower() in available_names
                status_icon = "✓" if is_available else "✗"

                # Mark current model
                current_marker = " (current)" if model.name == self.settings.llm.model else ""

                print(f"      {status_icon} {model.name}{current_marker}")

            print()

        # Show legend
        print("Legend:")
        print("  ✓ AVAILABLE   - Model is accessible and ready to use")
        print("  ✗ UNAVAILABLE - Model is configured but not accessible")
        print("\n💡 Use 'llm:model:use <model-name>' to set active model")
        print("💡 Use 'llm:model:remove <model-name>' to remove from list")

        return True

    async def _cmd_model_discover(self, args: str) -> bool:
        """Discover and add all available models from provider to curated list.

        This command fetches all models currently available at the provider and adds
        them to the user's curated model list. Models must already be available:
        - For Ollama: Models must be pulled first with 'ollama pull <model-name>'
        - For OpenAI: Models must be accessible with your API key

        Args:
            args (str): Optional provider name argument (defaults to current provider).

        Returns:
            bool: True to continue the chat session.
        """
        try:
            args_def = self.commands['llm:model:discover']['args']
            parsed_args = self._parse_args(args, args_def)
            provider_name = parsed_args.get('provider-name', self.settings.llm.provider_enum.value)
            provider = LLMSettings.to_provider_enum(provider_name)

            # Check provider health first
            is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)
            if not is_healthy:
                print(f"❌ Provider '{provider.value}' is not accessible.")
                print(f"\nTroubleshooting:")
                if provider.value == "ollama":
                    print(f"  1. Check if Ollama is running: 'ollama list'")
                    print(f"  2. Verify connection settings:")
                    print(f"     - IP: {self.settings.ollama.ip}")
                    print(f"     - Port: {self.settings.ollama.port}")
                    print(f"  3. Update settings if needed:")
                    print(f"     settings:set ollama:ip <ip>")
                    print(f"     settings:set ollama:port <port>")
                elif provider.value == "openai":
                    print(f"  1. Verify OPENAI_API_KEY is set")
                    print(f"  2. Check internet connection")
                    print(f"  3. Verify API base URL: {self.settings.openai.api_base}")
                return True

            # Fetch available models from provider
            print(f"🔍 Discovering models from {provider.value}...")
            available_models = await ModelManagerAPI.list_available_models(provider, self.settings)

            if not available_models:
                print(f"⚠️  No models found at {provider.value}.")
                if provider.value == "ollama":
                    print(f"\nTo add models:")
                    print(f"  1. Pull a model: ollama pull <model-name>")
                    print(f"  2. Run discovery again: llm:model:discover")
                return True

            # Add models to curated list (skip duplicates)
            added_count = 0
            skipped_count = 0
            existing_model_keys = {(m.provider, m.name) for m in self.settings.llm.models}

            for model in available_models:
                model_key = (model.provider, model.name)
                if model_key not in existing_model_keys:
                    self.settings.llm.models.append(model)
                    added_count += 1
                else:
                    skipped_count += 1

            # Report results
            print(f"\n✅ Discovery complete!")
            print(f"  Added: {added_count} model(s)")
            if skipped_count > 0:
                print(f"  Skipped: {skipped_count} model(s) (already in list)")
            print(f"  Total models in curated list: {len(self.settings.llm.models)}")

            # Update command completions
            if added_count > 0:
                self.commands['llm:model:use']['args']['model-name']['values'] = [
                    model.name for model in self.settings.llm.models
                ]
                self.commands['llm:model:remove']['args']['model-name']['values'] = [
                    model.name for model in self.settings.llm.models
                ]
                print(f"\n💡 Use 'llm:model:list' to see all models")
                print(f"💡 Use 'llm:model:use <model-name>' to set active model")

        except Exception as e:
            self.logger.error(f"Error in model discover command: {e}")
            print(f"❌ Error during discovery: {e}")

        return True

    async def _cmd_model_add(self, args: str) -> bool:
        """Add a specific model to the curated list with validation.

        This command validates that the model exists at the provider before adding
        it to the curated list. Models must already be available:
        - For Ollama: Model must be pulled first with 'ollama pull <model-name>'
        - For OpenAI: Model must be accessible with your API key

        Args:
            args (str): Model name argument and optional provider.

        Returns:
            bool: True to continue the chat session.
        """
        try:
            args_def = self.commands['llm:model:add']['args']
            parsed_args = self._parse_args(args, args_def)

            model_name = parsed_args.get('model-name', '')
            provider_name = parsed_args.get('provider-name', self.settings.llm.provider_enum.value)
            provider = LLMSettings.to_provider_enum(provider_name)

            if not model_name:
                self.logger.error("Positional argument 'model-name' is required to add a model.")
                return True

            # Check provider health
            is_healthy = await ModelManagerAPI.check_provider_health(provider, self.settings)
            if not is_healthy:
                print(f"❌ Provider '{provider.value}' is not accessible.")
                print(f"\nTroubleshooting:")
                if provider.value == "ollama":
                    print(f"  1. Check if Ollama is running: 'ollama list'")
                    print(f"  2. Verify connection settings:")
                    print(f"     - IP: {self.settings.ollama.ip}")
                    print(f"     - Port: {self.settings.ollama.port}")
                elif provider.value == "openai":
                    print(f"  1. Verify OPENAI_API_KEY is set")
                    print(f"  2. Check internet connection")
                return True

            # Fetch available models from provider
            available_models = await ModelManagerAPI.list_available_models(provider, self.settings)

            # Check if model exists in available list
            model_found = None
            for model in available_models:
                if model.name.lower() == model_name.lower():
                    model_found = model
                    break

            if not model_found:
                print(f"❌ Model '{model_name}' not found at {provider.value}.")
                print(f"\nAvailable models at {provider.value}:")
                if available_models:
                    # Show first 10 models
                    for i, model in enumerate(available_models[:10]):
                        print(f"  - {model.name}")
                    if len(available_models) > 10:
                        print(f"  ... and {len(available_models) - 10} more")
                    print(f"\n💡 Use 'llm:model:discover' to add all available models")
                else:
                    if provider.value == "ollama":
                        print(f"  No models found. Pull a model first:")
                        print(f"  ollama pull <model-name>")
                return True

            # Check for duplicates
            existing_model_keys = {(m.provider, m.name) for m in self.settings.llm.models}
            model_key = (model_found.provider, model_found.name)

            if model_key in existing_model_keys:
                print(f"⚠️  Model '{model_name}' is already in your curated list.")
                print(f"💡 Use 'llm:model:list' to see all models")
                return True

            # Add model to curated list
            self.settings.llm.models.append(model_found)
            print(f"✅ Added '{model_name}' to your curated list.")

            # Update command completions
            self.commands['llm:model:use']['args']['model-name']['values'] = [
                model.name for model in self.settings.llm.models
            ]
            self.commands['llm:model:remove']['args']['model-name']['values'] = [
                model.name for model in self.settings.llm.models
            ]

            print(f"💡 Use 'llm:model:use {model_name}' to set it as active model")

        except Exception as e:
            self.logger.error(f"Error in model add command: {e}")
            print(f"❌ Error adding model: {e}")

        return True

    def _cmd_model_use(self, args: str) -> bool:
        """Set the default model to use for the current session.
        
        Args:
            args (str): Model name argument.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            args_def = self.commands['llm:model:use']['args']
            parsed_args = self._parse_args(args, args_def)
            model_name = parsed_args.get('model-name', '')
            force_confirmed = parsed_args.get('force-confirmed', False)

            if not model_name:
                self.logger.error("Positional argument 'model-name' is required to set the default model.")
                return True
            
            # Check if the model exists in the available models
            model_info = None
            for model in self.settings.llm.models:
                if model.name == model_name:
                    model_info = model
                    break

            if not model_info:
                self.logger.warning(f"Model '{model_name}' not found in available models. No action taken.")
                return True
            
            # Set the default model in the settings
            self.settings_registry.set_setting(
                "llm", "model", model_info.name, force=force_confirmed
            )
            self.settings_registry.set_setting(
                "llm", "provider_enum", model_info.provider, force=force_confirmed
            )

        except Exception as e:
            self.logger.error(f"Error in model use command: {e}")
            
        return True
    
    def _cmd_model_remove(self, args: str) -> bool:
        """Remove a model from the list of available models.
        
        Args:
            args (str): Model name argument.
            
        Returns:
            bool: True to continue the chat session.
        """
        try:
            args_def = self.commands['llm:model:remove']['args']
            parsed_args = self._parse_args(args, args_def)

            model_name = parsed_args.get('model-name', '')
            if not model_name:
                self.logger.error("Positional argument 'model-name' is required to remove a model.")
                return True
            
            # Find and remove the model
            for model_info in self.settings.llm.models:
                if model_info.name == model_name:
                    self.settings.llm.models.remove(model_info)
                    self.logger.info(f"Model '{model_name}' removed successfully.")
                    
                    # Update the command args values for autocompletion
                    self.commands['llm:model:use']['args']['model-name']['values'] = [model.name for model in self.settings.llm.models]
                    self.commands['llm:model:remove']['args']['model-name']['values'] = [model.name for model in self.settings.llm.models]
                    return True

            self.logger.warning(f"Model '{model_name}' not found in available models. No action taken.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model remove command: {e}")
            
        return True
    