import os
import yaml
import time
from typing import Dict, Any, Optional, List
from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting, Content, Part
from utils import load_config
from logger import Logger


class BaseProvider:
    """
    Base class for AI content generation providers.

    Implements common functionality for all providers including
    configuration loading, error handling, and content generation.
    """

    def __init__(self, provider_name: str):
        """
        Initialize a provider with its configuration.

        Args:
            provider_name (str): The name of the provider in the config file

        Raises:
            ValueError: If the provider configuration is invalid
            FileNotFoundError: If the config file doesn't exist
        """
        try:
            config = load_config()
            self.config = config["providers"].get(provider_name)

            if not self.config:
                raise ValueError(
                    f"Configuration for provider '{provider_name}' not found"
                )

            api_key_env_var = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(api_key_env_var)

            if not api_key:
                raise ValueError(
                    f"API key not found. Set the {api_key_env_var} environment variable"
                )

            self.client = genai.Client(api_key=api_key)
            self.provider_name = provider_name

        except FileNotFoundError:
            Logger.log(False, "Configuration file not found", "error")
            raise
        except Exception as e:
            Logger.log(
                False, f"Error initializing {provider_name} provider: {str(e)}", "error"
            )
            raise ValueError(f"Failed to initialize {provider_name} provider: {str(e)}")

    def _get_model_config(self) -> tuple:
        """
        Get the model configuration from the config file.

        Returns:
            tuple: (model_name, generate_config, safety_settings)

        Raises:
            ValueError: If the model configuration is invalid
        """
        try:
            model_name = self.config["models"]["default"]
            model_config = (
                self.config["models"].get("model_configs", {}).get(model_name, {})
            )

            safety_settings = []
            if "safety_settings" in model_config:
                safety_settings = [
                    SafetySetting(**s) for s in model_config.pop("safety_settings", [])
                ]

            generate_config = GenerateContentConfig(
                **model_config, safety_settings=safety_settings or None
            )

            return model_name, generate_config

        except Exception as e:
            Logger.log(False, f"Error getting model configuration: {str(e)}", "error")
            raise ValueError(f"Invalid model configuration: {str(e)}")

    def generate_content(
        self, prompt: str, max_retries: int = 3, retry_delay: float = 2.0
    ) -> str:
        """
        Generate content using the configured AI model with retry logic.

        Args:
            prompt (str): The prompt to send to the model
            max_retries (int): Maximum number of retry attempts on failure
            retry_delay (float): Initial delay between retries in seconds

        Returns:
            str: The generated content

        Raises:
            ValueError: If content generation fails after all retries
        """
        model_name, generate_config = self._get_model_config()

        attempt = 0
        while attempt <= max_retries:
            try:
                Logger.log(
                    True,
                    f"Sending request to {self.provider_name} ({model_name})",
                    "debug",
                )

                response = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=[Content(parts=[Part(text=prompt)])],
                    config=generate_config,
                )

                result = ""
                print()
                print("", end="", flush=True)

                for chunk in response:
                    if chunk.text is not None:
                        print(chunk.text, end="", flush=True)
                        result += chunk.text
                print()

                if not result.strip():
                    raise ValueError("Empty response received from API")

                return result

            except Exception as e:
                attempt += 1
                if attempt <= max_retries:
                    Logger.log(
                        False,
                        f"API request failed (attempt {attempt}/{max_retries}): {str(e)}. Retrying...",
                        "warning",
                    )
                    time.sleep(retry_delay * (2 ** (attempt - 1)))
                else:
                    Logger.log(
                        False,
                        f"API request failed after {max_retries} attempts: {str(e)}",
                        "error",
                    )
                    raise ValueError(f"Failed to generate content: {str(e)}")


class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter API."""

    def __init__(self):
        """Initialize the OpenRouter provider."""
        super().__init__("openrouter")


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini API."""

    def __init__(self):
        """Initialize the Gemini provider."""
        super().__init__("gemini")
