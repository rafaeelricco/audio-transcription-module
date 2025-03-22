import os
import yaml
from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting, Content, Part


class OpenRouterProvider:
    def __init__(self):
        with open("config.yml") as f:
            config = yaml.safe_load(f)
        self.config = config["providers"]["openrouter"]
        self.client = genai.Client(api_key=os.getenv("OPENROUTER_API_KEY"))

    def generate_content(self, prompt: str):
        model_name = self.config["models"]["default"]
        model_config = (
            self.config["models"].get("model_configs", {}).get(model_name, {})
        )

        safety_settings = [
            SafetySetting(**s) for s in model_config.pop("safety_settings", [])
        ]

        generate_config = GenerateContentConfig(
            **model_config, safety_settings=safety_settings or None
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

        return "".join(chunk.text for chunk in response if chunk.text is not None)


class GeminiProvider:
    def __init__(self):
        with open("config.yml") as f:
            config = yaml.safe_load(f)
        self.config = config["providers"]["gemini"]
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_content(self, prompt: str):
        model_name = self.config["models"]["default"]
        model_config = (
            self.config["models"].get("model_configs", {}).get(model_name, {})
        )

        safety_settings = [
            SafetySetting(**s) for s in model_config.pop("safety_settings", [])
        ]

        generate_config = GenerateContentConfig(
            **model_config, safety_settings=safety_settings or None
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

        return "".join(chunk.text for chunk in response if chunk.text is not None)
