import os
import yaml
from google import genai

class OpenRouterProvider:
    def __init__(self):
        with open('config.yml') as f:
            config = yaml.safe_load(f)
        self.config = config['providers']['openrouter']
        self.client = genai.Client(api_key=os.getenv('OPENROUTER_API_KEY'))

    def generate_content(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.config['models']['default'],
            contents=prompt
        )
        return response.text

class GeminiProvider:
    def __init__(self):
        with open('config.yml') as f:
            config = yaml.safe_load(f)
        self.config = config['providers']['gemini']
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def generate_content(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.config['models']['default'],
            contents=prompt
        )
        return response.text
