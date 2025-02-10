import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def process_text(input_text, model="qwen/qwen2.5-vl-72b-instruct:free"):
    """
    Process text using an AI model via OpenRouter API.
    
    Args:
        input_text (str): The text to be processed
        model (str): The AI model to use for processing
        
    Returns:
        str: Processed text or None if an error occurs
        
    Raises:
        dict: Error information containing 'type' and 'message'
    """
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError({
                "type": "Configuration Error",
                "message": "OPENROUTER_API_KEY environment variable not found"
            })

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompt = f"""
        # Text Processing Instructions

        ## Content Organization
        - Divide the text into thematic sections based on discussed content
        - Organize text into clear, short paragraphs, avoiding long text blocks
        - Remove repetitions and unnecessary phrases
        - Add titles and subtitles using markdown formatting

        ## Content Enhancement
        - Identify and highlight main topics
        - Highlight technical or important terms using bold
        - Correct grammatical errors and confusing phrases
        - Add examples or lists where needed for clarity
        - Fix typos based on context (e.g., "Macron" -> "Crown")

        ## Visual Representation
        Create a Mermaid flowchart showing:
        - Main topics as nodes
        - Relationships between topics using arrows
        - Brief descriptions on connections where relevant

        Text to process:
        {input_text}
        """

        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )

        if not completion or not completion.choices or len(completion.choices) == 0:
            raise ValueError({
                "type": "API Error",
                "message": "Invalid response from OpenRouter API"
            })

        return completion.choices[0].message.content

    except Exception as e:
        if hasattr(e, 'args') and isinstance(e.args[0], dict):
            raise type(e)(e.args[0])
        else:
            raise ValueError({
                "type": "Processing Error",
                "message": f"Error processing text with OpenRouter: {str(e)}"
            })
