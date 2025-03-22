import os
from openai import OpenAI
from dotenv import load_dotenv
from logger import Logger

load_dotenv()


def process_text(input_text, model="google/gemini-2.0-flash-thinking-exp:free"):
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
            raise ValueError(
                {
                    "type": "Configuration Error",
                    "message": "OPENROUTER_API_KEY environment variable not found",
                }
            )

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompt = f"""
        Analyze this transcript thoroughly and create a complete summary in article format that effectively replaces the need to watch the original video/audio. Return the summary entirely formatted in Markdown. Completely adapt the structure based on the specific content, following these general guidelines:

        1. Determine the most appropriate main title that captures the essence of the content.

        2. Structure the article with logical and fluid sections, creating relevant subtitles that organically reflect the main themes of the content. Do not use predefined structures, but completely adapt to the specific material.

        3. Begin with a contextual introduction that presents the general overview of the subject, establishing the tone and scope of the content.

        4. Develop the body of the article cohesively, grouping related information into natural thematic sections. Determine if the content benefits more from:
        - Narrative paragraphs for conceptual content
        - Bullet lists for discrete points
        - Numbered sequences for processes or steps
        - Comparisons for different perspectives or approaches

        5. When relevant, include a section highlighting practical applications, benefits, main results, or implications.

        6. Conclude with a synthesis that connects the main points and offers perspective on the overall value of the content.

        Use Markdown formatting to highlight important elements:
        - **Bold** for key concepts
        - Hierarchical subtitles (# ## ###) for visual organization
        - Lists and bullets when appropriate to facilitate reading
        - Direct quotations only when essential to preserve the exact meaning

        The final format should be clean, professional, and pleasant to read, similar to a well-structured article. Completely adapt the approach to the specific type of content, whether it is educational, tutorial, interview, debate, presentation, or any other format.
        
        Apply this structure to the following text:
        {input_text}
        """

        Logger.log(True, "Sending request to AI model...")

        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        Logger.log(True, "Request confirmed by API")

        Logger.log(True, "Processing AI response...")

        if not completion or not completion.choices or len(completion.choices) == 0:
            raise ValueError(
                {"type": "API Error", "message": "Invalid response from OpenRouter API"}
            )
        Logger.log(True, "Processing completed")
        return completion.choices[0].message.content

    except Exception as e:
        if hasattr(e, "args") and isinstance(e.args[0], dict):
            raise type(e)(e.args[0])
        else:
            raise ValueError(
                {
                    "type": "Processing Error",
                    "message": f"Error processing text with OpenRouter: {str(e)}",
                }
            )
