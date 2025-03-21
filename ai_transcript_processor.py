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
        # Text Processing Instructions
        
        ## Structural Requirements
        - Create hierarchical document structure with collapsible sections
        - Implement automatic section numbering (1.1, 1.2, 2.1, etc.)
        - Generate topic flowcharts using Mermaid.js syntax
        - Add table of contents with anchor links
        - Include progress tracking milestones
        - Maintain original timestamps as metadata

        ## Content Organization
        1. Divide content into thematic sections with clear headings
        2. Create summary bullet points for each main topic
        3. Organize technical content in expandable/collapsible blocks
        4. Separate main content from auxiliary information using side notes
        5. Implement responsive layout considerations for HTML output

        ## Stylistic Guidelines
        - Format code blocks with syntax highlighting (specify language)
        - Use consistent typography:
          * Technical terms in **bold**
          * Important concepts in *italics*
          * Key quotes in blockquotes
        - Apply conditional formatting:
          ✅ Correct statements in green
          ⚠️ Uncertain elements in orange
          ❌ Contradictions in red
        - Add interactive elements for HTML reports:
          * Clickable section headers
          * Searchable term index
          * Dynamic content filtering

        ## Quality Assurance
        - Verify technical term consistency
        - Cross-check referenced sources
        - Maintain original content meaning
        - Ensure logical flow between sections
        - Validate all external links/resources
        - Preserve context while removing redundancies

        ## Output Format
        - Return the entire response in valid Markdown format.
        - Use appropriate Markdown syntax for headings, lists, code blocks, and other text elements.
        
        Text to process:
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
