import os
import argparse
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
        Analyze this transcript thoroughly and create a complete summary in article format that effectively replaces the need to watch the original video/audio. Return the summary entirely formatted in Markdown without any Markdown code block tags (```). Completely adapt the structure based on the specific content, following these general guidelines:

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
        - Return the content in Brazilian Portuguese (Português Brasileiro)

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


def read_file(file_path):
    """
    Read content from a text file.

    Args:
        file_path (str): Path to the text file

    Returns:
        str: Content of the file

    Raises:
        ValueError: If file cannot be read
    """
    try:
        Logger.log(True, f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        raise ValueError(
            {"type": "File Error", "message": f"Error reading file: {str(e)}"}
        )


def save_processed_text(processed_text, input_file_path):
    """
    Save the processed text to a markdown file.

    Args:
        processed_text (str): The processed text to save
        input_file_path (str): Original input file path (used to generate output filename)

    Returns:
        str: Path to the saved file
    """
    try:
        # Generate output filename based on input filename
        base_name = os.path.basename(input_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.join(os.path.dirname(input_file_path))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{name_without_ext}_processed.md")

        Logger.log(True, f"Saving processed text to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(processed_text)

        return output_file
    except Exception as e:
        Logger.log(False, f"Error saving processed text: {str(e)}")
        return None


def main():
    """
    Main function to process text from a file using command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process text files using AI")
    parser.add_argument(
        "--file", "-f", required=True, help="Path to the text file to process"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="google/gemini-2.0-flash-thinking-exp:free",
        help="AI model to use for processing",
    )

    args = parser.parse_args()

    try:
        # Read the file content
        file_content = read_file(args.file)

        # Process the text
        Logger.log(True, f"Processing text from file: {args.file}")
        processed_text = process_text(file_content, model=args.model)

        # Save the processed text
        output_file = save_processed_text(processed_text, args.file)

        if output_file:
            Logger.log(True, f"Processing complete. Output saved to: {output_file}")
        else:
            Logger.log(True, "Processing complete, but output could not be saved.")
            print(processed_text)

    except KeyboardInterrupt:
        Logger.log(False, "Process interrupted by user")
        print("\nProcess interrupted by user. Exiting gracefully...")

    except Exception as e:
        if hasattr(e, "args") and isinstance(e.args[0], dict):
            error = e.args[0]
            Logger.log(False, f"{error['type']}: {error['message']}")
        else:
            Logger.log(False, f"Error: {str(e)}")


if __name__ == "__main__":
    main()
