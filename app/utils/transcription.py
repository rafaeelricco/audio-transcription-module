import os
import argparse
import sys

from typing import Optional
from app.utils.providers import OpenRouterProvider, GeminiProvider
from dotenv import load_dotenv
from app.utils.logger import Logger
from app.utils.functions import load_config

load_dotenv()


def get_provider(provider_name: str):
    """
    Factory function to create provider instances by name.

    Creates and returns an instance of the specified AI provider class.
    This function implements the factory design pattern to encapsulate
    provider instantiation logic.

    Args:
        provider_name (str): Name of the provider to create (e.g., 'openrouter', 'gemini')

    Returns:
        BaseProvider: Provider instance of the requested type

    Raises:
        ValueError: If the specified provider is not supported
    """
    providers = {"openrouter": OpenRouterProvider, "gemini": GeminiProvider}

    if provider_name not in providers:
        supported = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Supported providers: {supported}"
        )

    return providers[provider_name]()


def load_prompt_template(input_text) -> str:
    prompt = f"""
    Analise esta transcrição e crie um resumo completo e organizado que substitua efetivamente a necessidade de assistir ao vídeo/áudio original. Retorne o resumo formatado em Markdown sem tags de bloco de código. Adapte a estrutura com base no conteúdo específico, seguindo estas diretrizes:

    ## Estrutura do Resumo

    • **Título Principal**: Crie um título conciso que capture a essência do conteúdo.

    • **Introdução**:
    - Apresente uma visão geral do tema (2-3 frases)
    - Identifique o objetivo principal ou questão central
    - Mencione brevemente o contexto relevante

    • **Pontos-Chave**: 
    - Liste os 4-12 conceitos mais importantes usando bullet points
    - Destaque **termos essenciais** em negrito
    - Limite cada ponto a 1-6 frases objetivas

    • **Síntese Final**:
    - Resuma as conexões entre os pontos principais (2-8 frases)
    - Mencione a relevância ou aplicabilidade do conteúdo

    • **Perguntas-Chave** (opcional):
    - Inclua 2-8 questões que estimulem reflexão adicional sobre o tema

    ## Formatação e Estilo

    • Utilize hierarquia de títulos (# ## ###) para organização visual
    • Empregue **negrito** para destacar conceitos fundamentais
    • Use bullet points (•) para informações discretas
    • Inclua numeração para processos sequenciais
    • Mantenha o resumo limitado a aproximadamente 25% do conteúdo original
    • Utilize linguagem clara e direta, sem repetições desnecessárias

    O formato final deve ser limpo, profissional e agradável de ler, semelhante a um artigo bem estruturado. Adapte completamente a abordagem ao tipo específico de conteúdo.

    Aplique esta estrutura ao seguinte texto:
    {input_text}
    """
    return prompt


def process_text(input_text: str) -> str:
    """
    Process text using the configured AI provider.

    Takes raw input text (typically a transcript) and processes it using
    the configured AI provider to improve organization, formatting,
    or other text qualities as defined in the prompt template.

    Args:
        input_text (str): The raw text to be processed by the AI provider

    Returns:
        str: Processed and enhanced text from the AI provider

    Raises:
        ValueError: If processing fails due to configuration issues, API errors,
                   or empty responses from the provider
    """
    try:
        config = load_config()
        provider_name = config["default_provider"]

        Logger.log(True, f"Using provider: {provider_name}")
        provider = get_provider(provider_name)

        template = load_prompt_template(input_text)
        prompt = template.format(input_text=input_text)

        Logger.log(True, "Sending request to AI model...")
        processed_text = provider.generate_content(prompt)

        if not processed_text:
            raise ValueError("Empty response received from AI provider")

        Logger.log(True, "Processing completed successfully")
        return processed_text

    except FileNotFoundError as e:
        Logger.log(False, str(e), "error")
        raise ValueError({"type": "Configuration Error", "message": str(e)})
    except ValueError as e:
        Logger.log(False, f"AI processing failed: {str(e)}", "error")
        if hasattr(e, "args") and isinstance(e.args[0], dict):
            raise
        else:
            raise ValueError(
                {
                    "type": "Processing Error",
                    "message": f"Error processing text with AI provider: {str(e)}",
                }
            )
    except Exception as e:
        Logger.log(False, f"Unexpected error: {str(e)}", "error")
        raise ValueError(
            {"type": "Unknown Error", "message": f"Unexpected error occurred: {str(e)}"}
        )


def read_file(file_path: str) -> str:
    """
    Read content from a text file.

    Safely reads and returns the content of a text file with proper error handling.

    Args:
        file_path (str): Path to the text file to read

    Returns:
        str: Content of the file as a string

    Raises:
        ValueError: If the file cannot be read due to permissions, encoding issues,
                  or if the file doesn't exist
    """
    try:
        Logger.log(True, f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        Logger.log(False, f"Error reading file: {str(e)}", "error")
        raise ValueError(
            {"type": "File Error", "message": f"Error reading file: {str(e)}"}
        )


def save_processed_text(processed_text: str, input_file_path: str) -> Optional[str]:
    """
    Save the processed text to a markdown file.

    Creates a new markdown file containing the processed text, using the
    original input file name as a base for the output filename.

    Args:
        processed_text (str): The processed text to save to file
        input_file_path (str): Original input file path (used to generate output filename)

    Returns:
        Optional[str]: Path to the saved file, or None if saving failed
    """
    try:
        base_name = os.path.basename(input_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(input_file_path)

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{name_without_ext}_processed.md")

        Logger.log(True, f"Saving processed text to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(processed_text)

        return output_file
    except Exception as e:
        Logger.log(False, f"Error saving processed text: {str(e)}", "error")
        return None


def main() -> None:
    """
    Main function to process text files using command-line arguments.

    Provides a command-line interface for the text processing functionality.
    This function parses command-line arguments, processes the specified file,
    and outputs the result to a new file or displays it if saving fails.

    Command-line arguments:
        --file, -f: Path to the text file to process (required)
        --verbose, -v: Enable verbose logging (optional)
    """
    parser = argparse.ArgumentParser(description="Process text files using AI")
    parser.add_argument(
        "--file", "-f", required=True, help="Path to the text file to process"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        Logger.set_verbose(True)

    try:
        file_content = read_file(args.file)

        Logger.log(True, f"Processing text from file: {args.file}")
        processed_text = process_text(file_content)

        output_file = save_processed_text(processed_text, args.file)

        if output_file:
            Logger.log(True, f"Processing complete. Output saved to: {output_file}")
        else:
            Logger.log(
                True,
                "Processing complete, but output could not be saved. Displaying result:",
            )
            print(processed_text)

    except KeyboardInterrupt:
        Logger.log(False, "Process interrupted by user", "warning")
        print("\nProcess interrupted by user. Exiting gracefully...")

    except ValueError as e:
        if hasattr(e, "args") and isinstance(e.args[0], dict):
            error_data = e.args[0]
            Logger.log(
                False,
                f"{error_data['type']}: {error_data['message']}",
                "error",
            )
        else:
            Logger.log(False, f"Error: {str(e)}", "error")
        sys.exit(1)
    except Exception as e:
        Logger.log(False, f"Unexpected error: {str(e)}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()
