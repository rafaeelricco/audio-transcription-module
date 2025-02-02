import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def process_text(input_text, model="anthropic/claude-3.5-sonnet"):
    """
    Processa texto usando um modelo via OpenRouter
    """
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Variável de ambiente OPENROUTER_API_KEY não encontrada")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompt = f"""
        Instruções:

        1. Divida o texto em seções temáticas com base no conteúdo discutido.
        2. Identifique e destaque os tópicos principais.
        3. Organize o texto em parágrafos curtos e claros, evitando blocos de texto longos.
        4. Remova repetições e frases desnecessárias, mantendo apenas o conteúdo relevante.
        5. Adicione títulos e subtítulos para cada seção, utilizando formatação em markdown.
        6. Destaque termos técnicos ou importantes usando negrito.
        7. Mantenha a linguagem natural, mas corrija erros gramaticais ou frases confusas.
        8. Inclua exemplos ou listas quando necessário para melhorar a clareza.
        9. Crie um fluxograma usando a sintaxe Mermaid que represente a sequência lógica dos principais tópicos e suas relações.
           - Use nós para representar os tópicos principais
           - Use setas para mostrar a relação entre os tópicos
           - Inclua breves descrições nas conexões quando relevante

        Texto a ser processado:
        {input_text}
        """

        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )

        if not completion or not completion.choices or len(completion.choices) == 0:
            raise ValueError("Resposta inválida da API do OpenRouter")

        return completion.choices[0].message.content

    except Exception as e:
        print(f"Erro ao processar texto com OpenRouter: {str(e)}")
        return None
