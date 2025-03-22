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
        # Guia de Estruturação para Resumos Eficientes
        
        ## Estrutura Geral do Documento
        - Crie um documento estruturado dividido em seções claras e hierárquicas
        - Utilize hierarquia de títulos com até 3 níveis (# Título, ## Subtítulo, ### Tópico)
        - Adicione um sumário no início com links para navegação rápida entre seções
        - Mantenha o resumo conciso (aproximadamente 25% do conteúdo original)
        - Organize as informações em ordem lógica e progressiva
        
        ## Métodos de Estruturação Recomendados
        1. **Método Cornell**: Divida o conteúdo em três áreas:
           - Coluna da esquerda: Palavras-chave, conceitos e perguntas (1/3 da largura)
           - Coluna principal: Conteúdo detalhado e explicações (2/3 da largura)
           - Seção inferior: Sumário que sintetiza os pontos principais da página
        
        2. **Estrutura Hierárquica**:
           - Inicie com conceitos principais, dividindo em subtópicos
           - Utilize numeração automática e sistemática (1.1, 1.2, 2.1, etc.)
           - Mantenha consistência na profundidade dos tópicos
           - Agrupe informações relacionadas sob o mesmo tópico
        
        3. **Estrutura em Mapa Mental**:
           - Conceito principal no centro do documento
           - Ramificações primárias para temas principais
           - Ramificações secundárias para subtemas
           - Conexões visuais entre conceitos relacionados
        
        ## Elementos Visuais para Melhorar a Estrutura
        - **Listas numeradas**: Para sequências, processos, ou hierarquias
        - **Listas com marcadores**: Para itens sem ordem específica
        - **Tabelas**: Para comparações ou dados estruturados
        - **Diagramas** (usando Mermaid.js): Para visualização de processos e relações
        - **Blocos de destaque**: Para enfatizar informações críticas
        
        ## Formatação para Clareza Estrutural
        - Use **negrito** para termos-chave e conceitos importantes
        - Aplique *itálico* para ênfase e definições
        - Utilize `código` para termos técnicos ou comandos
        - Empregue > citações para referências importantes
        - Aplique destaque de cores para categorização visual:
          * ✓ Conceitos confirmados/corretos
          * ⚠️ Pontos de atenção ou controverérsia
          * ✗ Erros comuns ou conceitos incorretos
        
        ## Estrutura Específica de Saída
        Organize seu resumo com as seguintes seções estruturais:
        
        1. **TÍTULO PRINCIPAL**: Nome claro do tema principal (heading nível 1)
        
        2. **CONCEITO PRINCIPAL**: 
           - Definição clara e concisa do tema central
           - Evite mais de 3-5 linhas nesta seção
        
        3. **CAUSAS/ORIGENS**: 
           - Liste fatores principais numerados por importância
           - Mantenha formato consistente para cada item
        
        4. **EVIDÊNCIAS/DADOS**: 
           - Apresente dados concretos e mensuráveis
           - Use formatação visual para números e estatísticas
        
        5. **IMPACTOS/CONSEQUÊNCIAS**: 
           - Divida por categorias ou áreas afetadas
           - Use marcadores consistentes para facilitar leitura
        
        6. **SOLUÇÕES/APLICAÇÕES**: 
           - Organize em abordagens práticas e teóricas
           - Numere por prioridade ou eficácia
        
        7. **PERGUNTAS-CHAVE**: 
           - Formule 3-5 perguntas para revisão e reflexão
           - Assegure que cubram diferentes níveis de complexidade
        
        Aplique esta estrutura ao seguinte texto, mantendo formatação Markdown adequada:
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
