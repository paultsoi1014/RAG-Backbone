from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class ContextualTemplate:
    """
    A template class for generating context for new chunks for contextual
    retrieval using the COSTAR framework.
    """

    @staticmethod
    def _system_message_template() -> str:
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are an expert document architect. You are working on a "
            "'Contextual Retrieval' system where small chunks of text are "
            "indexed for vector search. To improve retrieval accuracy, each "
            "chunk needs a short preamble that situates it within the larger "
            "document."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Analyze the provided document and a specific chunk from it. Your "
            "goal is to write a 2-3 sentence 'situating context' for that chunk.\n"
            "1. Identify the overarching topic of the document.\n"
            "2. Define the specific sub-topic or finding discussed in the chunk.\n"
            "3. Retain key figures, dates, or percentages that make the chunk "
            "unique."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Direct and informative. Avoid meta-talk. Never use phrases like "
            "'This chunk discusses' or 'This section provides'. Start the "
            "response immediately with the context."
        )

        # Tone - T
        tone = "=== TONE ===\n" "Professional, technical, and precise."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output is for an embedding model and an LLM. It must be "
            "optimized for semantic search relevance."
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Provide ONLY the 2-3 sentence context string. No preamble, no "
            "post-script, and no markdown formatting like 'Context:' labels."
        )

        # Combine all COSTAR components
        return (
            f"{context}\n\n"
            f"{objective}\n\n"
            f"{style}\n\n"
            f"{tone}\n\n"
            f"{audience}\n\n"
            f"{response}"
        )

    @staticmethod
    def _human_message_template() -> str:
        human_message = (
            "DOCUMENT:\n"
            "<document>\n"
            "{document}\n"
            "</document>\n\n"
            "CHUNK TO SITUATE:\n"
            "<chunk>\n"
            "{chunk}\n"
            "</chunk>\n\n"
            "Generate the situating context:"
        )
        return human_message

    @classmethod
    def formulate(cls) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate.from_template(
            cls._system_message_template()
        )
        human_message = HumanMessagePromptTemplate.from_template(
            cls._human_message_template()
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])
