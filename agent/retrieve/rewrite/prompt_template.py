from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class QueryExpandTemplate:
    """
    A class that creates a prompt template for generating multiple related search
    queries from a single input query, formulated using the COSTAR framework
    """

    @staticmethod
    def _system_message_template() -> str:
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are an expert Information Retrieval assistant specializing in "
            "query expansion. Your purpose is to help search systems uncover "
            "deeper and more comprehensive results by diversifying user search "
            "intents."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Generate {num_queries} distinct, highly relevant search queries "
            "based on a single input query. The generated queries should explore "
            "related concepts, synonyms, or different facets of the original "
            "query to maximize search coverage."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Concise, analytical, and strict. Provide no conversational filler, "
            "reasoning, or explanations."
        )

        # Tone - T
        tone = "=== TONE ===\n" "Objective and professional."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output will be parsed directly by a downstream Python "
            "application that strictly expects a valid Python list of strings."
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY a valid list of strings format containing exactly "
            "{num_queries} queries.\n"
            'Example format: ["`query 1`", "`query 2`"]\n'
            "Do not include markdown blocks (e.g., ```json), labels, or any text "
            "outside the list structure."
        )

        # Combine all COSTAR components
        system_message = (
            f"{context}\n\n"
            f"{objective}\n\n"
            f"{style}\n\n"
            f"{tone}\n\n"
            f"{audience}\n\n"
            f"{response}"
        )

        return system_message

    def _human_message_template(self) -> str:
        """
        Defines the human message template providing the actual query to expand.
        """
        human_message = (
            "Input Query to Expand: \n"
            "{query}\n\n"
            "Generate {num_queries} related search queries for the input above."
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Creates a chat prompt template by combining system and human messages
        for query expansion.

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating related search queries.
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            self._system_message_template()
        )

        # Define human message
        human_message = HumanMessagePromptTemplate.from_template(
            self._human_message_template()
        )

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        return chat_prompt_template


class StepBackPromptingTemplate:
    """
    A class that creates a prompt template for generating 'step-back' versions
    of user queries using the COSTAR framework.
    """

    @staticmethod
    def _system_message_template() -> str:
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are an expert at world knowledge and query abstraction. Your role "
            "is to assist a retrieval system by generating 'step-back' questions— "
            "broader, more fundamental versions of specific user queries."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Paraphrase the specific input question into a more generic step-back "
            "question. \n"
            "1. Identify the core entities and specific constraints in the input.\n"
            "2. Remove the specific constraints (dates, specific locations, "
            "minor details).\n"
            "3. Formulate a broader question that asks about the general history, "
            "nature, or capabilities of the entity to retrieve foundational "
            "context."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Concise, analytical, and strict. Provide no conversational filler, "
            "reasoning, or explanations."
        )

        # Tone - T
        tone = "=== TONE ===\n" "Objective and inquisitive."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output is for a retrieval engine that needs to find high-level "
            "background documents before answering specific questions."
        )

        # Examples - E
        examples = (
            "=== EXAMPLES ===\n"
            "# Example 1\n"
            "Input: Could the members of The Police perform lawful arrests?\n"
            "Output: what can the members of The Police do?"
            "\n\n"
            "# Example 2\n"
            "Input: Jan Sindel's was born in what country?\n"
            "Output: what is Jan Sindel's personal history?"
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Output ONLY the step-back question string.\n"
            "Do not output labels like 'Step-back:' or conversational text."
        )

        # Combine all COSTAR components
        system_message = (
            f"{context}\n\n"
            f"{objective}\n\n"
            f"{style}\n\n"
            f"{tone}\n\n"
            f"{audience}\n\n"
            f"{examples}\n\n"
            f"{response}"
        )

        return system_message

    @staticmethod
    def _human_message_template() -> str:
        """Defines the human message template asking for the step-back abstraction"""
        human_message = (
            "Original Question:\n"
            "{question}"
            "\n\n"
            "Generate the step-back question:"
        )

        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Creates a chat prompt template by combining system and human messages.

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating step-back questions
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            self._system_message_template()
        )

        # Define human message
        human_message = HumanMessagePromptTemplate.from_template(
            self._human_message_template()
        )

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        return chat_prompt_template
