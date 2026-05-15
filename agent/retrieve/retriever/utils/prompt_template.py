from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class GetCollectionTemplate:
    """
    Prompt builder for finding all relevant collections based on the questions
    asking for, revised to follow the COSTAR framework
    """

    def _system_message_template(self) -> str:
        # Context (C)
        context = (
            "=== CONTEXT ===\n"
            "You are an intelligent retrieval assistant designed to optimize "
            "data querying within a RAG (Retrieval Augmented Generation) system. "
            "You have access to a registry of data 'Collections' and must "
            "determine which ones contain information relevant to a user query."
        )

        # Objective (O)
        objective = (
            "=== OBJECTIVE ===\n"
            "Your task is to analyze the User queries and Metadata to identify "
            "the most relevant Collections. Follow this reasoning process:"
            "\n\n"
            "1. Extract Core Concepts: Identify the main subject(s) and entities "
            "in the query, stripping away question words (what, how, when, "
            "where, why), articles (a, an, the), and auxiliary verbs (is, are, "
            "do, does). Focus on the substantive content—nouns, key terms, and "
            "technical domain-specific vocabulary.\n"
            "2. Parse Collection Names: Examine each collection name for "
            "meaningful content words that indicate what topics or information "
            "it contains. Collection names may include:\n"
            "- Descriptive labels (e.g., 'Chapter 4', 'Introduction to')\n"
            "- Multiple topic keywords combined together\n"
            "- Related concepts grouped in a single name\n"
            "Look for any words in the collection name that relate to subject "
            "matter.\n"
            "3. Apply Flexible Matching: Match query concepts to collections "
            "using:\n"
            "- Exact word matches (case-insensitive)\n"
            "- Morphological variations (singular ↔ plural, different verb forms)\n"
            "- Substring containment (query term appears within collection name "
            "words)\n"
            "- Root word matching (query 'biology' matches collection "
            "'biological')\n"
            "- Semantic relationships (synonyms, related concepts, broader topics)\n"
            "- Topical overlap (collections covering domains that include the "
            "query concept)\n"
            "4. Consider Information Hierarchy: Include collections that would "
            "logically contain answers to the query:\n"
            "- Direct topic matches (query about X, collection titled with X)\n"
            "- Broader categories (query about X, collection about field "
            "containing X)\n"
            "- Related domains (query about X, collection about Y where X is "
            "discussed)\n"
            "5. Be Inclusive with Uncertainty: When relevance is unclear or "
            "borderline, include the collection. Missing relevant information "
            "is worse than including extra collections for downstream filtering."
        )

        # Style (S)
        style = (
            "=== STYLE ===\n"
            "Analytical and inclusive. Prioritize comprehensive recall—favor "
            "including potentially relevant collections over strict exclusion. "
            "Use semantic understanding rather than rigid keyword matching."
        )

        # Tone (T)
        tone = "=== TONE ===\n" "Professional, objective, and computational."

        # Audience (A)
        audience = (
            "=== AUDIENCE ===\n"
            "Your output will be processed by a downstream automated database "
            "router. Precision and strict adherence to the output format are "
            "critical to prevent system errors"
        )

        # Response (R)
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY a valid list of strings containing the names of the "
            "relevant collections. Do not include markdown formatting like ``` "
            "or conversational filler.\n"
            'Example Output: ["collection_a", "collection_b"]'
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

    @staticmethod
    def _human_message_template():
        human_message = (
            "Available Collections:\n"
            "{collections}"
            "\n\n"
            "User Queries:\n"
            "{queries}"
            "\n\n"
            "**Metadata:**\n"
            "{metadata}"
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Combines system and human messages to form a chat prompt template

        Returns
        -------
        ChatPromptTemplate
            The fully constructed chat prompt template following COSTAR framework
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
