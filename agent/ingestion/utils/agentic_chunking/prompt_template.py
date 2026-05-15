from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from typing import Tuple


class FindRelevantChunkTemplate:
    """
    A template class for finding relevant chunks for propositions, revised to
    follow the COSTAR framework
    """

    @staticmethod
    def _system_message_template() -> str:
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are an intelligent classification assistant managing a "
            "knowledge base of 'chunks'. Each chunk represents a semantic group "
            "of sentences with a shared topic, direction, or intention."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Determine whether the provided 'Proposition' belongs to any of the "
            "existing chunks outlined in the context. Your goal is to group "
            "semantically similar items together.\n"
            "1. Analyze the meaning, direction, and intention of the Proposition.\n"
            "2. Compare it against the summaries and names of the Current Chunks.\n"
            "3. If a match is found, identify the Chunk ID.\n"
            "4. If no match is found, determine that it belongs to 'No chunks'."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Decisive, strict, and computational. Do not provide reasoning or "
            "explanations."
        )

        # TOne - T
        tone = "=== TONE ===\n" "Objective and binary."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output will be processed by a downstream python script that "
            "strictly expects a 5-letter ID string or the exact string 'No chunks'."
        )

        # Examples
        examples = (
            "=== EXAMPLES ===\n"
            "# Example 1"
            "Input Proposition: 'Greg really likes hamburgers'\n"
            "Current Chunks:\n"
            "- Chunk ID: 2n4l3d\n"
            "- Chunk Name: Places in San Francisco\n"
            "- Chunk Summary: Overview of the things to do with San Francisco Places"
            "\n\n"
            "- Chunk ID: 93833k\n"
            "- Chunk Name: Food Greg likes\n"
            "- Chunk Summary: Lists of the food and dishes that Greg likes\n"
            "Output: 93833k"
            "\n\n"
            "# Example 2"
            'Input Proposition: "The weather in London is rainy"\n'
            "Current Chunks:\n"
            "- Chunk ID: 93833k\n"
            "- Chunk Name: Food Greg likes\n"
            "- Chunk Summary: Lists of the food and dishes that Greg likes\n"
            "Output: No chunks"
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY the 5-letter Chunk ID if a match is found.\n"
            'Return ONLY the string "No chunks" if no match is found.\n'
            "Do not include markdown formatting, punctuation, or labels."
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
    def _human_message_template() -> Tuple[str, str]:
        """
        Constructs two human message templates that provide instructions for
        checking if a proposition belongs to a chunk
        """
        human_message_1 = (
            "**Current Chunks:**\n"
            "--Start of current chunks--\n"
            "{chunk_outline}\n"
            "--End of current chunks--"
        )

        human_message_2 = (
            "**Proposition to Evaluate:**\n"
            "{proposition}\n\n"
            "Determine if this proposition belongs to one of the chunks above."
        )

        return human_message_1, human_message_2

    def formulate(self) -> ChatPromptTemplate:
        """
        Combines system and human messages to form a chat prompt template
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            self._system_message_template()
        )

        # Define human messages
        _human_message_1, _human_message_2 = self._human_message_template()
        human_message_1 = HumanMessagePromptTemplate.from_template(_human_message_1)
        human_message_2 = HumanMessagePromptTemplate.from_template(_human_message_2)

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message_1, human_message_2]
        )

        return chat_prompt_template


class GetNewChunkTitleSummaryTemplate:
    """
    A template class for generating summaries for new chunks based on added
    propositions, revised to follow the COSTAR framework
    """

    @staticmethod
    def _system_message_template() -> str:
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are the steward of a dynamic information architecture. Your "
            "role is to maintain cohesive 'chunks' of information—semantic "
            "groups of sentences that share a common topic. As new atomic "
            "propositions are added, you must synthesize their identity into a "
            "concise title and a descriptive summary."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Your task is to analyze the provided proposition and generate two "
            "distinct outputs:"
            "\n\n"
            "1. **Chunk Summary**: A single sentence informing viewers what the "
            "chunk is about, while anticipating generalization (e.g., 'apples' "
            "-> 'food').\n"
            "2. **Chunk Title**: A very brief (2-5 words) label that "
            "encompasses the chunk's topic. It must be abstract enough to hold "
            "future related propositions (e.g., 'Date & Times' instead of "
            "'Friday Meetings')."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Concise, professional, and organizational. Avoid conversational "
            "filler. Focus strictly on semantic classification."
        )

        # Tone - T
        tone = "=== TONE ===\n" "Objective and administrative."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output will be parsed by an automated system, so strict "
            "adherence to the JSON schema is critical."
        )

        # Example - E
        examples = (
            "=== EXAMPLES ===\n"
            'Input Proposition: "Greg likes to eat pizza."\n'
            "Output:\n"
            "{{\n"
            '    "title": "Food Preferences",\n'
            '    "summary": "This chunk contains information about the types of '
            '                food Greg likes to eat."\n'
            "}}\n\n"
            'Input Proposition: "The meeting is scheduled for 3 PM on Friday."\n'
            "Output:\n"
            "{{\n"
            '    "title": "Dates & Times",\n'
            '    "summary": "This chunk aggregates details regarding schedules, '
            '                dates, and times for events."\n'
            "}}"
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY a valid JSON object with keys 'title' and 'summary'.\n"
            "Format:\n"
            "{{\n"
            '     "title": "<Your Title Here>", \n'
            '     "summary": "<Your Summary Here>"\n'
            "}}"
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
        """
        Creates a human message template asking for both title and summary
        """
        message = (
            "**New Proposition to Add:**\n"
            "{proposition}\n\n"
            "Generate the title and summary for the new chunk"
        )

        return message

    def formulate(self) -> ChatPromptTemplate:
        """
        Combines system and human messages to form a chat prompt template
        for chunk summary generation.

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


class PropositionIndexingTemplate:
    """
    A unified template class for decomposing text into atomic propositions and
    cleaning them in a single step, following the COSTAR framework
    """

    @staticmethod
    def _system_message_template():
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are an expert in Natural Language Processing and Text "
            "Normalization. Your role is to prepare raw text by breaking it "
            "down into 'atomic propositions'—sentences that are factual, "
            "self-contained, and chemically pure of formatting artifacts."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Perform a process transformation on the input text:"
            "\n\n"
            "1. Decomposition & Decontextualization:\n"
            "- Split compound sentences into simple, standalone sentences.\n"
            "- Resolve pronouns (e.g., 'he', 'it', 'they', 'this') by replacing "
            "them with the specific entity names they refer to.\n"
            "- Ensure every proposition makes sense without needing the "
            "surrounding context."
            "\n\n"
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Linguistic, precise, and syntactically simple. Maintain the "
            "original phrasing where possible, but prioritize clarity and "
            "independence of facts"
        )

        # Tone - T
        tone = "=== TONE ===\n" "Objective and analytical"

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output will be ingested by an embedding model for semantic "
            "search. Ambiguity, pronouns, and special characters will degrade "
            "search performance"
        )

        # Example - E
        examples = (
            "=== EXAMPLE ===\n"
            "Input:\n"
            "The earliest evidence for the Easter Hare was recorded in 1678 by "
            "Georg Franck von Franckenau, a professor of medicine. He noted "
            "that it remained unknown elsewhere until the 18th century. Later, "
            "German immigrants exported the custom to America where it evolved "
            "into the Easter Bunny."
            "\n\n"
            "Output:\n"
            "[\n"
            '    "The earliest evidence for the Easter Hare was recorded in '
            '     1678 by Georg Franck von Franckenau.",\n'
            '    "Georg Franck von Franckenau was a professor of medicine.",\n'
            '    "Georg Franck von Franckenau noted that the Easter Hare '
            '     remained unknown elsewhere until the 18th century.",\n'
            '    "German immigrants exported the custom of the Easter Hare to '
            '     America.",\n'
            '    "The custom of the Easter Hare evolved into the Easter Bunny '
            '     in America."\n'
            "]"
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY a valid Python list of strings, where each item is "
            "enclosed in double quotes.\n"
            "Do not include markdown formatting (e.g., ```).\n"
            "Do not include explanations, comments, or any text outside the list.\n"
            "Example format:\n"
            '["string 1", "string 2", "string 3"]'
        )

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
        human_message = (
            "**Content to Decompose and Refine:**\n"
            "{input}\n\n"
            "Return the list of atomic, cleaned propositions."
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Combines system and human messages to form a chat prompt template.
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


class UpdateChunkTitleSummaryTemplate:
    """
    A unified template class for updating BOTH the title and summary of an
    existing chunk when a new proposition is added. Follows the COSTAR framework
    """

    @staticmethod
    def _system_message_template():
        # Context - C
        context = (
            "=== CONTEXT ===\n"
            "You are the steward of a dynamic information architecture. Your "
            "role is to maintain cohesive 'chunks' of information—semantic "
            "groups of sentences that share a common topic. As new propositions "
            "are added to an existing chunk, you must refine its identity by "
            "updating its title and summary."
        )

        # Objective - O
        objective = (
            "=== OBJECTIVE ===\n"
            "Your task is to synthesize the 'Existing Chunk Details' with the "
            "'New Proposition' to generate two updated outputs:\n\n"
            "1. **Updated Summary**: A single sentence that accurately reflects "
            "the *combined* content of the chunk. It should be broad enough to "
            "encompass both the old and new data (e.g., generalizing 'apples' "
            "and 'oranges' to 'fruit').\n"
            "2. **Updated Title**: A concise (2-5 words) label that acts as a "
            "high-level header for the group. It must remain stable yet "
            "descriptive enough for retrieval."
        )

        # Style - S
        style = (
            "=== STYLE ===\n"
            "Concise, professional, and abstract. Avoid listing specific items; "
            "instead, use categorical terms (e.g., use 'inventory' instead of "
            "'list of items')."
        )

        # Tone - T
        tone = "=== TONE ===\n" "Objective and administrative."

        # Audience - A
        audience = (
            "=== AUDIENCE ===\n"
            "The output will be parsed by an automated system. Strict adherence "
            "to the JSON schema is required."
        )

        # Examples - E
        examples = (
            "=== EXAMPLES ===\n"
            "# Example 1:\n"
            "Input:\n"
            "- Current Title: Greg's Favs\n"
            "- Current Summary: This chunk lists foods Greg likes.\n"
            "- New Proposition: Greg also enjoys hiking in the mountains.\n"
            "Output:\n"
            "{{\n"
            '    "title": "Greg\'s Interests",\n'
            '    "summary": "This chunk aggregates information about Greg\'s food preferences and hobbies."\n'
            "}}"
            "\n\n"
            "# Example 2:\n"
            "Input:\n"
            "- Current Title: Meeting Times\n"
            "- Current Summary: Schedule of Monday meetings.\n"
            "- New Proposition: The project review is on Friday at 2 PM.\n"
            "Output:\n"
            "{{\n"
            '    "title": "Project Schedule",\n'
            '    "summary": "This chunk contains dates and times for all project-related meetings."\n'
            "}}"
        )

        # Response - R
        response = (
            "=== RESPONSE ===\n"
            "Return ONLY a valid JSON object with keys 'title' and 'summary'.\n"
            "Format:\n"
            "{{\n"
            '    "title": "<Updated Title>",\n'
            '    "summary": "<Updated Summary>"\n'
            "}}"
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
        """
        Creates a human message template asking for the updated title and summary
        """
        message = (
            "**Existing Chunk Details:**\n"
            "Title: {current_title}\n"
            "Summary: {current_summary}\n\n"
            "**New Proposition to Add:**\n"
            "{proposition}\n\n"
            "Generate the updated title and summary that encompasses all this "
            "information."
        )

        return message

    def formulate(self) -> ChatPromptTemplate:
        """
        Combines system and human messages to form a chat prompt template.
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            self._system_message_template()
        )

        # Define human message
        human_message = HumanMessagePromptTemplate.from_template(
            self._human_message_template()
        )

        # Define chat prompt template (No separate few-shot component needed)
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        return chat_prompt_template
