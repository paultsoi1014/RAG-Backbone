from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class DocumentParseRefineTemplate:
    """
    A class for constructing a chat prompt template that refines parsed markdown
    documents into plain text
    """

    @staticmethod
    def _system_message_template() -> str:
        """Defines the system message template for refining parsed document content"""
        persona = (
            "You are a detail-oriented and efficient writer, focused on "
            "transforming markdown documents into clear, plain text. Your "
            "objective is to rewrite content without retaining any markdown "
            "syntax such as '#' or '*', and present tables and lists as "
            "descriptive, well-structured paragraphs. You prioritize "
            "readability, accuracy, and logical flow while retaining the "
            "original meaning of the content."
        )

        guideline = (
            "Step-by-Step Guide:\n\n"
            "1. Identify Key Information: Focus on the main ideas in each "
            "section, and ensure essential details are included in the "
            "rewritten text.\n\n"
            "2. Rewrite Tables into Paragraphs: Interpret the information in "
            "each table and rewrite it into a short, coherent paragraph that "
            "retain the original meaning.\n\n"
            "3. Combine Bullet Points into Paragraphs: Merge related ideas from "
            "bullet points into a single, cohesive paragraph. Do not use bullet "
            "points in the rewritten text.\n\n"
            "4. Remove Markdown Syntax: Strip all markdown-specific symbols, "
            "such as '#', '**', '-', and '|', while maintaining the context.\n\n"
            "5. Preserve Original Meaning: Ensure that the rewritten content "
            "accurately conveys the same ideas and details as the original "
            "document.\n\n"
            "6. Simplify and Clarify: Rewrite in a concise and clear manner, "
            "using simple language while keeping all critical information "
            "intact."
        )

        system_message = (
            f"{persona}\n\n"
            "Your goal is to rewrite a document with bullet points into plain "
            "text, while converting the bullet points into concise, clear "
            "paragraphs without losing any important information. Follow the "
            "following step-by-step guides to rewrite the document\n\n"
            f"Step-by-Step Guide: {guideline}\n\n"
            "Warning: You are NOT allow to output any table or markdown specific "
            "symbols. Please understand the context and rewrite it into plain "
            "text. Please treat your job seriously. You will be tipped $1000 "
            "dollar if you successfully achieve the task."
        )
        return system_message

    @staticmethod
    def _human_message_template() -> str:
        """
        Defines the human message template for the document transformation task
        """
        human_message = (
            "Below is a markdown document that needs to be converted into plain "
            "text. Please rewrite it by removing all markdown formatting, "
            "integrating tables and bullet points into descriptive paragraphs, "
            "and ensuring clarity and logical structure. Only output the "
            "rewritten document without brackets or markdown syntax.\n\n"
            "Document: {document}\n\n"
            "Rewritten Document:"
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Constructs and returns a ChatPromptTemplate for document refinement
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
