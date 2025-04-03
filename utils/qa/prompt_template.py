from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class PromptTemplate:
    @staticmethod
    def define_persona(target_subject: str) -> str:
        """
        Defines a persona prompt for the language model to act as a teacher of the
        specified subject, creating examination questions and answers

        Parameters
        ----------
        target_subject: str
            The subject for which the persona will act as a teacher (e.g.,
            'Mathematics', 'Biology')

        Returns
        -------
        str
            A prompt instructing the language model to act as a teacher of the
            specified subject, with a focus on constructing exam questions and
            answers
        """
        persona_prompt = (
            f"I want you to act as a {target_subject} teacher. I will provide "
            f"some topics related to the study of {target_subject}, and it will "
            "be your job to construct examination question and answer for students "
            "to learn and test their capability."
        )

        return persona_prompt

    def formulate(self) -> ChatPromptTemplate:
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


class GeneralResponseTemplate(PromptTemplate):
    """
    A class to generate prompt templates for AI-based assistance. The templates
    define the structure of messages exchanged between the system and the user
    """

    @staticmethod
    def _system_message_template() -> str:
        """
        A static method that returns the system message template used to guide
        the AI assistant's behavior in responding to user queries based on provided
        context
        """
        system_message = (
            "You are a help AI assistant, please answer the query based on the "
            "given context information"
        )

        return system_message

    @staticmethod
    def _human_message_template() -> str:
        """
        A static method that returns the human message template, which includes
        the user's query and the context information
        """
        human_message = (
            "The following is the user query and the context information. Please "
            "treat the task seriously. You will be tipped $100 if you answer the "
            "question correctly and precisely\n\n"
            "User Query: {query}\n\n"
            "Information: {relevant_info}\n\n"
            "Response: "
        )

        return human_message


class GeneralStructureResponseTemplate(PromptTemplate):
    @staticmethod
    def _system_message_template() -> str:
        """
        A static method that returns the system message template used to guide
        the AI assistant's behavior in responding to user queries based on provided
        context
        """
        system_message = (
            "You are a help AI assistant, please answer the query based on the "
            "given context information"
        )

        return system_message

    @staticmethod
    def _human_message_template() -> str:
        """
        A static method that returns the human message template, which includes
        the user's query and the context information
        """
        template = """{{"response": <Your Response Here>}}"""

        human_message = (
            "The following is the user query and the context information. Please "
            "treat the task seriously. You will be tipped $100 if you answer the "
            "question correctly and precisely. The user query, information will "
            "be plaintest while the response will be formatted as JSON following "
            "the provided template. Do not output the bracket ``` to identify "
            "it is JSON output.\n\n"
            "User Query: {query}\n\n"
            "Information: {relevant_info}\n\n"
            f"Response Template: {template}\n\n"
            "Response: "
        )

        return human_message
