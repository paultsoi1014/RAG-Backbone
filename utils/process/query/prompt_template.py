import json

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pathlib import Path


class PromptTemplate:
    """
    A base template class that provides methods to generate few-shot prompt
    formats and retrieve few-shot examples for use in various prompt templates
    """

    @staticmethod
    def _read_json(filepath: str) -> dict:
        """
        Read a JSON file from the specified filepath and return its contents

        Parameters
        ----------
        filepath: str
            The path to the JSON file to be read

        Returns
        -------
        dict
            The contents of the JSON file parsed into a Python dictionary
        """
        with open(filepath, "r") as file:
            json_file = json.load(file)

        return json_file

    @staticmethod
    def _few_shot_prompt_format() -> ChatPromptTemplate:
        """
        Generates a few-shot prompt format

        Returns
        -------
        ChatPromptTemplate
            A chat prompt template with human-AI interaction format
        """
        few_shot_message = [("human", "{input}"), ("ai", "{output}")]

        return ChatPromptTemplate.from_messages(few_shot_message)

    def _get_few_shot_example(self, keys: str) -> dict:
        """
        Retrieves a few-shot example for the specified key from a JSON file

        Parameters
        ----------
        keys : str
            The key corresponding to the few-shot example in the JSON file

        Returns
        -------
        dict
            The few-shot example mapped to the provided key
        """

        few_shot_example = self._read_json(
            f"{Path(__file__).parent}/few_shot_examples.json"
        )

        return few_shot_example.get(keys)


class KeywordExtractionTemplate(PromptTemplate):

    @staticmethod
    def _system_message_template():
        """Defines the system message template for keyword extraction"""
        system_message = (
            "You are an AI assistant designed to extract the most relevant "
            "keywords from the provided query. These keywords should capture "
            "the in depth concepts of the query and in english. Your output "
            "should be in noun phrase, string format and each keyword is in "
            "lowercase and separated by commas. DO NOT output anything other "
            "than the keywords. DO NOT output any keywords in Chinese even "
            "though the user query are in Chinese. You will be given $1000 if "
            "you successfully done the job."
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """Defines the human message template for keyword extraction"""
        human_message = (
            "Your task is to extract at most 2 relevant keywords in english "
            "from the query. Please strictly following the example format. "
            "Below are the query for your reference:\n\n"
            "Query: {query}\n"
            "Keywords: "
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Creates a chat prompt template by combining system and human messages
        for keyword extraction

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for extracting keywords
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            self._system_message_template()
        )

        # Define human message
        human_message = HumanMessagePromptTemplate.from_template(
            self._human_message_template()
        )

        # Define few shot prompt template and example
        few_shot_prompt_example = self._get_few_shot_example("keyword_extraction")
        few_shot_prompt_format = self._few_shot_prompt_format()
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_prompt_format, examples=few_shot_prompt_example
        )

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, few_shot_prompt, human_message]
        )

        return chat_prompt_template


class KeywordSearchTemplate(PromptTemplate):
    @staticmethod
    def _output_format_template() -> str:
        template = """
            {{
                "keywords": ["Insert keywords here"]
            }}
        """

        return template

    def _system_message_template(self):
        # Define output formate template
        output_template = self._output_format_template()

        # Define step by step guideline on how to extract keywords
        guideline = (
            "# Step 1: Analyze the User Query - Carefully read the user query "
            "and understand what they're asking about in relation to exam "
            "question generation.\n\n"
            "# Step 2: Identify Relevant Keywords - Select keywords from the "
            "provided keyword list that are most relevant to the underlying "
            "topics and concepts discussed in the user query and/or provided "
            "topics. Ignore the information from user query or provided topics "
            "if it is in empty string \n\n"
            "# Step 3: Prioritize Accuracy - Focus on selecting keywords that "
            "accurately reflect the meaning and context of the user query. Do "
            "not add keywords that are not in the provided list.\n\n"
            "# Step 4: Provide the selected keywords as a comma-separated list.\n\n"
            "The most relevant keywords should be placed at the front while "
            "the least relevant keywords should be placed at the last. If no "
            "keywords from the list are relevant, output an empty list. The "
            "output should be in JSON following the provided template"
            "should be in double quotes."
        )

        system_message = (
            "You are an advanced keyword extraction AI. Your task is to analyze "
            "a user query and/or topics, and extract all relevant keywords "
            "based on a predefined keyword list. Follow the following steps to "
            "extract all relevant keywords:\n"
            f"{guideline}\n\n"
            "Here is an example template to guide you:\n"
            f"{output_template}\n\n"
            "YOU ARE ONLY REQUIRED TO OUTPUT KEYWORDS IN JSON FORMAT."
            "You will be penalize $1000 if keywords cannot be retrieved from "
            "the given keyword list."
        )

        return system_message

    @staticmethod
    def _human_message_template():
        human_message = (
            "Extract relevant keywords from the user query or topics that "
            "relate to generating exam questions. Only use keywords from the "
            "provided keyword list.\n\n"
            "Keyword List: {keywords}\n\n"
            "User Query: {query}\n\n"
            "Topics: {topics}\n\n"
            "Extracted Keywords:"
        )

        return human_message

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


class QueryExpandTemplate(PromptTemplate):
    """
    A class that creates a prompt template for generating multiple related search
    queries from a single input query
    """

    @staticmethod
    def _system_message_template():
        """Defines the system message template for generating multiple search queries"""

        system_message = (
            "You are a helpful assistant that generates multiple search queries "
            "based on a single input query."
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """Defines the human message template that asks for a specific number of
        related search queries"""

        human_message = (
            "Generate {num_queries} search queries, one on each line, related to "
            "the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )
        return human_message

    def formulate(self):
        """
        Creates a chat prompt template by combining system and human messages
        for query expansion

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating related search queries
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


class QueryLanguageClassificationTemplate(PromptTemplate):
    @staticmethod
    def _system_message_template():
        """Defines the system message template that classify the language of user
        query"""
        system_message = (
            "You are a helpful assistant that classify the language of the user "
            "query, please output the language either in 'traditional_chinese' "
            "or 'english'"
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """Defines the human message template that classify the language of user
        query"""

        human_message = (
            "Detect and classify the language of the user query it is either "
            "'english' or 'chinese'. "
            "PLEASE PRINT ONLY THE DETECTED LANGUAGE WITHOUT ANY OTHER MESSAGE"
            "The following is the input query:\n"
            "Query: {query}\n"
            "Detected Language: \n"
        )
        return human_message

    @classmethod
    def formulate(cls):
        """
        Creates a chat prompt template by combining system and human messages
        for query expansion

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating related search queries
        """
        # Define system message
        system_message = SystemMessagePromptTemplate.from_template(
            cls._system_message_template()
        )

        # Define human message
        human_message = HumanMessagePromptTemplate.from_template(
            cls._human_message_template()
        )

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        return chat_prompt_template


class QueryTranslateTemplate:
    @staticmethod
    def _system_message_template():
        """Defines the system message template to translate user query"""
        system_message = (
            "You are a helpful assistant that try to translate user input to "
            "target language. Please only output the best translated sentence. "
            "Please treat the job seriously and don't lose the original meaning "
            "from users input. The output must maintain the structure of the "
            "user input. If you are dealing with dictionary, please translate "
            "the values only and do not translate the keys. "
            ""
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """Defines the human message template that translate the user query"""

        human_message = (
            "Please translate the user input to {language}. "
            "PLEASE PRINT ONLY THE OUTPUT WITHOUT ANY OTHER MESSAGE "
            "The following is the user input:\n"
            "User Input: {query}\n"
            "Output:\n"
        )
        return human_message

    @classmethod
    def formulate(self):
        """
        Creates a chat prompt template by combining system and human messages
        for query expansion

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating related search queries
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


class QuestionTranslateTemplate:
    @staticmethod
    def _system_message_template():
        """Defines the system message template to translate generated questions"""
        system_message = (
            "You are a helpful assistant that try to translate generated "
            "question to target language. Please only output the best "
            "translated sentence. Please treat the job seriously and don't loss "
            "the original meaning, and JSON structure. The output must maintain "
            "the structure of the question. Please translate the values only "
            "and do not translate the keys. Do not translate the answer if you "
            "are working on 'tf' true-false questions. Otherwise, please "
            "translate the answers."
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """Defines the human message template that translate the generated
        questions to target language"""

        human_message = (
            "Please translate the user input to {language}. "
            "PLEASE OUTPUT THE TRANSLATED QUESTION WITHOUT ANY OTHER MESSAGE "
            "LIKE 'Question:' IN THE BEGINNING. "
            "The following is the question to be translated:\n"
            "Question: {question}\n"
            "Translated Question: \n\n"
        )
        return human_message

    @classmethod
    def formulate(self):
        """
        Creates a chat prompt template by combining system and human messages
        for query expansion

        Returns
        -------
        ChatPromptTemplate
            A complete chat prompt template for generating related search queries
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


class StepBackPromptingTemplate(PromptTemplate):
    """
    A class that creates a prompt template for step-back versions of the users
    queries
    """

    @staticmethod
    def _system_message_template():
        """Defines the system message template for step-back question generation"""
        system_message = (
            "You are an expert at world knowledge. Please only generate step-back "
            "version of this question without generating samples"
        )
        return system_message

    @staticmethod
    def _human_message_template():
        """
        Defines the human message template asking for a paraphrased, more generic
        version of the question
        """
        human_message = (
            "Your task is to step back and paraphrase a question to a more generic "
            "step-back question, which is easier to answer. Here are the question:\n\n"
            "Question: {question}"
        )
        return human_message

    def formulate(self) -> ChatPromptTemplate:
        """
        Creates a chat prompt template by combining system and human messages,
        along with few-shot examples for the step-back question generation task

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

        # Define few shot prompt template and example
        few_shot_prompt_example = self._get_few_shot_example("step_back_prompting")
        few_shot_prompt_format = self._few_shot_prompt_format()
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_prompt_format, examples=few_shot_prompt_example
        )

        # Define chat prompt template
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, few_shot_prompt, human_message]
        )

        return chat_prompt_template
