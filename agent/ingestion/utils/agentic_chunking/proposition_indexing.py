import ast
import asyncio
import logging

from typing import List, Optional, Union

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langfuse.langchain import CallbackHandler

from .prompt_template import PropositionIndexingTemplate

logger = logging.getLogger(__name__)


class PropositionIndexingPipeline:
    """
    A pipeline designed to deconstruct text into atomic propositions for
    improved semantic indexing in RAG (Retrieval-Augmented Generation) systems.

    Attributes
    ----------
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        The LangChain-compatible LLM client used for text transformation
    config: dict
        Execution configuration containing callbacks for tracing and logging.
    proposition_indexing_chain: RunnableSequence
        The composed LangChain sequence (Prompt | LLM) that executes
        the extraction logic
    """

    __slots__ = ("lc_client", "config", "proposition_indexing_chain")

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        lf_handler: Optional[CallbackHandler],
    ):
        """
        Initializes the pipeline with a specific LLM provider and optional
        Langfuse tracing

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            An initialized LangChain LLM client
        lf_handler: Optional[CallbackHandler]
            A Langfuse callback handler for observability and cost tracking
        """
        self.lc_client = lc_client
        self.config = {"callbacks": [lf_handler]} if lf_handler else {}

        # Define chain of operations
        self.proposition_indexing_chain = (
            PropositionIndexingTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "PropositionIndexing"})

    async def _get_propositions(self, prompt: str) -> List[str]:
        """
        Submits a single text segment to the LLM to extract individual
        propositions

        Parameters
        ----------
        prompt: str
            A segment of text (usually a paragraph) to be atomized

        Returns
        -------
        List[str]
            A list of atomic propositions extracted from the segment

        Raises
        ------
        SyntaxError
            If the LLM response is not a valid Python-literal list string
        """
        # Get propositions
        try:
            response = await self.proposition_indexing_chain.ainvoke(
                {"input": prompt}, config=self.config
            )

            # convert into list of string
            propositions_content = response.content
            propositions = ast.literal_eval(propositions_content)

            return propositions

        except Exception as e:
            logger.error(f"Error during proposition indexing: {e}")

    async def _aproposition_indexing(self, text: str) -> List[str]:
        """
        Orchestrates the concurrent extraction of propositions across multiple
        text paragraphs.

        Parameters
        ----------
        text: str
            The raw text content to be processed

        Returns
        -------
        List[str]
            A flattened list of all atomic propositions found in the text
        """
        # text preprocessing
        paragraphs = text.split("\n\n")

        # Extract propositions from text concurrently
        tasks = [self._get_propositions(para) for para in paragraphs]
        results = await asyncio.gather(*tasks)

        # Flatten results
        essay_propositions = []
        for propositions in results:
            essay_propositions.extend(propositions)

        logger.info(f"You have {len(essay_propositions)} propositions")
        return essay_propositions

    async def aindex(self, document: str) -> List[str]:
        """
        The primary entry point for asynchronously indexing a document into
        atomic propositions

        Parameters
        ----------
        document: str
            The full document string to process

        Returns
        -------
        List[str]
            A list of indexed propositions ready for vector store insertion

        Raises
        ------
        ValueError
            If there is a failure in parsing the LLM's structured output
        Exception
            For general failures during the LLM invocation or network requests
        """
        try:
            # Index data based on their proposition
            propositions = await self._aproposition_indexing(document)
            return propositions
        except ValueError as e:
            logging.error(f"Error parsing propositions: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during proposition indexing: {e}")
            raise
