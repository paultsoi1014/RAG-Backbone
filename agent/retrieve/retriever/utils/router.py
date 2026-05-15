import ast
import logging

from typing import List, Optional, Union

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

from .prompt_template import GetCollectionTemplate

logger = logging.getLogger(__name__)


class CollectionsRouter:
    """
    Routes user queries to the most relevant document collections using an LLM

    Attributes
    ----------
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        The LangChain client used to generate and evaluate routing prompts
    config: dict
        Configuration dictionary, including Langfuse callbacks if provided
    runnable_chain: RunnableSequence
        The compiled LangChain pipeline for collection routing
    """

    __slots__ = ("lc_client", "config", "runnable_chain")

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        lf_handler: Optional[CallbackHandler],
    ):
        """
        Initializes the CollectionsRouter and its evaluation chain

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            The LangChain client used to generate and evaluate prompts
        lf_handler: Optional[CallbackHandler]
            Handler for logging and tracing LLM interactions via Langfuse
        """
        self.lc_client = lc_client
        self.config = {"callbacks": [lf_handler]} if lf_handler else {}

        # Init chain for collection routing
        self.runnable_chain = (
            GetCollectionTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "GetRelevantCollections"})

    def get_relevant_collections(
        self,
        collections: List[str],
        queries: Union[str, List[str]],
        metadata: Optional[dict] = None,
    ) -> List[str]:
        """
        Retrieves relevant document collections based on the provided query

        Parameters
        ----------
        collections: List[str]
            A list of collections retrieved from the vector database
        queries: Union[str, List[str]]
            The user queries to determine relevant collections
        metadata: Optional[dict], default=None
            Additional metadata to assist in collection selection

        Returns
        -------
        List[str]
            A list of relevant document collections based on the query
        """
        # Get response from LLM
        response = self.runnable_chain.invoke(
            {
                "collections": collections,
                "queries": queries,
                "metadata": metadata.get("tags", []),
            },
            config=self.config,
        )

        # Parse response to get collection names
        return ast.literal_eval(response.content)

    async def aget_relevant_collections(
        self,
        collections: List[str],
        queries: Union[str, List[str]],
        metadata: Optional[dict] = None,
    ) -> List[str]:
        """
        Asynchronously retrieves relevant document collections based on the
        query

        Parameters
        ----------
        collections: List[str]
            A list of collections retrieved from the vector database
        queries: Union[str, List[str]]
            The user queries to determine relevant collections
        metadata: Optional[dict], default=None
            Additional metadata to assist in collection selection

        Returns
        -------
        List[str]
            A list of relevant document collections based on the query
        """
        # Get response from LLM
        try:
            response = await self.runnable_chain.ainvoke(
                {
                    "collections": collections,
                    "queries": queries,
                    "metadata": metadata.get("tags", []),
                },
                config=self.config,
            )

            # Parse response to get collection names
            return ast.literal_eval(response.content)

        except Exception as e:
            logger.error(f"Error when executing collections routing: {e}")
