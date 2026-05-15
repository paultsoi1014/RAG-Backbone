import asyncio
import logging

from typing import List, Optional, Union

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langfuse.langchain import CallbackHandler

from .prompt_template import ContextualTemplate

logger = logging.getLogger(__name__)


class ContextualChunker:
    """
    An asynchronous contextual chunking processor that enriches text chunks with
    generated context using language models

    The contextual chunking approach enhances each chunk by prepending relevant
    context or summary information, making chunks more semantically meaningful
    when used in downstream tasks like retrieval or indexing

    Attributes
    ----------
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        An instance of the LangChain chat model client.
    config: dict
        Configuration dictionary containing callbacks (e.g., Langfuse handler)
        for the runnable chain
    runnable_chain: Runnable
        A configured processing chain for performing contextual chunking
    """

    __slots__ = ("lc_client", "config", "runnable_chain")

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        lf_handler: Optional[CallbackHandler],
    ):
        """
        Initializes the ContextualChunker with a language model client and
        an optional logging handler

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            A LangChain chat model instance used for generating contextual
            information for chunks.
        lf_handler: Optional[CallbackHandler]
            Optional Langfuse callback handler for monitoring, tracing, and
            logging LLM API calls. Pass None to disable logging.
        """
        # Init langchain LLM clients
        self.lc_client = lc_client

        # Init langfuse callback handler
        self.config = {"callbacks": [lf_handler]} if lf_handler else {}

        # Define runnable chain for contextual chunking
        self.runnable_chain = (
            ContextualTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "ContextualChunking"})

    async def _generate_context(self, document: str, chunk: str) -> str:
        """
        Asynchronously generates context for a given chunk of text from the
        document using a language model

        Parameters
        ----------
        document: str
            The entire document from which context is derived
        chunk: str
            The specific chunk for which context is generated

        Returns
        -------
        str
            The generated context relevant to the given chunk

        Raises
        ------
        Exception
            If the language model fails to generate context
        """
        try:
            # Get context from LLM response asynchronously
            response = await self.runnable_chain.ainvoke(
                {"document": document, "chunk": chunk}, config=self.config
            )
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate context for chunk: {e}")
            raise

    async def _generate_contextual_chunks(
        self, document: str, chunks: List[str]
    ) -> List[str]:
        """
        Asynchronously generates a list of contextualized chunks by appending
        generated context to each chunk

        Parameters
        ----------
        document: str
            The entire document from which context is derived
        chunks: List[str]
            A list of text chunks to contextualize

        Returns
        -------
        List[str]
            A list of contextualized chunks, each consisting of the generated
            context followed by the original chunk

        Raises
        ------
        Exception
            If the generation of contextual chunks fails
        """
        try:
            # Create tasks for all chunks to process them concurrently
            tasks = [
                self._generate_context(document=document, chunk=chunk)
                for chunk in chunks
            ]
            contexts = await asyncio.gather(*tasks)

            # Concatenate contexts with their respective chunks
            contextual_chunks = [
                f"{context}\n\n{chunk}" for context, chunk in zip(contexts, chunks)
            ]

            return contextual_chunks
        except Exception as e:
            logger.error(f"Failed to generate contextual chunks: {e}")
            raise

    def _merge_summary(self, chunks: List[str], chunk_summary: List[str]) -> List[str]:
        """
        Generates a list of contextualized chunks by merging the chunks with
        pre-generated chunk summaries (e.g., title and summary)

        Parameters
        ----------
        chunks: List[str]
            A list of text chunks to contextualize
        chunk_summary: List[str]
            A list of strings containing chunk titles and summaries

        Returns
        -------
        List[str]
            A list of contextualized chunks, each consisting of the chunk
            summary followed by the original chunk
        """
        contextual_chunks = []

        # Merge chunk and summary together
        for chunk, summary in zip(chunks, chunk_summary):
            contextualized_content = f"{summary}\n\n{chunk}"
            contextual_chunks.append(contextualized_content)

        return contextual_chunks

    async def achunk(
        self,
        document: str,
        chunks: List[str],
        chunks_summary: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Asynchronously processes the document and chunks to produce a list of
        contextualized chunks

        Parameters
        ----------
        document: str
            The entire document to be processed.
        chunks: List[str]
            A list of text chunks to be contextualized.
        chunks_summary: Optional[List[str]], default=None
            An optional list of text including the title and summary from
            agentic chunking. If provided, overrides dynamic context generation

        Returns
        -------
        List[str]
            A list of contextualized chunks

        Raises
        ------
        Exception
            If contextual chunk generation fails
        """
        logger.info("Contextual chunking is enabled. Start contextual chunking")

        try:
            if chunks_summary:
                contextual_chunks = self._merge_summary(
                    chunks=chunks, chunk_summary=chunks_summary
                )
            else:
                contextual_chunks = await self._generate_contextual_chunks(
                    document=document, chunks=chunks
                )

            logger.info("Contextual chunking successful")
            return contextual_chunks
        except Exception as e:
            logger.error(f"Contextual chunking failed: {e}")
            raise
