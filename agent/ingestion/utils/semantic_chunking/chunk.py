import asyncio
import logging
import re

import numpy as np

from typing import Optional, List, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langfuse.langchain import CallbackHandler

from agent.ingestion.utils.base_chunker import BaseChunker
from agent.ingestion.utils.contextual_chunking import ContextualChunker
from vector_db import ElasticVectorDB, QdrantVectorDB

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Segments and embeds document text into semantically meaningful chunks.

    This class splits a document into sentences, combines them into contextually
    relevant groups using a rolling window, computes sentence embeddings, and
    chunks the text based on semantic similarity distances for ingestion into
    a vector store

    Attributes
    ----------
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        The LangChain language model client used for processing and generating
        chunk metadata
    embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding, OllamaEmbedding, OpenAIEmbedding]
        The embedding model used to generate vector representations of sentences
    lf_handler: CallbackHandler
        Handler for logging and tracing LLM interactions via Langfuse
    vector_store: Union[ElasticVectorDB, QdrantVectorDB]
        The vector store backend used for storing document embeddings
    storage_context: StorageContext
        The LlamaIndex storage context for the ingestion pipeline
    breakpoint_percentile_threshold: int
        The percentile threshold for sentence similarity distance, used to
        segment chunks
    buffer_size: int
        The number of sentences to include before and after each sentence to
        provide context
    enable_contextual_chunking: bool
        Flag indicating whether secondary LLM-based contextual chunking is
        enabled
    """

    __slots__ = (
        "breakpoint_percentile_threshold",
        "buffer_size",
        "embed_model",
        "enable_contextual_chunking",
        "lc_client",
        "lf_handler",
    )

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        embed_model: Union[
            FastEmbedEmbedding, HuggingFaceEmbedding, OllamaEmbedding, OpenAIEmbedding
        ],
        lf_handler: CallbackHandler,
        vector_db: Union[ElasticVectorDB, QdrantVectorDB],
        collection_name: str,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
        enable_contextual_chunking: Optional[bool] = True,
    ):
        """
        Initializes the SemanticChunker with required models and database
        configurations

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            The LangChain LLM client
        embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                      OllamaEmbedding, OpenAIEmbedding]
            The embedding model used to encode text
        lf_handler: CallbackHandler
            Langfuse callback handler for tracing
        vector_db: Union[ElasticVectorDB, QdrantVectorDB]
            The vector database client for indexing and retrieval
        collection_name: str
            The base name of the collection where documents will be stored
        breakpoint_percentile_threshold: Optional[int], default=95
            Percentile threshold for cosine distance to trigger a new chunk
        buffer_size: Optional[int], default=1
            Number of neighboring sentences to combine for context windowing
        enable_contextual_chunking: Optional[bool], default=True
            Whether to apply an additional layer of contextual chunking via LLM
        """
        super().__init__(vector_db, collection_name, "_semantic")

        # Init langchain LLM clients & langfuse handler for llm tracing
        self.lc_client = lc_client
        self.lf_handler = lf_handler

        # Initialize embedding model
        self.embed_model = embed_model

        # Init llama index vector store and storage context to store chunks
        revised_collection_name = f"{collection_name}_semantic"
        self.vector_store, self.storage_context = vector_db.init_ingestion(
            revised_collection_name
        )

        # Initialize semantic chunking specific parameters
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = buffer_size

        # Enable contextual chunking or not
        self.enable_contextual_chunking = enable_contextual_chunking

    def _build_chunks(self, sentences: list, distances: list) -> list:
        """
        Segments sentences into chunks based on semantic distance breakpoints

        Parameters
        ----------
        sentences: list
            A list of sentence dictionaries extracted from the document
        distances: list
            A list of cosine distances between consecutive sentences

        Returns
        -------
        list
            A list of strings, where each string is a semantically meaningful
            chunk
        """
        chunks = []
        if len(distances) > 0:
            # Get the distance threshold that considered as outliers
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            # Get the index of distances that are above the threshold.
            indices_above_thresh = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            # Iterate through the breakpoints to slice the sentences
            for index in indices_above_thresh:
                group = sentences[start_index : index + 1]
                combined_text = " ".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                # Update the start index for next group
                start_index = index + 1

            # For the last groups
            if start_index < len(sentences):
                combined_text = " ".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)

        else:
            # treat the whole document as a single node if distance cannot be obtained
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks

    @staticmethod
    def _calculate_cosine_distances(sentences: list) -> Tuple[list, list]:
        """
        Computes cosine distances between consecutive sentence embeddings

        Parameters
        ----------
        sentences: list
            A list of sentence dictionaries containing sentence embeddings

        Returns
        -------
        Tuple[list, list]
            - A list of cosine distances between each pair of consecutive sentences
            - The updated list of sentence dictionaries with 'distance_to_next' added
        """
        distances = []
        for i in range(len(sentences) - 1):
            # Retrieve embedding for current and next sentences
            sentence_embedding_current = sentences[i]["combined_sentence_embedding"]
            sentence_embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            # Calculate cosine similarity
            similarity = cosine_similarity(
                [sentence_embedding_current], [sentence_embedding_next]
            )[0][0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    def _combine_sentence(self, sentences: List[dict]) -> List[dict]:
        """
        Combines sentences with their neighbors to capture broader context
        and reduce noise before embedding.

        Parameters
        ----------
        sentences: List[dict]
            A list of dictionaries, each containing a sentence and its index

        Returns
        -------
        List[dict]
            The list of dictionaries updated with a 'combined_sentence' key
        """
        for i in range(len(sentences)):
            # Hold all sentences which are joined
            combined_sentence = ""

            # Add sentence before the current one based on the buffer size
            for j in range(i - self.buffer_size, i):
                # Check if it the index out of range
                if j >= 0:
                    # Add sentence at index j to combined sentence
                    combined_sentence += sentences[j]["sentence"] + " "

            # Add the current sentence
            combined_sentence += sentences[i]["sentence"]

            # Add sentence after the current one based on the buffer size
            for j in range(i + 1, i + 1 + self.buffer_size):
                # Check whether j is within the range of sentence list
                if j < len(sentences):
                    # Add the sentence at index j to the combined sentence
                    combined_sentence += " " + sentences[j]["sentence"]

            # Store the combined sentence in the current sentence dict
            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    async def _aget_sentence_embedding(self, sentences: List[dict]) -> List[dict]:
        """
        Asynchronously computes embeddings for each contextually combined
        sentence

        This parallelizes API calls to the embedding model using asyncio.gather
        to optimize network I/O

        Parameters
        ----------
        sentences: List[dict]
            A list of dictionaries, each containing a 'combined_sentence'

        Returns
        -------
        List[dict]
            The list of sentences updated with a 'combined_sentence_embedding'
            key
        """
        tasks = [
            self.embed_model.aget_text_embedding(sentence["combined_sentence"])
            for sentence in sentences
        ]

        # Compute embeddings concurrently
        embeddings = await asyncio.gather(*tasks)

        # Assign each embedding to the corresponding sentence dictionary
        for sentence, embedding in zip(sentences, embeddings):
            sentence["combined_sentence_embedding"] = embedding

        return sentences

    def _pipeline(self) -> IngestionPipeline:
        """
        Defines the ingestion pipeline configuration for vector store insertion

        Returns
        -------
        IngestionPipeline
            An ingestion pipeline configured for the specific vector store
            instance
        """
        return IngestionPipeline(
            vector_store=self.vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            docstore=SimpleDocumentStore(),
        )

    @staticmethod
    def _sentence_splitter(document: str):
        """
        Splits a raw document string into individual sentences using regex

        Parameters
        ----------
        document: str
            The raw document text to be split

        Returns
        -------
        List[dict]
            A list of dictionaries, each containing the 'sentence' text and its
            'index'
        """
        # Split document content into list of single sentences
        sentence_list = re.split(r"(?<=[.?!])\s+", document)
        logger.info(f"{len(sentence_list)} sentences were found")

        # Convert sentence into list of dictionary format
        sentences = [{"sentence": x, "index": i} for i, x in enumerate(sentence_list)]

        return sentences

    async def achunk(self, document: str) -> None:
        """
        Executes the asynchronous semantic chunking pipeline on a document and
        ingests the resulting chunks into the vector store.

        Parameters
        ----------
        document: str
            The raw text of the document to be processed
        """
        # Separate document into  multiple sentences
        sentences = self._sentence_splitter(document)

        # Sentence combine to reduce noise and capture relationship between sentence
        sentences = self._combine_sentence(sentences)

        # Embed the sentence
        sentences = await self._aget_sentence_embedding(sentences=sentences)

        # Calculate cosine distance of each sentences
        distances, sentences = self._calculate_cosine_distances(sentences)

        # Get chunks for each of the sentence
        chunks = self._build_chunks(sentences, distances)

        # Check if conduct contextual chunking
        if self.enable_contextual_chunking:
            # Contextual chunking
            contextual_chunker = ContextualChunker(
                self.lc_client, lf_handler=self.lf_handler
            )
            chunks = await contextual_chunker.achunk(document=document, chunks=chunks)

        self.process_and_ingest(chunks=chunks)
        logger.info("Semantic chunking successful")
