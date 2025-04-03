import logging
import re

import numpy as np

from typing import Optional, List, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.vector_stores.qdrant import QdrantVectorStore


class SemanticChunker:
    __slots__ = [
        "vector_store",
        "storage_context",
        "embed_model",
        "enable_contextual_chunking",
        "breakpoint_percentile_threshold",
        "buffer_size",
    ]

    def __init__(
        self,
        vector_store: Union[QdrantVectorStore],
        storage_context: StorageContext,
        embed_model: Union[
            FastEmbedEmbedding, HuggingFaceEmbedding, OllamaEmbedding, OpenAIEmbedding
        ],
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
    ):
        """
        A class for segmenting and embedding document text into semantically
        meaningful chunks.

        The SemanticChunker class provides methods to split a document into
        sentences, combine sentences into contextually relevant groups, compute
        sentence embeddings, and chunk sentences based on similarity for further
        ingestion into a vector store index

        Attributes
        ----------
        vector_store : Union[QdrantVectorStore]
            Vector storage backend for storing document embeddings
        storage_context : StorageContext
            Context for storage configuration and management
        embed_model : Union[FastEmbedEmbedding, HuggingFaceEmbedding,
                      OllamaEmbedding, OpenAIEmbedding]
            Embedding model to generate vector representations of sentences
        breakpoint_percentile_threshold : Optional[int]
            Percentile threshold for sentence similarity, used to segment chunks
        buffer_size : Optional[int]
            Number of sentences to include before and after each sentence for
            combining
        """
        self.vector_store = vector_store
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = buffer_size

    def _build_chunks(self, sentences: list, distances: list) -> list:
        """
        Segments sentences into chunks based on similarity distances and a
        specified threshold

        Parameters
        ----------
        sentences : list
            A list of sentence extracted from the document
        distances : list
            A list of cosine distances between consecutive sentences

        Returns
        -------
        list
            A list of sentence groups where each group represents a semantically
            meaningful chunk
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
        sentences : list
            A list of sentence dictionaries containing sentence embeddings

        Returns
        -------
        Tuple[list, list]
            A list of cosine distances between each pair of consecutive sentences
            and The updated list of sentence dictionaries with distance metadata
            added
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
        Combining sentence to reduce noise and capture more of the relationships
        between sequential sentences

        Parameters
        ----------
        sentences: List[dict]
            A list of dictionary containing each of the sentence and the index number

        Returns
        -------
        List[dict]
            The list of sentences updated with combined sentence information
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

    def _get_sentence_embedding(self, sentences: List[dict]) -> List[dict]:
        """
        Computes embeddings for each sentence using the embedding model

        Parameters
        ----------
        sentences : List[dict]
            A list of dictionaries, each containing a sentence

        Returns
        -------
        List[dict]
            The list of sentences with added embeddings
        """
        # Compute embeddings for all sentences
        embeddings = [
            self.embed_model.get_text_embedding(sentence["combined_sentence"])
            for sentence in sentences
        ]

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
            An ingestion pipeline configured for the specified vector store
        """
        return IngestionPipeline(
            vector_store=self.vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
            docstore=SimpleDocumentStore(),
        )

    @staticmethod
    def _sentence_splitter(document: str):
        """
        Splits a document into individual sentences and stores them in dictionary
        format

        Parameters
        ----------
        document : str
            The document text to be split

        Returns
        -------
        sentences : List[dict]
            A list of dictionaries, each containing a sentence and its index
        """
        # Split document content into list of single sentences
        sentence_list = re.split(r"(?<=[.?!])\s+", document)
        logging.info(f"{len(sentence_list)} sentences were found")

        # Convert sentence into list of dictionary format
        sentences = [{"sentence": x, "index": i} for i, x in enumerate(sentence_list)]

        return sentences

    def __call__(self, document: str) -> VectorStoreIndex:
        """
        Processes a document through the chunking pipeline, creates chunks, and
        ingests them into a vector store

        Parameters
        ----------
        document : str
            The document text to be processed.

        Returns
        -------
        VectorStoreIndex
            An index of the ingested document chunks stored in the vector store
        """
        # Separate document into  multiple sentences
        sentences = self._sentence_splitter(document)

        # Sentence combine to reduce noise and capture relationship between sentence
        sentences = self._combine_sentence(sentences)

        # Embed the sentence
        sentences = self._get_sentence_embedding(sentences=sentences)

        # Calculate cosine distance of each sentences
        distances, sentences = self._calculate_cosine_distances(sentences)

        # Get chunks for each of the sentence
        chunks = self._build_chunks(sentences, distances)

        # Load data from an iterable of strings to Document type #TODO: Reorganize later
        chunk_documents = StringIterableReader().load_data(chunks)

        # Define ingestion pipeline and run
        nodes = self._pipeline().run(documents=chunk_documents, show_progress=True)
        logging.info(f"Ingested {len(nodes)} Nodes")

        # build llama index
        return VectorStoreIndex.from_documents(
            chunk_documents, storage_context=self.storage_context
        )
