from typing import Tuple
from llama_index.core import VectorStoreIndex


class SimpleRetriever:
    __slots__ = ["retriever"]

    def __init__(self, retriever: VectorStoreIndex):
        """
        A simple retriever class that interacts with a vector store and an LLM
        client to retrieve relevant information from a vector database based on
        a given query

        Attributes
        ----------
        retriever: VectorStoreIndex
            The vector store index used to perform similarity-based retrieval
        """
        self.retriever = retriever

    def retrieve(self, query: str) -> Tuple[list, str]:
        """
        Retrieves relevant information from the vector database based on the
        query's similarity and returns both the raw nodes and a concatenated
        string of their contents

        Parameters
        ----------
        query: str
            The input query string to search for in the vector database

        Returns
        -------
        Tuple[list, str]
            A tuple where the first element is a list of nodes retrieved from the
            vector database, and the second element is a string containing the
            concatenated content of those nodes
        """
        # Run retrieval from vector database
        nodes = self.retriever.retrieve(query)

        # Retrieve text content from each node
        text = "\n\n".join(
            [node_with_scores.node.get_content() for node_with_scores in nodes]
        )

        return nodes, text
