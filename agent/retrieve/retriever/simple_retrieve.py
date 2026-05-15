import asyncio

from typing import Tuple, Union, List
from llama_index.core import VectorStoreIndex

from .utils.repack import repack_document


class SimpleRetriever:
    __slots__ = ("retriever", "repacking_strategy")

    def __init__(self, retriever: VectorStoreIndex, repacking_strategy: str):
        """
        A simple retriever class that interacts with a vector store and an LLM
        client to retrieve relevant information from a vector database based on
        a given query

        Attributes
        ----------
        retriever: VectorStoreIndex
            The vector store index used to perform similarity-based retrieval
        repacking_strategy: str
            The strategy used for packing or ordering retrieved document chunks
            within the context window
        """
        self.retriever = retriever
        self.repacking_strategy = repacking_strategy

    def retrieve(self, queries: Union[str, List[str]]) -> Tuple[list, str]:
        """
        Retrieves relevant information from the vector database based on the
        query's similarity and returns both the raw nodes and a concatenated
        string of their contents

        Parameters
        ----------
        queries: str
            The input query string to search for in the vector database

        Returns
        -------
        Tuple[list, str]
            A tuple where the first element is a list of nodes retrieved from the
            vector database, and the second element is a string containing the
            concatenated content of those nodes
        """
        queries = [queries] if isinstance(queries, str) else queries

        all_nodes = []
        seen_node_ids = set()

        # Run retrieval from vector database for each query
        for q in queries:
            nodes = self.retriever.retrieve(q)
            for node in nodes:
                # Deduplicate nodes to prevent context bloat
                if node.node_id not in seen_node_ids:
                    all_nodes.append(node)
                    seen_node_ids.add(node.node_id)

        # Apply repacking strategy
        repacked_nodes, repacked_text = repack_document(
            seen_node_ids, self.repacking_strategy
        )

        return repacked_nodes, repacked_text

    async def aretrieve(self, queries: str) -> Tuple[list, str]:
        """
        Asynchronously Retrieves relevant information from the vector database
        based on the query's similarity and returns both the raw nodes and a
        concatenated string of their contents

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
        queries = [queries] if isinstance(queries, str) else queries

        all_nodes = []
        seen_node_ids = set()

        # Run retrieval from vector database concurrently for all queries
        tasks = [self.retriever.aretrieve(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Process and deduplicate the nested results
        for nodes in results:
            for node in nodes:
                chunking_index = node.metadata.get("chunk_index")

                if chunking_index not in seen_node_ids:
                    all_nodes.append(node)
                    seen_node_ids.add(chunking_index)

        # Apply repacking strategy
        repacked_nodes, repacked_text = repack_document(
            all_nodes, self.repacking_strategy
        )

        return repacked_nodes, repacked_text
