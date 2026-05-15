import asyncio
import nest_asyncio
import itertools

from typing import Optional, List, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .utils.repack import repack_document

nest_asyncio.apply()


class FusionRetriever:
    """
    An asynchronous ensemble retriever that combines results from multiple
    vector store retrievers using reciprocal rank fusion (RBF)

    This retriever executes queries concurrently across multiple retrievers,
    fuses their results using rank-based scoring, and returns the top-ranked
    documents. The fusion process gives higher weight to documents that appear
    highly ranked across multiple retrievers

    Attributes
    ----------
    retrievers: List[VectorStoreIndex]
        List of vector store retrievers to query concurrently
    top_n: int
        Number of top-ranked results to return after fusion
    k: float
        Reciprocal rank fusion constant for score calculation
    repacking_strategy: str
        Strategy for repacking document chunks ('forward', 'reverse', or 'sides')
    """

    __slots__ = ("retrievers", "top_n", "k", "repacking_strategy")

    def __init__(
        self,
        retrievers: List[VectorStoreIndex],
        repacking_strategy: str,
        top_n: Optional[int] = 2,
        k: Optional[float] = 60.0,
    ):
        """
        An ensemble retriever that combines results from multiple retrievers
        using reciprocal rank fusion. The class performs asynchronous queries to
        multiple vector store retrievers and fuses the results based on their
        ranks and scores

        Parameters
        ----------
        retrievers: List[VectorStoreIndex]
            List of vector store retrievers to query. Each retriever should
            support asynchronous retrieval via the `aretrieve` method
        repacking_strategy: Optional[str]
            The strategy used for packing or ordering retrieved document chunks
            within the context window
            - 'forward': Original order
            - 'reverse': Reversed order
            - 'sides': Alternating from edges inward
        top_n: Optional[int]
            Number of top-ranked results to return after fusion and reranking.
            Default is 2
        k: Optional[float]
            Reciprocal rank fusion constant. Higher values reduce the impact of
            lower-ranked documents. The score for rank r is computed as 1/(r + k).
            Default is 60
        """
        self.retrievers = retrievers
        self.top_n = top_n
        self.k = k
        self.repacking_strategy = repacking_strategy

    async def _arun_queries(self, queries: List[str]) -> dict:
        """
        Asynchronously runs the given queries across all retrievers and returns
        their results in a dictionary

        Parameters
        ----------
        queries: List[str]
            A list of queries to run against the retrievers

        Returns
        -------
        dict
            A dictionary where the keys are query and index pairs, and the values
            are the corresponding retrieval results from each retriever
        """
        # Schedule all retrieval tasks asynchronously
        tasks = []
        for query in queries:
            for i, retriever in enumerate(self.retrievers):
                tasks.append(retriever.aretrieve(query))

        # Gather results asynchronously
        task_results = await asyncio.gather(*tasks)

        # Combine the results into a dictionary, with (query, index) as the key
        results_dict = {}
        for i, it in enumerate(itertools.product(queries, range(len(self.retrievers)))):
            results_dict[it] = task_results[i]

        return results_dict

    def _reciprocal_rank_fusion(self, results: dict) -> Tuple[dict, dict]:
        """
        Fuse the extracted content from vector database using reciprocal rank
        fusion which is through assigning a score based on its rank position

        Parameters
        ----------
        results: dict
            Dictionary of query results, where each key is a query and its
            associated index, and the value is a list of results from different
            retrievers

        Returns
        -------
        Tuple[dict, dict]
            - fused_scores: dict
                A dictionary where keys are the text content and values are
                their corresponding fusion scores
            - text_to_node: dict
                A mapping of the text and its corresponding node (with score)
        """
        fused_scores = {}
        text_to_node = {}

        # Iterate over all results for fusion
        for node_with_scores in results.values():
            for rank, node_with_score in enumerate(
                sorted(node_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                # Get text content of the node
                text = node_with_score.node.get_content()

                # map the text to its corresponding node
                text_to_node[text] = node_with_score

                # Initialize score if not already present
                if text not in fused_scores:
                    fused_scores[text] = 0.0

                # Apply reciprocal rank score with the 'k' adjustment
                fused_scores[text] += 1.0 / (rank + self.k)

        return fused_scores, text_to_node

    def _fuse_results(self, results: dict) -> List[NodeWithScore]:
        """
        Fuses the query results using reciprocal rank fusion and reranks them
        based on the calculated scores

        Parameters
        ----------
        results: dict
            The dictionary of query results to be fused and reranked

        Returns
        -------
        List[NodeWithScore]
            A list of the top n reranked nodes
        """
        # Perform reciprocal rank fusion
        fused_scores, text_to_node = self._reciprocal_rank_fusion(results)

        # Adjust the node scores based on the fused scores
        reranked_nodes = self._reranking(fused_scores, text_to_node)

        # Return only top n result
        return reranked_nodes[: self.top_n]

    def _reranking(self, fused_score: dict, text_to_node: dict) -> List[NodeWithScore]:
        """
        Rerank the retrieved node based on the calculated fusion score

        Parameters
        ----------
        fused_score: dict
            A dictionary contains the retrieved text and its corresponding score
        text_to_node: dict
            A dictionary contains the mapping between the retrieved text and its
            corresponding nodes

        Returns
        -------
        List[NodeWithScore]
            - A list of reranked nodes based on the fusion score
        """
        # Rerank the order of retrieved text based on score
        reranked_results = dict(
            sorted(fused_score.items(), key=lambda x: x[1], reverse=True)
        )

        # Adjust the order of node and its corresponding scores based on the fused scores
        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score

        return reranked_nodes

    async def aretrieve(
        self, queries: List[str]
    ) -> Tuple[List[NodeWithScore], List[str]]:
        """
        Asynchronously retrieve and fuse results for given queries

        Parameters
        ----------
        queries: List[str]
            List of query strings to retrieve information for

        Returns
        -------
        Tuple[List[NodeWithScore], List[str]]
            - List of top-n reranked nodes with fusion scores
            - List of repacked text content from the nodes
        """
        # Execute queries across all retrievers concurrently
        results = await self._arun_queries(queries)

        # Fuse and rerank results
        fused_nodes = self._fuse_results(results)

        # Repack documents using specified strategy
        repacked_nodes, repacked_text = repack_document(
            fused_nodes, self.repacking_strategy
        )

        return repacked_nodes, repacked_text

    def retrieve(self, queries: List[str]) -> Tuple[List[NodeWithScore], List[str]]:
        """
        Synchronous wrapper for aretrieve. Retrieves and fuses results for given
        queries

        Parameters
        ----------
        queries: List[str]
            List of query strings to retrieve information for

        Returns
        -------
        Tuple[List[NodeWithScore], List[str]]
            - List of top-n reranked nodes with fusion scores
            - List of repacked text content from the nodes
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.create_task(self.aretrieve(queries))
        except RuntimeError:
            return asyncio.run(self.aretrieve(queries))
