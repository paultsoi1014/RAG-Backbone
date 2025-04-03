import asyncio
from typing import Optional, List, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from langfuse.callback import CallbackHandler

from utils.process.query.rewrite import QueryRewrite


class FusionRetriever:
    __slots__ = ["retrievers", "top_n", "k", "query_rewrite"]

    def __init__(
        self,
        lc_client: Union[ChatAnthropic, ChatOpenAI],
        lf_handler: CallbackHandler,
        retrievers: List[VectorStoreIndex],
        top_n: Optional[int] = 2,
        k: Optional[float] = 60.0,
    ):
        """
        An ensemble retriever that combines results from multiple retrievers
        using reciprocal rank fusion. The class performs asynchronous queries to
        multiple vector store retrievers and fuses the results based on their
        ranks and scores

        Attributes
        ----------
        retrievers: List[VectorStoreIndex]
            A list of retrievers to perform retrievals from
        top_n: Optional[int]
            The number of top results to return after fusion. Default is 2
        k: Optional[float]
            A constant to reduce the impact of low-ranked items, ensure
            high-ranked text contributes more to the final score

        Parameters
        ----------
        lc_client: Union[ChatAnthropic, ChatOpenAI]
            The langchain language model client used for query rewriting
        lf_handler : langfuse.callback.CallbackHandler
            Handler for logging and tracing LLM interactions via Langfuse
        """
        self.retrievers = retrievers
        self.top_n = top_n
        self.k = k

        # Init query rewriter
        self.query_rewrite = QueryRewrite(lc_client=lc_client, lf_handler=lf_handler)

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
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result

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

    def _fuse_results(self, results: dict) -> Tuple[List[NodeWithScore], List[str]]:
        """
        Fuses the query results using reciprocal rank fusion and reranks them
        based on the calculated scores

        Parameters
        ----------
        results: dict
            The dictionary of query results to be fused and reranked

        Returns
        -------
        Tuple[List[NodeWithScore], List[str]]
            - A list of the top n reranked nodes
            - A string containing the concatenated content of the top n nodes
        """
        # Perform reciprocal rank fusion
        fused_scores, text_to_node = self._reciprocal_rank_fusion(results)

        # Adjust the node scores based on the fused scores
        reranked_nodes, reranked_results = self._reranking(fused_scores, text_to_node)

        # Return only top n result
        return reranked_nodes[: self.top_n], "\n\n".join(reranked_results[: self.top_n])

    async def _async_get_query_rewrite(
        self, query: str, num_queries: Optional[int] = 3
    ) -> List[str]:
        """
        Asynchronously generates rewritten queries for the given query

        Parameters
        ----------
        query: str
            The original query string to be rewritten
        num_queries : Optional[int]
            The number of rewritten queries to generate (default is 3)

        Returns
        -------
        List[str]
            A list of rewritten queries based on the original input query
        """
        return await self.query_rewrite.arewrite(query=query, num_queries=num_queries)

    def _get_query_rewrite(
        self, query: str, num_queries: Optional[int] = 3
    ) -> List[str]:
        """
        Generate alternative rewritten queries based on the original query

        Parameters
        ----------
        query: str
            The original query string to be rewritten
        num_queries : Optional[int]
            The number of rewritten queries to generate (default is 3)

        Returns
        -------
        List[str]
            A list of rewritten queries based on the original input query
        """
        return self.query_rewrite.rewrite(query=query, num_queries=num_queries)

    def _reranking(
        self, fused_score: dict, text_to_node: dict
    ) -> Tuple[List[NodeWithScore], List[str]]:
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
        Tuple[List[NodeWithScore], List[str]]
            - A list of reranked nodes based on the fusion score
            - A list of the corresponding text content, ordered by score
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

        # Get the retrieved text at the order after reranking
        reranked_text = list(reranked_results.keys())

        return reranked_nodes, reranked_text

    def retrieve(
        self, query: str, query_rewrite: Optional[bool] = True
    ) -> Tuple[List[NodeWithScore], List[str]]:
        """
        Retrieves results for a given query by querying all retrievers, fusing
        the results, and reranking them

        Parameters
        ----------
        query: str
            The input query string to retrieve information for
        query_rewrite: Optional[bool]
            Whether to rewrite the query (default is True)

        Returns
        -------
        Tuple[List[NodeWithScore], str]
            - A list of the top n reranked nodes
            - A concatenated string of the content from the top n nodes
        """
        # Create a new event loop for this thread if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Query rewrite
        if query_rewrite:
            query_revised = loop.run_until_complete(
                self._async_get_query_rewrite(query=query)
            )
        else:
            query_revised = [query]

        # Retrieve context from vector db each each retriever
        results = loop.run_until_complete(self._arun_queries(query_revised))

        # Retrieve information fusion
        fuse_node, fuse_text = self._fuse_results(results)

        return fuse_node, fuse_text
