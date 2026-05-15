import ast
import asyncio
import logging

from typing import Annotated, List, Optional, TypedDict, Union

from langchain_core.runnables import RunnableSequence
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from langfuse.langchain import CallbackHandler

from .prompt_template import QueryExpandTemplate, StepBackPromptingTemplate

logger = logging.getLogger(__name__)


class QueryRewriteState(TypedDict):
    query: str
    num_queries: int
    rewrite_queries: Annotated[list, add_messages]


class QueryRewriter:
    """
    Rewrites user queries using techniques like query expansion and step-back
    generation

    Attributes
    ----------
    lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
        The LangChain LLM client used for generating queries
    config: dict
        Configuration dictionary, including Langfuse callbacks if provided
    query_rewrite_chain: RunnableSequence
        Chain that expands an input query into multiple variations
    step_back_rewrite_chain: RunnableSequence
        Chain that generates a broader, "step-back" version of the query
    rewrite_workflow: CompiledStateGraph
        The compiled LangGraph workflow managing the rewrite execution
    """

    __slots__ = (
        "lc_client",
        "config",
        "query_rewrite_chain",
        "step_back_rewrite_chain",
        "rewrite_workflow",
    )

    def __init__(
        self,
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama],
        lf_handler: Optional[CallbackHandler],
    ):
        """
        Initializes the QueryRewriter chains and graph workflow.

        Parameters
        ----------
        lc_client: Union[ChatDeepSeek, ChatOpenAI, ChatOllama]
            The LangChain LLM client.
        lf_handler: Optional[CallbackHandler]
            Optional Langfuse handler for tracing LLM interactions.
        """
        # Init llm client and langfuse for LLM tracing
        self.lc_client = lc_client
        self.config = {"callbacks": [lf_handler]} if lf_handler else {}

        # Define runnable chain for query rewrite and step back rewrite
        self.query_rewrite_chain: RunnableSequence = (
            QueryExpandTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "QueryExpandRewrite"})

        self.step_back_rewrite_chain: RunnableSequence = (
            StepBackPromptingTemplate().formulate() | self.lc_client
        ).with_config({"run_name": "StepBackRewrite"})

        # Define query rewrite workflow
        self.rewrite_workflow = self.build_workflow()

    async def query_expansion(self, state: QueryRewriteState) -> QueryRewriteState:
        """
        LangGraph node that expands the input query into multiple variations

        Parameters
        ----------
        state: QueryRewriteState
            The current graph state containing the 'query' and 'num_queries'

        Returns
        -------
        QueryRewriteState
            A partial state dict mapping 'rewrite_queries' to the generated list
        """
        try:
            response = await self.query_rewrite_chain.ainvoke(
                {"query": state["query"], "num_queries": state["num_queries"]},
                config=self.config,
            )
            queries = ast.literal_eval(response.content)

            return {
                "rewrite_queries": queries if isinstance(queries, list) else [queries]
            }

        except asyncio.TimeoutError as e:
            logger.error("Query expansion timed out for query: '%s'")
            raise

        except Exception as e:
            logger.exception("Unexpected error during query expansion: %s", e)
            raise

    async def step_back_query(self, state: QueryRewriteState) -> QueryRewriteState:
        """
        LangGraph node that generates a "step-back" version of the input query

        Parameters
        ----------
        state: QueryRewriteState
            The current graph state containing the 'query'

        Returns
        -------
        QueryRewriteState
            A partial state dict mapping 'rewrite_queries' to the step-back query
        """
        try:
            response = await self.step_back_rewrite_chain.ainvoke(
                {"question": state["query"]}, config=self.config
            )
            step_back_query = response.content

            return {
                "rewrite_queries": (
                    step_back_query
                    if isinstance(step_back_query, list)
                    else [step_back_query]
                )
            }

        except asyncio.TimeoutError as e:
            logger.error(
                "Step-back query generation timed out for query: '%s'", state["query"]
            )
            raise

        except Exception as e:
            logger.exception(
                "Unexpected error during step-back query generation: %s", e
            )
            raise

    def build_workflow(self) -> CompiledStateGraph:
        """
        Builds and compiles the parallel query rewriting state graph

        Returns
        -------
        CompiledStateGraph
            The compiled LangGraph workflow
        """
        builder = StateGraph(QueryRewriteState)
        builder.add_node("query_expansion", self.query_expansion)
        builder.add_node("step_back_query", self.step_back_query)

        builder.add_edge(START, "query_expansion")
        builder.add_edge(START, "step_back_query")
        builder.add_edge("query_expansion", END)
        builder.add_edge("step_back_query", END)

        workflow = builder.compile()

        return workflow

    async def arewrite(self, query: str, num_queries: Optional[int] = 3) -> List[str]:
        """
        Asynchronously executes the rewrite workflow (expansion + step-back).

        Parameters
        ----------
        query: str
            The original user query to rewrite
        num_queries: Optional[int], default=3
            The number of expanded query variations to generate

        Returns
        -------
        List[str]
            A flattened list of all generated query variations
        """
        try:
            response = await self.rewrite_workflow.ainvoke(
                {"query": query, "num_queries": num_queries},
                config=self.config,
            )

            return [query.content for query in response["rewrite_queries"]]

        except asyncio.TimeoutError as e:
            logger.error("Query rewrite timed out for query: '%s'", query)
            raise

        except Exception as e:
            logger.exception("Unexpected error during query rewrite: %s", e)
            raise

    def rewrite(self, query: str, num_queries: Optional[int] = 3) -> List[str]:
        """
        Synchronous wrapper for `arewrite`

        Parameters
        ----------
        query : str
            The original user query to rewrite
        num_queries : Optional[int], default=3
            The number of expanded query variations to generate

        Returns
        -------
        List[str]
            A flattened list of all generated query variations
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        rewrite_queries = loop.run_until_complete(self.arewrite(query, num_queries))

        return rewrite_queries
