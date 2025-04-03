import json
import logging

from typing import List, Optional, Any
from pydantic import BaseModel

from langchain_core.outputs import Generation
from langchain_core.output_parsers import BaseGenerationOutputParser


class JsonContentModel(BaseModel):
    content: Any


class StructureJsonOutputParser(BaseGenerationOutputParser[str]):
    """
    A parser that format LLM output into as a Json string
    """

    def parse_result(
        self, messages: List[Generation], *, partial: Optional[bool] = False
    ) -> str:
        """
        Parse the LLM output into a JSON formatted string

        Parameters
        ----------
        messages : List[Generation]
            A list of `Generation` objects representing the LLM's output. The
            list is assumed to contain one generation
        partial : Optional[bool]
            Whether to allow partial results. Defaults to False. This is
            currently not used but can be extended to support streaming outputs

        Returns
        -------
        str
            A JSON string containing the LLM's response under the "content" key
        """
        logging.info(f"Raw output from LLM: {messages[0].message.content}")
        return JsonContentModel(content=json.loads(messages[0].message.content))

    async def aparse_result(self, result: List[Generation], *, partial=False):
        """
        Asynchronously parse the output from the LLM into a JSON-formatted string

        Parameters
        ----------
        result: List[Generation]
            A list of `Generation` objects representing the LLM's output. The
            list is assumed to contain one generation
        partial : Optional[bool]
            Whether to allow partial results. Defaults to False. This is
            currently not used but can be extended to support streaming outputs

        Returns
        -------
        str
           A JSON string containing the LLM's response under the "content" key
        """
        logging.info(f"Raw output from LLM: {result[0].message.content}")
        return JsonContentModel(content=json.loads(result[0].message.content))
