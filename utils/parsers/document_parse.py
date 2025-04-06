import glob
import logging
import os
import requests

from config import PARSER_URL
from typing import Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langfuse.callback import CallbackHandler

from utils.parsers.prompt_template import DocumentParseRefineTemplate


class DocumentParser:
    __slots__ = ["lf_handler", "parser_mapper", "refinement_chain", "parser_url"]

    def __init__(
        self,
        lc_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI],
        lf_handler: CallbackHandler,
    ):
        """
        A class to parse and process documents of various file types (txt, pdf)

        Attributes
        ----------
        lf_handler: langfuse.callback.CallbackHandler
            Handler for logging and tracing LLM interactions via Langfuse
        parser_mapper: dict
            A dictionary mapping file extensions to their corresponding parser
            methods
        parser_url: str
            The base URL of the remote parsing API server

        Parameters
        ----------
        lc_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]
            A langchain client instance for interacting with the language model
        """
        # langfuse handler
        self.lf_handler = lf_handler

        # parser format mapper
        self.parser_mapper = {"txt": self._txt_parser, "pdf": self._pdf_parser}

        # Init langchain LLM client for parsed document refinement
        self.refinement_chain = (
            DocumentParseRefineTemplate().formulate() | lc_client
        ).with_config({"run_name": "DocumentParseRefinement"})

        # init and check the api server for Philo Document Parser
        self.parser_url = PARSER_URL
        self._check_server_accessibility()

    def parse(self, folder_path: str) -> Tuple[str, list]:
        """
        Parses all files in the specified folder and combines their content into
        a single string

        Parameters
        ----------
        folder_path: str
            The path to the folder containing the files to be processed

        Returns
        -------
        Tuple[str, list]
            The combined content of all successfully parsed files
        """
        # Retrieve all the file in the folder
        files = glob.glob(f"{folder_path}/*")
        logging.info(f"{len(files)} files is going to be processed")

        # Start document parsing
        combined_text = ""
        for file in files:
            # Extract file extension
            file_extension = os.path.basename(file).split(".")[-1]

            # File parsing
            content = self.parser_mapper[file_extension](file)

            if not isinstance(content, str):
                raise ValueError(f"Failed to parse file: {file}")

            # Concate the parsed content
            combined_text += self.parser_mapper[file_extension](file) + "\n"

        return combined_text

    def _parse_refinement(self, context: str) -> str:
        """
        Refines the extracted document content using a language model

        Parameters
        ----------
        context : str
            The raw extracted text from a document

        Returns
        -------
        str
            The refined text with improved structure and clarity
        """
        return self.refinement_chain.invoke(
            {"document": context}, config={"callbacks": [self.lf_handler]}
        ).content

    def _check_server_accessibility(self):
        """
        Checks if the server is accessible by sending a GET request to the
        specified URL
        """
        api_url = os.path.join(self.parser_url, "health")

        try:
            response = requests.get(api_url, timeout=5)

            # Check if the response is 200 (ok)
            if response.status_code == 200:
                logging.info("Document Parser Server is accessible!")
            else:
                logging.warning(
                    f"Document Parser Server responded with status code: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            logging.warning("Failed to connect to the Document Parser server")

        except requests.exceptions.Timeout:
            logging.warning("The request for Document Parser is timed out")

        except requests.exceptions.RequestException as e:
            logging.warning(f"An error occurred in Document Parser: {e}")

    def _pdf_parser(self, filepath: str) -> str:
        """
        Parses the content of a PDF file using the remote API.

        Parameters
        ----------
        filepath: str
            The path to the PDF file

        Returns
        -------
        str
            The extracted text from the PDF file
        """
        # api endpoint
        api_url = os.path.join(self.parser_url, "parse/pdf")

        # Open the PDF file in binary mode
        with open(filepath, "rb") as file:
            # Send a POST request with the file
            context = requests.post(api_url, files={"file": file})

        # Parsed result refine
        refined_context = self._parse_refinement(context.json()["text"])

        return refined_context

    def _txt_parser(self, filepath: str) -> str:
        """
        Parses the content of a plain text file.

        Parameters
        ----------
        filepath: str
            The path to the text file

        Returns
        -------
        str
            The content of the text file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        return text
