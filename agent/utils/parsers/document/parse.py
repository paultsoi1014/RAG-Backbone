import glob
import json
import logging
import os
import requests

from agent.base import RAGAgentBase

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parses and processes documents of various file types (e.g., txt, pdf).

    This class routes files to the appropriate parsing strategy based on their
    extension, handling local plain text extraction and remote API-based PDF
    parsing

    Attributes
    ----------
    parser_mapper: dict
        A dictionary mapping file extensions to their corresponding parser
        methods
    parser_url: str
        The full endpoint URL of the remote parsing API server
    """

    __slots__ = ("parser_mapper", "parser_url")

    def __init__(self, agent_base: RAGAgentBase):
        """
        Initializes the DocumentParser with routing configurations and API
        endpoints
        """
        # parser format mapper
        self.parser_mapper = {"txt": self._txt_parser, "pdf": self._pdf_parser}
        api_routes = {"marker": "marker-parser/parse", "vlm": "vlm-parser/pdf"}

        # init and check the api server for Philo Document Parser
        parser_settings = agent_base.get_parser_settings()
        parser_type = parser_settings["type"]
        route_path = api_routes.get(parser_type)

        self.parser_url = os.path.join(parser_settings["url"], route_path)

    @staticmethod
    def _join_natural_text(context: dict) -> str:
        """
        Joins all natural_text values from a dictionary of pages

        Parameters
        ----------
        context: dict
            Dictionary with page keys containing JSON strings

        Returns
        -------
        str
            A single string with all natural_text content joined together
        """
        texts = []

        # for page_key in context:
        for page_key in context:
            page_content = json.loads(context[page_key])
            if "natural_text" in page_content:
                texts.append(page_content["natural_text"])

        return "".join(texts)

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
        # PDF parsing via API
        try:
            with open(filepath, "rb") as file:
                # Send a POST request with the file
                response = requests.post(self.parser_url, files={"file": file})

        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(f"Request to {self.parser_url} timed out")
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Failed to connect to {self.parser_url}"
            )
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error during API request: {e}")

        # Extract the content
        try:
            parsed_content = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON response: {exc}")

        return parsed_content["processed_document"]

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

    def parse(self, folder_path: str) -> str:
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
        logger.info(f"{len(files)} files is going to be processed")

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
            combined_text += content + "\n"

        return combined_text
