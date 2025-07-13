import uuid

from langfuse.callback import CallbackHandler

from agent.base import RAGAgentBase
from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

from utils.ingestion.pipeline import RAGIngestionPipeline


class RAGAgentIngest(RAGAgentBase):
    def __init__(self):
        """
        A class that extends `RAGAgentBase` to handle the ingestion of
        documents into a vector database for later retrieval

        Attributes
        ----------
        qdrant_db: Union[QdrantVectorDB]
            An instance of the vector database for storing and retrieving
            document collections
        langchain_client: Union[ChatAnthropic, ChatDeepSeek, ChatGroq, ChatOpenAI]
            A client instance for interacting with the language model, inherited
            from `RAGAgentBase`
        llamaindex_embed_model: Union[FastEmbedEmbedding, HuggingFaceEmbedding
                                 OpenAIEmbedding, OllamaEmbedding]
            An instance of the LlamaIndexEmbedModel for generating embeddings
        langfuse_handler: langfuse.callback.CallbackHandler
            Handler for logging and tracing LLM interactions via Langfuse
        """
        RAGAgentBase.__init__(self)

        # Init langfuse handler for llm tracing & management
        self.langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            session_id=str(uuid.uuid4()),
        )

    def file_ingestion(self, folder_path: str, collection_name: str, mode: str):
        """
        Ingests files from a specified folder into the vector database

        This method processes documents from the specified folder, chunking them
        into recursively or semantically, and stores them in a vector database
        for later retrieval

        Parameters
        ----------
        folder_dir: str
            The directory containing the files to be ingested
        collection_name: str
            The name of the collection for the ingested documents
        mode: Optional[str]
            The mode of chunking to be used. Options are 'recursive' or 'semantic'.
            Default is using recursive chunking method
        """
        # init ingestion process of db
        vector_store, storage_context = self.vector_db.init_ingestion(collection_name)

        # Init RAG ingestion pipeline
        rag_ingestion = RAGIngestionPipeline(
            lc_client=self.langchain_client,
            li_embed_model=self.llamaindex_embed_model,
            lf_handler=self.langfuse_handler,
            vector_store=vector_store,
            storage_context=storage_context,
        )
        rag_ingestion(folder_path, mode=mode)
