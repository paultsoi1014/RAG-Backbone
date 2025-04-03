from agent.ingestion.ingest import RAGAgentIngest
from agent.general_response.response import RAGAgentResponse


class RAGAgent(RAGAgentIngest, RAGAgentResponse):
    def __init__(self):
        """
        A comprehensive agent class that integrates capabilities for both
        document ingestion and response generation within a Retrieval-Augmented
        Generation (RAG) framework
        """
        RAGAgentIngest.__init__(self)
        RAGAgentResponse.__init__(self)
