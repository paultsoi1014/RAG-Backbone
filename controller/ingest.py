import logging
import json
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import List

from agent.core import RAGAgent
from schema.ingestion import IngestionRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

rag_agent = RAGAgent()


@router.post("/file")
async def file_ingestion(
    files: List[UploadFile] = File(...), metadata: str = Form(...)
):
    """
    Ingest documents into vector database

    Parameters
    ----------
    request: IngestionRequest
        Contains the files to ingest, the target collection name, and
        ingestion mode

    Returns
    -------
    dict
        A success message if ingestion is completed successfully

    Raise
    -----
    HTTPException
        If any error occurs during the ingestion process
    """
    try:
        metadata_dict = json.loads(metadata)
        request = IngestionRequest.model_validate(metadata_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {str(e)}")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded files to temp directory
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()

            # Write file to temp folder
            with open(file_path, "wb") as buffer:
                buffer.write(content)

        # Process the files
        await rag_agent.afolder_ingestion(
            folder_path=temp_dir,
            collection_name=request.collection_name,
            mode=request.mode,
        )

        return {"status": "success", "message": "Files ingested successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/text")
async def text_ingestion(text: str, metadata: str = Form(...)):
    try:
        metadata_dict = json.loads(metadata)
        request = IngestionRequest.model_validate(metadata_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {str(e)}")

    try:
        await rag_agent.atext_ingestion(
            text=text,
            collection_name=request.collection_name,
            mode=request.mode,
        )

        return {"status": "success", "message": "Text ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
