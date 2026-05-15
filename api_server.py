import uvicorn

from fastapi import FastAPI

from config import AppConfig
from controller.ingest import router as ingest_router
from controller.retrieve import router as retrieve_router
from controller.search_relevancy import router as search_router

# Initialize FastAPI app
app = FastAPI()
app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(search_router)

api_config = AppConfig.from_env()

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=api_config.api_port)
