# Start qdrant vector db server
docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant

