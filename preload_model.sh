!#/bin/bash

# Pre-load Gemma3 model - makes first inference faster
echo "Pre-loading Gemma3 model..."
ollama run gemma3 &

# Start the vector database
echo "Starting ChromaDB..."
chromadb start --persist-directory ./chroma_db &
