from typing import List, Optional, Any
import time
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import streamlit as st
from config import PINECONE_DIMENSION, PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    The function attempts to split text at natural boundaries (paragraphs,
    sentences, or words) while maintaining the specified chunk size and overlap.
    
    Args:
        text (str): Input text to be chunked
        chunk_size (int, optional): Maximum size of each chunk
        overlap (int, optional): Number of characters to overlap between chunks
    
    Returns:
        List[str]: List of text chunks
        
    Example:
        >>> text = "This is a long piece of text. It contains multiple sentences."
        >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
        >>> print(chunks)
        ['This is a long', 'long piece of text', 'text. It contains', 'contains multiple']
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Adjust chunk to end at a sentence or paragraph break if possible
        if end < len(text):
            for separator in ['\n\n', '\n', '.', ' ']:
                last_separator = chunk.rfind(separator)
                if last_separator != -1:
                    end = start + last_separator + 1
                    chunk = text[start:end]
                    break
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def create_embeddings(chunks: List[str], openai_client: OpenAI) -> List[List[float]]:
    """
    Create vector embeddings for text chunks using OpenAI's embedding model.
    
    Args:
        chunks (List[str]): List of text chunks to embed
        openai_client (OpenAI): Initialized OpenAI client
    
    Returns:
        List[List[float]]: List of embedding vectors
        
    Note:
        Uses the 'text-embedding-ada-002' model which produces 1536-dimensional
        vectors optimized for semantic similarity search.
    """
    embeddings = []
    for chunk in chunks:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def setup_pinecone(api_key: str, index_name: str = "real-estate") -> Optional[Any]:
    """
    Initialize Pinecone client and ensure the required index exists.
    
    This function handles:
        1. Client initialization
        2. Index creation if it doesn't exist
        3. Waiting for index to be ready
    
    Args:
        api_key (str): Pinecone API key
        index_name (str, optional): Name of the Pinecone index
    
    Returns:
        Optional[Any]: Initialized Pinecone index object or None if setup fails
    """
    try:
        if not api_key:
            return None
        pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        existing_indexes = pc.list_indexes()
        if not any(index.name == index_name for index in existing_indexes):
            pc.create_index(
                name=index_name,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            while not any(index.name == index_name and index.status['ready'] 
                        for index in pc.list_indexes()):
                time.sleep(1)
        
        return pc.Index(index_name)
    
    except Exception as e:
        st.error(f"Pinecone setup error: {e}")
        return None

def upload_to_pinecone(
    index: Any,
    chunks: List[str],
    embeddings: List[List[float]],
    namespace: str = "knowledge-base"
    ) -> bool:
    """
    Upload text chunks and their embeddings to Pinecone.
    
    Args:
        index (Any): Initialized Pinecone index
        chunks (List[str]): List of text chunks
        embeddings (List[List[float]]): List of embedding vectors
        namespace (str, optional): Pinecone namespace for organization
    
    Returns:
        bool: True if upload successful, False otherwise
        
    Note:
        Each vector in Pinecone includes:
        - id: Unique identifier for the chunk
        - values: The embedding vector
        - metadata: Original text and any additional metadata
    """
    try:
        vectors = [
            {
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to Pinecone: {e}")
        return False

def get_relevant_context(
    query: str,
    index: Any,
    openai_client: OpenAI,
    namespace: str = "knowledge-base",
    top_k: int = 3
    ) -> str:
    """
    Retrieve relevant context from Pinecone for a given query.
    
    This function:
        1. Creates an embedding for the query
        2. Searches for similar vectors in Pinecone
        3. Combines the most relevant text chunks
    
    Args:
        query (str): User's input query
        index (Any): Initialized Pinecone index
        openai_client (OpenAI): Initialized OpenAI client
        namespace (str, optional): Pinecone namespace to search
        top_k (int, optional): Number of most relevant chunks to retrieve
    
    Returns:
        str: Combined relevant context or empty string if retrieval fails
    """
    try:
        # Create query embedding
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        ).data[0].embedding
        
        # Query Pinecone
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Combine relevant contexts
        contexts = [match.metadata["text"] for match in results.matches]
        return "\n\n".join(contexts)
    
    except Exception as e:
        st.error(f"Error getting context: {e}")
        return ""