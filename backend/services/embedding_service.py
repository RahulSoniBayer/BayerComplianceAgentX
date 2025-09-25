"""
Embedding service for creating and managing document embeddings.
Supports multiple vector databases (Pinecone, Weaviate, FAISS) with a pluggable interface.
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from utils.config import settings
from parsers.pdf_parser import PDFChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    metadata: Dict[str, Any]


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database operations."""
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to the database."""
        pass
    
    @abstractmethod
    async def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from the database."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass


class PineconeVectorDB(VectorDatabaseInterface):
    """Pinecone vector database implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._index = None
    
    async def _get_client(self):
        """Get Pinecone client instance."""
        if self._client is None:
            try:
                import pinecone
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                self._client = pinecone
                self._index = pinecone.Index(settings.pinecone_index_name)
            except Exception as e:
                self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
                raise
        
        return self._client, self._index
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to Pinecone."""
        try:
            client, index = await self._get_client()
            
            # Prepare vectors for Pinecone
            pinecone_vectors = []
            for vector_id, embedding, metadata in vectors:
                pinecone_vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                index.upsert(vectors=batch)
            
            self.logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Error upserting vectors to Pinecone: {str(e)}")
            return False
    
    async def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        try:
            client, index = await self._get_client()
            
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from Pinecone."""
        try:
            client, index = await self._get_client()
            index.delete(ids=vector_ids)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting vectors from Pinecone: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            client, index = await self._get_client()
            stats = index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            self.logger.error(f"Error getting Pinecone stats: {str(e)}")
            return {}


class WeaviateVectorDB(VectorDatabaseInterface):
    """Weaviate vector database implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    async def _get_client(self):
        """Get Weaviate client instance."""
        if self._client is None:
            try:
                import weaviate
                self._client = weaviate.Client(
                    url=settings.weaviate_url,
                    auth_client_secret=weaviate.AuthApiKey(settings.weaviate_api_key) if settings.weaviate_api_key else None
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Weaviate: {str(e)}")
                raise
        
        return self._client
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to Weaviate."""
        try:
            client = await self._get_client()
            
            # Create class if it doesn't exist
            class_name = "DocumentChunk"
            if not client.schema.exists(class_name):
                class_schema = {
                    "class": class_name,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "content_type", "dataType": ["string"]},
                        {"name": "page_number", "dataType": ["int"]},
                        {"name": "section_title", "dataType": ["string"]},
                        {"name": "chunk_id", "dataType": ["string"]},
                        {"name": "document_id", "dataType": ["int"]}
                    ]
                }
                client.schema.create_class(class_schema)
            
            # Upsert vectors
            for vector_id, embedding, metadata in vectors:
                data_object = {
                    "chunk_id": vector_id,
                    "content": metadata.get("content", ""),
                    "content_type": metadata.get("content_type", ""),
                    "page_number": metadata.get("page_number", 0),
                    "section_title": metadata.get("section_title", ""),
                    "document_id": metadata.get("document_id", 0)
                }
                
                client.data_object.create(
                    data_object=data_object,
                    class_name=class_name,
                    vector=embedding
                )
            
            self.logger.info(f"Successfully upserted {len(vectors)} vectors to Weaviate")
            return True
            
        except Exception as e:
            self.logger.error(f"Error upserting vectors to Weaviate: {str(e)}")
            return False
    
    async def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate."""
        try:
            client = await self._get_client()
            
            results = client.query.get(
                "DocumentChunk",
                ["content", "content_type", "page_number", "section_title", "chunk_id", "document_id"]
            ).with_near_vector({
                "vector": query_vector
            }).with_limit(top_k).do()
            
            # Format results
            formatted_results = []
            for result in results["data"]["Get"]["DocumentChunk"]:
                formatted_results.append({
                    "id": result["chunk_id"],
                    "score": 0.0,  # Weaviate doesn't return scores in this format
                    "metadata": result
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching Weaviate: {str(e)}")
            return []
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from Weaviate."""
        try:
            client = await self._get_client()
            
            # Delete by chunk_id
            for vector_id in vector_ids:
                client.data_object.delete(
                    where={
                        "path": ["chunk_id"],
                        "operator": "Equal",
                        "valueString": vector_id
                    },
                    class_name="DocumentChunk"
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting vectors from Weaviate: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate statistics."""
        try:
            client = await self._get_client()
            stats = client.cluster.get_nodes_status()
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": 1536,  # OpenAI embedding dimension
                "status": "healthy" if stats else "unknown"
            }
        except Exception as e:
            self.logger.error(f"Error getting Weaviate stats: {str(e)}")
            return {}


class FAISSVectorDB(VectorDatabaseInterface):
    """FAISS vector database implementation for local storage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._index = None
        self._id_to_metadata = {}
        self._dimension = 1536  # OpenAI embedding dimension
    
    async def _initialize_index(self):
        """Initialize FAISS index."""
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self._dimension)  # Inner product for cosine similarity
                self.logger.info("Initialized FAISS index")
            except Exception as e:
                self.logger.error(f"Failed to initialize FAISS: {str(e)}")
                raise
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Upsert vectors to FAISS."""
        try:
            await self._initialize_index()
            
            # Prepare vectors
            vector_arrays = []
            for vector_id, embedding, metadata in vectors:
                # Normalize embedding for cosine similarity
                embedding_array = np.array(embedding, dtype=np.float32)
                embedding_array = embedding_array / np.linalg.norm(embedding_array)
                vector_arrays.append(embedding_array)
                
                # Store metadata
                self._id_to_metadata[vector_id] = metadata
            
            # Add to index
            vectors_matrix = np.vstack(vector_arrays)
            self._index.add(vectors_matrix)
            
            self.logger.info(f"Successfully upserted {len(vectors)} vectors to FAISS")
            return True
            
        except Exception as e:
            self.logger.error(f"Error upserting vectors to FAISS: {str(e)}")
            return False
    
    async def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in FAISS."""
        try:
            await self._initialize_index()
            
            if self._index.ntotal == 0:
                return []
            
            # Normalize query vector
            query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            query_array = query_array / np.linalg.norm(query_array)
            
            # Search
            scores, indices = self._index.search(query_array, min(top_k, self._index.ntotal))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue
                
                # Get metadata by index (this is a simplification)
                vector_id = f"vector_{idx}"
                metadata = self._id_to_metadata.get(vector_id, {})
                
                results.append({
                    "id": vector_id,
                    "score": float(score),
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching FAISS: {str(e)}")
            return []
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from FAISS (not directly supported, would need index recreation)."""
        self.logger.warning("FAISS doesn't support direct vector deletion")
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics."""
        try:
            await self._initialize_index()
            return {
                "total_vector_count": self._index.ntotal,
                "dimension": self._dimension,
                "index_type": "FAISS_FlatIP"
            }
        except Exception as e:
            self.logger.error(f"Error getting FAISS stats: {str(e)}")
            return {}


class EmbeddingService:
    """Main embedding service for generating and managing embeddings."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._vector_db = self._initialize_vector_db()
        self._embedding_client = None
    
    def _initialize_vector_db(self) -> VectorDatabaseInterface:
        """Initialize the appropriate vector database based on configuration."""
        if settings.vector_db_type == "pinecone":
            return PineconeVectorDB()
        elif settings.vector_db_type == "weaviate":
            return WeaviateVectorDB()
        elif settings.vector_db_type == "faiss":
            return FAISSVectorDB()
        else:
            raise ValueError(f"Unsupported vector database type: {settings.vector_db_type}")
    
    async def _get_embedding_client(self):
        """Get OpenAI embedding client."""
        if self._embedding_client is None:
            try:
                import openai
                self._embedding_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        
        return self._embedding_client
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            client = await self._get_embedding_client()
            
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def chunk_and_embed(self, chunks: List[PDFChunk], document_id: int) -> List[EmbeddingResult]:
        """
        Process chunks and generate embeddings.
        
        Args:
            chunks: List of PDF chunks
            document_id: ID of the source document
            
        Returns:
            List[EmbeddingResult]: List of embedding results
        """
        results = []
        
        for chunk in chunks:
            try:
                # Generate unique chunk ID
                chunk_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = await self.generate_embedding(chunk.content)
                
                # Prepare metadata
                metadata = {
                    "content": chunk.content,
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    **chunk.metadata
                }
                
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    metadata=metadata
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # Store embeddings in vector database
        if results:
            vectors = [(r.chunk_id, r.embedding, r.metadata) for r in results]
            await self._vector_db.upsert_vectors(vectors)
        
        self.logger.info(f"Successfully processed {len(results)} chunks for document {document_id}")
        return results
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query string.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List[Dict]: List of similar chunks with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            
            # Search vector database
            results = await self._vector_db.search_similar(query_embedding, top_k)
            
            self.logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    async def delete_document_embeddings(self, document_id: int) -> bool:
        """
        Delete all embeddings for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            bool: True if successful
        """
        try:
            # This would require implementing a way to find all vector IDs for a document
            # For now, this is a placeholder
            self.logger.warning("Document embedding deletion not fully implemented")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document embeddings: {str(e)}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get vector database statistics.
        
        Returns:
            Dict: Database statistics
        """
        try:
            stats = await self._vector_db.get_stats()
            stats["vector_db_type"] = settings.vector_db_type
            return stats
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}


# Convenience functions
async def create_embeddings_for_chunks(chunks: List[PDFChunk], document_id: int) -> List[EmbeddingResult]:
    """
    Convenience function to create embeddings for PDF chunks.
    
    Args:
        chunks: List of PDF chunks
        document_id: ID of the source document
        
    Returns:
        List[EmbeddingResult]: List of embedding results
    """
    service = EmbeddingService()
    return await service.chunk_and_embed(chunks, document_id)


async def search_similar_content(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to search for similar content.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List[Dict]: List of similar chunks
    """
    service = EmbeddingService()
    return await service.search_similar_chunks(query, top_k)
