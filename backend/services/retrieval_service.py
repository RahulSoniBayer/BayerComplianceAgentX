"""
Retrieval service for finding relevant document chunks using vector similarity search.
Handles query processing and result ranking.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    relevance_reason: str


class RetrievalService:
    """Main retrieval service for finding relevant document content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._embedding_service = EmbeddingService()
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        content_type_filter: Optional[str] = None,
        document_filter: Optional[List[int]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: Search query (placeholder text or related terms)
            top_k: Number of results to return
            content_type_filter: Filter by content type ('text', 'table', 'image')
            document_filter: Filter by specific document IDs
            
        Returns:
            List[RetrievalResult]: List of relevant chunks with scores
        """
        try:
            self.logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            # Search for similar chunks
            search_results = await self._embedding_service.search_similar_chunks(query, top_k * 2)
            
            # Filter and process results
            filtered_results = self._filter_results(
                search_results,
                content_type_filter,
                document_filter
            )
            
            # Take top results
            top_results = filtered_results[:top_k]
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in top_results:
                retrieval_result = self._process_retrieval_result(result, query)
                retrieval_results.append(retrieval_result)
            
            self.logger.info(f"Retrieved {len(retrieval_results)} relevant chunks")
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _filter_results(
        self,
        search_results: List[Dict[str, Any]],
        content_type_filter: Optional[str],
        document_filter: Optional[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Filter search results based on criteria.
        
        Args:
            search_results: Raw search results from vector database
            content_type_filter: Filter by content type
            document_filter: Filter by document IDs
            
        Returns:
            List[Dict]: Filtered results
        """
        filtered = []
        
        for result in search_results:
            metadata = result.get("metadata", {})
            
            # Apply content type filter
            if content_type_filter:
                if metadata.get("content_type") != content_type_filter:
                    continue
            
            # Apply document filter
            if document_filter:
                document_id = metadata.get("document_id")
                if document_id not in document_filter:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _process_retrieval_result(self, result: Dict[str, Any], query: str) -> RetrievalResult:
        """
        Process a raw retrieval result into a RetrievalResult object.
        
        Args:
            result: Raw result from vector database
            query: Original search query
            
        Returns:
            RetrievalResult: Processed result with relevance reasoning
        """
        metadata = result.get("metadata", {})
        content = metadata.get("content", "")
        
        # Generate relevance reason
        relevance_reason = self._generate_relevance_reason(result, query)
        
        return RetrievalResult(
            chunk_id=result.get("id", ""),
            content=content,
            score=result.get("score", 0.0),
            metadata=metadata,
            relevance_reason=relevance_reason
        )
    
    def _generate_relevance_reason(self, result: Dict[str, Any], query: str) -> str:
        """
        Generate a human-readable explanation of why this chunk is relevant.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            str: Relevance explanation
        """
        metadata = result.get("metadata", {})
        score = result.get("score", 0.0)
        
        # Base relevance explanation
        if score > 0.8:
            relevance = "Highly relevant"
        elif score > 0.6:
            relevance = "Moderately relevant"
        else:
            relevance = "Somewhat relevant"
        
        # Add context information
        context_parts = []
        
        content_type = metadata.get("content_type", "")
        if content_type:
            context_parts.append(f"contains {content_type} content")
        
        page_number = metadata.get("page_number")
        if page_number:
            context_parts.append(f"from page {page_number}")
        
        section_title = metadata.get("section_title")
        if section_title:
            context_parts.append(f"in section '{section_title}'")
        
        # Combine relevance and context
        if context_parts:
            context_str = " and ".join(context_parts)
            return f"{relevance} - {context_str}"
        else:
            return relevance
    
    async def retrieve_for_placeholder(
        self,
        placeholder_text: str,
        context_type: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks specifically for a placeholder.
        
        Args:
            placeholder_text: Text of the placeholder
            context_type: Type of context ('table' or 'section')
            top_k: Number of results to return
            
        Returns:
            List[RetrievalResult]: Relevant chunks for the placeholder
        """
        try:
            # Enhance query based on placeholder and context type
            enhanced_query = self._enhance_placeholder_query(placeholder_text, context_type)
            
            # Retrieve relevant chunks
            results = await self.retrieve_relevant_chunks(
                query=enhanced_query,
                top_k=top_k,
                content_type_filter=self._get_content_type_for_context(context_type)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving for placeholder: {str(e)}")
            return []
    
    def _enhance_placeholder_query(self, placeholder_text: str, context_type: str) -> str:
        """
        Enhance placeholder text to create a better search query.
        
        Args:
            placeholder_text: Original placeholder text
            context_type: Type of context
            
        Returns:
            str: Enhanced search query
        """
        # Clean the placeholder text
        query = placeholder_text.strip().lower()
        
        # Add context-specific terms
        if context_type == "table":
            # For table placeholders, look for structured data
            query += " data information facts details"
        else:
            # For section placeholders, look for comprehensive content
            query += " description explanation overview details"
        
        # Remove common placeholder artifacts
        query = query.replace("{{", "").replace("}}", "")
        query = query.replace("[[", "").replace("]]", "")
        query = query.replace("<", "").replace(">", "")
        query = query.replace("%", "")
        
        return query
    
    def _get_content_type_for_context(self, context_type: str) -> Optional[str]:
        """
        Determine the best content type filter for a given context type.
        
        Args:
            context_type: Type of context ('table' or 'section')
            
        Returns:
            Optional[str]: Content type filter
        """
        if context_type == "table":
            return "table"
        else:
            return None  # No filter for sections, include all types
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievalResult]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Optional[RetrievalResult]: Chunk if found
        """
        try:
            # This would require implementing a way to retrieve by ID from the vector database
            # For now, this is a placeholder
            self.logger.warning("Chunk retrieval by ID not fully implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunk by ID: {str(e)}")
            return None
    
    async def get_document_chunks(self, document_id: int) -> List[RetrievalResult]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List[RetrievalResult]: All chunks for the document
        """
        try:
            # This would require implementing a way to retrieve all chunks for a document
            # For now, this is a placeholder
            self.logger.warning("Document chunk retrieval not fully implemented")
            return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving document chunks: {str(e)}")
            return []
    
    async def search_similar_placeholders(
        self,
        placeholder_text: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Search for chunks that might be relevant to a placeholder.
        
        Args:
            placeholder_text: Text of the placeholder
            top_k: Number of results to return
            
        Returns:
            List[RetrievalResult]: Similar chunks
        """
        try:
            # Use the placeholder text directly as a search query
            results = await self.retrieve_relevant_chunks(
                query=placeholder_text,
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar placeholders: {str(e)}")
            return []
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval service.
        
        Returns:
            Dict: Statistics about the retrieval service
        """
        try:
            # Get database stats from embedding service
            db_stats = await self._embedding_service.get_database_stats()
            
            stats = {
                "total_chunks": db_stats.get("total_vector_count", 0),
                "vector_dimension": db_stats.get("dimension", 0),
                "vector_db_type": db_stats.get("vector_db_type", "unknown"),
                "service_status": "active"
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting retrieval stats: {str(e)}")
            return {"service_status": "error", "error": str(e)}


# Convenience functions
async def retrieve_relevant_content(
    query: str,
    top_k: int = 5,
    content_type_filter: Optional[str] = None
) -> List[RetrievalResult]:
    """
    Convenience function to retrieve relevant content.
    
    Args:
        query: Search query
        top_k: Number of results
        content_type_filter: Filter by content type
        
    Returns:
        List[RetrievalResult]: Relevant chunks
    """
    service = RetrievalService()
    return await service.retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        content_type_filter=content_type_filter
    )


async def retrieve_for_placeholder_context(
    placeholder_text: str,
    context_type: str,
    top_k: int = 3
) -> List[RetrievalResult]:
    """
    Convenience function to retrieve context for a placeholder.
    
    Args:
        placeholder_text: Placeholder text
        context_type: Context type
        top_k: Number of results
        
    Returns:
        List[RetrievalResult]: Relevant chunks
    """
    service = RetrievalService()
    return await service.retrieve_for_placeholder(
        placeholder_text=placeholder_text,
        context_type=context_type,
        top_k=top_k
    )
