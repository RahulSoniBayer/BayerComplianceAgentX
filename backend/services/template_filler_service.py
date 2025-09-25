"""
Template filler service for orchestrating the complete document automation workflow.
Handles placeholder extraction, retrieval, LLM generation, and document filling.
"""

import asyncio
import logging
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from parsers.docx_parser import DOCXParser, Placeholder
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService, LLMRequest
from utils.config import settings
from utils.validators import ContentValidator

logger = logging.getLogger(__name__)


class TemplateFillerService:
    """Main service for filling document templates with AI-generated content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._docx_parser = DOCXParser()
        self._retrieval_service = RetrievalService()
        self._llm_service = LLMService()
    
    async def process_template_file(
        self,
        template_path: str,
        user_context: Optional[str] = None,
        process_flow_description: Optional[str] = None,
        process_flow_image: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a single template file and fill all placeholders.
        
        Args:
            template_path: Path to the template file
            user_context: Additional user context
            process_flow_description: Description of process flow
            process_flow_image: Base64 encoded process flow image
            progress_callback: Callback function for progress updates
            
        Returns:
            Dict: Processing result with file path and metadata
        """
        try:
            self.logger.info(f"Processing template file: {template_path}")
            
            # Validate template file
            is_valid, error_msg = self._docx_parser.validate_docx(template_path)
            if not is_valid:
                raise ValueError(f"Invalid template file: {error_msg}")
            
            # Extract placeholders
            placeholders = self._docx_parser.extract_placeholders(template_path)
            if not placeholders:
                raise ValueError("No placeholders found in template")
            
            self.logger.info(f"Found {len(placeholders)} placeholders to fill")
            
            # Update progress
            if progress_callback:
                await progress_callback({
                    "status": "extracting_placeholders",
                    "message": f"Found {len(placeholders)} placeholders",
                    "progress": 10
                })
            
            # Analyze process flow image if provided
            if process_flow_image and not process_flow_description:
                self.logger.info("Analyzing process flow image")
                process_flow_description = await self._llm_service.analyze_process_flow_image(process_flow_image)
                
                if progress_callback:
                    await progress_callback({
                        "status": "analyzing_process_flow",
                        "message": "Process flow analysis completed",
                        "progress": 20
                    })
            
            # Fill placeholders
            placeholder_fills = {}
            total_placeholders = len(placeholders)
            
            for i, placeholder in enumerate(placeholders):
                try:
                    self.logger.info(f"Processing placeholder {i+1}/{total_placeholders}: {placeholder.text[:50]}...")
                    
                    # Update progress
                    if progress_callback:
                        progress = 20 + (i / total_placeholders) * 60
                        await progress_callback({
                            "status": "filling_placeholders",
                            "message": f"Processing placeholder {i+1}/{total_placeholders}",
                            "progress": progress,
                            "current_placeholder": placeholder.text
                        })
                    
                    # Retrieve relevant chunks
                    retrieved_chunks = await self._retrieval_service.retrieve_for_placeholder(
                        placeholder_text=placeholder.text,
                        context_type=placeholder.context_type,
                        top_k=3
                    )
                    
                    # Prepare LLM request
                    llm_request = LLMRequest(
                        prompt=placeholder.text,
                        context_type=placeholder.context_type,
                        retrieved_chunks=[{
                            "metadata": chunk.metadata,
                            "content": chunk.content,
                            "score": chunk.score
                        } for chunk in retrieved_chunks],
                        user_context=user_context,
                        process_flow_description=process_flow_description,
                        image_data=process_flow_image
                    )
                    
                    # Generate content using LLM
                    llm_response = await self._llm_service.fill_placeholder(llm_request)
                    
                    if llm_response.success:
                        placeholder_fills[placeholder.text] = llm_response.content
                        self.logger.info(f"Successfully filled placeholder: {placeholder.text[:50]}...")
                    else:
                        self.logger.error(f"Failed to fill placeholder: {llm_response.error_message}")
                        # Use fallback content
                        placeholder_fills[placeholder.text] = self._generate_fallback_content(placeholder)
                    
                except Exception as e:
                    self.logger.error(f"Error processing placeholder {placeholder.text}: {str(e)}")
                    # Use fallback content
                    placeholder_fills[placeholder.text] = self._generate_fallback_content(placeholder)
            
            # Update progress
            if progress_callback:
                await progress_callback({
                    "status": "generating_document",
                    "message": "Generating final document",
                    "progress": 80
                })
            
            # Generate output file path
            output_filename = f"filled_{Path(template_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            output_path = os.path.join(settings.generated_dir, output_filename)
            
            # Fill the template
            success = self._docx_parser.fill_placeholders(
                docx_path=template_path,
                placeholder_fills=placeholder_fills,
                output_path=output_path
            )
            
            if not success:
                raise RuntimeError("Failed to generate filled document")
            
            # Update progress
            if progress_callback:
                await progress_callback({
                    "status": "completed",
                    "message": "Document generation completed",
                    "progress": 100
                })
            
            result = {
                "success": True,
                "output_path": output_path,
                "output_filename": output_filename,
                "placeholders_filled": len(placeholder_fills),
                "total_placeholders": total_placeholders,
                "user_context": user_context,
                "process_flow_description": process_flow_description,
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully processed template: {template_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing template file {template_path}: {str(e)}")
            
            if progress_callback:
                await progress_callback({
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "progress": 0
                })
            
            raise
    
    async def process_multiple_templates(
        self,
        template_files: List[str],
        user_context: Optional[str] = None,
        process_flow_description: Optional[str] = None,
        process_flow_image: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple template files concurrently.
        
        Args:
            template_files: List of template file paths
            user_context: Additional user context
            process_flow_description: Description of process flow
            process_flow_image: Base64 encoded process flow image
            progress_callback: Callback function for progress updates
            
        Returns:
            Dict: Processing results for all files
        """
        try:
            self.logger.info(f"Processing {len(template_files)} template files")
            
            # Create task ID for tracking
            task_id = str(uuid.uuid4())
            
            results = {
                "task_id": task_id,
                "total_files": len(template_files),
                "completed_files": 0,
                "failed_files": 0,
                "results": [],
                "started_at": datetime.now().isoformat()
            }
            
            # Process files concurrently
            semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
            
            async def process_single_file(file_path: str, file_index: int):
                """Process a single template file."""
                async with semaphore:
                    try:
                        file_progress_callback = None
                        if progress_callback:
                            file_progress_callback = lambda progress: progress_callback({
                                "task_id": task_id,
                                "file_index": file_index,
                                "file_path": file_path,
                                "file_progress": progress
                            })
                        
                        result = await self.process_template_file(
                            template_path=file_path,
                            user_context=user_context,
                            process_flow_description=process_flow_description,
                            process_flow_image=process_flow_image,
                            progress_callback=file_progress_callback
                        )
                        
                        result["file_index"] = file_index
                        result["file_path"] = file_path
                        results["results"].append(result)
                        results["completed_files"] += 1
                        
                        # Notify completion
                        if progress_callback:
                            await progress_callback({
                                "task_id": task_id,
                                "status": "file_completed",
                                "file_index": file_index,
                                "file_path": file_path,
                                "filename": result["output_filename"],
                                "completed_files": results["completed_files"],
                                "total_files": results["total_files"]
                            })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing file {file_path}: {str(e)}")
                        results["failed_files"] += 1
                        
                        error_result = {
                            "file_index": file_index,
                            "file_path": file_path,
                            "success": False,
                            "error": str(e)
                        }
                        results["results"].append(error_result)
                        
                        # Notify failure
                        if progress_callback:
                            await progress_callback({
                                "task_id": task_id,
                                "status": "file_failed",
                                "file_index": file_index,
                                "file_path": file_path,
                                "error": str(e),
                                "failed_files": results["failed_files"],
                                "total_files": results["total_files"]
                            })
            
            # Start all processing tasks
            tasks = [
                process_single_file(file_path, i)
                for i, file_path in enumerate(template_files)
            ]
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            results["completed_at"] = datetime.now().isoformat()
            results["success"] = results["completed_files"] > 0
            
            self.logger.info(f"Completed processing {results['completed_files']}/{results['total_files']} files")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch template processing: {str(e)}")
            raise
    
    def _generate_fallback_content(self, placeholder: Placeholder) -> str:
        """
        Generate fallback content when LLM fails.
        
        Args:
            placeholder: Placeholder object
            
        Returns:
            str: Fallback content
        """
        if placeholder.context_type == "table":
            return f"[{placeholder.text}] - Content not available"
        else:
            return f"Content for '{placeholder.text}' is not available at this time. Please refer to the source documents for more information."
    
    async def validate_template(self, template_path: str) -> Dict[str, Any]:
        """
        Validate a template file and return analysis.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Dict: Validation results and analysis
        """
        try:
            # Validate file format
            is_valid, error_msg = self._docx_parser.validate_docx(template_path)
            
            if not is_valid:
                return {
                    "valid": False,
                    "error": error_msg
                }
            
            # Extract placeholders
            placeholders = self._docx_parser.extract_placeholders(template_path)
            
            # Analyze placeholders
            placeholder_analysis = {
                "total_count": len(placeholders),
                "by_type": {},
                "by_context": {},
                "complexity_score": 0
            }
            
            for placeholder in placeholders:
                # Count by type
                ptype = placeholder.placeholder_type
                placeholder_analysis["by_type"][ptype] = placeholder_analysis["by_type"].get(ptype, 0) + 1
                
                # Count by context
                context = placeholder.context_type
                placeholder_analysis["by_context"][context] = placeholder_analysis["by_context"].get(context, 0) + 1
                
                # Calculate complexity score
                complexity = len(placeholder.text.split()) + (2 if placeholder.metadata.get("is_in_table") else 1)
                placeholder_analysis["complexity_score"] += complexity
            
            # Get document metadata
            doc_metadata = self._docx_parser.get_document_metadata(template_path)
            
            return {
                "valid": True,
                "placeholders": placeholder_analysis,
                "document_metadata": doc_metadata,
                "estimated_processing_time": self._estimate_processing_time(placeholders)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating template {template_path}: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _estimate_processing_time(self, placeholders: List[Placeholder]) -> int:
        """
        Estimate processing time based on placeholders.
        
        Args:
            placeholders: List of placeholders
            
        Returns:
            int: Estimated time in seconds
        """
        # Base time per placeholder
        base_time_per_placeholder = 5  # seconds
        
        # Complexity multipliers
        complexity_multipliers = {
            "table": 1.0,
            "section": 1.5,
            "inline": 0.8
        }
        
        total_time = 0
        for placeholder in placeholders:
            multiplier = complexity_multipliers.get(placeholder.placeholder_type, 1.0)
            total_time += base_time_per_placeholder * multiplier
        
        return int(total_time)
    
    async def close(self):
        """Close all services."""
        await self._llm_service.close()


# Convenience functions
async def fill_template_file(
    template_path: str,
    user_context: Optional[str] = None,
    process_flow_description: Optional[str] = None,
    process_flow_image: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to fill a single template file.
    
    Args:
        template_path: Path to the template file
        user_context: Additional user context
        process_flow_description: Description of process flow
        process_flow_image: Base64 encoded process flow image
        
    Returns:
        Dict: Processing result
    """
    service = TemplateFillerService()
    result = await service.process_template_file(
        template_path=template_path,
        user_context=user_context,
        process_flow_description=process_flow_description,
        process_flow_image=process_flow_image
    )
    await service.close()
    return result


async def process_template_batch(
    template_files: List[str],
    user_context: Optional[str] = None,
    process_flow_description: Optional[str] = None,
    process_flow_image: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to process multiple template files.
    
    Args:
        template_files: List of template file paths
        user_context: Additional user context
        process_flow_description: Description of process flow
        process_flow_image: Base64 encoded process flow image
        
    Returns:
        Dict: Batch processing results
    """
    service = TemplateFillerService()
    result = await service.process_multiple_templates(
        template_files=template_files,
        user_context=user_context,
        process_flow_description=process_flow_description,
        process_flow_image=process_flow_image
    )
    await service.close()
    return result
