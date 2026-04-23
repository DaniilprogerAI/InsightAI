from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from models.comment import Comment, CommentCreate, CommentResponse, CommentAnalysis, CommentSearch
from utils.db import get_async_db, db_manager
from pipeline.rag_pipeline import rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=CommentResponse)
async def create_comment(
    comment: CommentCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new comment and process it with AI"""
    try:
        # Create comment in database
        db_comment = await db_manager.create_comment(comment.dict())
        
        # Add AI processing to background tasks
        background_tasks.add_task(process_comment_ai, db_comment.id)
        
        return CommentResponse.from_orm(db_comment)
        
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{comment_id}", response_model=CommentResponse)
async def get_comment(comment_id: int, db: AsyncSession = Depends(get_async_db)):
    """Get a specific comment by ID"""
    comment = await db_manager.get_comment(comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    return CommentResponse.from_orm(comment)

@router.get("/operator/{operator_id}", response_model=List[CommentResponse])
async def get_comments_by_operator(
    operator_id: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db)
):
    """Get comments by operator ID"""
    try:
        comments = await db_manager.get_comments_by_operator(operator_id, limit)
        return [CommentResponse.from_orm(comment) for comment in comments]
    except Exception as e:
        logger.error(f"Error getting comments by operator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/event/{event_id}", response_model=List[CommentResponse])
async def get_comments_by_event(
    event_id: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db)
):
    """Get comments by production event ID"""
    try:
        comments = await db_manager.get_comments_by_event(event_id, limit)
        return [CommentResponse.from_orm(comment) for comment in comments]
    except Exception as e:
        logger.error(f"Error getting comments by event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=List[CommentResponse])
async def search_comments(search: CommentSearch, db: AsyncSession = Depends(get_async_db)):
    """Search for similar comments using vector similarity"""
    try:
        # Search for similar comments using RAG pipeline
        similar_comments = rag_pipeline.search_similar_comments(
            query=search.query,
            n_results=search.limit,
            threshold=search.threshold
        )
        
        # Get full comment data from database
        comment_responses = []
        for similar_comment in similar_comments:
            comment = await db_manager.get_comment(similar_comment["comment_id"])
            if comment:
                response = CommentResponse.from_orm(comment)
                comment_responses.append(response)
        
        return comment_responses
        
    except Exception as e:
        logger.error(f"Error searching comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{comment_id}/analyze", response_model=CommentAnalysis)
async def analyze_comment(comment_id: int, db: AsyncSession = Depends(get_async_db)):
    """Analyze a comment with AI"""
    try:
        # Get comment from database
        comment = await db_manager.get_comment(comment_id)
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        # Process comment with RAG pipeline
        result = await rag_pipeline.process_comment(
            comment_id=comment.id,
            text=comment.text,
            metadata={
                "operator_id": comment.operator_id,
                "production_event_id": comment.production_event_id,
                "event_type": comment.event_type,
                "severity": comment.severity
            }
        )
        
        # Update comment with AI analysis
        await db_manager.update_comment_analysis(
            comment_id=comment.id,
            analysis=result["analysis"],
            confidence_score=result["confidence_score"]
        )
        
        return CommentAnalysis(
            comment_id=comment_id,
            analysis=result["analysis"],
            confidence_score=result["confidence_score"],
            recommendations=result["recommendations"],
            related_events=[comment.production_event_id] if comment.production_event_id else []
        )
        
    except Exception as e:
        logger.error(f"Error analyzing comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/process")
async def batch_process_comments(background_tasks: BackgroundTasks):
    """Process all unprocessed comments in batch"""
    try:
        # Add batch processing to background tasks
        background_tasks.add_task(batch_process_unprocessed_comments)
        
        return {"message": "Batch processing started"}
        
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/overview")
async def get_stats_overview(db: AsyncSession = Depends(get_async_db)):
    """Get overview statistics"""
    try:
        # Get unprocessed comments count
        unprocessed_comments = await db_manager.get_unprocessed_comments(limit=1000)
        
        # This is a simplified stats - in production you'd want more sophisticated queries
        stats = {
            "total_comments": len(unprocessed_comments),  # This would be total count in real implementation
            "unprocessed_comments": len(unprocessed_comments),
            "processed_comments": 0,  # This would be calculated in real implementation
            "ai_analysis_available": True
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def process_comment_ai(comment_id: int):
    """Background task to process comment with AI"""
    try:
        # Get comment from database
        comment = await db_manager.get_comment(comment_id)
        if not comment:
            logger.error(f"Comment {comment_id} not found for AI processing")
            return
        
        # Process comment with RAG pipeline
        result = await rag_pipeline.process_comment(
            comment_id=comment.id,
            text=comment.text,
            metadata={
                "operator_id": comment.operator_id,
                "production_event_id": comment.production_event_id,
                "event_type": comment.event_type,
                "severity": comment.severity
            }
        )
        
        # Update comment with AI analysis
        await db_manager.update_comment_analysis(
            comment_id=comment.id,
            analysis=result["analysis"],
            confidence_score=result["confidence_score"]
        )
        
        logger.info(f"Successfully processed comment {comment_id} with AI")
        
    except Exception as e:
        logger.error(f"Error processing comment {comment_id} with AI: {e}")

async def batch_process_unprocessed_comments():
    """Background task to process all unprocessed comments"""
    try:
        # Get unprocessed comments
        unprocessed_comments = await db_manager.get_unprocessed_comments(limit=50)
        
        logger.info(f"Starting batch processing for {len(unprocessed_comments)} comments")
        
        # Process each comment
        for comment in unprocessed_comments:
            await process_comment_ai(comment.id)
        
        logger.info(f"Completed batch processing for {len(unprocessed_comments)} comments")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")