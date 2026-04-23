from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

Base = declarative_base()

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    operator_id = Column(String, index=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    production_event_id = Column(String, index=True)
    event_type = Column(String)
    severity = Column(String)
    embedding = Column(Text)  # JSON string of vector embedding
    processed = Column(Boolean, default=False)
    ai_analysis = Column(Text)  # AI-generated analysis
    confidence_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class CommentCreate(BaseModel):
    operator_id: str = Field(..., description="ID of the operator who made the comment")
    text: str = Field(..., description="The comment text")
    production_event_id: Optional[str] = Field(None, description="Related production event ID")
    event_type: Optional[str] = Field(None, description="Type of production event")
    severity: Optional[str] = Field(None, description="Severity level of the event")

class CommentResponse(BaseModel):
    id: int
    operator_id: str
    text: str
    timestamp: datetime
    production_event_id: Optional[str]
    event_type: Optional[str]
    severity: Optional[str]
    processed: bool
    ai_analysis: Optional[str]
    confidence_score: Optional[float]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class CommentAnalysis(BaseModel):
    comment_id: int
    analysis: str
    confidence_score: float
    recommendations: List[str]
    related_events: List[str]

class CommentSearch(BaseModel):
    query: str = Field(..., description="Search query for comments")
    limit: int = Field(10, description="Maximum number of results")
    threshold: float = Field(0.7, description="Similarity threshold")