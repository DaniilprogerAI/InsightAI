from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from models.comment import Base
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create engines
engine = create_engine(DATABASE_URL)
async_engine = create_async_engine(ASYNC_DATABASE_URL)

# Create session makers
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

def get_db() -> Session:
    """Get synchronous database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncSession:
    """Get asynchronous database session"""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections"""
    await async_engine.dispose()

class DatabaseManager:
    """Database manager for handling operations"""
    
    def __init__(self):
        self.async_engine = async_engine
        self.SessionLocal = AsyncSessionLocal
    
    async def create_comment(self, comment_data):
        """Create a new comment"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            comment = Comment(**comment_data)
            session.add(comment)
            await session.commit()
            await session.refresh(comment)
            return comment
    
    async def get_comment(self, comment_id: int):
        """Get a comment by ID"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            from sqlalchemy import select
            result = await session.execute(select(Comment).where(Comment.id == comment_id))
            return result.scalar_one_or_none()
    
    async def get_comments_by_operator(self, operator_id: str, limit: int = 100):
        """Get comments by operator ID"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            from sqlalchemy import select
            result = await session.execute(
                select(Comment)
                .where(Comment.operator_id == operator_id)
                .order_by(Comment.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def get_comments_by_event(self, event_id: str, limit: int = 100):
        """Get comments by production event ID"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            from sqlalchemy import select
            result = await session.execute(
                select(Comment)
                .where(Comment.production_event_id == event_id)
                .order_by(Comment.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def update_comment_analysis(self, comment_id: int, analysis: str, confidence_score: float):
        """Update comment with AI analysis"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            from sqlalchemy import select, update
            await session.execute(
                update(Comment)
                .where(Comment.id == comment_id)
                .values(
                    ai_analysis=analysis,
                    confidence_score=confidence_score,
                    processed=True
                )
            )
            await session.commit()
    
    async def get_unprocessed_comments(self, limit: int = 50):
        """Get comments that haven't been processed by AI"""
        async with self.SessionLocal() as session:
            from models.comment import Comment
            from sqlalchemy import select
            result = await session.execute(
                select(Comment)
                .where(Comment.processed == False)
                .order_by(Comment.created_at)
                .limit(limit)
            )
            return result.scalars().all()

# Global database manager instance
db_manager = DatabaseManager()