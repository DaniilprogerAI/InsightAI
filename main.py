from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.comments import router as comments_router
from utils.db import init_db
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="InsightAI",
    description="Intelligent operator comment analysis system with production event context",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(comments_router, prefix="/api/comments", tags=["comments"])

@app.get("/")
async def root():
    return {"message": "InsightAI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
