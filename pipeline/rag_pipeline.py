import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for comment analysis and retrieval"""
    
    def __init__(self):
        # Initialize Google Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="comments",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_comment_to_vector_store(self, comment_id: int, text: str, metadata: Dict = None):
        """Add comment to vector store"""
        try:
            embedding = self.create_embedding(text)
            
            doc_metadata = {
                "comment_id": str(comment_id),
                "text": text,
                **(metadata or {})
            }
            
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
                ids=[str(comment_id)]
            )
            
            logger.info(f"Added comment {comment_id} to vector store")
            
        except Exception as e:
            logger.error(f"Error adding comment to vector store: {e}")
            raise
    
    def search_similar_comments(self, query: str, n_results: int = 10, threshold: float = 0.7) -> List[Dict]:
        """Search for similar comments"""
        try:
            query_embedding = self.create_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            similar_comments = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    if results['distances'][0][i] <= (1 - threshold):  # Convert distance to similarity
                        similar_comments.append({
                            "comment_id": int(metadata["comment_id"]),
                            "text": metadata["text"],
                            "similarity": 1 - results['distances'][0][i],
                            "metadata": metadata
                        })
            
            return similar_comments
            
        except Exception as e:
            logger.error(f"Error searching similar comments: {e}")
            return []
    
    async def analyze_comment_with_gemini(self, comment_text: str, context: Optional[str] = None) -> Dict:
        """Analyze comment using Google Gemini"""
        try:
            prompt = f"""
            Analyze the following operator comment and provide insights:
            
            Comment: "{comment_text}"
            
            {f'Context: {context}' if context else ''}
            
            Please provide:
            1. Analysis of the comment's meaning and intent
            2. Potential issues or concerns identified
            3. Recommended actions
            4. Confidence score (0.0-1.0)
            
            Format your response as JSON with keys: analysis, issues, recommendations, confidence_score
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback if response is not valid JSON
                result = {
                    "analysis": response.text,
                    "issues": [],
                    "recommendations": [],
                    "confidence_score": 0.5
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing comment with Gemini: {e}")
            return {
                "analysis": f"Error analyzing comment: {str(e)}",
                "issues": [],
                "recommendations": [],
                "confidence_score": 0.0
            }
    
    async def process_comment(self, comment_id: int, text: str, metadata: Dict = None) -> Dict:
        """Process a comment through the complete RAG pipeline"""
        try:
            # Step 1: Find similar comments for context
            similar_comments = self.search_similar_comments(text, n_results=5)
            
            # Step 2: Build context from similar comments
            context = ""
            if similar_comments:
                context = "Similar comments from the past:\n"
                for comment in similar_comments[:3]:
                    context += f"- {comment['text']}\n"
            
            # Step 3: Analyze with Gemini
            analysis_result = await self.analyze_comment_with_gemini(text, context)
            
            # Step 4: Add to vector store
            self.add_comment_to_vector_store(comment_id, text, metadata)
            
            # Step 5: Return comprehensive result
            return {
                "comment_id": comment_id,
                "analysis": analysis_result.get("analysis", ""),
                "issues": analysis_result.get("issues", []),
                "recommendations": analysis_result.get("recommendations", []),
                "confidence_score": analysis_result.get("confidence_score", 0.0),
                "similar_comments": similar_comments,
                "embedding_created": True
            }
            
        except Exception as e:
            logger.error(f"Error processing comment {comment_id}: {e}")
            return {
                "comment_id": comment_id,
                "analysis": f"Error processing comment: {str(e)}",
                "issues": [],
                "recommendations": [],
                "confidence_score": 0.0,
                "similar_comments": [],
                "embedding_created": False
            }
    
    async def batch_process_comments(self, comments: List[Dict]) -> List[Dict]:
        """Process multiple comments in batch"""
        results = []
        
        for comment in comments:
            result = await self.process_comment(
                comment_id=comment["id"],
                text=comment["text"],
                metadata=comment.get("metadata", {})
            )
            results.append(result)
        
        return results
    
    def get_embedding_for_comment(self, comment_text: str) -> List[float]:
        """Get embedding for a comment text"""
        return self.create_embedding(comment_text)
    
    def update_comment_in_vector_store(self, comment_id: int, text: str, metadata: Dict = None):
        """Update an existing comment in vector store"""
        try:
            # Delete existing comment
            self.collection.delete(ids=[str(comment_id)])
            
            # Add updated comment
            self.add_comment_to_vector_store(comment_id, text, metadata)
            
            logger.info(f"Updated comment {comment_id} in vector store")
            
        except Exception as e:
            logger.error(f"Error updating comment in vector store: {e}")
            raise

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()