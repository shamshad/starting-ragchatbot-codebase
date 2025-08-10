import pytest
import sys
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "./test_chroma"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = "test_api_key"
        self.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.MAX_HISTORY = 2


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()


@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for testing"""
    mock_rag = Mock()
    
    mock_rag.query.return_value = (
        "This is a test response about the course material.",
        ["Python Basics - Lesson 1|http://example.com/lesson1"]
    )
    
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Advanced Python"]
    }
    
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System", "status": "running"}
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data for testing"""
    return {
        "query": "What are Python variables?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with session ID for testing"""
    return {
        "query": "How do I create functions in Python?", 
        "session_id": "test_session_456"
    }


@pytest.fixture
def expected_query_response():
    """Expected query response structure for testing"""
    return {
        "answer": "This is a test response about the course material.",
        "sources": ["Python Basics - Lesson 1|http://example.com/lesson1"],
        "session_id": "test_session_123"
    }


@pytest.fixture
def expected_course_stats():
    """Expected course stats response for testing"""
    return {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Advanced Python"]
    }


@pytest.fixture(autouse=True)
def mock_startup_event():
    """Mock the startup event to prevent document loading during tests"""
    with patch('app.startup_event'):
        yield