import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import os
from typing import Any, Dict, List, Optional

from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


# API Endpoints


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    print("Starting document loading process...")
    print(f"Current working directory: {os.getcwd()}")

    # Try multiple possible paths for docs folder
    possible_docs_paths = [
        "../docs",  # Standard path when running from backend/
        "docs",  # If running from project root
        "../../docs",  # If running from nested directory
        "./docs",  # Alternative current directory
    ]

    docs_loaded = False

    for docs_path in possible_docs_paths:
        print(f"Checking for docs folder at: {docs_path}")
        if os.path.exists(docs_path):
            print(f"✓ Found docs folder at {docs_path}")
            try:
                # List files in docs folder for debugging
                files = os.listdir(docs_path)
                course_files = [
                    f for f in files if f.lower().endswith((".txt", ".pdf", ".docx"))
                ]
                print(f"Found {len(course_files)} course files: {course_files}")

                if course_files:
                    print("Loading course documents...")
                    courses, chunks = rag_system.add_course_folder(
                        docs_path, clear_existing=False
                    )
                    print(
                        f"✓ Successfully loaded {courses} courses with {chunks} chunks"
                    )
                    docs_loaded = True
                    break
                else:
                    print(f"⚠ No course files found in {docs_path}")
            except Exception as e:
                print(f"✗ Error loading documents from {docs_path}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"✗ Docs folder not found at {docs_path}")

    if not docs_loaded:
        print("⚠ WARNING: No course documents were loaded!")
        print("The RAG system will not be able to answer content-specific questions.")
        print("Available files in current directory:", os.listdir("."))

        # Check if vector store has existing content
        try:
            analytics = rag_system.get_course_analytics()
            if analytics["total_courses"] > 0:
                print(
                    f"✓ Found {analytics['total_courses']} existing courses in vector store"
                )
                docs_loaded = True
            else:
                print("✗ No existing courses found in vector store")
        except Exception as e:
            print(f"Error checking existing courses: {e}")

    # Final status
    if docs_loaded:
        analytics = rag_system.get_course_analytics()
        print(
            f"🎉 Startup complete! System ready with {analytics['total_courses']} courses"
        )
    else:
        print("⚠ System started but no course content available")


import os
from pathlib import Path

from fastapi.responses import FileResponse

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
