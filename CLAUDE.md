# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Setup and Dependencies
```bash
# Install uv package manager (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Add new dependencies (use uv, not pip)
uv add package_name

# Remove dependencies
uv remove package_name

# Create environment file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

**Important**: Always use `uv` for dependency management. Do not use `pip` directly in this project.

### Code Quality Tools
```bash
# Format code (black + isort)
./scripts/format.sh

# Run all linting checks (flake8, mypy, black --check, isort --check)
./scripts/lint.sh

# Complete quality workflow (format then lint)
./scripts/quality.sh

# Individual tools
uv run black backend/ main.py        # Format with black
uv run isort backend/ main.py         # Sort imports
uv run flake8 backend/ main.py        # Lint with flake8
uv run mypy backend/ main.py          # Type checking
```

**Code Standards**:
- Line length: 88 characters (black standard)
- Import sorting: isort with black profile
- Type hints encouraged but not required
- All Python code is automatically formatted with black and isort

### Development Server
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Server runs with auto-reload enabled for development

## Architecture Overview

This is a **Course Materials RAG System** built with a modular backend architecture and simple frontend interface.

### Core Processing Pipeline
The system follows a **retrieval-augmented generation** flow:
1. **Document Processing**: Course materials are parsed, chunked, and embedded
2. **Vector Storage**: ChromaDB stores embeddings for semantic search
3. **Query Processing**: User queries trigger Claude API with search tools
4. **Tool-Based Retrieval**: Claude dynamically searches vector database when needed
5. **Response Generation**: Final answers incorporate retrieved context

### Key Components

**RAG System (`rag_system.py`)**: Central orchestrator that coordinates all components
- Manages document ingestion and vector storage
- Handles query processing with AI generation
- Provides session management and conversation history

**Document Processor (`document_processor.py`)**: Structured document parsing
- Expects format: Course Title/Link/Instructor → Lesson markers → Content
- Sentence-based chunking with configurable size/overlap (800/100 chars)
- Context enhancement: adds course/lesson metadata to chunks

**Vector Store (`vector_store.py`)**: ChromaDB integration
- Uses `all-MiniLM-L6-v2` embeddings by default
- Supports course-specific and lesson-specific filtering
- Semantic search with metadata preservation

**AI Generator (`ai_generator.py`)**: Claude API integration
- Tool calling enabled for dynamic search execution
- System prompt optimized for educational content
- Handles conversation context and session history

**Search Tools (`search_tools.py`)**: Tool interface for Claude
- `CourseSearchTool`: Semantic search with course/lesson filtering
- Returns formatted results with source attribution
- Extensible tool framework for adding new capabilities

### Data Flow Architecture
```
Frontend (JS) → FastAPI → RAG System → AI Generator → Claude API
                                   ↓
                         Tool Manager → Search Tool → Vector Store → ChromaDB
```

### Configuration (`config.py`)
Key settings in `Config` class:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks  
- `MAX_RESULTS: 5` - Search results returned
- `MAX_HISTORY: 2` - Conversation context length
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`

### Document Structure Expected
Course documents should follow this format:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: Introduction
Lesson Link: [URL]
[Content...]

Lesson 1: [Title]
[Content...]
```

### Session Management
- Sessions created automatically on first query
- Conversation history maintained per session
- Session IDs returned to frontend for continuity

## Important Implementation Details

### Tool Calling Pattern
The system uses Claude's tool calling capability rather than traditional RAG embedding similarity search. Claude decides when to search based on query content.

### Error Handling
- Document processing gracefully handles encoding issues (UTF-8 fallback)
- Missing metadata defaults to filename/unknown values
- Empty search results return clear "no content found" messages

### Vector Database
- ChromaDB persists to `./chroma_db` directory
- Existing courses detected to avoid re-processing on startup
- Supports clearing and rebuilding entire database

### Frontend Integration
- Static files served directly by FastAPI
- Real-time loading indicators during query processing
- Source attribution displayed with collapsible sections
- Session persistence across page reloads