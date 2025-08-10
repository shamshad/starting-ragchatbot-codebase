import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, Lesson
from rag_system import RAGSystem


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


class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        self.config = MockConfig()

        # Create patches for all major dependencies
        self.patches = {
            "document_processor": patch("rag_system.DocumentProcessor"),
            "vector_store": patch("rag_system.VectorStore"),
            "ai_generator": patch("rag_system.AIGenerator"),
            "session_manager": patch("rag_system.SessionManager"),
            "course_search_tool": patch("rag_system.CourseSearchTool"),
            "course_outline_tool": patch("rag_system.CourseOutlineTool"),
            "tool_manager": patch("rag_system.ToolManager"),
        }

        # Start all patches and store mocks
        self.mocks = {}
        for name, patch_obj in self.patches.items():
            self.mocks[name] = patch_obj.start()

        # Configure mock returns
        self.mock_tool_manager_instance = Mock()
        self.mocks["tool_manager"].return_value = self.mock_tool_manager_instance

        self.mock_ai_generator_instance = Mock()
        self.mocks["ai_generator"].return_value = self.mock_ai_generator_instance

        self.mock_session_manager_instance = Mock()
        self.mocks["session_manager"].return_value = self.mock_session_manager_instance

        # Initialize RAG system
        self.rag_system = RAGSystem(self.config)

    def tearDown(self):
        """Clean up patches"""
        for patch_obj in self.patches.values():
            patch_obj.stop()

    def test_init_creates_all_components(self):
        """Test that RAGSystem initialization creates all required components"""
        # Verify all components were initialized
        self.mocks["document_processor"].assert_called_once_with(
            self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP
        )

        self.mocks["vector_store"].assert_called_once_with(
            self.config.CHROMA_PATH,
            self.config.EMBEDDING_MODEL,
            self.config.MAX_RESULTS,
        )

        self.mocks["ai_generator"].assert_called_once_with(
            self.config.ANTHROPIC_API_KEY, self.config.ANTHROPIC_MODEL
        )

        self.mocks["session_manager"].assert_called_once_with(self.config.MAX_HISTORY)

        # Verify tools were registered
        self.mock_tool_manager_instance.register_tool.assert_any_call(
            self.rag_system.search_tool
        )
        self.mock_tool_manager_instance.register_tool.assert_any_call(
            self.rag_system.outline_tool
        )

    def test_query_successful_content_search(self):
        """Test successful query execution with content search"""
        # Mock AI generator response
        self.mock_ai_generator_instance.generate_response.return_value = (
            "Here's information about Python variables..."
        )

        # Mock tool manager sources
        self.mock_tool_manager_instance.get_last_sources.return_value = [
            "Python Basics - Lesson 1|http://lesson1.com",
            "Python Basics - Lesson 2|http://lesson2.com",
        ]

        # Mock session manager
        self.mock_session_manager_instance.get_conversation_history.return_value = None

        response, sources = self.rag_system.query("Explain Python variables")

        # Verify AI generator was called with correct parameters
        self.mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = self.mock_ai_generator_instance.generate_response.call_args

        # Check query formatting
        self.assertIn(
            "Answer this question about course materials: Explain Python variables",
            call_args[1]["query"],
        )

        # Check tools and tool manager were provided
        self.assertIsNotNone(call_args[1]["tools"])
        self.assertEqual(call_args[1]["tool_manager"], self.mock_tool_manager_instance)

        # Verify sources were retrieved and reset
        self.mock_tool_manager_instance.get_last_sources.assert_called_once()
        self.mock_tool_manager_instance.reset_sources.assert_called_once()

        # Check response
        self.assertEqual(response, "Here's information about Python variables...")
        self.assertEqual(len(sources), 2)
        self.assertIn("Python Basics - Lesson 1", sources[0])

    def test_query_with_session_history(self):
        """Test query execution with session history"""
        session_id = "test_session_123"

        # Mock conversation history
        self.mock_session_manager_instance.get_conversation_history.return_value = (
            "Previous conversation context"
        )

        self.mock_ai_generator_instance.generate_response.return_value = (
            "Response with context"
        )
        self.mock_tool_manager_instance.get_last_sources.return_value = []

        response, sources = self.rag_system.query(
            "Follow-up question", session_id=session_id
        )

        # Verify conversation history was retrieved and used
        self.mock_session_manager_instance.get_conversation_history.assert_called_once_with(
            session_id
        )

        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertEqual(
            call_args["conversation_history"], "Previous conversation context"
        )

        # Verify conversation was updated
        self.mock_session_manager_instance.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Response with context"
        )

    def test_query_without_session_history(self):
        """Test query execution without session history"""
        self.mock_ai_generator_instance.generate_response.return_value = (
            "Response without history"
        )
        self.mock_tool_manager_instance.get_last_sources.return_value = []

        response, sources = self.rag_system.query("Standalone question")

        # Verify no session operations were called
        self.mock_session_manager_instance.get_conversation_history.assert_not_called()
        self.mock_session_manager_instance.add_exchange.assert_not_called()

        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertIsNone(call_args["conversation_history"])

    def test_query_ai_generator_exception(self):
        """Test query handling when AI generator raises exception"""
        # Mock AI generator to raise exception
        self.mock_ai_generator_instance.generate_response.side_effect = Exception(
            "API Error: Invalid API key"
        )

        # This should propagate the exception
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")

        self.assertIn("API Error", str(context.exception))

        # Sources should not be retrieved if AI generation fails
        self.mock_tool_manager_instance.get_last_sources.assert_not_called()

    def test_query_tool_manager_sources_handling(self):
        """Test proper handling of tool manager sources"""
        self.mock_ai_generator_instance.generate_response.return_value = "Response"

        # Test with various source formats
        test_sources = [
            "Course 1 - Lesson 1",  # Without link
            "Course 2 - Lesson 2|http://example.com/lesson2",  # With link
            "General Course",  # No lesson number
        ]

        self.mock_tool_manager_instance.get_last_sources.return_value = test_sources

        response, sources = self.rag_system.query("Test query")

        # Verify sources are passed through correctly
        self.assertEqual(sources, test_sources)

        # Verify sources were reset after retrieval
        self.mock_tool_manager_instance.reset_sources.assert_called_once()

    def test_add_course_document_success(self):
        """Test successful course document addition"""
        # Mock document processor
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://test.com",
            lessons=[],
        )
        mock_chunks = ["chunk1", "chunk2", "chunk3"]

        mock_doc_processor = self.rag_system.document_processor
        mock_doc_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Mock vector store
        mock_vector_store = self.rag_system.vector_store

        course, chunk_count = self.rag_system.add_course_document("/path/to/course.txt")

        # Verify document processing
        mock_doc_processor.process_course_document.assert_called_once_with(
            "/path/to/course.txt"
        )

        # Verify vector store operations
        mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)

        # Check return values
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 3)

    def test_add_course_document_failure(self):
        """Test course document addition with processing error"""
        # Mock document processor to raise exception
        mock_doc_processor = self.rag_system.document_processor
        mock_doc_processor.process_course_document.side_effect = Exception(
            "File not found"
        )

        course, chunk_count = self.rag_system.add_course_document("/invalid/path.txt")

        # Should return None and 0 on failure
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)

        # Vector store should not be called
        self.rag_system.vector_store.add_course_metadata.assert_not_called()
        self.rag_system.vector_store.add_course_content.assert_not_called()

    def test_get_course_analytics(self):
        """Test getting course analytics"""
        # Mock vector store responses
        mock_vector_store = self.rag_system.vector_store
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = [
            "Python Basics",
            "Advanced Python",
            "Data Science",
            "Web Development",
            "Machine Learning",
        ]

        analytics = self.rag_system.get_course_analytics()

        # Verify correct calls were made
        mock_vector_store.get_course_count.assert_called_once()
        mock_vector_store.get_existing_course_titles.assert_called_once()

        # Check analytics structure
        self.assertEqual(analytics["total_courses"], 5)
        self.assertEqual(len(analytics["course_titles"]), 5)
        self.assertIn("Python Basics", analytics["course_titles"])


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for RAG System with realistic scenarios"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.config = MockConfig()

        # Use real tool manager and mock only external dependencies
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            # Configure mocks
            self.mock_vector_store_instance = Mock()
            mock_vector_store.return_value = self.mock_vector_store_instance

            self.mock_ai_generator_instance = Mock()
            mock_ai_gen.return_value = self.mock_ai_generator_instance

            self.mock_session_manager_instance = Mock()
            mock_session_mgr.return_value = self.mock_session_manager_instance

            # Initialize RAG system
            self.rag_system = RAGSystem(self.config)

    def test_end_to_end_content_query_flow(self):
        """Test complete end-to-end flow for content query"""
        # This tests the integration between components for a typical content query

        query = "Explain Python functions from the Python Basics course"

        # Mock AI generator to simulate tool usage
        self.mock_ai_generator_instance.generate_response.return_value = (
            "Python functions are reusable blocks of code..."
        )

        # Mock tool manager to simulate search tool execution
        search_tool_mock = Mock()
        search_tool_mock.last_sources = ["Python Basics - Lesson 3|http://lesson3.com"]

        # Replace the search tool with our mock
        self.rag_system.search_tool = search_tool_mock
        self.rag_system.tool_manager.tools["search_course_content"] = search_tool_mock
        self.rag_system.tool_manager.get_last_sources = Mock(
            return_value=search_tool_mock.last_sources
        )

        response, sources = self.rag_system.query(query)

        # Verify the complete flow
        self.mock_ai_generator_instance.generate_response.assert_called_once()

        # Check that query was formatted correctly
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertIn("Answer this question about course materials", call_args["query"])
        self.assertIn(query, call_args["query"])

        # Verify tools were provided
        self.assertIsNotNone(call_args["tools"])
        self.assertIsNotNone(call_args["tool_manager"])

        # Check response and sources
        self.assertEqual(response, "Python functions are reusable blocks of code...")
        self.assertEqual(len(sources), 1)
        self.assertIn("Python Basics - Lesson 3", sources[0])

    def test_query_failure_scenarios(self):
        """Test various failure scenarios that could cause 'query failed'"""

        # Scenario 1: API Key error
        self.mock_ai_generator_instance.generate_response.side_effect = Exception(
            "AuthenticationError: Invalid API key"
        )

        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")

        self.assertIn("AuthenticationError", str(context.exception))

        # Reset for next test
        self.mock_ai_generator_instance.generate_response.side_effect = None

        # Scenario 2: Vector store connection error
        # This would be caught in the search tool, not here directly
        self.mock_ai_generator_instance.generate_response.return_value = (
            "I couldn't find information"
        )

        response, sources = self.rag_system.query("Test query")

        # Should still work but might not have useful content
        self.assertEqual(response, "I couldn't find information")

    def test_session_management_integration(self):
        """Test session management integration"""
        session_id = "integration_test_session"

        # Mock session history
        self.mock_session_manager_instance.get_conversation_history.return_value = (
            "User: What is Python?\nAssistant: Python is a programming language."
        )

        self.mock_ai_generator_instance.generate_response.return_value = (
            "Functions in Python are defined with the def keyword..."
        )

        response, sources = self.rag_system.query(
            "How do I create functions?", session_id=session_id
        )

        # Verify session operations
        self.mock_session_manager_instance.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify conversation history was passed to AI generator
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertIn(
            "Python is a programming language", call_args["conversation_history"]
        )

        # Verify conversation was updated
        self.mock_session_manager_instance.add_exchange.assert_called_once_with(
            session_id,
            "How do I create functions?",
            "Functions in Python are defined with the def keyword...",
        )

    def test_tool_registration_and_definitions(self):
        """Test that tools are properly registered and accessible"""
        # Verify tools are registered
        tool_manager = self.rag_system.tool_manager

        # Should have search tool and outline tool
        self.assertIn("search_course_content", tool_manager.tools)
        self.assertIn("get_course_outline", tool_manager.tools)

        # Verify tool definitions are available for AI
        definitions = tool_manager.get_tool_definitions()

        self.assertEqual(len(definitions), 2)

        # Find search tool definition
        search_def = next(
            d for d in definitions if d["name"] == "search_course_content"
        )
        self.assertIn("description", search_def)
        self.assertIn("input_schema", search_def)

        # Check required parameters
        required_params = search_def["input_schema"]["required"]
        self.assertIn("query", required_params)


if __name__ == "__main__":
    unittest.main()
