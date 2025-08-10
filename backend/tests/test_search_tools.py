import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted for Anthropic API"""
        definition = self.search_tool.get_tool_definition()
        
        self.assertIsInstance(definition, dict)
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        
        # Check required parameters
        required_params = definition["input_schema"]["required"]
        self.assertIn("query", required_params)
        self.assertEqual(len(required_params), 1)  # Only query is required
        
        # Check optional parameters exist
        properties = definition["input_schema"]["properties"]
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)
    
    def test_execute_successful_search(self):
        """Test successful search execution with results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Content from course 1", "More content"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = self.search_tool.execute("python variables")
        
        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="python variables",
            course_name=None,
            lesson_number=None
        )
        
        # Check result formatting
        self.assertIsInstance(result, str)
        self.assertIn("Python Basics", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Content from course 1", result)
        
        # Check sources were stored
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertIn("Python Basics - Lesson 1", self.search_tool.last_sources[0])
    
    def test_execute_with_course_filter(self):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="decorators",
            course_name="Advanced Python"
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        self.assertIn("Advanced Python", result)
    
    def test_execute_with_lesson_filter(self):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="pandas dataframes",
            lesson_number=5
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="pandas dataframes",
            course_name=None,
            lesson_number=5
        )
        
        self.assertIn("Lesson 5", result)
    
    def test_execute_empty_results(self):
        """Test search execution with no results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        self.assertIn("No relevant content found", result)
        self.assertEqual(len(self.search_tool.last_sources), 0)
    
    def test_execute_empty_results_with_filters(self):
        """Test empty results message includes filter information"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        # Test with course filter
        result = self.search_tool.execute(
            query="topic",
            course_name="Machine Learning"
        )
        
        self.assertIn("No relevant content found", result)
        self.assertIn("in course 'Machine Learning'", result)
        
        # Test with lesson filter
        result = self.search_tool.execute(
            query="topic",
            lesson_number=3
        )
        
        self.assertIn("in lesson 3", result)
    
    def test_execute_search_error(self):
        """Test search execution with error from vector store"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Vector store connection failed"
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        self.assertEqual(result, "Vector store connection failed")
    
    def test_format_results_with_links(self):
        """Test result formatting includes lesson links when available"""
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Web Development", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"
        
        result = self.search_tool.execute("HTML basics")
        
        # Check that get_lesson_link was called
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Web Development", 2)
        
        # Check source includes link information
        self.assertEqual(len(self.search_tool.last_sources), 1)
        source_data = self.search_tool.last_sources[0]
        self.assertIn("Web Development - Lesson 2", source_data)
        self.assertIn("https://example.com/lesson2", source_data)
    
    def test_format_results_without_links(self):
        """Test result formatting when no lesson link is available"""
        mock_results = SearchResults(
            documents=["Content without link"],
            metadata=[{"course_title": "unknown", "lesson_number": None}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("general topic")
        
        # Should not call get_lesson_link for unknown course or no lesson number
        self.mock_vector_store.get_lesson_link.assert_not_called()
        
        # Check source format without link
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(self.search_tool.last_sources[0], "unknown")


class TestToolManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool"
        }
        self.mock_tool.execute.return_value = "test result"
    
    def test_register_tool(self):
        """Test tool registration"""
        self.tool_manager.register_tool(self.mock_tool)
        
        self.assertIn("test_tool", self.tool_manager.tools)
        self.assertEqual(self.tool_manager.tools["test_tool"], self.mock_tool)
    
    def test_register_tool_without_name(self):
        """Test tool registration fails without name in definition"""
        self.mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with self.assertRaises(ValueError):
            self.tool_manager.register_tool(self.mock_tool)
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        self.tool_manager.register_tool(self.mock_tool)
        
        definitions = self.tool_manager.get_tool_definitions()
        
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "test_tool")
    
    def test_execute_tool(self):
        """Test tool execution"""
        self.tool_manager.register_tool(self.mock_tool)
        
        result = self.tool_manager.execute_tool("test_tool", param1="value1")
        
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        self.assertEqual(result, "test result")
    
    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool returns error message"""
        result = self.tool_manager.execute_tool("nonexistent_tool")
        
        self.assertEqual(result, "Tool 'nonexistent_tool' not found")
    
    def test_get_last_sources(self):
        """Test getting sources from tools that track them"""
        # Mock a tool with sources
        tool_with_sources = Mock()
        tool_with_sources.get_tool_definition.return_value = {"name": "source_tool"}
        tool_with_sources.last_sources = ["source1", "source2"]
        
        self.tool_manager.register_tool(tool_with_sources)
        
        sources = self.tool_manager.get_last_sources()
        
        self.assertEqual(sources, ["source1", "source2"])
    
    def test_get_last_sources_no_sources(self):
        """Test getting sources when no tools have sources"""
        self.tool_manager.register_tool(self.mock_tool)
        
        sources = self.tool_manager.get_last_sources()
        
        self.assertEqual(sources, [])
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        tool_with_sources = Mock()
        tool_with_sources.get_tool_definition.return_value = {"name": "source_tool"}
        tool_with_sources.last_sources = ["source1", "source2"]
        
        self.tool_manager.register_tool(tool_with_sources)
        
        self.tool_manager.reset_sources()
        
        self.assertEqual(tool_with_sources.last_sources, [])


if __name__ == '__main__':
    unittest.main()