import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


class MockAnthropicResponse:
    """Mock Anthropic API response object"""

    def __init__(self, content_text=None, stop_reason="end_turn", tool_use_blocks=None):
        self.stop_reason = stop_reason

        if tool_use_blocks:
            # Mock response with tool use
            self.content = tool_use_blocks
        else:
            # Mock simple text response
            mock_content = Mock()
            mock_content.text = content_text or "Default response"
            self.content = [mock_content]


class MockToolUseBlock:
    """Mock tool use content block"""

    def __init__(self, tool_name, tool_input, block_id="tool_123"):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input
        self.id = block_id


class TestAIGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            self.mock_client = Mock()
            mock_anthropic_class.return_value = self.mock_client

            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_init(self):
        """Test AIGenerator initialization"""
        self.assertEqual(self.ai_generator.model, self.model)
        self.assertEqual(self.ai_generator.base_params["model"], self.model)
        self.assertEqual(self.ai_generator.base_params["temperature"], 0)
        self.assertEqual(self.ai_generator.base_params["max_tokens"], 800)

    def test_generate_response_simple_query(self):
        """Test generating response for simple query without tools"""
        # Mock API response
        mock_response = MockAnthropicResponse("This is a simple answer")
        self.mock_client.messages.create.return_value = mock_response

        result = self.ai_generator.generate_response("What is Python?")

        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]

        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["messages"][0]["content"], "What is Python?")
        self.assertIn(self.ai_generator.SYSTEM_PROMPT, call_args["system"])
        self.assertNotIn("tools", call_args)  # No tools for simple query

        self.assertEqual(result, "This is a simple answer")

    def test_generate_response_with_conversation_history(self):
        """Test generating response with conversation history"""
        mock_response = MockAnthropicResponse("Response with history")
        self.mock_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"

        result = self.ai_generator.generate_response(
            "Follow-up question", conversation_history=history
        )

        call_args = self.mock_client.messages.create.call_args[1]

        # Verify history is included in system prompt
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn(history, call_args["system"])

        self.assertEqual(result, "Response with history")

    def test_generate_response_with_tools(self):
        """Test generating response with tools available"""
        mock_response = MockAnthropicResponse("Response using tools")
        self.mock_client.messages.create.return_value = mock_response

        tools = [{"name": "search_tool", "description": "Search for content"}]

        result = self.ai_generator.generate_response(
            "Search for Python basics", tools=tools
        )

        call_args = self.mock_client.messages.create.call_args[1]

        # Verify tools are included in API call
        self.assertEqual(call_args["tools"], tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})

        self.assertEqual(result, "Response using tools")

    def test_generate_response_with_tool_execution(self):
        """Test generating response that requires tool execution"""
        # Mock initial response with tool use
        tool_block = MockToolUseBlock(
            "search_course_content",
            {"query": "python variables", "course_name": "Python Basics"},
        )
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )

        # Mock final response after tool execution
        final_response = MockAnthropicResponse(
            "Here's what I found about Python variables..."
        )

        # Configure mock client to return responses in sequence
        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about variables"

        tools = [{"name": "search_course_content", "description": "Search content"}]

        result = self.ai_generator.generate_response(
            "Explain Python variables", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify initial API call
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python variables",
            course_name="Python Basics",
        )

        # Verify final response
        self.assertEqual(result, "Here's what I found about Python variables...")

    def test_generate_response_tool_execution_failure(self):
        """Test handling of tool execution errors"""
        # Mock tool use response
        tool_block = MockToolUseBlock("search_course_content", {"query": "test"})
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )

        # Mock final response
        final_response = MockAnthropicResponse("I couldn't find relevant information")

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search error: Database connection failed"
        )

        tools = [{"name": "search_course_content", "description": "Search content"}]

        result = self.ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool was called and error was handled
        mock_tool_manager.execute_tool.assert_called_once()
        self.assertEqual(result, "I couldn't find relevant information")

    def test_generate_response_api_exception(self):
        """Test handling of Anthropic API exceptions"""
        # Mock API exception
        self.mock_client.messages.create.side_effect = Exception(
            "API Error: Invalid API key"
        )

        with self.assertRaises(Exception) as context:
            self.ai_generator.generate_response("Test query")

        self.assertIn("API Error", str(context.exception))

    def test_handle_tool_execution_multiple_tools(self):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool uses
        tool_block_1 = MockToolUseBlock(
            "search_course_content", {"query": "python"}, "tool_1"
        )
        tool_block_2 = MockToolUseBlock(
            "get_course_outline", {"course_title": "Python"}, "tool_2"
        )

        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block_1, tool_block_2]
        )

        final_response = MockAnthropicResponse("Combined results from both tools")

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search results",
            "Course outline",
        ]

        tools = [
            {"name": "search_course_content", "description": "Search"},
            {"name": "get_course_outline", "description": "Get outline"},
        ]

        result = self.ai_generator.generate_response(
            "Tell me about Python course", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify both tools were called
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python"
        )

        self.assertEqual(result, "Combined results from both tools")

    def test_handle_tool_execution_message_structure(self):
        """Test correct message structure for sequential tool execution"""
        tool_block = MockToolUseBlock("search_course_content", {"query": "test"})
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )

        # Second response: Claude decides not to use more tools (normal end)
        final_response = MockAnthropicResponse(
            content_text="Final answer",
            stop_reason="end_turn",  # Claude finishes without more tools
        )

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [{"name": "search_course_content", "description": "Search"}]

        self.ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # With sequential tool calling, there are 2 API calls total
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Check the first API call (initial query with tools)
        first_call_args = self.mock_client.messages.create.call_args_list[0][1]
        self.assertIn("tools", first_call_args)
        self.assertIn("tool_choice", first_call_args)

        # Check the second API call structure (Claude reasoning with tool results)
        second_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]

        # Should have: [user_query, assistant_tool_response, user_tool_results]
        self.assertEqual(len(messages), 3)

        # Check user's original query
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Test query")

        # Check assistant's tool use response
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], [tool_block])

        # Check tool results
        self.assertEqual(messages[2]["role"], "user")
        tool_results = messages[2]["content"]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0]["type"], "tool_result")
        self.assertEqual(tool_results[0]["tool_use_id"], "tool_123")
        self.assertEqual(tool_results[0]["content"], "Tool result")

        # Second call still has tools available (Claude can choose to use or not)
        self.assertIn("tools", second_call_args)
        self.assertIn("tool_choice", second_call_args)

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check key elements are present
        self.assertIn("course materials", system_prompt.lower())
        self.assertIn("tool usage", system_prompt.lower())
        self.assertIn("course content search", system_prompt.lower())
        self.assertIn("course outline", system_prompt.lower())
        self.assertIn("sequential tool calling", system_prompt.lower())
        self.assertIn("multi-step reasoning", system_prompt.lower())
        self.assertIn("brief, concise and focused", system_prompt.lower())

    def test_base_params_configuration(self):
        """Test that base parameters are configured correctly"""
        base_params = self.ai_generator.base_params

        self.assertEqual(base_params["model"], "claude-sonnet-4-20250514")
        self.assertEqual(base_params["temperature"], 0)  # Deterministic responses
        self.assertEqual(base_params["max_tokens"], 800)  # Reasonable limit

    def test_no_tool_manager_with_tool_use(self):
        """Test behavior when tools are provided but no tool manager"""
        tool_use_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_blocks=[
                MockToolUseBlock("search_course_content", {"query": "test"})
            ],
        )

        self.mock_client.messages.create.return_value = tool_use_response

        tools = [{"name": "search_course_content", "description": "Search"}]

        # With new sequential implementation: if no tool_manager, skip sequential rounds
        # and fall back to single API call behavior
        result = self.ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=None  # No tool manager provided
        )

        # Should make single API call and return fallback message
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
        # Should return fallback message since tool use blocks don't have .text attribute
        self.assertEqual(
            result,
            "Tools were requested but no tool manager was provided to execute them.",
        )


class TestAIGeneratorIntegration(unittest.TestCase):
    """Integration tests that test AIGenerator with realistic scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            self.mock_client = Mock()
            mock_anthropic_class.return_value = self.mock_client

            self.ai_generator = AIGenerator("test_key", "claude-sonnet-4-20250514")

            # Set up realistic tool definitions
            self.tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials with smart course name matching and lesson filtering",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for",
                            },
                            "course_name": {
                                "type": "string",
                                "description": "Course title (partial matches work)",
                            },
                            "lesson_number": {
                                "type": "integer",
                                "description": "Specific lesson number",
                            },
                        },
                        "required": ["query"],
                    },
                }
            ]

    def test_content_query_triggers_tool_use(self):
        """Test that content-specific queries trigger tool usage"""
        # This is a critical test - content queries should use tools

        tool_block = MockToolUseBlock(
            "search_course_content",
            {"query": "python functions", "course_name": "Python Basics"},
        )

        initial_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )

        final_response = MockAnthropicResponse(
            "Python functions are reusable blocks of code that perform specific tasks..."
        )

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Functions are defined with def keyword..."
        )

        result = self.ai_generator.generate_response(
            "Explain Python functions from the Python Basics course",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify tool was called - this is what we expect for content queries
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python functions",
            course_name="Python Basics",
        )

        self.assertIn("Python functions are reusable", result)

    def test_general_query_no_tool_use(self):
        """Test that general knowledge queries don't trigger tools"""
        # General queries should not use tools

        direct_response = MockAnthropicResponse(
            "Python is a high-level programming language..."
        )

        self.mock_client.messages.create.return_value = direct_response

        mock_tool_manager = Mock()

        result = self.ai_generator.generate_response(
            "What is Python programming language?",  # General question
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify no tool was called for general query
        mock_tool_manager.execute_tool.assert_not_called()

        self.assertIn("Python is a high-level", result)

    def test_error_handling_in_realistic_scenario(self):
        """Test error handling in a realistic failure scenario"""
        # Simulate API key error or other API failure

        self.mock_client.messages.create.side_effect = Exception(
            "AuthenticationError: Invalid API key"
        )

        mock_tool_manager = Mock()

        with self.assertRaises(Exception) as context:
            self.ai_generator.generate_response(
                "Search for Python basics",
                tools=self.tools,
                tool_manager=mock_tool_manager,
            )

        self.assertIn("AuthenticationError", str(context.exception))
        mock_tool_manager.execute_tool.assert_not_called()


class TestSequentialToolCalling(unittest.TestCase):
    """Tests for sequential tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            self.mock_client = Mock()
            mock_anthropic_class.return_value = self.mock_client

            self.ai_generator = AIGenerator("test_key", "claude-sonnet-4-20250514")

            self.tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials",
                },
                {"name": "get_course_outline", "description": "Get course outline"},
            ]

    def test_sequential_two_round_tool_calling(self):
        """Test successful two-round sequential tool calling"""
        # First round: Tool use response
        first_tool_block = MockToolUseBlock(
            "get_course_outline", {"course_title": "Python Basics"}, "tool_1"
        )
        first_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[first_tool_block]
        )

        # Second round: Another tool use response
        second_tool_block = MockToolUseBlock(
            "search_course_content",
            {"query": "functions", "course_name": "Advanced Python"},
            "tool_2",
        )
        second_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[second_tool_block]
        )

        # Final response
        final_response = MockAnthropicResponse(
            "Here's the comparison between the courses..."
        )

        # Configure mock client responses
        self.mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline with lesson 4: Functions",
            "Functions content from Advanced Python course",
        ]

        result = self.ai_generator.generate_response(
            "Compare lesson 4 topics of Python Basics with Advanced Python course",
            tools=self.tools,
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        # Verify both tool calls were made
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python Basics"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="functions", course_name="Advanced Python"
        )

        # Verify 3 API calls were made (2 rounds + final synthesis)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Verify final response
        self.assertEqual(result, "Here's the comparison between the courses...")

    def test_single_round_completion(self):
        """Test that single round completes without forcing second round"""
        # First round: Direct response (no tool use)
        direct_response = MockAnthropicResponse("This is a direct answer without tools")

        self.mock_client.messages.create.return_value = direct_response

        mock_tool_manager = Mock()

        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        # Verify only one API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

        # Verify no tools were called
        mock_tool_manager.execute_tool.assert_not_called()

        self.assertEqual(result, "This is a direct answer without tools")

    def test_max_rounds_reached_with_final_synthesis(self):
        """Test that max rounds triggers final synthesis call"""
        # First round: Tool use
        first_tool_block = MockToolUseBlock(
            "search_course_content", {"query": "python"}
        )
        first_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[first_tool_block]
        )

        # Second round: Tool use
        second_tool_block = MockToolUseBlock(
            "get_course_outline", {"course_title": "Python"}
        )
        second_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[second_tool_block]
        )

        # Final synthesis (no tools)
        final_response = MockAnthropicResponse("Final synthesized answer")

        self.mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Search result 2",
        ]

        result = self.ai_generator.generate_response(
            "Complex query requiring multiple searches",
            tools=self.tools,
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        # Verify 2 tool executions + 1 final synthesis call
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Check final API call has no tools
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", final_call_args)
        self.assertNotIn("tool_choice", final_call_args)

        self.assertEqual(result, "Final synthesized answer")

    def test_tool_execution_failure_terminates_early(self):
        """Test that tool execution failure terminates the sequence"""
        # First round: Tool use that will fail
        tool_block = MockToolUseBlock("search_course_content", {"query": "test"})
        response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[tool_block]
        )

        self.mock_client.messages.create.return_value = response

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        result = self.ai_generator.generate_response(
            "Test query", tools=self.tools, tool_manager=mock_tool_manager, max_rounds=2
        )

        # Verify early termination with error message
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

        self.assertIn("encountered an error", result)

    def test_message_structure_across_rounds(self):
        """Test correct message structure is maintained across rounds"""
        # Two rounds of tool use
        first_tool = MockToolUseBlock("search_course_content", {"query": "test1"})
        second_tool = MockToolUseBlock("get_course_outline", {"course_title": "test"})

        first_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[first_tool]
        )
        second_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[second_tool]
        )
        final_response = MockAnthropicResponse("Final answer")

        self.mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        self.ai_generator.generate_response(
            "Test query", tools=self.tools, tool_manager=mock_tool_manager
        )

        # Check final API call message structure
        final_call_args = self.mock_client.messages.create.call_args_list[2][1]
        messages = final_call_args["messages"]

        # Should have: [user_query, assistant_response1, tool_results1, assistant_response2, tool_results2]
        self.assertEqual(len(messages), 5)

        # Verify message roles and types
        expected_roles = ["user", "assistant", "user", "assistant", "user"]
        actual_roles = [msg["role"] for msg in messages]
        self.assertEqual(actual_roles, expected_roles)

    def test_backward_compatibility_no_tools(self):
        """Test that queries without tools still work as before"""
        response = MockAnthropicResponse("Simple answer")
        self.mock_client.messages.create.return_value = response

        result = self.ai_generator.generate_response("What is Python?")

        # Should make single API call without tools
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertNotIn("tools", call_args)

        self.assertEqual(result, "Simple answer")

    def test_custom_max_rounds(self):
        """Test custom max_rounds parameter"""
        # Mock responses for custom 1-round limit
        tool_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_blocks=[
                MockToolUseBlock("search_course_content", {"query": "test"})
            ],
        )
        final_response = MockAnthropicResponse("Answer after 1 round")

        self.mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        result = self.ai_generator.generate_response(
            "Test query",
            tools=self.tools,
            tool_manager=mock_tool_manager,
            max_rounds=1,  # Custom limit
        )

        # Should only do 1 round + final synthesis
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        self.assertEqual(result, "Answer after 1 round")


class TestSequentialToolCallingIntegration(unittest.TestCase):
    """Integration tests for realistic multi-step educational queries"""

    def setUp(self):
        """Set up realistic test fixtures"""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            self.mock_client = Mock()
            mock_anthropic_class.return_value = self.mock_client

            self.ai_generator = AIGenerator("test_key", "claude-sonnet-4-20250514")

            # Realistic tool definitions
            self.tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials with smart course name matching and lesson filtering",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "course_name": {"type": "string"},
                            "lesson_number": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_course_outline",
                    "description": "Get comprehensive course structure and lesson list",
                    "input_schema": {
                        "type": "object",
                        "properties": {"course_title": {"type": "string"}},
                        "required": ["course_title"],
                    },
                },
            ]

    def test_course_comparison_workflow(self):
        """Test realistic course comparison requiring multiple searches"""
        # Step 1: Get outline for first course
        outline_tool = MockToolUseBlock(
            "get_course_outline", {"course_title": "Python Basics"}, "outline_call"
        )
        outline_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[outline_tool]
        )

        # Step 2: Search second course based on first course content
        search_tool = MockToolUseBlock(
            "search_course_content",
            {"query": "object-oriented programming", "course_name": "Advanced Python"},
            "search_call",
        )
        search_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[search_tool]
        )

        # Step 3: Final comparison synthesis
        final_response = MockAnthropicResponse(
            "Both courses cover OOP concepts. Python Basics introduces classes in lesson 8, while Advanced Python covers inheritance and polymorphism extensively."
        )

        # Configure responses
        self.mock_client.messages.create.side_effect = [
            outline_response,
            search_response,
            final_response,
        ]

        # Mock realistic tool results
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            # Course outline result
            """Course: Python Basics
            Lesson 1: Variables and Data Types
            Lesson 8: Introduction to Classes
            Lesson 12: File Handling""",
            # Search result for Advanced Python
            """Found 3 matches in Advanced Python:
            Lesson 4: Advanced OOP Concepts - Inheritance, polymorphism, encapsulation
            Lesson 7: Design Patterns - Factory, Observer patterns using classes""",
        ]

        result = self.ai_generator.generate_response(
            "Compare the object-oriented programming coverage between Python Basics and Advanced Python courses",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify the complete workflow
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python Basics"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="object-oriented programming",
            course_name="Advanced Python",
        )

        self.assertIn("Both courses cover OOP", result)
        self.assertIn("lesson 8", result.lower())

    def test_prerequisite_analysis_workflow(self):
        """Test prerequisite analysis requiring course outline then content search"""
        # Step 1: Get advanced course outline
        outline_tool = MockToolUseBlock(
            "get_course_outline",
            {"course_title": "Machine Learning with Python"},
            "prereq_outline",
        )
        outline_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[outline_tool]
        )

        # Step 2: Search for prerequisite topics
        prereq_search = MockToolUseBlock(
            "search_course_content",
            {"query": "numpy pandas statistics", "course_name": "Python Data Science"},
            "prereq_search",
        )
        prereq_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[prereq_search]
        )

        # Final analysis
        final_response = MockAnthropicResponse(
            "To take Machine Learning with Python, you should first complete Python Data Science which covers the required NumPy and Pandas foundations."
        )

        self.mock_client.messages.create.side_effect = [
            outline_response,
            prereq_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            # Advanced course outline showing prerequisites needed
            """Course: Machine Learning with Python
            Prerequisites: NumPy, Pandas, Basic Statistics
            Lesson 1: Data Preprocessing with Pandas
            Lesson 3: Linear Regression using NumPy""",
            # Search results for prerequisites
            """Found prerequisite coverage in Python Data Science:
            Lesson 5: NumPy Arrays and Mathematical Operations
            Lesson 8: Pandas DataFrames and Data Manipulation  
            Lesson 12: Statistical Analysis with Python""",
        ]

        result = self.ai_generator.generate_response(
            "What prerequisites do I need before taking the Machine Learning with Python course?",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify prerequisite analysis workflow
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Machine Learning with Python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="numpy pandas statistics",
            course_name="Python Data Science",
        )

        self.assertIn("Python Data Science", result)
        self.assertIn("NumPy and Pandas", result)

    def test_lesson_topic_discovery_workflow(self):
        """Test finding courses with similar topics to a specific lesson"""
        # Step 1: Get specific lesson content
        lesson_search = MockToolUseBlock(
            "search_course_content",
            {
                "query": "lesson content",
                "course_name": "Web Development",
                "lesson_number": 5,
            },
            "lesson_lookup",
        )
        lesson_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[lesson_search]
        )

        # Step 2: Search for similar topics in other courses
        topic_search = MockToolUseBlock(
            "search_course_content",
            {"query": "API design REST endpoints"},
            "similar_topics",
        )
        topic_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[topic_search]
        )

        # Final recommendations
        final_response = MockAnthropicResponse(
            "Similar API design topics are covered in Backend Development (lesson 7) and Microservices Architecture (lessons 3-4)."
        )

        self.mock_client.messages.create.side_effect = [
            lesson_response,
            topic_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            # Specific lesson content
            """Lesson 5: RESTful API Design
            Topics: HTTP methods, endpoint structure, status codes, API versioning
            Hands-on: Building a user management API""",
            # Similar topics in other courses
            """Found related content:
            Backend Development - Lesson 7: Advanced API Patterns (GraphQL, pagination)  
            Microservices Architecture - Lesson 3: API Gateway Design
            Microservices Architecture - Lesson 4: Service Communication Patterns""",
        ]

        result = self.ai_generator.generate_response(
            "Find other courses that cover topics similar to lesson 5 of the Web Development course",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Verify topic discovery workflow
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson content",
            course_name="Web Development",
            lesson_number=5,
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="API design REST endpoints"
        )

        self.assertIn("Backend Development", result)
        self.assertIn("Microservices Architecture", result)
        self.assertIn("lesson 7", result.lower())

    def test_error_handling_in_educational_workflow(self):
        """Test graceful error handling during multi-step educational queries"""
        # First tool succeeds
        outline_tool = MockToolUseBlock(
            "get_course_outline", {"course_title": "Python"}
        )
        outline_response = MockAnthropicResponse(
            stop_reason="tool_use", tool_use_blocks=[outline_tool]
        )

        self.mock_client.messages.create.return_value = outline_response

        # Second tool fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection timeout"
        )

        result = self.ai_generator.generate_response(
            "Compare Python course with Java course structure",
            tools=self.tools,
            tool_manager=mock_tool_manager,
        )

        # Should handle error gracefully
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)
        self.assertIn("encountered an error", result)


if __name__ == "__main__":
    unittest.main()
