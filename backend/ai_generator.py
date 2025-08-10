from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools.

Tool Usage:
- **Sequential Tool Calling**: You can make multiple tool calls across up to 2 separate reasoning rounds to gather comprehensive information
- **Course Content Search**: Use for questions about specific course content or detailed educational materials
- **Course Outline**: Use for questions about course structure, lesson lists, course overviews, or curriculum information
- **Multi-step Reasoning**: First gather information, then reason about results to make additional searches if needed
- Synthesize results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Calling Examples:
- "Compare topics in lesson 4 of course X with course Y" → Get outline/content of course X lesson 4 → Search course Y for similar topics → Synthesize comparison
- "Find courses discussing the same topic as lesson 3 of MCP course" → Get lesson 3 content/title → Search for courses covering that topic → Present results
- "What prerequisites are needed before taking the advanced Python course?" → Get course outline → Search for prerequisite topics in other courses → Compile requirements

Course Outline Tool:
- Returns: Course title, course link, instructor, and complete lesson list with numbers and titles
- Use for queries about:
  - "What lessons are in [course]?"
  - "Course outline for [course]"
  - "What topics does [course] cover?"
  - "Course structure of [course]"

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool(s) first, then answer
- **Multi-step queries**: Use first tool call to gather initial information, evaluate if additional searches are needed for complete answer
- **Course outline queries**: Always include the course title, course link, and complete numbered lesson list
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool calling rounds (default 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # If tools are available and tool manager is provided, use sequential rounds
        if tools and tool_manager:
            return self._execute_sequential_rounds(
                query, system_content, tools, tool_manager, max_rounds
            )

        # For non-tool queries, make single API call
        response = self.client.messages.create(**api_params)

        # Handle case where tools were provided but no tool manager
        # In this case, just return the first text content if available
        for content_block in response.content:
            if hasattr(content_block, "text"):
                return content_block.text

        # Fallback for tool use responses without tool manager
        return "Tools were requested but no tool manager was provided to execute them."

    def _execute_sequential_rounds(
        self,
        query: str,
        system_content: str,
        tools: List,
        tool_manager,
        max_rounds: int,
    ) -> str:
        """
        Execute up to max_rounds of tool calling with Claude reasoning between each.

        Args:
            query: The user's original question
            system_content: System prompt content
            tools: Available tools for Claude to use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds to execute

        Returns:
            Final response text after all rounds completed
        """
        messages = [{"role": "user", "content": query}]
        current_round = 0

        while current_round < max_rounds:
            # Make API call WITH tools available
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            response = self.client.messages.create(**api_params)

            # Check if Claude wants to use tools
            if response.stop_reason != "tool_use":
                # No tool use requested - Claude is done reasoning
                return response.content[0].text

            # Add Claude's response with tool calls to message history
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = self._execute_tools(response, tool_manager)
            if tool_results is None:  # Tool execution failed
                return "I encountered an error while searching for information. Please try again."

            # Add tool results to message history
            messages.append({"role": "user", "content": tool_results})
            current_round += 1

        # Final round without tools to force Claude to synthesize answer
        final_api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_api_params)
        return final_response.content[0].text

    def _execute_tools(self, response, tool_manager) -> Optional[List[Dict]]:
        """
        Execute all tool calls in a response and return formatted results.

        Args:
            response: Claude's response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dictionaries, or None if execution fails
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    # Log error and return None to signal failure
                    print(f"Tool execution error: {e}")
                    return None

        return tool_results

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
