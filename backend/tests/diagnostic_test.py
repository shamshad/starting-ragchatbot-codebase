#!/usr/bin/env python3
"""
Diagnostic script to test the RAG system and identify where "query failed" occurs
"""

import os
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_config():
    """Test configuration loading"""
    try:
        from config import config

        print("✓ Config loaded successfully")
        print(f"  - API Key present: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"  - Model: {config.ANTHROPIC_MODEL}")
        print(f"  - ChromaDB path: {config.CHROMA_PATH}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_vector_store():
    """Test VectorStore initialization and basic operations"""
    try:
        from config import config
        from vector_store import VectorStore

        # Initialize vector store
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        print("✓ VectorStore initialized successfully")

        # Test getting existing courses
        existing_courses = vector_store.get_existing_course_titles()
        print(f"  - Existing courses count: {len(existing_courses)}")
        if existing_courses:
            print(f"  - Sample course titles: {existing_courses[:3]}")

        # Test a simple search
        if existing_courses:
            results = vector_store.search("python")
            print(
                f"  - Search test results count: {len(results.documents) if results else 0}"
            )
            if results and results.error:
                print(f"  - Search error: {results.error}")

        return True
    except Exception as e:
        print(f"✗ VectorStore test failed: {e}")
        traceback.print_exc()
        return False


def test_ai_generator():
    """Test AI generator initialization (without API call)"""
    try:
        from ai_generator import AIGenerator
        from config import config

        if not config.ANTHROPIC_API_KEY:
            print("⚠ AI Generator test skipped: No API key configured")
            return False

        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        print("✓ AIGenerator initialized successfully")
        print(f"  - Model: {ai_generator.model}")
        print(f"  - Base params configured: {bool(ai_generator.base_params)}")

        return True
    except Exception as e:
        print(f"✗ AIGenerator test failed: {e}")
        traceback.print_exc()
        return False


def test_search_tools():
    """Test search tools initialization"""
    try:
        from config import config
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import VectorStore

        # Initialize components
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(vector_store)
        tool_manager = ToolManager()

        # Register tool
        tool_manager.register_tool(search_tool)

        print("✓ Search tools initialized successfully")

        # Test tool definition
        tool_def = search_tool.get_tool_definition()
        print(f"  - Tool name: {tool_def['name']}")
        print(f"  - Required params: {tool_def['input_schema']['required']}")

        # Test tool execution with simple query
        result = search_tool.execute("python")
        print(f"  - Test query result length: {len(result) if result else 0}")
        if "No relevant content found" in result:
            print("  - ⚠ No content found in vector store")
        elif "error" in result.lower():
            print(f"  - ⚠ Search error: {result[:100]}...")
        else:
            print("  - ✓ Search returned content")

        return True
    except Exception as e:
        print(f"✗ Search tools test failed: {e}")
        traceback.print_exc()
        return False


def test_rag_system():
    """Test RAG system initialization and basic query"""
    try:
        from config import config
        from rag_system import RAGSystem

        if not config.ANTHROPIC_API_KEY:
            print("⚠ RAG System test skipped: No API key configured")
            return False

        # Initialize RAG system
        rag_system = RAGSystem(config)
        print("✓ RAG System initialized successfully")

        # Test analytics
        analytics = rag_system.get_course_analytics()
        print(f"  - Total courses loaded: {analytics['total_courses']}")

        if analytics["total_courses"] == 0:
            print("  - ⚠ No courses loaded in vector store")
            return False

        print("  - Sample course titles:", analytics["course_titles"][:2])

        # Test a simple query (this will call the API)
        print("  - Testing simple query (will call Anthropic API)...")
        try:
            response, sources = rag_system.query("What is Python?")
            print(f"  - ✓ Query successful, response length: {len(response)}")
            print(f"  - Sources returned: {len(sources)}")
            if response == "query failed":
                print("  - ✗ ISSUE FOUND: Query returned 'query failed'")
                return False
        except Exception as query_error:
            print(f"  - ✗ Query failed with error: {query_error}")
            traceback.print_exc()
            return False

        return True
    except Exception as e:
        print(f"✗ RAG System test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run diagnostic tests"""
    print("=== RAG System Diagnostic Tests ===\n")

    tests = [
        ("Configuration", test_config),
        ("VectorStore", test_vector_store),
        ("AIGenerator", test_ai_generator),
        ("Search Tools", test_search_tools),
        ("RAG System", test_rag_system),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()

    print("\n=== Summary ===")
    for test_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {'PASS' if success else 'FAIL'}")

    failed_tests = [name for name, success in results.items() if not success]
    if failed_tests:
        print(f"\n⚠ Issues found in: {', '.join(failed_tests)}")
        print("These components need investigation for the 'query failed' issue.")
    else:
        print(
            "\n✓ All components working - issue might be in API integration or frontend."
        )


if __name__ == "__main__":
    main()
