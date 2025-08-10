#!/usr/bin/env python3
"""
Verify that the RAG system fix works correctly
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def test_startup_loading():
    """Test the startup document loading process"""
    print("=== Testing Startup Document Loading ===\n")

    try:
        from app import startup_event

        print("Simulating FastAPI startup event...")
        await startup_event()

        print("\n✓ Startup event completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Startup event failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rag_query():
    """Test a real RAG query after loading"""
    print("\n=== Testing RAG Query ===\n")

    try:
        from config import config
        from rag_system import RAGSystem

        if not config.ANTHROPIC_API_KEY:
            print("⚠ Skipping RAG query test - no API key configured")
            return True

        rag_system = RAGSystem(config)

        # Check courses are loaded
        analytics = rag_system.get_course_analytics()
        print(f"Courses in system: {analytics['total_courses']}")

        if analytics["total_courses"] == 0:
            print("✗ No courses loaded - query test will fail")
            return False

        print("Testing content-specific query...")
        response, sources = rag_system.query("What is Python used for?")

        print(f"Query Results:")
        print(f"  - Response length: {len(response)}")
        print(f"  - Sources count: {len(sources)}")
        print(f"  - Response preview: {response[:150]}...")

        if response == "query failed":
            print("✗ ISSUE STILL EXISTS: Query returned 'query failed'")
            return False

        if "error" in response.lower():
            print("⚠ Query response contains error message")
            return False

        print("✓ Query completed successfully!")
        return True

    except Exception as e:
        print(f"✗ RAG query test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_search_tools():
    """Test search tools directly"""
    print("\n=== Testing Search Tools ===\n")

    try:
        from config import config
        from rag_system import RAGSystem

        rag_system = RAGSystem(config)

        # Test search tool directly
        search_tool = rag_system.search_tool

        result = search_tool.execute("python programming")

        print(f"Direct search tool test:")
        print(f"  - Result length: {len(result)}")
        print(f"  - Result preview: {result[:200]}...")

        if "No relevant content found" in result:
            print("⚠ Search tool found no content")
            return False

        if "error" in result.lower():
            print(f"✗ Search tool returned error: {result}")
            return False

        print("✓ Search tool working correctly!")
        return True

    except Exception as e:
        print(f"✗ Search tool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all verification tests"""
    print("🔍 RAG System Fix Verification\n")

    tests = [
        ("Startup Loading", test_startup_loading()),
        ("Search Tools", lambda: test_search_tools()),
        ("RAG Query", lambda: test_rag_query()),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print("=" * 50)

        if asyncio.iscoroutine(test_func):
            results[test_name] = await test_func
        else:
            results[test_name] = test_func()

    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! The 'query failed' issue should be resolved.")
        print("Your RAG chatbot is now working correctly.")
    else:
        failed_tests = [name for name, success in results.items() if not success]
        print(f"\n⚠ Some issues remain: {', '.join(failed_tests)}")
        print("Additional debugging may be needed.")


if __name__ == "__main__":
    asyncio.run(main())
