#!/usr/bin/env python3
"""
Test course document loading to identify why vector store is empty
"""

import os
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_docs_folder():
    """Check if docs folder exists and has content"""
    docs_path = "../../docs"  # Fixed path to go up to project root
    full_docs_path = os.path.join(os.path.dirname(__file__), docs_path)

    print(f"Checking docs folder at: {full_docs_path}")

    if not os.path.exists(full_docs_path):
        print("✗ Docs folder does not exist")
        return False

    print("✓ Docs folder exists")

    # List files in docs folder
    files = os.listdir(full_docs_path)
    print(f"Files in docs folder: {files}")

    # Check for course files
    course_files = [f for f in files if f.lower().endswith((".txt", ".pdf", ".docx"))]
    print(f"Course files found: {course_files}")

    if not course_files:
        print("✗ No course files found in docs folder")
        return False

    print(f"✓ Found {len(course_files)} course files")
    return True


def test_document_processing():
    """Test document processor with actual course files"""
    try:
        from config import config
        from document_processor import DocumentProcessor

        processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print("✓ DocumentProcessor initialized")

        # Find first course file
        docs_path = "../../docs"
        full_docs_path = os.path.join(os.path.dirname(__file__), docs_path)

        if not os.path.exists(full_docs_path):
            print("✗ Docs folder not found")
            return False

        course_files = [
            f
            for f in os.listdir(full_docs_path)
            if f.lower().endswith((".txt", ".pdf", ".docx"))
        ]
        if not course_files:
            print("✗ No course files found")
            return False

        # Test processing first file
        test_file = os.path.join(full_docs_path, course_files[0])
        print(f"Testing document processing with: {course_files[0]}")

        course, chunks = processor.process_course_document(test_file)

        if course is None:
            print("✗ Document processing returned None")
            return False

        print(f"✓ Course processed successfully:")
        print(f"  - Title: {course.title}")
        print(f"  - Instructor: {course.instructor}")
        print(f"  - Lessons: {len(course.lessons)}")
        print(f"  - Chunks created: {len(chunks)}")

        if len(chunks) == 0:
            print("⚠ No chunks created - this might be an issue")
            return False

        # Show sample chunk
        if chunks:
            print(
                f"  - Sample chunk content (first 100 chars): {chunks[0].content[:100]}..."
            )

        return True

    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        traceback.print_exc()
        return False


def test_course_loading_to_vector_store():
    """Test loading courses into vector store"""
    try:
        from config import config
        from rag_system import RAGSystem

        rag_system = RAGSystem(config)
        print("✓ RAG System initialized")

        # Check current state
        analytics = rag_system.get_course_analytics()
        print(f"Current courses in vector store: {analytics['total_courses']}")

        # Try to load courses from docs folder
        docs_path = "../../docs"
        full_docs_path = os.path.join(os.path.dirname(__file__), docs_path)

        if not os.path.exists(full_docs_path):
            print("✗ Docs folder not found")
            return False

        print(f"Attempting to load courses from: {full_docs_path}")

        # Load courses (this will skip existing ones)
        courses_added, chunks_added = rag_system.add_course_folder(
            full_docs_path, clear_existing=False
        )

        print(f"Load results:")
        print(f"  - Courses added: {courses_added}")
        print(f"  - Chunks added: {chunks_added}")

        # Check final state
        final_analytics = rag_system.get_course_analytics()
        print(f"Final courses in vector store: {final_analytics['total_courses']}")
        print(f"Course titles: {final_analytics['course_titles']}")

        if final_analytics["total_courses"] == 0:
            print("✗ No courses were loaded into vector store")
            return False

        print("✓ Courses successfully loaded into vector store")

        # Test search functionality
        print("Testing search with loaded courses...")
        results = rag_system.vector_store.search("python")
        print(f"Search results count: {len(results.documents)}")

        if results.error:
            print(f"Search error: {results.error}")
            return False

        if len(results.documents) == 0:
            print("⚠ Search returned no results even with loaded courses")
        else:
            print("✓ Search working with loaded courses")

        return True

    except Exception as e:
        print(f"✗ Course loading test failed: {e}")
        traceback.print_exc()
        return False


def test_api_query():
    """Test an actual API query"""
    try:
        from config import config
        from rag_system import RAGSystem

        if not config.ANTHROPIC_API_KEY:
            print("⚠ Skipping API test - no API key")
            return True

        rag_system = RAGSystem(config)

        # Ensure courses are loaded first
        docs_path = "../../docs"
        full_docs_path = os.path.join(os.path.dirname(__file__), docs_path)
        if os.path.exists(full_docs_path):
            rag_system.add_course_folder(full_docs_path, clear_existing=False)

        analytics = rag_system.get_course_analytics()
        if analytics["total_courses"] == 0:
            print("⚠ Skipping API test - no courses loaded")
            return True

        print("Testing actual API query with loaded courses...")

        # Test a content-specific query
        response, sources = rag_system.query("What is covered in lesson 1?")

        print(f"API Query Results:")
        print(f"  - Response length: {len(response)}")
        print(f"  - Sources count: {len(sources)}")
        print(f"  - Response preview: {response[:200]}...")

        if response == "query failed":
            print("✗ FOUND THE ISSUE: API query returned 'query failed'")
            return False

        if "I couldn't find" in response or "no relevant content" in response.lower():
            print("⚠ API query couldn't find content, but didn't fail completely")
        else:
            print("✓ API query returned meaningful content")

        return True

    except Exception as e:
        print(f"✗ API query test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run course loading diagnostic tests"""
    print("=== Course Loading Diagnostic Tests ===\n")

    tests = [
        ("Docs Folder Check", test_docs_folder),
        ("Document Processing", test_document_processing),
        ("Course Loading to Vector Store", test_course_loading_to_vector_store),
        ("API Query Test", test_api_query),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()

    print("\n=== Summary ===")
    for test_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {'PASS' if success else 'FAIL'}")

    failed_tests = [name for name, success in results.items() if not success]
    if failed_tests:
        print(f"\n⚠ Issues found in: {', '.join(failed_tests)}")
    else:
        print("\n✓ All course loading tests passed!")


if __name__ == "__main__":
    main()
