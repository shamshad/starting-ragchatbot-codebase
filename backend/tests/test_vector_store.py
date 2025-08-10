import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults(unittest.TestCase):

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB query results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "Python"}, {"course": "JS"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        self.assertEqual(results.documents, ["doc1", "doc2"])
        self.assertEqual(results.metadata, [{"course": "Python"}, {"course": "JS"}])
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Connection failed")

        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertEqual(results.error, "Connection failed")

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(["doc"], [{}], [0.1])

        self.assertTrue(empty_results.is_empty())
        self.assertFalse(non_empty_results.is_empty())


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "test_chroma")

        # Create mock ChromaDB client and collections
        with (
            patch("chromadb.PersistentClient") as mock_client_class,
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding_func,
        ):

            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client

            self.mock_catalog_collection = Mock()
            self.mock_content_collection = Mock()

            # Set up collection creation to return our mocks
            def get_or_create_collection(name, embedding_function):
                if name == "course_catalog":
                    return self.mock_catalog_collection
                elif name == "course_content":
                    return self.mock_content_collection
                return Mock()

            self.mock_client.get_or_create_collection.side_effect = (
                get_or_create_collection
            )

            # Initialize VectorStore
            self.vector_store = VectorStore(
                chroma_path=self.chroma_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_collections(self):
        """Test that VectorStore initialization creates required collections"""
        # Verify client was created with correct path
        self.mock_client.get_or_create_collection.assert_any_call(
            name="course_catalog",
            embedding_function=self.vector_store.embedding_function,
        )
        self.mock_client.get_or_create_collection.assert_any_call(
            name="course_content",
            embedding_function=self.vector_store.embedding_function,
        )

    def test_search_successful(self):
        """Test successful search operation"""
        # Mock successful content collection query
        self.mock_content_collection.query.return_value = {
            "documents": [["content 1", "content 2"]],
            "metadatas": [[{"course_title": "Python"}, {"course_title": "JS"}]],
            "distances": [[0.1, 0.2]],
        }

        results = self.vector_store.search("test query")

        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=None
        )

        self.assertFalse(results.is_empty())
        self.assertEqual(len(results.documents), 2)
        self.assertIsNone(results.error)

    def test_search_with_course_name_filter(self):
        """Test search with course name filtering"""
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            "documents": [["Python Basics"]],
            "metadatas": [[{"title": "Python Basics"}]],
        }

        # Mock content search
        self.mock_content_collection.query.return_value = {
            "documents": [["python content"]],
            "metadatas": [[{"course_title": "Python Basics"}]],
            "distances": [[0.1]],
        }

        results = self.vector_store.search("variables", course_name="Python")

        # Verify course name resolution was called
        self.mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Python"], n_results=1
        )

        # Verify content search was called with resolved course title
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["variables"],
            n_results=5,
            where={"course_title": "Python Basics"},
        )

    def test_search_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        self.mock_content_collection.query.return_value = {
            "documents": [["lesson content"]],
            "metadatas": [[{"lesson_number": 3}]],
            "distances": [[0.1]],
        }

        results = self.vector_store.search("functions", lesson_number=3)

        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["functions"], n_results=5, where={"lesson_number": 3}
        )

    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        # Mock course resolution
        self.mock_catalog_collection.query.return_value = {
            "documents": [["Advanced Python"]],
            "metadatas": [[{"title": "Advanced Python"}]],
        }

        self.mock_content_collection.query.return_value = {
            "documents": [["specific content"]],
            "metadatas": [[{"course_title": "Advanced Python", "lesson_number": 5}]],
            "distances": [[0.1]],
        }

        results = self.vector_store.search(
            "decorators", course_name="Advanced", lesson_number=5
        )

        expected_filter = {
            "$and": [{"course_title": "Advanced Python"}, {"lesson_number": 5}]
        }

        self.mock_content_collection.query.assert_called_once_with(
            query_texts=["decorators"], n_results=5, where=expected_filter
        )

    def test_search_course_not_found(self):
        """Test search when course name resolution fails"""
        # Mock failed course resolution
        self.mock_catalog_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        results = self.vector_store.search("test", course_name="NonexistentCourse")

        self.assertTrue(results.is_empty())
        self.assertIn("No course found matching 'NonexistentCourse'", results.error)

        # Content search should not be called
        self.mock_content_collection.query.assert_not_called()

    def test_search_exception_handling(self):
        """Test search handles ChromaDB exceptions"""
        self.mock_content_collection.query.side_effect = Exception("ChromaDB error")

        results = self.vector_store.search("test query")

        self.assertTrue(results.is_empty())
        self.assertIn("Search error", results.error)
        self.assertIn("ChromaDB error", results.error)

    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        self.mock_catalog_collection.query.return_value = {
            "documents": [["Python Programming"]],
            "metadatas": [[{"title": "Python Programming Fundamentals"}]],
        }

        resolved_title = self.vector_store._resolve_course_name("Python")

        self.assertEqual(resolved_title, "Python Programming Fundamentals")
        self.mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Python"], n_results=1
        )

    def test_resolve_course_name_not_found(self):
        """Test course name resolution when no matches found"""
        self.mock_catalog_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        resolved_title = self.vector_store._resolve_course_name("Unknown")

        self.assertIsNone(resolved_title)

    def test_build_filter_no_filters(self):
        """Test filter building with no filters"""
        filter_dict = self.vector_store._build_filter(None, None)
        self.assertIsNone(filter_dict)

    def test_build_filter_course_only(self):
        """Test filter building with course title only"""
        filter_dict = self.vector_store._build_filter("Python", None)
        self.assertEqual(filter_dict, {"course_title": "Python"})

    def test_build_filter_lesson_only(self):
        """Test filter building with lesson number only"""
        filter_dict = self.vector_store._build_filter(None, 3)
        self.assertEqual(filter_dict, {"lesson_number": 3})

    def test_build_filter_both(self):
        """Test filter building with both course and lesson"""
        filter_dict = self.vector_store._build_filter("Python", 3)
        expected = {"$and": [{"course_title": "Python"}, {"lesson_number": 3}]}
        self.assertEqual(filter_dict, expected)

    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        # Create test course
        lessons = [
            Lesson(
                lesson_number=1,
                title="Intro",
                lesson_link="http://lesson1.com",
                content="",
            ),
            Lesson(
                lesson_number=2,
                title="Variables",
                lesson_link="http://lesson2.com",
                content="",
            ),
        ]
        course = Course(
            title="Python Basics",
            instructor="John Doe",
            course_link="http://course.com",
            lessons=lessons,
        )

        self.vector_store.add_course_metadata(course)

        # Verify catalog collection was called correctly
        self.mock_catalog_collection.add.assert_called_once()

        call_args = self.mock_catalog_collection.add.call_args
        self.assertEqual(call_args[1]["documents"], ["Python Basics"])
        self.assertEqual(call_args[1]["ids"], ["Python Basics"])

        # Check metadata structure
        metadata = call_args[1]["metadatas"][0]
        self.assertEqual(metadata["title"], "Python Basics")
        self.assertEqual(metadata["instructor"], "John Doe")
        self.assertEqual(metadata["course_link"], "http://course.com")
        self.assertEqual(metadata["lesson_count"], 2)
        self.assertIn("lessons_json", metadata)

    def test_add_course_content(self):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                course_title="Python Basics",
                lesson_number=1,
                chunk_index=0,
                content="This is chunk 1",
            ),
            CourseChunk(
                course_title="Python Basics",
                lesson_number=1,
                chunk_index=1,
                content="This is chunk 2",
            ),
        ]

        self.vector_store.add_course_content(chunks)

        self.mock_content_collection.add.assert_called_once()

        call_args = self.mock_content_collection.add.call_args
        self.assertEqual(
            call_args[1]["documents"], ["This is chunk 1", "This is chunk 2"]
        )
        self.assertEqual(call_args[1]["ids"], ["Python_Basics_0", "Python_Basics_1"])

        expected_metadata = [
            {"course_title": "Python Basics", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Python Basics", "lesson_number": 1, "chunk_index": 1},
        ]
        self.assertEqual(call_args[1]["metadatas"], expected_metadata)

    def test_add_course_content_empty_chunks(self):
        """Test adding empty chunks list does nothing"""
        self.vector_store.add_course_content([])
        self.mock_content_collection.add.assert_not_called()

    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        self.mock_catalog_collection.get.return_value = {
            "ids": ["Course 1", "Course 2", "Course 3"]
        }

        titles = self.vector_store.get_existing_course_titles()

        self.assertEqual(titles, ["Course 1", "Course 2", "Course 3"])
        self.mock_catalog_collection.get.assert_called_once()

    def test_get_existing_course_titles_empty(self):
        """Test getting course titles when catalog is empty"""
        self.mock_catalog_collection.get.return_value = {"ids": []}

        titles = self.vector_store.get_existing_course_titles()

        self.assertEqual(titles, [])

    def test_get_existing_course_titles_exception(self):
        """Test getting course titles handles exceptions"""
        self.mock_catalog_collection.get.side_effect = Exception("DB error")

        titles = self.vector_store.get_existing_course_titles()

        self.assertEqual(titles, [])

    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_catalog_collection.get.return_value = {
            "ids": ["Course 1", "Course 2"]
        }

        count = self.vector_store.get_course_count()

        self.assertEqual(count, 2)

    def test_get_course_count_empty(self):
        """Test getting course count when empty"""
        self.mock_catalog_collection.get.return_value = {"ids": []}

        count = self.vector_store.get_course_count()

        self.assertEqual(count, 0)

    def test_clear_all_data(self):
        """Test clearing all data and recreating collections"""
        self.vector_store.clear_all_data()

        # Verify collections were deleted
        self.mock_client.delete_collection.assert_any_call("course_catalog")
        self.mock_client.delete_collection.assert_any_call("course_content")

        # Verify collections were recreated (additional calls beyond setUp)
        self.assertEqual(
            self.mock_client.get_or_create_collection.call_count, 4
        )  # 2 in setUp + 2 in clear

    def test_get_lesson_link_success(self):
        """Test getting lesson link successfully"""
        lessons_json = '[{"lesson_number": 1, "lesson_link": "http://lesson1.com"}, {"lesson_number": 2, "lesson_link": "http://lesson2.com"}]'

        self.mock_catalog_collection.get.return_value = {
            "metadatas": [{"lessons_json": lessons_json}]
        }

        link = self.vector_store.get_lesson_link("Python Basics", 2)

        self.assertEqual(link, "http://lesson2.com")
        self.mock_catalog_collection.get.assert_called_once_with(ids=["Python Basics"])

    def test_get_lesson_link_not_found(self):
        """Test getting lesson link when lesson not found"""
        lessons_json = '[{"lesson_number": 1, "lesson_link": "http://lesson1.com"}]'

        self.mock_catalog_collection.get.return_value = {
            "metadatas": [{"lessons_json": lessons_json}]
        }

        link = self.vector_store.get_lesson_link("Python Basics", 3)

        self.assertIsNone(link)

    def test_get_lesson_link_course_not_found(self):
        """Test getting lesson link when course not found"""
        self.mock_catalog_collection.get.return_value = {"metadatas": []}

        link = self.vector_store.get_lesson_link("Nonexistent Course", 1)

        self.assertIsNone(link)


if __name__ == "__main__":
    unittest.main()
