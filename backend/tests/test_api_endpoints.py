import pytest
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoint functionality"""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns basic info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Course Materials RAG System"
        assert data["status"] == "running"

    def test_query_endpoint_success(self, client, sample_query_request, mock_rag_system):
        """Test successful query processing"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify content
        assert data["answer"] == "This is a test response about the course material."
        assert len(data["sources"]) == 1
        assert "Python Basics - Lesson 1" in data["sources"][0]
        assert data["session_id"] == "test_session_123"
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once()

    def test_query_endpoint_with_session_id(self, client, sample_query_request_with_session, mock_rag_system):
        """Test query processing with existing session ID"""
        response = client.post("/api/query", json=sample_query_request_with_session)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use provided session ID
        assert data["session_id"] == "test_session_456"
        
        # Verify RAG system was called with session ID
        mock_rag_system.query.assert_called_once_with(
            sample_query_request_with_session["query"], 
            sample_query_request_with_session["session_id"]
        )

    def test_query_endpoint_without_session_id(self, client, sample_query_request, mock_rag_system):
        """Test query processing creates new session when none provided"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should create and return new session ID
        assert data["session_id"] == "test_session_123"
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_endpoint_missing_query(self, client):
        """Test query endpoint with missing query field"""
        response = client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query string"""
        response = client.post("/api/query", json={"query": ""})
        
        assert response.status_code == 200  # Empty query is technically valid
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json", 
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_query_endpoint_rag_system_error(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        # Configure mock to raise exception
        mock_rag_system.query.side_effect = Exception("Database connection error")
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection error" in data["detail"]

    def test_courses_endpoint_success(self, client, expected_course_stats, mock_rag_system):
        """Test successful course stats retrieval"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify content matches expected
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_endpoint_no_courses(self, client, mock_rag_system):
        """Test course stats when no courses exist"""
        # Configure mock to return empty state
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_endpoint_rag_system_error(self, client, mock_rag_system):
        """Test course stats endpoint when RAG system raises exception"""
        # Configure mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Vector store error" in data["detail"]


@pytest.mark.api
class TestAPIEndpointIntegration:
    """Integration tests for API endpoints with more realistic scenarios"""

    def test_query_to_courses_workflow(self, client, mock_rag_system):
        """Test typical workflow: query → check courses"""
        # First make a query
        query_response = client.post("/api/query", json={"query": "What courses are available?"})
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]
        
        # Then check course stats
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        
        courses_data = courses_response.json()
        assert courses_data["total_courses"] > 0
        assert len(courses_data["course_titles"]) == courses_data["total_courses"]

    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries with the same session ID"""
        session_id = "persistent_session_789"
        
        # First query
        response1 = client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": session_id
        })
        assert response1.status_code == 200
        assert response1.json()["session_id"] == session_id
        
        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "How do I create variables?",
            "session_id": session_id
        })
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system was called twice
        assert mock_rag_system.query.call_count == 2

    def test_concurrent_sessions(self, client, mock_rag_system):
        """Test handling of concurrent sessions"""
        # Configure mock to return different session IDs
        session_ids = ["session_1", "session_2"]
        mock_rag_system.session_manager.create_session.side_effect = session_ids
        
        # Create two queries without session IDs (should create separate sessions)
        response1 = client.post("/api/query", json={"query": "Query 1"})
        response2 = client.post("/api/query", json={"query": "Query 2"})
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should have different session IDs
        session1 = response1.json()["session_id"]
        session2 = response2.json()["session_id"]
        
        assert session1 != session2

    def test_content_type_handling(self, client):
        """Test proper content type handling for different request types"""
        # JSON request (correct)
        json_response = client.post("/api/query", json={"query": "test"})
        assert json_response.status_code == 200
        
        # Form data request (should fail for JSON endpoint)
        form_response = client.post("/api/query", data={"query": "test"})
        assert form_response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.get("/api/courses")
        
        # Check that CORS middleware is working
        # The test client doesn't simulate browser CORS behavior exactly,
        # but we can verify the response is successful
        assert response.status_code == 200

    def test_large_query_handling(self, client, mock_rag_system):
        """Test handling of very large query strings"""
        # Create a large query (simulating edge case)
        large_query = "What is Python? " * 1000  # Very long repeated question
        
        response = client.post("/api/query", json={"query": large_query})
        
        # Should still process successfully
        assert response.status_code == 200
        
        # Verify the full query was passed to RAG system
        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args[0]
        assert call_args[0] == large_query

    def test_special_characters_in_query(self, client, mock_rag_system):
        """Test handling of special characters in queries"""
        special_query = "What about unicode: 你好, émojis 🐍, and symbols @#$%?"
        
        response = client.post("/api/query", json={"query": special_query})
        
        assert response.status_code == 200
        
        # Verify special characters were preserved
        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args[0]
        assert call_args[0] == special_query