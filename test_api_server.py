import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

# Import the FastAPI app
from api_server import app

# Create test client
client = TestClient(app)

# Mock data for tests
sample_user_id = "test_user_123"
now = datetime.now()
tomorrow = now + timedelta(days=1)

sample_task = {
    "id": "task1",
    "title": "Test Task",
    "priority": "High",
    "estimated_duration": 60,
    "due": tomorrow.isoformat()
}

sample_event = {
    "id": "event1",
    "title": "Test Event",
    "start": now.isoformat(),
    "end": (now + timedelta(hours=1)).isoformat()
}

sample_constraints = {
    "work_hours": {
        "start": "09:00",
        "end": "17:00"
    },
    "max_continuous_work_min": 90
}

sample_scheduled_task = {
    "id": "task1",
    "title": "Test Task",
    "start": now.isoformat(),
    "end": (now + timedelta(hours=1)).isoformat(),
    "priority": "High",
    "estimated_duration": 60,
    "mandatory": True
}

sample_feedback = {
    "mood_score": 4,
    "energy_level": 3,
    "adjusted_tasks": [],
    "completed_tasks": ["task1"],
    "task_specific_feedback": {"task1": "Too long"}
}

# Tests for /optimize_schedule endpoint
@patch('api_server.ml_learner.get_user_parameters')
@patch('api_server.TaskScheduler')
def test_optimize_schedule_success(mock_scheduler_class, mock_get_params):
    # Setup mocks
    mock_scheduler = MagicMock()
    mock_scheduler_class.return_value = mock_scheduler
    mock_get_params.return_value = {
        "break_importance": 1.0,
        "max_continuous_work": 90,
        "continuous_work_penalty": 2.0,
        "evening_work_penalty": 3.0,
        "early_completion_bonus": 2.0
    }
    mock_scheduler.schedule_tasks.return_value = {
        "status": "success",
        "scheduled_tasks": [sample_scheduled_task]
    }
    
    # Create request payload
    request_payload = {
        "user_id": sample_user_id,
        "tasks": [sample_task],
        "calendar_events": [sample_event],
        "constraints": sample_constraints,
        "optimization_goal": "maximize_wellbeing"
    }
    
    # Make the request
    response = client.post("/optimize_schedule", json=request_payload)
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["scheduled_tasks"]) == 1
    assert response.json()["scheduled_tasks"][0]["id"] == "task1"
    
    # Verify mocks were called correctly
    mock_get_params.assert_called_once_with(sample_user_id)
    mock_scheduler.schedule_tasks.assert_called_once()

@patch('api_server.ml_learner.get_user_parameters')
@patch('api_server.TaskScheduler')
def test_optimize_schedule_error(mock_scheduler_class, mock_get_params):
    # Setup mocks
    mock_scheduler = MagicMock()
    mock_scheduler_class.return_value = mock_scheduler
    mock_get_params.return_value = {
        "break_importance": 1.0,
        "max_continuous_work": 90,
        "continuous_work_penalty": 2.0,
        "evening_work_penalty": 3.0,
        "early_completion_bonus": 2.0
    }
    mock_scheduler.schedule_tasks.return_value = {
        "status": "error",
        "message": "No feasible solution found. Solver status: INFEASIBLE"
    }
    
    # Create request payload
    request_payload = {
        "user_id": sample_user_id,
        "tasks": [sample_task],
        "calendar_events": [sample_event],
        "constraints": sample_constraints
    }
    
    # Make the request
    response = client.post("/optimize_schedule", json=request_payload)
    
    # Check response
    assert response.status_code == 200  # API returns 200 even for business logic errors
    assert response.json()["status"] == "error"
    assert "message" in response.json()
    assert "No feasible solution found" in response.json()["message"]

@patch('api_server.ml_learner.get_user_parameters')
@patch('api_server.TaskScheduler')
def test_optimize_schedule_exception(mock_scheduler_class, mock_get_params):
    # Setup mocks to raise exception
    mock_get_params.return_value = {}
    mock_scheduler = MagicMock()
    mock_scheduler_class.return_value = mock_scheduler
    mock_scheduler.schedule_tasks.side_effect = Exception("Internal server error")
    
    # Create request payload
    request_payload = {
        "user_id": sample_user_id,
        "tasks": [sample_task],
        "calendar_events": [sample_event],
        "constraints": sample_constraints
    }
    
    # Make the request
    response = client.post("/optimize_schedule", json=request_payload)
    
    # Check response
    assert response.status_code == 200  # API returns 200 as defined in the handler
    assert response.json()["status"] == "error"
    assert "message" in response.json()
    assert "Internal server error" in response.json()["message"]

def test_optimize_schedule_invalid_schema():
    # Create an invalid request missing required fields
    invalid_request = {
        "user_id": sample_user_id
        # Missing tasks, calendar_events, and constraints
    }
    
    # Make the request
    response = client.post("/optimize_schedule", json=invalid_request)
    
    # Check response - should be 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
    assert "detail" in response.json()

# Tests for /record_feedback endpoint
@patch('api_server.ml_learner.record_feedback')
def test_record_feedback_success(mock_record_feedback):
    # Setup mock
    mock_record_feedback.return_value = None  # Doesn't return anything on success
    
    # Create request payload
    request_payload = {
        "user_id": sample_user_id,
        "schedule_data": {
            "scheduled_tasks": [sample_scheduled_task]
        },
        "feedback_data": sample_feedback
    }
    
    # Make the request
    response = client.post("/record_feedback", json=request_payload)
    
    # Check response
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Feedback recorded successfully"
    
    # Verify mock was called once (don't check exact parameters due to Pydantic conversion)
    assert mock_record_feedback.call_count == 1
    # Check that user_id and schedule_data were passed correctly
    call_args = mock_record_feedback.call_args
    assert call_args[1]['user_id'] == sample_user_id
    assert call_args[1]['schedule_data'] == request_payload["schedule_data"]
    # We don't check feedback_data specifically because it's converted to a Pydantic model

@patch('api_server.ml_learner.record_feedback')
def test_record_feedback_error(mock_record_feedback):
    # Setup mock to raise exception
    mock_record_feedback.side_effect = Exception("Database error")
    
    # Create request payload
    request_payload = {
        "user_id": sample_user_id,
        "schedule_data": {"scheduled_tasks": []},
        "feedback_data": sample_feedback
    }
    
    # Make the request
    response = client.post("/record_feedback", json=request_payload)
    
    # Check response - should be 500 Internal Server Error
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Database error" in response.json()["detail"]

def test_record_feedback_invalid_schema():
    # Create an invalid request missing required fields
    invalid_request = {
        "user_id": sample_user_id
        # Missing schedule_data and feedback_data
    }
    
    # Make the request
    response = client.post("/record_feedback", json=invalid_request)
    
    # Check response - should be 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422
    assert "detail" in response.json()

def test_feedback_field_constraints():
    # Test that field constraints (min/max values) are enforced
    invalid_request = {
        "user_id": sample_user_id,
        "schedule_data": {"scheduled_tasks": []},
        "feedback_data": {
            "mood_score": 6,  # Invalid: should be 1-5
            "energy_level": 3,
            "adjusted_tasks": [],
            "completed_tasks": []
        }
    }
    
    response = client.post("/record_feedback", json=invalid_request)
    
    # Should fail validation
    assert response.status_code == 422
    assert "mood_score" in str(response.json())