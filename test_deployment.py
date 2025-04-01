import requests
import json

# Base URL of your API
base_url = "https://task-scheduler-qpbq.onrender.com"

# Test optimize_schedule endpoint
def test_optimize_schedule():
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Complete report", "priority": "High", "estimated_duration": 60, "due": "2025-04-01T17:00:00Z"},
            {"id": "task2", "title": "Review code", "priority": "Medium", "estimated_duration": 45}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Team meeting", "start": "2025-04-01T10:00:00Z", "end": "2025-04-01T11:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

# Test record_feedback endpoint
def test_record_feedback(schedule_data):
    endpoint = f"{base_url}/record_feedback"
    
    payload = {
        "user_id": "test_user",
        "schedule_data": schedule_data,
        "feedback_data": {
            "mood_score": 4,
            "adjusted_tasks": [],
            "completed_tasks": ["task1"]
        }
    }
    
    response = requests.post(endpoint, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

# Run the tests
if __name__ == "__main__":
    # First test the scheduler
    schedule_result = test_optimize_schedule()
    
    # If successful, test the feedback with the returned schedule
    if schedule_result["status"] == "success":
        test_record_feedback({"scheduled_tasks": schedule_result["scheduled_tasks"]})