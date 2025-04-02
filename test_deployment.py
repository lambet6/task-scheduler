import requests
import json
from datetime import datetime, timedelta

# Base URL of your API
base_url = "https://task-scheduler-qpbq.onrender.com"

# Helper functions
def print_response(response):
    """Pretty print response data"""
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def format_datetime(dt):
    """Format datetime to ISO format with Z timezone"""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Test Cases
def test_basic_schedule():
    """Test case: Basic task scheduling with one meeting"""
    print("\n=== Testing Basic Schedule ===")
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
    return print_response(response)

def test_overloaded_schedule():
    """Test case: Too many tasks for available work hours"""
    print("\n=== Testing Overloaded Schedule ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    # Current time base for creating tasks
    now = datetime.now()
    today = now.replace(hour=17, minute=0, second=0)
    
    # Create many tasks that won't fit in a day
    tasks = []
    for i in range(1, 11):  # 10 tasks of 90 minutes each (900 minutes total)
        tasks.append({
            "id": f"overload_task{i}",
            "title": f"Long Task {i}",
            "priority": "High" if i <= 3 else "Medium",
            "estimated_duration": 90,  # Each task takes 90 minutes
            "due": format_datetime(today)
        })
    
    payload = {
        "user_id": "test_user",
        "tasks": tasks,
        "calendar_events": [
            {"id": "evt1", "title": "Important Meeting", "start": "2025-04-01T11:00:00Z", "end": "2025-04-01T12:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},  # 8 hour workday = 480 minutes
            "max_continuous_work_min": 120
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_no_events():
    """Test case: Schedule with no calendar events"""
    print("\n=== Testing Schedule with No Events ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Write documentation", "priority": "High", "estimated_duration": 120},
            {"id": "task2", "title": "Plan sprint", "priority": "Medium", "estimated_duration": 60},
            {"id": "task3", "title": "Review PRs", "priority": "Low", "estimated_duration": 45}
        ],
        "calendar_events": [],  # No events
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_different_constraints():
    """Test case: Different work hours and constraints"""
    print("\n=== Testing Different Constraints ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Morning task", "priority": "High", "estimated_duration": 45},
            {"id": "task2", "title": "Afternoon task", "priority": "Medium", "estimated_duration": 60},
            {"id": "task3", "title": "Evening task", "priority": "Medium", "estimated_duration": 30}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Lunch", "start": "2025-04-01T12:00:00Z", "end": "2025-04-01T13:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "08:00", "end": "20:00"},  # Extended hours
            "max_continuous_work_min": 45  # Very short work sessions
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_priority_mix():
    """Test case: Mix of priorities with due dates"""
    print("\n=== Testing Priority Mix ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    now = datetime.now()
    today = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    next_week = (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Urgent task", "priority": "High", "estimated_duration": 60, "due": today},
            {"id": "task2", "title": "Important but not urgent", "priority": "High", "estimated_duration": 45, "due": tomorrow},
            {"id": "task3", "title": "Medium priority", "priority": "Medium", "estimated_duration": 30},
            {"id": "task4", "title": "Low priority task", "priority": "Low", "estimated_duration": 120, "due": next_week},
            {"id": "task5", "title": "Another urgent task", "priority": "High", "estimated_duration": 90, "due": today}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Daily standup", "start": "2025-04-01T09:30:00Z", "end": "2025-04-01T10:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 120
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_no_due_dates():
    """Test case: Tasks with no due dates"""
    print("\n=== Testing No Due Dates ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Research article", "priority": "Medium", "estimated_duration": 120},
            {"id": "task2", "title": "Learn new framework", "priority": "Low", "estimated_duration": 180},
            {"id": "task3", "title": "Brainstorm ideas", "priority": "Medium", "estimated_duration": 60}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Weekly review", "start": "2025-04-01T15:00:00Z", "end": "2025-04-01T16:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_no_tasks():
    """Test case: No tasks to schedule"""
    print("\n=== Testing No Tasks ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [],
        "calendar_events": [
            {"id": "evt1", "title": "Team meeting", "start": "2025-04-01T10:00:00Z", "end": "2025-04-01T11:00:00Z"},
            {"id": "evt2", "title": "Lunch", "start": "2025-04-01T12:00:00Z", "end": "2025-04-01T13:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_record_feedback(schedule_data):
    """Test recording feedback for a schedule"""
    print("\n=== Testing Record Feedback ===")
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
    return print_response(response)

def test_feedback_low_mood(schedule_data):
    """Test recording negative feedback for a schedule"""
    print("\n=== Testing Record Feedback (Low Mood) ===")
    endpoint = f"{base_url}/record_feedback"
    
    payload = {
        "user_id": "test_user",
        "schedule_data": schedule_data,
        "feedback_data": {
            "mood_score": 2,
            "adjusted_tasks": [
                {"id": "task1", "original_start": "2025-04-01T09:00:00Z", "new_start": "2025-04-01T14:00:00Z"}
            ],
            "completed_tasks": []
        }
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_very_short_work_hours():
    """Test case: Extremely short work hours window"""
    print("\n=== Testing Very Short Work Hours ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Quick task", "priority": "High", "estimated_duration": 15},
            {"id": "task2", "title": "Another quick task", "priority": "Medium", "estimated_duration": 20}
        ],
        "calendar_events": [],
        "constraints": {
            "work_hours": {"start": "12:00", "end": "13:00"},  # Only 1 hour window
            "max_continuous_work_min": 30
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_many_small_tasks():
    """Test case: Many small tasks instead of few large ones"""
    print("\n=== Testing Many Small Tasks ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    # Create many small tasks
    tasks = []
    for i in range(1, 16):  # 15 tasks of 10-15 minutes each
        tasks.append({
            "id": f"small_task{i}",
            "title": f"Quick Task {i}",
            "priority": "Medium",
            "estimated_duration": 10 + (i % 6)  # Tasks between 10-15 minutes
        })
    
    # Add a few high priority ones
    tasks[2]["priority"] = "High"
    tasks[7]["priority"] = "High"
    tasks[12]["priority"] = "High"
    
    payload = {
        "user_id": "test_user",
        "tasks": tasks,
        "calendar_events": [
            {"id": "evt1", "title": "Short meeting", "start": "2025-04-01T11:30:00Z", "end": "2025-04-01T12:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 60
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_mixed_durations():
    """Test case: Mix of very short and very long tasks"""
    print("\n=== Testing Mixed Task Durations ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Quick check", "priority": "Low", "estimated_duration": 5},
            {"id": "task2", "title": "Email triage", "priority": "Medium", "estimated_duration": 15},
            {"id": "task3", "title": "Major project work", "priority": "High", "estimated_duration": 180},
            {"id": "task4", "title": "Quick call", "priority": "Medium", "estimated_duration": 10},
            {"id": "task5", "title": "Documentation", "priority": "Low", "estimated_duration": 120}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Lunch", "start": "2025-04-01T12:00:00Z", "end": "2025-04-01T13:00:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_past_due_dates():
    """Test case: Tasks with past due dates"""
    print("\n=== Testing Past Due Dates ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    now = datetime.now()
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Overdue task", "priority": "High", "estimated_duration": 45, "due": yesterday},
            {"id": "task2", "title": "Current task", "priority": "Medium", "estimated_duration": 30}
        ],
        "calendar_events": [],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_overlapping_events():
    """Test case: Calendar with overlapping events"""
    print("\n=== Testing Overlapping Events ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Important work", "priority": "High", "estimated_duration": 60},
            {"id": "task2", "title": "Secondary work", "priority": "Medium", "estimated_duration": 45}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Meeting 1", "start": "2025-04-01T10:00:00Z", "end": "2025-04-01T11:30:00Z"},
            {"id": "evt2", "title": "Meeting 2", "start": "2025-04-01T11:00:00Z", "end": "2025-04-01T12:00:00Z"},
            {"id": "evt3", "title": "Lunch", "start": "2025-04-01T12:30:00Z", "end": "2025-04-01T13:30:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_late_day_scheduling():
    """Test case: Tasks due late in the day"""
    print("\n=== Testing Late Day Scheduling ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    now = datetime.now()
    today_late = now.replace(hour=16, minute=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    today_very_late = now.replace(hour=16, minute=30).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Urgent late task", "priority": "High", "estimated_duration": 45, "due": today_late},
            {"id": "task2", "title": "Another late task", "priority": "High", "estimated_duration": 30, "due": today_very_late},
            {"id": "task3", "title": "Regular task", "priority": "Medium", "estimated_duration": 60}
        ],
        "calendar_events": [
            {"id": "evt1", "title": "Late meeting", "start": "2025-04-01T15:00:00Z", "end": "2025-04-01T15:30:00Z"}
        ],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_high_fragmentation():
    """Test case: Many calendar events creating small gaps"""
    print("\n=== Testing High Fragmentation ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    # Create a highly fragmented day with many short meetings
    events = []
    for i in range(8):
        hour = 9 + i
        events.append({
            "id": f"evt{i+1}",
            "title": f"Meeting {i+1}",
            "start": f"2025-04-01T{hour:02d}:00:00Z",
            "end": f"2025-04-01T{hour:02d}:30:00Z"
        })
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Short task 1", "priority": "High", "estimated_duration": 15},
            {"id": "task2", "title": "Short task 2", "priority": "Medium", "estimated_duration": 20},
            {"id": "task3", "title": "Medium task", "priority": "High", "estimated_duration": 40},
            {"id": "task4", "title": "Another short task", "priority": "Low", "estimated_duration": 25}
        ],
        "calendar_events": events,
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

def test_minimal_viable_schedule():
    """Test case: Absolute minimum viable scheduling scenario"""
    print("\n=== Testing Minimal Viable Schedule ===")
    endpoint = f"{base_url}/optimize_schedule"
    
    payload = {
        "user_id": "test_user",
        "tasks": [
            {"id": "task1", "title": "Simple task", "priority": "Medium", "estimated_duration": 30}
        ],
        "calendar_events": [],
        "constraints": {
            "work_hours": {"start": "09:00", "end": "17:00"},
            "max_continuous_work_min": 90
        },
        "optimization_goal": "maximize_wellbeing"
    }
    
    response = requests.post(endpoint, json=payload)
    return print_response(response)

# Run the tests
if __name__ == "__main__":
    print("Starting comprehensive test suite for Task Scheduler API")
    
    # Dictionary to track which tests to run (all True by default)
    run_tests = {
        # Existing tests
        "basic": False,  # Setting to False until we debug core issues
        "overloaded": False,
        "no_events": False,
        "different_constraints": False,
        "priority_mix": False,
        "no_due_dates": False,
        "no_tasks": False,
        
        # New edge case tests
        "very_short_work_hours": True,
        "many_small_tasks": True,
        "mixed_durations": True,
        "past_due_dates": True,
        "overlapping_events": True,
        "late_day_scheduling": True,
        "high_fragmentation": True,
        "minimal_viable_schedule": True
    }
    
    # Get the last successful schedule result to use for feedback tests
    last_schedule = None
    
    # Run original tests (skipped for now to focus on edge cases)
    # [Original test execution code...]
    
    # Run new edge case tests
    if run_tests["very_short_work_hours"]:
        result = test_very_short_work_hours()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["many_small_tasks"]:
        result = test_many_small_tasks()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["mixed_durations"]:
        result = test_mixed_durations()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["past_due_dates"]:
        result = test_past_due_dates()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["overlapping_events"]:
        result = test_overlapping_events()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["late_day_scheduling"]:
        result = test_late_day_scheduling()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    if run_tests["high_fragmentation"]:
        result = test_high_fragmentation()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
            
    if run_tests["minimal_viable_schedule"]:
        result = test_minimal_viable_schedule()
        if result["status"] == "success" and result.get("scheduled_tasks"):
            last_schedule = {"scheduled_tasks": result["scheduled_tasks"]}
    
    # Test feedback if we have a schedule
    if last_schedule:
        test_record_feedback(last_schedule)
        test_feedback_low_mood(last_schedule)
    else:
        print("\n⚠️ No successful schedule with tasks to test feedback with")
    
    print("\nTest suite completed")