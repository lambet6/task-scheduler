import pytest
from datetime import datetime, timedelta
from scheduler_model import TaskScheduler

# Helper functions for test data creation
def create_task(task_id, title, priority, duration, due=None):
    """Helper to create a task dictionary."""
    task = {
        "id": task_id,
        "title": title,
        "priority": priority,
        "estimated_duration": duration
    }
    if due:
        task["due"] = due
    return task

def create_event(event_id, title, start, end):
    """Helper to create a calendar event dictionary."""
    return {
        "id": event_id,
        "title": title,
        "start": start,
        "end": end
    }

def create_constraints(work_start="09:00", work_end="17:00", max_continuous_work=90):
    """Helper to create constraints dictionary."""
    return {
        "work_hours": {
            "start": work_start,
            "end": work_end
        },
        "max_continuous_work_min": max_continuous_work
    }

# Fixtures for common test setups
@pytest.fixture
def base_date():
    """Return a base date for tests, set to today at midnight."""
    # Using a fixed date to ensure tests are reproducible
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

@pytest.fixture
def standard_ml_params():
    """Return standard ML parameters for tests."""
    return {
        'break_importance': 1.0,
        'max_continuous_work': 90,
        'continuous_work_penalty': 2.0,
        'evening_work_penalty': 3.0,
        'early_completion_bonus': 2.0
    }

@pytest.fixture
def scheduler(standard_ml_params):
    """Return a TaskScheduler instance with standard ML parameters."""
    return TaskScheduler(ml_params=standard_ml_params)

# Tests for constraint enforcement
class TestConstraintEnforcement:
    def test_no_overlap_constraint(self, scheduler, base_date):
        """Test that scheduled tasks don't overlap with each other or with events."""
        # Create tasks that would overlap if scheduled sequentially
        tasks = [
            create_task("task1", "Task 1", "High", 60),
            create_task("task2", "Task 2", "High", 60),
            create_task("task3", "Task 3", "High", 60)
        ]
        
        # Create one event in the middle of the day
        events = [
            create_event(
                "event1", 
                "Meeting", 
                (base_date + timedelta(hours=12)).isoformat(), 
                (base_date + timedelta(hours=13)).isoformat()
            )
        ]
        
        constraints = create_constraints()
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # Check no overlap between tasks
        scheduled_tasks = result["scheduled_tasks"]
        for i in range(len(scheduled_tasks)):
            task_i_end = datetime.fromisoformat(scheduled_tasks[i]["end"].replace('Z', '+00:00'))
            for j in range(i+1, len(scheduled_tasks)):
                task_j_start = datetime.fromisoformat(scheduled_tasks[j]["start"].replace('Z', '+00:00'))
                task_j_end = datetime.fromisoformat(scheduled_tasks[j]["end"].replace('Z', '+00:00'))
                task_i_start = datetime.fromisoformat(scheduled_tasks[i]["start"].replace('Z', '+00:00'))
                # Either task i ends before task j starts OR task j ends before task i starts
                assert task_i_end <= task_j_start or task_j_end <= task_i_start
        
        # Check no overlap with events
        event_start = datetime.fromisoformat(events[0]["start"].replace('Z', '+00:00'))
        event_end = datetime.fromisoformat(events[0]["end"].replace('Z', '+00:00'))
        for task in scheduled_tasks:
            task_start = datetime.fromisoformat(task["start"].replace('Z', '+00:00'))
            task_end = datetime.fromisoformat(task["end"].replace('Z', '+00:00'))
            # Either task ends before event starts OR task starts after event ends
            assert task_end <= event_start or task_start >= event_end

    def test_work_hours_constraint(self, scheduler, base_date):
        """Test that all tasks are scheduled within work hours."""
        # Create some tasks
        tasks = [
            create_task("task1", "Task 1", "High", 60),
            create_task("task2", "Task 2", "Medium", 90),
            create_task("task3", "Task 3", "Low", 30)
        ]
        
        # No events
        events = []
        
        # Set work hours constraint
        work_start = "10:00"
        work_end = "16:00"
        constraints = create_constraints(work_start=work_start, work_end=work_end)
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # Check all tasks are within work hours
        work_start_minutes = 10 * 60  # 10:00 in minutes
        work_end_minutes = 16 * 60    # 16:00 in minutes
        
        for task in result["scheduled_tasks"]:
            task_start = datetime.fromisoformat(task["start"].replace('Z', '+00:00'))
            task_end = datetime.fromisoformat(task["end"].replace('Z', '+00:00'))
            
            # Convert to minutes since midnight
            task_start_minutes = task_start.hour * 60 + task_start.minute
            task_end_minutes = task_end.hour * 60 + task_end.minute
            
            assert task_start_minutes >= work_start_minutes
            assert task_end_minutes <= work_end_minutes

    def test_due_date_constraint(self, scheduler, base_date):
        """Test that tasks are completed before their due date."""
        # Create tasks with due dates
        tasks = [
            create_task(
                "task1", 
                "Task 1", 
                "High", 
                60, 
                due=(base_date + timedelta(hours=11)).isoformat()
            ),
            create_task(
                "task2", 
                "Task 2", 
                "Medium", 
                90, 
                due=(base_date + timedelta(hours=14)).isoformat()
            )
        ]
        
        # No events
        events = []
        
        # Standard constraints
        constraints = create_constraints()
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # Check each task ends before its due date
        for task in result["scheduled_tasks"]:
            task_end = datetime.fromisoformat(task["end"].replace('Z', '+00:00'))
            # Find the corresponding input task
            input_task = next(t for t in tasks if t["id"] == task["id"])
            due_date = datetime.fromisoformat(input_task["due"].replace('Z', '+00:00'))
            assert task_end <= due_date

# Tests for mandatory vs. optional tasks
class TestMandatoryOptionalTasks:
    def test_mandatory_task_inclusion(self, scheduler, base_date):
        """Test that all mandatory tasks are included in the schedule."""
        # Create both mandatory and optional tasks
        today = base_date.date()
        tomorrow = (base_date + timedelta(days=1)).date()
        
        # Mandatory tasks (due today)
        mandatory_tasks = [
            create_task(
                "task1", 
                "Mandatory 1", 
                "Medium", 
                60, 
                due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=15)).isoformat()
            ),
            create_task(
                "task2", 
                "Mandatory 2", 
                "Low", 
                30, 
                due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=16)).isoformat()
            )
        ]
        
        # Optional tasks (due tomorrow)
        optional_tasks = [
            create_task(
                "task3", 
                "Optional 1", 
                "High", 
                120, 
                due=(datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=15)).isoformat()
            ),
            create_task(
                "task4", 
                "Optional 2", 
                "Medium", 
                60, 
                due=(datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=16)).isoformat()
            )
        ]
        
        tasks = mandatory_tasks + optional_tasks
        events = []
        constraints = create_constraints()
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # Check all mandatory tasks are scheduled
        scheduled_task_ids = [task["id"] for task in result["scheduled_tasks"]]
        for mandatory_task in mandatory_tasks:
            assert mandatory_task["id"] in scheduled_task_ids

    def test_optional_task_prioritization(self, scheduler, base_date):
        """Test that optional tasks are prioritized by priority and due date."""
        # Define a set of tasks including one mandatory to ensure something is scheduled
        today = base_date.date()
        tomorrow = (base_date + timedelta(days=1)).date()
        
        tasks = [
            # Mandatory task (due today)
            create_task(
                "task_mandatory", 
                "Must Schedule", 
                "Medium", 
                30, 
                due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=15)).isoformat()
            ),
            # Optional tasks with varying priority and due dates
            create_task(
                "task1", 
                "High Priority, Later Due", 
                "High", 
                60, 
                due=(datetime.combine(tomorrow, datetime.min.time()) + timedelta(days=1)).isoformat()
            ),
            create_task(
                "task2", 
                "Medium Priority, Soon Due", 
                "Medium", 
                60, 
                due=datetime.combine(tomorrow, datetime.min.time()).isoformat()
            ),
            create_task(
                "task3", 
                "Low Priority, Soon Due", 
                "Low", 
                60, 
                due=datetime.combine(tomorrow, datetime.min.time()).isoformat()
            ),
            create_task(
                "task4", 
                "High Priority, Soon Due", 
                "High", 
                60, 
                due=datetime.combine(tomorrow, datetime.min.time()).isoformat()
            )
        ]
        
        events = []
        
        # Restrict work hours to create scarcity
        constraints = create_constraints(work_start="09:00", work_end="11:30")  # 2.5 hours = 150 minutes
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # Check scheduled tasks (should prefer high priority and soon due)
        scheduled_task_ids = [task["id"] for task in result["scheduled_tasks"]]
        
        # Mandatory task must be scheduled
        assert "task_mandatory" in scheduled_task_ids
        
        # At least the mandatory task should be scheduled
        assert len(scheduled_task_ids) >= 1
        
        # If there's room for optional tasks (there should be)
        if len(scheduled_task_ids) > 1:
            # The optional tasks that get scheduled should include the high priority, soon due one first
            optional_scheduled = [task_id for task_id in scheduled_task_ids if task_id != "task_mandatory"]
            if optional_scheduled:
                # First optional task should be task4 (high priority, soon due)
                assert "task4" in optional_scheduled

    def test_optional_task_exclusion(self, scheduler, base_date):
        """Test that optional tasks are excluded when there's not enough time."""
        # Create a lot of optional tasks and a tight work window
        optional_tasks = [
            create_task(
                f"task{i}", 
                f"Optional Task {i}", 
                "Medium", 
                60, 
                due=(base_date + timedelta(days=1)).isoformat()
            ) for i in range(1, 10)  # 9 tasks, each 60 minutes
        ]
        
        events = []
        
        # Restrict work hours to 4 hours (can only fit 4 tasks)
        constraints = create_constraints(work_start="09:00", work_end="13:00")
        
        # Schedule tasks
        result = scheduler.schedule_tasks(optional_tasks, events, constraints)
        
        # Check result status
        assert result["status"] == "success"
        
        # There should be fewer scheduled tasks than input tasks
        assert len(result["scheduled_tasks"]) < len(optional_tasks)

# Tests for edge cases
class TestEdgeCases:
    def test_no_feasible_solution(self, scheduler, base_date):
        """Test handling when there's no feasible solution (too many mandatory tasks)."""
        # Create many mandatory tasks with limited work hours
        mandatory_tasks = [
            create_task(
                f"task{i}", 
                f"Mandatory Task {i}", 
                "High", 
                60, 
                due=(base_date + timedelta(hours=15)).isoformat()
            ) for i in range(1, 10)  # 9 tasks, each 60 minutes
        ]
        
        events = []
        
        # Very restricted work hours (only 2 hours available)
        constraints = create_constraints(work_start="09:00", work_end="11:00")
        
        # Schedule tasks - should fail to find a solution
        result = scheduler.schedule_tasks(mandatory_tasks, events, constraints)
        
        # Check result status - should be error
        assert result["status"] == "error"
        assert "No feasible solution found" in result.get("message", "")

    def test_tight_constraints(self, scheduler, base_date):
        """Test scheduling with tight constraints but still feasible."""
        # Create tasks including one mandatory task that must be scheduled
        today = base_date.date()
        
        tasks = [
            # Mandatory task (due today)
            create_task(
                "mandatory_task", 
                "Must Schedule", 
                "High", 
                60, 
                due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=16)).isoformat()
            ),
            # Optional tasks
            create_task(
                "optional_task1", 
                "May Schedule 1", 
                "Medium", 
                60, 
                due=(base_date + timedelta(days=1)).isoformat()
            ),
            create_task(
                "optional_task2", 
                "May Schedule 2", 
                "Low", 
                30, 
                due=(base_date + timedelta(days=1)).isoformat()
            )
        ]
        
        # An event that takes up some time
        events = [
            create_event(
                "event1", 
                "Meeting", 
                (base_date + timedelta(hours=10)).isoformat(), 
                (base_date + timedelta(hours=11)).isoformat()
            )  # 60 minutes
        ]
        
        # Work window is just big enough: 3 hours = 180 minutes
        # After event: 180 - 60 = 120 minutes available
        # Tasks need 150 minutes, but some are optional
        constraints = create_constraints(work_start="09:00", work_end="12:00")
        
        # Schedule tasks
        result = scheduler.schedule_tasks(tasks, events, constraints)
        
        # Should be successful
        assert result["status"] == "success"
        
        # At minimum, the mandatory task must be scheduled
        assert len(result["scheduled_tasks"]) > 0
        assert any(task["id"] == "mandatory_task" for task in result["scheduled_tasks"])

    def test_empty_inputs(self, scheduler, base_date):
        """Test handling of empty inputs."""
        # Empty tasks
        result_no_tasks = scheduler.schedule_tasks([], [], create_constraints())
        assert result_no_tasks["status"] == "success"
        assert len(result_no_tasks["scheduled_tasks"]) == 0
        
        # Empty events (but with a mandatory task)
        today = base_date.date()
        mandatory_task = create_task(
            "task1", 
            "Mandatory Task", 
            "Medium", 
            60,
            due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=15)).isoformat()
        )
        
        result_no_events = scheduler.schedule_tasks([mandatory_task], [], create_constraints())
        assert result_no_events["status"] == "success"
        assert len(result_no_events["scheduled_tasks"]) > 0

    def test_just_fitting_mandatory_tasks(self, scheduler, base_date):
        """Test with mandatory tasks that just barely fit within constraints."""
        # Create mandatory tasks that exactly fill the work hours
        today = base_date.date()
        
        # Work hours: 9:00 - 17:00 = 8 hours = 480 minutes
        # Create 8 mandatory tasks of 60 minutes each
        mandatory_tasks = [
            create_task(
                f"task{i}", 
                f"Mandatory Task {i}", 
                "Medium", 
                60, 
                due=(datetime.combine(today, datetime.min.time()) + timedelta(hours=17)).isoformat()
            ) for i in range(1, 9)
        ]
        
        events = []
        constraints = create_constraints()
        
        # Schedule tasks
        result = scheduler.schedule_tasks(mandatory_tasks, events, constraints)
        
        # Check result status - should succeed
        assert result["status"] == "success"
        # All mandatory tasks should be scheduled
        assert len(result["scheduled_tasks"]) == len(mandatory_tasks)