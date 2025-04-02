from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import os
import json
from datetime import datetime

# Import our scheduler and ML components
from scheduler_model import TaskScheduler
from ml_constraint_learner import MLConstraintLearner

app = FastAPI(title="Task Scheduler API")

# Add CORS middleware to allow requests from React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML constraint learner
data_dir = os.environ.get("DATA_DIR", "./user_data")
ml_learner = MLConstraintLearner(data_dir=data_dir)

# Pydantic models for request/response validation
class TaskBase(BaseModel):
    id: Union[str, int]
    title: str
    priority: str = "Medium"  # High, Medium, Low
    estimated_duration: int  # Duration in minutes

class TaskInput(TaskBase):
    due: Optional[str] = None  # ISO datetime string for due date/time

class EventInput(BaseModel):
    id: Union[str, int]
    title: str
    start: str  # ISO datetime string
    end: str    # ISO datetime string

class WorkHours(BaseModel):
    start: str  # HH:MM format
    end: str    # HH:MM format

class ScheduleConstraints(BaseModel):
    work_hours: WorkHours
    max_continuous_work_min: Optional[int] = 90

class ScheduleRequest(BaseModel):
    user_id: str
    tasks: List[TaskInput]
    calendar_events: List[EventInput]
    constraints: ScheduleConstraints
    optimization_goal: Optional[str] = "maximize_wellbeing"
    target_date: Optional[str] = None  # New field for explicit target date

class ScheduledTask(BaseModel):
    id: Union[str, int]
    title: str
    priority: str  # High, Medium, Low
    estimated_duration: int  # Duration in minutes
    start: str  # ISO datetime string
    end: str    # ISO datetime string
    mandatory: bool = False

class ScheduleResponse(BaseModel):
    status: str
    scheduled_tasks: List[ScheduledTask] = []
    message: Optional[str] = None

class FeedbackItem(BaseModel):
    mood_score: int = Field(..., ge=1, le=5)  # 1-5 scale
    adjusted_tasks: List[Dict[str, Any]] = []
    completed_tasks: List[Union[str, int]] = []

class FeedbackRequest(BaseModel):
    user_id: str
    schedule_data: Dict[str, Any]
    feedback_data: FeedbackItem

@app.post("/optimize_schedule", response_model=ScheduleResponse)
async def optimize_schedule(request: ScheduleRequest):
    try:
        # Get ML-derived parameters for this user
        ml_params = ml_learner.get_user_parameters(request.user_id)
        
        # Initialize scheduler with ML parameters
        scheduler = TaskScheduler(ml_params=ml_params)
        
        # Convert Pydantic models to dictionaries
        tasks_as_dicts = [task.model_dump() for task in request.tasks]
        events_as_dicts = [event.model_dump() for event in request.calendar_events]
        
        # Prepare additional context for the scheduler
        scheduling_context = {}
        
        # If a target_date was provided, parse it to datetime
        if request.target_date:
            try:
                # Parse the ISO string to datetime (will be used by scheduler to set the base date)
                target_date = datetime.fromisoformat(request.target_date.replace('Z', '+00:00'))
                scheduling_context['target_date'] = target_date
                print(f"Using explicit target date: {target_date}")
            except ValueError as e:
                print(f"Warning: Could not parse target_date '{request.target_date}': {e}")
        
        # Call the scheduler with dictionaries and context
        result = scheduler.schedule_tasks(
            tasks=tasks_as_dicts,
            calendar_events=events_as_dicts,
            constraints=request.constraints.model_dump(),
            # Pass any additional context
            **scheduling_context
        )
        
        if result['status'] == 'error':
            # Log diagnostic information if available
            if 'diagnostics' in result:
                print(f"Scheduling diagnostics: {result['diagnostics']}")
                
            return ScheduleResponse(
                status="error",
                message=result['message']
            )
        
        # Handle partial schedules
        if result['status'] == 'partial':
            return ScheduleResponse(
                status="partial",
                message=result['message'],
                scheduled_tasks=result['scheduled_tasks']
            )
        
        # Success case
        return ScheduleResponse(
            status="success",
            scheduled_tasks=result['scheduled_tasks']
        )
    
    except Exception as e:
        # Log the error (in a production system)
        print(f"Error scheduling tasks: {str(e)}")
        
        # Return error response
        return ScheduleResponse(
            status="error",
            message=f"Failed to schedule tasks: {str(e)}"
        )

@app.post("/record_feedback", status_code=200)
async def record_feedback(request: FeedbackRequest):
    try:
        # Convert feedback_data Pydantic model to dict
        feedback_dict = request.feedback_data.model_dump()
        
        # Record feedback for ML learning
        ml_learner.record_feedback(
            user_id=request.user_id,
            schedule_data=request.schedule_data,
            feedback_data=feedback_dict  # Pass as dictionary instead of Pydantic model
        )
        
        return {"status": "success", "message": "Feedback recorded successfully"}
    
    except Exception as e:
        # Log the error
        print(f"Error recording feedback: {str(e)}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Create data directory if it doesn't exist
    data_dir = os.environ.get("DATA_DIR", "./user_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)