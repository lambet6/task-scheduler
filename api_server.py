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

# Add CORS middleware to allow requests from your React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML constraint learner
ml_learner = MLConstraintLearner(data_dir="./user_data")

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
    energy_level: int = Field(..., ge=1, le=5)  # 1-5 scale
    adjusted_tasks: List[Dict[str, Any]] = []
    completed_tasks: List[Union[str, int]] = []
    task_specific_feedback: Optional[Dict[str, str]] = None

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
        tasks_as_dicts = [task.dict() for task in request.tasks]
        events_as_dicts = [event.dict() for event in request.calendar_events]
        
        # Call the scheduler with dictionaries
        result = scheduler.schedule_tasks(
            tasks=tasks_as_dicts,
            calendar_events=events_as_dicts,
            constraints=request.constraints.dict()
        )
        
        if result['status'] == 'error':
            return ScheduleResponse(
                status="error",
                message=result['message']
            )
        
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
        # Record feedback for ML learning
        ml_learner.record_feedback(
            user_id=request.user_id,
            schedule_data=request.schedule_data,
            feedback_data=request.feedback_data
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

@app.post("/predict_duration")
async def predict_duration(task_data: Dict[str, Any], user_id: str = Body(...)):
    """
    Predict task duration using ML model.
    This is a placeholder - in a full implementation, you'd
    use a separate ML model trained on historical task completion data.
    """
    # Very simple baseline prediction based on task type
    task_type = task_data.get("type", "").lower()
    
    if "meeting" in task_type:
        return {"predicted_duration": 60}  # 60 minutes for meetings
    elif "email" in task_type:
        return {"predicted_duration": 15}  # 15 minutes for emails
    elif "report" in task_type:
        return {"predicted_duration": 120}  # 2 hours for reports
    else:
        return {"predicted_duration": 30}  # Default 30 minutes

if __name__ == "__main__":
    import uvicorn
    
    # Create data directory if it doesn't exist
    os.makedirs("./user_data", exist_ok=True)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)