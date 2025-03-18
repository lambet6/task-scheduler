# AI-Powered Task Scheduler: Technical Documentation

## System Overview

This task scheduling system implements an intelligent API service that optimizes daily task scheduling based on user preferences, calendar events, and learned constraints. The system integrates machine learning to personalize scheduling parameters based on user feedback.

### Key Components

1. **FastAPI Backend Service** (`api_server.py`) - RESTful API for scheduling tasks and collecting feedback
2. **Machine Learning Component** (`ml_constraint_learner.py`) - Learns personalized scheduling parameters from user feedback
3. **Task Scheduler Engine** (`scheduler_model.py`) - Constraint solver for optimal task scheduling

### Core Features

- **Personalized Scheduling** - Adapts to user preferences through machine learning
- **Calendar Integration** - Respects existing calendar events when scheduling
- **Feedback-Based Learning** - Continuously improves schedules based on user feedback
- **Prioritization** - Schedules tasks based on priority and due dates
- **Wellbeing Optimization** - Enforces breaks and prevents overwork

## Architecture Details

### API Service (`api_server.py`)

The API service provides endpoints for task scheduling and feedback collection, serving as the interface between clients and the scheduling engine.

#### Endpoints

1. **`/optimize_schedule`** (POST)
   - Schedules tasks based on user constraints and learns ML parameters
   - Request includes tasks, calendar events, and constraints
   - Returns optimized schedule

2. **`/record_feedback`** (POST)
   - Collects user feedback to improve future scheduling
   - Includes mood, energy level, and task-specific feedback

3. **`/predict_duration`** (POST)
   - Predicts task duration based on task type and user history

#### Data Models

- **`TaskInput`** - Task with title, priority, estimated duration, and optional due date
- **`EventInput`** - Calendar event with title, start and end times
- **`ScheduleConstraints`** - Work hours and maximum continuous work time
- **`ScheduleResponse`** - Optimized schedule with list of scheduled tasks
- **`FeedbackItem`** - User feedback on schedule quality and task completion

### Machine Learning Component (`ml_constraint_learner.py`)

The ML component learns personalized scheduling parameters from user feedback to optimize future schedules.

#### Key Parameters Learned

- **`break_importance`** - How strongly to value break time
- **`max_continuous_work`** - Maximum desired continuous work period
- **`continuous_work_penalty`** - Penalty for exceeding max continuous work
- **`evening_work_penalty`** - Penalty for scheduling tasks late in the day
- **`early_completion_bonus`** - Bonus for scheduling high-priority tasks earlier

#### Learning Process

1. **Feature Extraction** - Extracts schedule characteristics like work duration, break time, etc.
2. **Feedback Collection** - Records user mood, energy, and task adjustments
3. **Model Training** - Uses RandomForest to correlate schedule features with user satisfaction
4. **Parameter Adjustment** - Updates scheduling parameters based on feature importance and correlations

### Scheduling Engine (`scheduler_model.py`)

The scheduling engine uses Google's OR-Tools constraint solver to optimize task placement.

#### Key Optimization Factors

1. **Task Prioritization** - Schedules higher priority tasks first
2. **Due Date Proximity** - Prioritizes tasks with closer due dates
3. **Break Optimization** - Ensures adequate breaks between tasks
4. **Work Distribution** - Avoids excessive continuous work periods
5. **Time Preference** - Considers user preference for early/late scheduling

#### Constraint Processing

- Uses CP-SAT solver to handle complex constraint satisfaction
- Represents tasks and events as interval variables
- Enforces non-overlap between tasks and events
- Maximizes objective function combining multiple weighted factors

## Implementation Details

### Task Scheduling Algorithm

The scheduler uses a constraint programming approach to optimize task placement:

1. **Task Classification** - Tasks are classified as mandatory (due today) or optional
2. **Interval Creation** - Tasks and events are converted to interval variables
3. **Constraint Application** - Apply non-overlap, work hours, and due date constraints
4. **Objective Maximization** - Optimize for:
   - Break time (weighted by `break_importance`)
   - Optional task inclusion (weighted by priority and due date)
   - Early completion for high-priority tasks
   - Penalties for evening work and excessive continuous time

### Machine Learning Approach

The ML component uses a Random Forest regressor to learn relationships between schedule features and user satisfaction:

1. **Features** - `avg_task_duration`, `total_work_minutes`, `break_minutes`, etc.
2. **Target** - User `mood_score` from feedback (1-5 scale)
3. **Parameter Adjustment** - Updates scheduling parameters based on feature importance and correlation with mood

For example:
- If `actual_break_minutes` correlates positively with mood, increase `break_importance`
- If `excess_work` correlates negatively with mood, increase `continuous_work_penalty`

### Data Storage

- User feedback stored in CSV format: `./user_data/user_{user_id}_feedback.csv`
- Trained models saved as pickle files: `./user_data/user_{user_id}_mood_predictor.pkl`
- Learned parameters stored as JSON: `./user_data/user_{user_id}_params.json`

## Integration Guide

### Adding the Scheduler to Your Application

1. **API Integration**
   - Send POST requests to `/optimize_schedule` with tasks and constraints
   - Record user feedback with POST to `/record_feedback`

2. **Required Data Structures**
   ```python
   # Schedule Request
   {
     "user_id": "user123",
     "tasks": [
       {"id": "task1", "title": "Important meeting", "priority": "High", "estimated_duration": 60, "due": "2025-03-09T17:00:00Z"},
       {"id": "task2", "title": "Write report", "priority": "Medium", "estimated_duration": 120}
     ],
     "calendar_events": [
       {"id": "evt1", "title": "Team standup", "start": "2025-03-09T10:00:00Z", "end": "2025-03-09T10:30:00Z"}
     ],
     "constraints": {
       "work_hours": {"start": "09:00", "end": "17:00"},
       "max_continuous_work_min": 90
     }
   }
   ```

3. **Feedback Structure**
   ```python
   {
     "user_id": "user123",
     "schedule_data": {...},  # The schedule that was created
     "feedback_data": {
       "mood_score": 4,
       "energy_level": 3,
       "completed_tasks": ["task1"],
       "adjusted_tasks": []
     }
   }
   ```

## Performance Considerations

### Scalability

- The CP-SAT solver has a 30-second timeout to ensure reasonable response times
- ML model training only occurs after 5+ data points to ensure meaningful learning
- Consider implementing caching for user parameters to reduce database access

### Limitations

- Scheduling is done for a single day at a time
- No multi-user optimization (each user's schedule is optimized independently)
- Simplified model for predicting task duration based only on task type

## Future Enhancements

1. **Multi-day Scheduling** - Extend scheduling window beyond a single day
2. **More Sophisticated ML** - Add neural network models for better personalization
3. **Collaborative Scheduling** - Optimize schedules across team members
4. **Enhanced Metrics** - Track more detailed wellness and productivity metrics
5. **Integration with Productivity Tools** - Connect with popular task management apps

## Deployment Requirements

- Python 3.7+
- FastAPI
- Google OR-Tools
- scikit-learn
- pandas, numpy, joblib
- uvicorn (for serving)

## Operational Notes

- Create user_data directory to store user feedback and trained models
- Consider adding authentication middleware for production use
- Set appropriate CORS settings for your frontend domain