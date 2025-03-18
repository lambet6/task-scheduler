# Scheduler Algorithm: Code Walkthrough

## Introduction

This document provides a detailed walkthrough of the task scheduling algorithm implemented in `scheduler_model.py`. The algorithm uses constraint programming to create optimal schedules based on user preferences and constraints.

## Constraint Programming Background

The scheduler uses Google's OR-Tools CP-SAT solver, which is based on constraint programming with satisfiability techniques. Key concepts:

1. **Variables**: Represent unknown values (like task start times)
2. **Domains**: Possible values for each variable
3. **Constraints**: Restrictions on variable assignments
4. **Objective Function**: The value to optimize (maximize in our case)

## Key Components

### 1. Variable Representation

Tasks and events are represented as interval variables with:

- **Start Time**: When the task begins
- **End Time**: When the task completes
- **Duration**: Fixed length of the task
- **Presence** (for optional tasks): Boolean indicating if the task is scheduled

### 2. Objective Function Components

The objective function balances several factors:

- **Break Time Reward**: Rewarding adequate breaks
- **Optional Task Inclusion**: Reward for scheduling optional tasks based on priority
- **Early Completion Bonus**: Rewarding completing high-priority tasks earlier
- **Evening Work Penalty**: Discouraging work late in the day
- **Continuous Work Penalty**: Discouraging long stretches without breaks

## Algorithm Walkthrough

### Initialization and Setup

```python
def schedule_tasks(self, tasks, calendar_events, constraints):
    model = cp_model.CpModel()
    
    # Time horizon (assume single day scheduling)
    horizon = 24 * 60
    
    work_start = self._time_to_minutes(constraints['work_hours']['start'])
    work_end = self._time_to_minutes(constraints['work_hours']['end'])
```

We initialize the constraint model and convert the work hours to minutes since midnight for easier processing.

### Task Classification and Variable Creation

```python
task_vars = {}

for t in tasks:
    task_id = t['id']
    duration = t['estimated_duration']
    
    # Parse due date
    if 'due' in t and t['due']:
        due_dt = self._parse_datetime(t['due'])
        due_date = due_dt.date()
    else:
        # If no due date provided, treat it as future
        due_dt = today_midnight + datetime.timedelta(days=9999)
        due_date = due_dt.date()
    
    # Check if the due date is "today"
    is_today = (due_date == today_date)
    
    # Days from now to the due date
    days_diff = (due_dt.date() - today_date).days
    
    # Build interval variables
    start_var = model.NewIntVar(work_start, work_end - duration, f"start_{task_id}")
    end_var = model.NewIntVar(work_start + duration, work_end, f"end_{task_id}")
    
    # If mandatory => "NewIntervalVar", if optional => "NewOptionalIntervalVar"
    if is_today or days_diff <= 0:
        # Mandatory
        interval_var = model.NewIntervalVar(start_var, duration, end_var, f"interval_{task_id}")
        presence_var = None  # Always present
        is_mandatory = True
    else:
        # Optional
        presence_var = model.NewBoolVar(f"presence_{task_id}")
        interval_var = model.NewOptionalIntervalVar(start_var, duration, end_var, presence_var, f"interval_{task_id}")
        is_mandatory = False
        
        # Weighted by priority + how soon it's due
        score_val = self._compute_task_score(priority_val, days_diff)
```

This code:
1. Processes each task to determine if it's mandatory (due today) or optional
2. Creates interval variables with appropriate domains for start and end times
3. For optional tasks, creates a boolean "presence" variable that determines if the task is scheduled
4. Computes a score for optional tasks based on priority and due date proximity

### Event Processing

```python
event_intervals = []
for evt in calendar_events:
    evt_id = evt['id']
    start_dt = self._parse_datetime(evt['start'])
    end_dt = self._parse_datetime(evt['end'])
    
    start_min = self._datetime_to_minutes(start_dt)
    end_min = self._datetime_to_minutes(end_dt)
    
    duration = max(0, end_min - start_min)
    if duration > 0:
        total_event_duration += duration
        # Create a fixed start variable
        start_var = model.NewIntVar(start_min, start_min, f"start_event_{evt_id}")
        # Use NewFixedSizeIntervalVar
        fixed_iv = model.NewFixedSizeIntervalVar(start_var, duration, f"event_{evt_id}")
        event_intervals.append(fixed_iv)
```

This creates fixed interval variables for calendar events that cannot be moved.

### Non-Overlap Constraint

```python
all_intervals = [tv["interval"] for tv in task_vars.values()] + event_intervals
model.AddNoOverlap(all_intervals)
```

This single constraint ensures no tasks overlap with each other or with calendar events.

### Work Hours Constraints

```python
for tv in task_vars.values():
    model.Add(tv["start"] >= work_start)
    model.Add(tv["end"] <= work_end)
    if not tv["is_mandatory"]:
        model.Add(tv["start"] >= work_start).OnlyEnforceIf(tv["presence"])
        model.Add(tv["end"] <= work_end).OnlyEnforceIf(tv["presence"])
```

These constraints ensure all tasks are scheduled within work hours.

### Objective Function Construction

The objective function is built from multiple components:

#### 1. Scheduled Time Tracking

```python
scheduled_time_var = model.NewIntVar(0, work_end - work_start, "scheduled_time")
partial_sum = []

for task_id, tv in task_vars.items():
    if tv["is_mandatory"]:
        partial_sum.append(tv["duration"])
    else:
        dur_contrib = model.NewIntVar(0, tv["duration"], f"dur_contrib_{task_id}")
        model.Add(dur_contrib == tv["duration"]).OnlyEnforceIf(tv["presence"])
        model.Add(dur_contrib == 0).OnlyEnforceIf(tv["presence"].Not())
        partial_sum.append(dur_contrib)

model.Add(scheduled_time_var == sum(partial_sum))
```

This calculates the total scheduled time, properly handling optional tasks.

#### 2. Break Time Reward

```python
available_work_window = (work_end - work_start) - total_event_duration
break_importance = self.ml_params['break_importance']
break_time_expr = model.NewIntVar(0, available_work_window, "break_time_expr")
model.Add(break_time_expr == (available_work_window - scheduled_time_var))
objective_terms.append(break_importance * break_time_expr)
```

This rewards schedules that leave adequate break time.

#### 3. Optional Task Inclusion Reward

```python
for task_id, tv in task_vars.items():
    if not tv["is_mandatory"]:
        presence_score = tv["score_val"] * 100
        objective_terms.append(presence_score * tv["presence"])
```

This rewards including optional tasks, weighted by their priority and due date proximity.

#### 4. Early Completion Bonus

```python
early_completion_bonus = self.ml_params['early_completion_bonus']
for tv in task_vars.values():
    objective_terms.append(
        -1 * tv["end"] * tv["priority_value"] * early_completion_bonus
    )
```

This rewards scheduling high-priority tasks earlier in the day.

#### 5. Evening Work Penalty

```python
evening_cutoff = work_end - 60
evening_penalty = self.ml_params['evening_work_penalty']
for task_id, tv in task_vars.items():
    is_evening = model.NewBoolVar(f"evening_{task_id}")
    model.Add(tv["end"] > evening_cutoff).OnlyEnforceIf(is_evening)
    model.Add(tv["end"] <= evening_cutoff).OnlyEnforceIf(is_evening.Not())
    objective_terms.append(-1 * is_evening * evening_penalty * 100)
```

This penalizes scheduling tasks that end late in the day.

#### 6. Continuous Work Penalty

```python
max_cont_work = self.ml_params['max_continuous_work']
cont_penalty = self.ml_params['continuous_work_penalty']

excess_work = model.NewIntVar(0, work_end - work_start, "excess_work")
model.Add(excess_work >= scheduled_time_var - max_cont_work)
model.Add(excess_work >= 0)

objective_terms.append(-1 * cont_penalty * 10 * excess_work)
```

This penalizes schedules where the total scheduled time exceeds the preferred maximum continuous work time.

### Finalizing the Objective Function

```python
# Combine into final objective
model.Maximize(sum(objective_terms))
```

The objective terms are combined into a single expression to maximize.

### Solving the Model

```python
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30  # can adjust
status = solver.Solve(model)
```

The solver attempts to find an optimal solution within the time limit of 30 seconds.

### Processing the Solution

```python
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    scheduled_tasks = []
    base_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for task_id, tv in task_vars.items():
        if tv["is_mandatory"]:
            # Always scheduled
            start_val = solver.Value(tv["start"])
            end_val   = solver.Value(tv["end"])
            scheduled_tasks.append({
                'id': task_id,
                'title': tv["title"],
                'start': (base_date + datetime.timedelta(minutes=start_val)).isoformat(),
                'end':   (base_date + datetime.timedelta(minutes=end_val)).isoformat(),
                'priority': self._value_to_priority(tv["priority_value"]),
                'estimated_duration': tv["duration"],
                'mandatory': True
            })
        else:
            # Optional => only if presence=1
            presence_val = solver.Value(tv["presence"])
            if presence_val == 1:
                start_val = solver.Value(tv["start"])
                end_val   = solver.Value(tv["end"])
                scheduled_tasks.append({
                    'id': task_id,
                    'title': tv["title"],
                    'start': (base_date + datetime.timedelta(minutes=start_val)).isoformat(),
                    'end':   (base_date + datetime.timedelta(minutes=end_val)).isoformat(),
                    'priority': self._value_to_priority(tv["priority_value"]),
                    'estimated_duration': tv["duration"],
                    'mandatory': False
                })
```

When a solution is found, the values of the decision variables are extracted and converted back to a user-friendly format.

## Algorithm Analysis

### Time Complexity

The constraint solving process is NP-hard in general, which is why a time limit is set. The actual runtime depends on:

- Number of tasks and events
- Complexity of constraints
- How constrained the problem is (tightly constrained problems can be faster to solve)

### Space Complexity

The solver creates variables and constraints for:
- Each task (start, end, interval, presence variables)
- Each event (fixed intervals)
- Objective function components

This leads to O(n) variables and constraints where n is the number of tasks and events.

### Optimality vs. Feasibility

The solver attempts to find an optimal solution, but if time runs out:
- If at least one feasible solution was found, it returns that
- If no feasible solution was found, it returns an error

### Approximate Constraints

Some constraints like "avoid continuous work" are approximated:
- Rather than tracking exact consecutive work time, we penalize total scheduled time exceeding a threshold
- This simplification makes the model more tractable while preserving the intent

## Core ML Parameter Influence

Each ML parameter directly influences the objective function:

1. **`break_importance`**: Multiplier for break time reward
   ```python
   objective_terms.append(break_importance * break_time_expr)
   ```

2. **`early_completion_bonus`**: Multiplier for early completion reward
   ```python
   objective_terms.append(-1 * tv["end"] * tv["priority_value"] * early_completion_bonus)
   ```

3. **`evening_work_penalty`**: Penalty for work ending late
   ```python
   objective_terms.append(-1 * is_evening * evening_work_penalty * 100)
   ```

4. **`continuous_work_penalty`**: Penalty for exceeding continuous work threshold
   ```python
   objective_terms.append(-1 * cont_penalty * 10 * excess_work)
   ```

5. **`max_continuous_work`**: Threshold for excess work calculation
   ```python
   model.Add(excess_work >= scheduled_time_var - max_cont_work)
   ```

## Optimizing the Optimizer

### Performance Improvements

1. **Tighter Variable Domains**: Narrowing the domains of start/end variables improves performance
   ```python
   # Instead of full 24h range:
   start_var = model.NewIntVar(work_start, work_end - duration, f"start_{task_id}")
   ```

2. **Strategic Constraints**: Adding the no-overlap constraint is more efficient than manually preventing all pairwise overlaps

3. **Solver Parameters**: The 30-second timeout balances solution quality with user experience

### Possible Enhancements

1. **Real Continuous Work Tracking**: Use additional interval variables to track actual continuous work periods

2. **Time-of-Day Preferences**: Add parameters for morning/afternoon preferences

3. **Task Dependencies**: Add support for tasks that must be completed before others

4. **Multiple Day Scheduling**: Extend to schedule across multiple days

## Algorithm Limitations

1. **Approximation of Continuous Work**: The model approximates continuous work rather than modeling it exactly

2. **Fixed Task Durations**: The model assumes fixed durations rather than ranges

3. **Single Day Focus**: Currently optimizes for a single day at a time

4. **Limited Precedence Constraints**: The current implementation doesn't handle dependencies between tasks

## Conclusion

The scheduling algorithm successfully balances multiple objectives to create personalized, optimized schedules. The ML component continuously tunes the objective function weights based on user feedback, leading to increasingly tailored schedules.

The combination of constraint programming and machine learning creates a powerful adaptive system that respects hard constraints while optimizing for user preferences.