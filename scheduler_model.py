# scheduler_model.py

from ortools.sat.python import cp_model
import datetime
import math


class TaskScheduler:
    def __init__(self, ml_params=None):
        """
        Initialize the task scheduler with ML-derived parameters.
        
        Args:
            ml_params: Dictionary containing learned weights and parameters:
                - break_importance: How strongly we value total break time
                - max_continuous_work: Desired max minutes of continuous work
                - continuous_work_penalty: Penalty applied if total scheduled time
                  exceeds max_continuous_work (approximation, not strictly consecutive)
                - evening_work_penalty: Penalty for tasks ending late in the work window
                - early_completion_bonus: Reward for finishing tasks earlier
        """
        # Default parameters (will be overridden if provided)
        self.ml_params = {
            'break_importance': 1.0,
            'max_continuous_work': 90,
            'continuous_work_penalty': 2.0,
            'evening_work_penalty': 3.0,
            'early_completion_bonus': 2.0
        }
        
        if ml_params:
            self.ml_params.update(ml_params)
    
    def _time_to_minutes(self, time_str):
        """Convert a 'HH:MM' time string to minutes since midnight."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def _datetime_to_minutes(self, dt):
        """Convert a datetime to minutes since midnight."""
        return dt.hour * 60 + dt.minute
    
    def _parse_datetime(self, dt_str):
        """Parse an ISO datetime string to a Python datetime object.
        Ensure consistent timezone handling."""
        return datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    def _extract_date_from_tasks(self, tasks):
        """Extract the target date from the tasks list.
        Assumes all tasks are meant to be scheduled on the same day.
        Returns a datetime.date object.
        """
        if not tasks:
            # Default to today if no tasks provided
            return datetime.datetime.now().date()
        
        # Find a task with a due date
        for task in tasks:
            if 'due' in task and task['due']:
                due_dt = self._parse_datetime(task['due'])
                return due_dt.date()
        
        # If no tasks have due dates, default to today
        return datetime.datetime.now().date()
    
    def _priority_to_value(self, priority: str) -> int:
        """Convert string priority to numeric scale."""
        priority_map = {
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        return priority_map.get(priority, 1)
    
    def _value_to_priority(self, priority_val: int) -> str:
        """Convert numeric scale back to string priority."""
        priority_map = {
            3: 'High',
            2: 'Medium',
            1: 'Low'
        }
        return priority_map.get(priority_val, 'Medium')
    
    def _compute_task_score(self, base_priority: int, days_to_due: int) -> int:
        """
        Compute a scoring factor for optional tasks, combining priority and how soon the due date is.
        For example, an approaching due date may increase the score.
        You can refine this as you see fit.
        """
        # Simple logic: If due is in 0 or negative days_to_due => higher score.
        # If due is in the future, we weight it less.
        
        # If due is in 1 day, less reduction. If 2 days away, reduce a bit more, etc.
        score = base_priority * 100 + max(0, 5 - days_to_due) * 200
        return max(score, 100)  # Higher base minimum value
    
    def schedule_tasks(self, tasks, calendar_events, constraints, target_date=None):
        """
        Schedule tasks using OR-Tools CP-SAT, implementing:
        1) No overlap among tasks or events
        2) Tasks prioritized by their priority (High, Medium, Low) AND due date
        3) Break time rewarded in objective
        4) Penalties/rewards for continuous work, evening work, and early completion
        
        Args:
            tasks: List of task dictionaries
            calendar_events: List of event dictionaries
            constraints: Dictionary of scheduling constraints
            target_date: Optional datetime object specifying the target date for scheduling
                          (if provided, this overrides date extraction from tasks)
        
        Returns:
            dict with "status": "success" or "error",
            and "scheduled_tasks": [...]
        """
        model = cp_model.CpModel()
        
        # Time horizon (single day scheduling)
        horizon = 24 * 60
        
        # Get work hours from constraints
        work_start = self._time_to_minutes(constraints['work_hours']['start'])
        work_end = self._time_to_minutes(constraints['work_hours']['end'])
        
        # Collect total event durations, to help compute break time later
        total_event_duration = 0
        
        # Use the provided target_date or extract from tasks if not provided
        if target_date:
            if isinstance(target_date, datetime.datetime):
                schedule_date = target_date.date()
            else:
                schedule_date = target_date
        else:
            schedule_date = self._extract_date_from_tasks(tasks)
        
        # Identify "today" for deciding priority weighting 
        # This ensures backward compatibility with tests that expect "today" to be special
        today_date = datetime.datetime.now().date()
        
        # ---- CREATE INTERVALS FOR TASKS ----
        task_vars = {}
        
        for t in tasks:
            task_id = t['id']
            duration = t['estimated_duration']
            priority_val = self._priority_to_value(t.get('priority', 'Medium'))
            
            # Parse due date
            if 'due' in t and t['due']:
                due_dt = self._parse_datetime(t['due'])
                due_date = due_dt.date()
            else:
                # If no due date provided, treat it as future
                due_dt = datetime.datetime.combine(schedule_date, datetime.time.max)
                due_date = due_dt.date()
            
            # Check if the due date is "today" - used for maintaining backward compatibility
            is_today = (due_date == today_date)
            
            # Convert due_dt to a minutes-since-midnight
            due_time_in_minutes = min(self._datetime_to_minutes(due_dt), work_end)
            
            if due_time_in_minutes < work_start:
                # Edge case: if the due time is earlier than work_start,
                # we clamp it so the user doesn't end up with negative start range.
                due_time_in_minutes = work_start
            
            # Days from now to the due date
            days_diff = (due_date - today_date).days
            
            # A task is mandatory if it's high priority OR due today/overdue 
            # (this maintains backward compatibility with tests)
            is_mandatory = (priority_val == 3) or (is_today or days_diff <= 0)
            
            # Build interval variables - ALL tasks are optional in the model
            start_var = model.NewIntVar(work_start, work_end - duration, f"start_{task_id}")
            end_var = model.NewIntVar(work_start + duration, work_end, f"end_{task_id}")
            
            # Make everything optional but with penalty for not scheduling
            presence_var = model.NewBoolVar(f"presence_{task_id}")
            interval_var = model.NewOptionalIntervalVar(
                start_var, duration, end_var, presence_var, f"interval_{task_id}"
            )
            
            # Calculate score/penalty:
            # - For high priority tasks: very high penalty for not scheduling
            # - For other tasks: score based on priority and due date proximity
            if is_mandatory:
                # Higher priority = higher penalty for not scheduling
                # 1000 base penalty for high priority or due today makes these tasks strongly preferred
                score_val = priority_val * 1000
            else:
                # Regular optional task
                score_val = self._compute_task_score(priority_val, days_diff)
            
            task_vars[task_id] = {
                "start": start_var,
                "end": end_var,
                "interval": interval_var,
                "presence": presence_var,
                "duration": duration,
                "title": t['title'],
                "priority_value": priority_val,
                "is_mandatory": is_mandatory,
                "score_val": score_val
            }
            
            # Constrain end by the earlier of the due time or work_end if present
            upper_bound = min(due_time_in_minutes, work_end)
            model.Add(end_var <= upper_bound).OnlyEnforceIf(presence_var)
        
        # ---- CREATE INTERVALS FOR CALENDAR EVENTS (FIXED) ----
        event_intervals = []
        for evt in calendar_events:
            evt_id = evt['id']
            start_dt = self._parse_datetime(evt['start'])
            end_dt = self._parse_datetime(evt['end'])
            
            start_min = self._datetime_to_minutes(start_dt)
            end_min = self._datetime_to_minutes(end_dt)
            
            # Filter events by work hours - only consider overlap with work hours
            # If event is completely before work hours or after work hours, skip it
            if end_min <= work_start or start_min >= work_end:
                continue
            
            # Adjust start and end times to only include overlap with work hours
            start_min = max(start_min, work_start)
            end_min = min(end_min, work_end)
            
            duration = max(0, end_min - start_min)
            if duration > 0:
                total_event_duration += duration
                # Create a fixed start variable
                start_var = model.NewIntVar(start_min, start_min, f"start_event_{evt_id}")
                fixed_iv = model.NewFixedSizeIntervalVar(start_var, duration, f"event_{evt_id}")
                event_intervals.append(fixed_iv)

        # Block out time before work_start and after work_end
        if work_start > 0:
            morning_start = model.NewIntVar(0, 0, "morning_start")
            morning_iv = model.NewFixedSizeIntervalVar(morning_start, work_start, "non_work_morning")
            event_intervals.append(morning_iv)
        if work_end < horizon:
            evening_start = model.NewIntVar(work_end, work_end, "evening_start")
            evening_iv = model.NewFixedSizeIntervalVar(evening_start, horizon - work_end, "non_work_evening")
            event_intervals.append(evening_iv)
        
        # ---- NO OVERLAP CONSTRAINT ----
        all_intervals = [tv["interval"] for tv in task_vars.values()] + event_intervals
        model.AddNoOverlap(all_intervals)
        
        # ---- MUST START AND END WITHIN WORK HOURS ----
        for tv in task_vars.values():
            model.Add(tv["start"] >= work_start).OnlyEnforceIf(tv["presence"])
            model.Add(tv["end"] <= work_end).OnlyEnforceIf(tv["presence"])
        
        # ---- OBJECTIVE CONSTRUCTION ----
        objective_terms = []
        
        # 1) Sum up total scheduled time
        scheduled_time_var = model.NewIntVar(0, work_end - work_start, "scheduled_time")
        partial_sum = []
        
        for task_id, tv in task_vars.items():
            dur_contrib = model.NewIntVar(0, tv["duration"], f"dur_contrib_{task_id}")
            model.Add(dur_contrib == tv["duration"]).OnlyEnforceIf(tv["presence"])
            model.Add(dur_contrib == 0).OnlyEnforceIf(tv["presence"].Not())
            partial_sum.append(dur_contrib)
        
        model.Add(scheduled_time_var == sum(partial_sum))
        
        # 2) Break time calculation
        break_importance = self.ml_params['break_importance'] * 0.1  # REDUCED by 10x
        available_work_window = (work_end - work_start) - total_event_duration
        break_time_expr = model.NewIntVar(0, available_work_window, "break_time_expr")
        model.Add(break_time_expr == (available_work_window - scheduled_time_var))
        objective_terms.append(break_importance * break_time_expr)
        
        # 3) Task scheduling rewards/penalties
        for task_id, tv in task_vars.items():
            if tv["is_mandatory"]:
                # Penalty for NOT scheduling mandatory tasks (high penalty)
                # This will be added only when presence=0
                objective_terms.append(-tv["score_val"] * 50 * tv["presence"].Not())  # INCREASED by 50x
            else:
                # Reward for scheduling optional tasks
                presence_score = tv["score_val"] * 500  # INCREASED by 5x
                objective_terms.append(presence_score * tv["presence"])
        
        # 4) Early completion bonus
        early_completion_bonus = self.ml_params['early_completion_bonus']
        for task_id, tv in task_vars.items():
            end_term = model.NewIntVar(0, work_end, f"end_term_{task_id}")
            model.Add(end_term == tv["end"]).OnlyEnforceIf(tv["presence"])
            model.Add(end_term == 0).OnlyEnforceIf(tv["presence"].Not())
            objective_terms.append(-1 * end_term * tv["priority_value"] * early_completion_bonus)
        
        # 5) Evening penalty
        evening_cutoff = work_end - 60
        evening_penalty = self.ml_params['evening_work_penalty'] * 0.5  # REDUCED by 50%
        for task_id, tv in task_vars.items():
            is_evening = model.NewBoolVar(f"evening_{task_id}")
            model.Add(tv["end"] > evening_cutoff).OnlyEnforceIf(is_evening)
            model.Add(tv["end"] <= evening_cutoff).OnlyEnforceIf(is_evening.Not())
            model.Add(is_evening == 0).OnlyEnforceIf(tv["presence"].Not())
            objective_terms.append(-1 * is_evening * evening_penalty * 50)  # REDUCED impact
        
        # 6) Continuous work penalty
        max_cont_work = self.ml_params['max_continuous_work']
        cont_penalty = self.ml_params['continuous_work_penalty'] * 0.2  # REDUCED by 80%
        excess_work = model.NewIntVar(0, work_end - work_start, "excess_work")
        model.Add(excess_work >= scheduled_time_var - max_cont_work)
        model.Add(excess_work >= 0)
        objective_terms.append(-1 * cont_penalty * 10 * excess_work)
        
        # Combine into final objective
        model.Maximize(sum(objective_terms))
        
        # ---- Solve the model ----
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30  # can adjust
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            scheduled_tasks = []
            
            # Create a base datetime representing midnight of the target date
            base_date = datetime.datetime.combine(schedule_date, datetime.time.min)
            
            for task_id, tv in task_vars.items():
                # Check if task was scheduled
                presence_val = solver.Value(tv["presence"])
                if presence_val == 1:
                    start_val = solver.Value(tv["start"])
                    end_val = solver.Value(tv["end"])
                    
                    # Create ISO strings using the target date
                    start_dt = base_date + datetime.timedelta(minutes=start_val)
                    end_dt = base_date + datetime.timedelta(minutes=end_val)

                    # Format with 'Z' for UTC without actually changing the datetime objects
                    # This ensures the timezone info is in the string without affecting datetime comparisons
                    start_iso = start_dt.isoformat() + 'Z'
                    end_iso = end_dt.isoformat() + 'Z'

                    scheduled_tasks.append({
                        'id': task_id,
                        'title': tv["title"],
                        'start': start_iso,
                        'end': end_iso,
                        'priority': self._value_to_priority(tv["priority_value"]),
                        'estimated_duration': tv["duration"],
                        'mandatory': tv["is_mandatory"]
                    })

            
            # Check if we scheduled all mandatory tasks
            mandatory_tasks = [tv for tv in task_vars.values() if tv["is_mandatory"]]
            mandatory_scheduled = all(
                solver.Value(tv["presence"]) == 1 
                for tv in task_vars.values() 
                if tv["is_mandatory"]
            )
            
            if not mandatory_scheduled and mandatory_tasks:
                return {
                    'status': 'partial',
                    'message': 'Could not schedule all high-priority tasks due to time constraints',
                    'scheduled_tasks': scheduled_tasks
                }
            
            return {
                'status': 'success',
                'scheduled_tasks': scheduled_tasks
            }
        else:
            # Return diagnostics with the error
            total_task_mins = sum(tv["duration"] for tv in task_vars.values())
            mandatory_mins = sum(tv["duration"] for tv in task_vars.values() if tv["is_mandatory"])
            available_mins = (work_end - work_start) - total_event_duration
            
            return {
                'status': 'error',
                'message': f'No feasible solution found. Solver status: {solver.StatusName(status)}',
                'diagnostics': {
                    'total_tasks': len(tasks),
                    'mandatory_tasks': sum(1 for tv in task_vars.values() if tv["is_mandatory"]),
                    'total_task_minutes': total_task_mins,
                    'mandatory_task_minutes': mandatory_mins,
                    'available_minutes': available_mins,
                    'calendar_event_minutes': total_event_duration
                }
            }