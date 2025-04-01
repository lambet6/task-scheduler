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
        """Parse an ISO datetime string to a Python datetime object."""
        return datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
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
        # Simple logic: If due is in 0 or negative days_to_due => must be done today (mandatory).
        # If due is in the future, we weight it less.
        # E.g., reduce the value by each day away, but not below 1.
        
        # If due is in 1 day, no reduction. If 2 days away, reduce a bit more, etc.        
        score = base_priority + max(0, 3 - days_to_due)
        return max(score, 1)
    
    def schedule_tasks(self, tasks, calendar_events, constraints):
        """
        Schedule tasks using OR-Tools CP-SAT, implementing:
        1) No overlap among tasks or events
        2) Mandatory tasks for those due today, optional otherwise
        3) Break time rewarded in objective
        4) Penalties/rewards for continuous work, evening work, and early completion
        5) Weighted optional tasks by due date proximity and priority
        
        Args:
            tasks (list): List of dicts with:
                {
                  "id": str or int,
                  "title": str,
                  "priority": str (High/Medium/Low),
                  "estimated_duration": int (minutes),
                  "due": optional str in ISO format
                }
            calendar_events (list): Each with start/end in ISO, for blocking time
            constraints (dict): Must have:
                {
                  "work_hours": {
                      "start": "HH:MM",
                      "end":   "HH:MM"
                  },
                  "max_continuous_work_min": int (unused if we rely on ml_params, but can read if we want)
                }
        
        Returns:
            dict with "status": "success" or "error",
            and "scheduled_tasks": [...]
        """
        model = cp_model.CpModel()
        
        # Time horizon (assume single day scheduling).
        horizon = 24 * 60
        
        work_start = self._time_to_minutes(constraints['work_hours']['start'])
        work_end = self._time_to_minutes(constraints['work_hours']['end'])
        
        # Collect total event durations, to help compute break time later.
        total_event_duration = 0
        
        # Identify "today" for deciding mandatory vs optional
        # (Assume 'today' is the current date at local midnight.)
        today_midnight = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Convert that to date for comparison
        today_date = today_midnight.date()
        
        # ---- CREATE INTERVALS FOR TASKS ----
        
        # We'll store a structure:
        # task_vars[task_id] = {
        #   "start": IntVar,
        #   "end": IntVar,
        #   "interval": IntervalVar,
        #   "presence": BoolVar or None,
        #   "duration": int,
        #   "score": optional weighting for optional tasks,
        #   "title": str,
        #   "priority_value": int,
        #   "is_mandatory": bool
        # }
        
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
            
            # Convert due_dt to a minutes-since-midnight
            due_time_in_minutes = min(self._datetime_to_minutes(due_dt), work_end)
            
            if due_time_in_minutes < work_start:
                # Edge case: if the due time is earlier than work_start,
                # we clamp it so the user doesn't end up with negative start range.
                due_time_in_minutes = work_start
            
            priority_val = self._priority_to_value(t.get('priority', 'Medium'))
            
            # Days from now to the due date
            days_diff = (due_dt.date() - today_date).days
            # Negative or zero means it's effectively due today or overdue => mandatory
            
            # Build interval variables
            start_var = model.NewIntVar(work_start, work_end - duration, f"start_{task_id}")
            end_var = model.NewIntVar(work_start + duration, work_end, f"end_{task_id}")
            
            # If mandatory => "NewIntervalVar", if optional => "NewOptionalIntervalVar"
            if is_today or days_diff <= 0:
                # Mandatory
                interval_var = model.NewIntervalVar(start_var, duration, end_var, f"interval_{task_id}")
                presence_var = None  # Always present
                is_mandatory = True
                score_val = 0  # We'll handle mandatory tasks differently in objective
            else:
                # Optional
                presence_var = model.NewBoolVar(f"presence_{task_id}")
                interval_var = model.NewOptionalIntervalVar(start_var, duration, end_var, presence_var, f"interval_{task_id}")
                is_mandatory = False
                
                # Weighted by priority + how soon it's due
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
            
            # Constrain end by the earlier of the due time or work_end if mandatory
            if is_mandatory:
                # Must finish by due_time_in_minutes
                model.Add(end_var <= due_time_in_minutes)
            else:
                # Optional tasks can also not exceed the day or the due_time
                # We'll clamp at min(due_time_in_minutes, work_end).
                # If the user wants to let them start after due_time, you can relax here, but typically no.
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
            
            duration = max(0, end_min - start_min)
            if duration > 0:
                total_event_duration += duration
                # Create a fixed start variable
                start_var = model.NewIntVar(start_min, start_min, f"start_event_{evt_id}")
                # Use NewFixedSizeIntervalVar instead of NewFixedInterval
                fixed_iv = model.NewFixedSizeIntervalVar(start_var, duration, f"event_{evt_id}")
                event_intervals.append(fixed_iv)

        # Also block out time before work_start and after work_end
        # so tasks can't go there
        if work_start > 0:
            # Morning non-work hours
            morning_start = model.NewIntVar(0, 0, "morning_start")
            morning_iv = model.NewFixedSizeIntervalVar(morning_start, work_start, "non_work_morning")
            event_intervals.append(morning_iv)
        if work_end < horizon:
            # Evening non-work hours
            evening_start = model.NewIntVar(work_end, work_end, "evening_start")
            evening_iv = model.NewFixedSizeIntervalVar(evening_start, horizon - work_end, "non_work_evening")
            event_intervals.append(evening_iv)
        
        # ---- NO OVERLAP CONSTRAINT ----
        all_intervals = [tv["interval"] for tv in task_vars.values()] + event_intervals
        model.AddNoOverlap(all_intervals)
        
        # ---- MUST START AND END WITHIN WORK HOURS ----
        for tv in task_vars.values():
            model.Add(tv["start"] >= work_start)
            model.Add(tv["end"] <= work_end)
            if tv["is_mandatory"]:
                # No presence var, so it always must exist
                pass
            else:
                # If presence_var=0, we skip constraints except for the optional interval
                # i.e. these constraints apply OnlyEnforceIf(presence_var)
                model.Add(tv["start"] >= work_start).OnlyEnforceIf(tv["presence"])
                model.Add(tv["end"] <= work_end).OnlyEnforceIf(tv["presence"])
        
        # ---- OBJECTIVE CONSTRUCTION ----
        objective_terms = []
        # We'll define some linear expressions for total scheduled time, presence, etc.
        
        # 1) Sum up total scheduled minutes for tasks
        #    If mandatory => always add its duration
        #    If optional => add duration * presence
        scheduled_time_var = model.NewIntVar(0, work_end - work_start, "scheduled_time")
        partial_sum = []
        
        for task_id, tv in task_vars.items():
            if tv["is_mandatory"]:
                # Always included
                partial_sum.append(tv["duration"])
            else:
                # presence * duration => we introduce an IntVar for that
                dur_contrib = model.NewIntVar(0, tv["duration"], f"dur_contrib_{task_id}")
                # dur_contrib == tv["duration"] if presence=1, else 0
                model.Add(dur_contrib == tv["duration"]).OnlyEnforceIf(tv["presence"])
                model.Add(dur_contrib == 0).OnlyEnforceIf(tv["presence"].Not())
                partial_sum.append(dur_contrib)
        
        model.Add(scheduled_time_var == sum(partial_sum))
        
        # 2) Break time = (available_work_time - total_event_duration) - scheduled_time_var
        #    We'll treat that as a LinearExpr and add reward:  break_importance * break_time
        available_work_window = (work_end - work_start) - total_event_duration
        # We clamp at 0 to avoid negative break if total_event_duration > window
        # but typically that would lead to infeasibility if events fill the day.
        
        # break_time_expr = break_importance * (available_work_window - scheduled_time_var)
        # We'll do it in objective. Need to ensure the expression doesn't go negative.
        
        # 3) Reward or penalty for optional tasks
        #    For each optional task, if presence=1, add (score_val * 100 or so) to objective
        #    For mandatory tasks, you might want an incentive for finishing earlier, which we do next.
        
        # 4) Early completion bonus for mandatory tasks (or all tasks if desired).
        #    e.g., -end_time * priority * early_completion_bonus => the solver tries to keep end_time small
        #    We'll store them individually in objective_terms.
        
        # 5) Evening penalty: If a task ends near the end of the day, apply a negative factor.
        #    We can define a threshold, e.g., last hour of work window is "evening".
        
        # 6) Continuous work penalty: approximate by punishing if scheduled_time_var > max_continuous_work
        #    We'll define an integer var: excess_continuous. Then:
        #       excess_continuous >= scheduled_time_var - max_continuous_work
        #    objective -= continuous_work_penalty * excess_continuous
        
        # ---- Let's implement them step by step. ----
        
        # 2) Break time
        break_importance = self.ml_params['break_importance']
        break_time_expr = model.NewIntVar(0, available_work_window, "break_time_expr")
        # break_time_expr = max(0, available_work_window - scheduled_time_var)
        model.Add(break_time_expr == (available_work_window - scheduled_time_var))
        
        # We'll do objective += ( break_importance * break_time_expr ).
        # CP-SAT wants integer coefficients, so we might do an integer approximation or scale up.
        # We'll just multiply the expression directly:
        objective_terms.append(break_importance * break_time_expr)
        
        # 3) Optional tasks: add big reward for scheduling them
        for task_id, tv in task_vars.items():
            if not tv["is_mandatory"]:
                # presence -> add tv["score_val"] * (some factor)
                # We'll multiply by, say, 100 so that each point is quite valuable
                presence_score = tv["score_val"] * 100
                # presence_score * presence_var
                # In CP-SAT, we can do a linear expression: presence_var is 0 or 1,
                # so we can do presence_score * presence_var. Let's define an IntVar:
                # or we can do an "AddToObjective" with presence_var * presence_score.
                # We can do: objective_terms.append(presence_var * presence_score)
                # We'll do a WeightedSum approach:
                objective_terms.append(presence_score * tv["presence"])
        
        # 4) Early completion bonus
        early_completion_bonus = self.ml_params['early_completion_bonus']
        # We'll do for all tasks (esp. mandatory), but you can do only mandatory if you want
        for tv in task_vars.values():
            # e.g. -end_var * priority_value * early_completion_bonus, but we want a positive reward
            # so we do => objective += -(end_var * priority * bonus). We'll do that in a single line:
            # We want to maximize => so we do negative end time (less end time => bigger objective).
            # We'll multiply by priority as well if we want higher-priority tasks to finish earlier.
            # We'll call it: - end * priority * bonus
            objective_terms.append(
                -1 * tv["end"] * tv["priority_value"] * early_completion_bonus
            )
        
        # 5) Evening penalty
        # Let's define "evening start" as e.g. 60 minutes before work_end. You can adjust.
        evening_cutoff = work_end - 60
        evening_penalty = self.ml_params['evening_work_penalty']
        for task_id, tv in task_vars.items():
            is_evening = model.NewBoolVar(f"evening_{task_id}")
            model.Add(tv["end"] > evening_cutoff).OnlyEnforceIf(is_evening)
            model.Add(tv["end"] <= evening_cutoff).OnlyEnforceIf(is_evening.Not())
            # If it's in evening, negative effect => objective -= evening_penalty * 100
            objective_terms.append(-1 * is_evening * evening_penalty * 100)
        
        # 6) Continuous work penalty (approx. if total scheduled > max_continuous_work)
        max_cont_work = self.ml_params['max_continuous_work']
        cont_penalty = self.ml_params['continuous_work_penalty']
        
        # Define an IntVar for "excess_work"
        excess_work = model.NewIntVar(0, work_end - work_start, "excess_work")
        # excess_work >= scheduled_time_var - max_cont_work
        model.Add(excess_work >= scheduled_time_var - max_cont_work)
        model.Add(excess_work >= 0)
        
        # objective -= cont_penalty * excess_work * 10 (scaled)
        objective_terms.append(-1 * cont_penalty * 10 * excess_work)
        
        # Combine into final objective
        model.Maximize(sum(objective_terms))
        
        # ---- Solve the model ----
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30  # can adjust
        status = solver.Solve(model)
        
        # Find this section in the schedule_tasks method (around line 400):
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
                        'priority': self._value_to_priority(tv["priority_value"]),  # Convert to string
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
                            'priority': self._value_to_priority(tv["priority_value"]),  # Convert to string
                            'estimated_duration': tv["duration"],
                            'mandatory': False
                        })
            
            return {
                'status': 'success',
                'scheduled_tasks': scheduled_tasks
            }
        else:
            return {
                'status': 'error',
                'message': f'No feasible solution found. Solver status: {solver.StatusName(status)}'
            }
