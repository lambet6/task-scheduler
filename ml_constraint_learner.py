import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
from datetime import datetime

class MLConstraintLearner:
    def __init__(self, data_dir='./user_data'):
        """
        Initialize the ML model for learning constraint weights.
        
        Args:
            data_dir: Directory to store user data and models
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Default constraint parameters (will be updated through learning)
        self.default_params = {
            'break_importance': 1.0,
            'max_continuous_work': 90,  # 90 minutes default
            'continuous_work_penalty': 2.0,
            'evening_work_penalty': 3.0,
            'early_completion_bonus': 2.0
        }
        
        # Initialize any fitted models (if you want separate per-parameter models, not used below)
        self.models = {
            'mood_predictor': None
        }
    
    def _priority_to_value(self, priority):
        """Convert string priority to numeric scale."""
        if isinstance(priority, int):
            return priority
        priority_map = {
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        return priority_map.get(priority, 1)
    
    def _get_user_data_path(self, user_id):
        """Get path to user's data file."""
        return os.path.join(self.data_dir, f'user_{user_id}_feedback.csv')
    
    def _get_user_model_path(self, user_id, model_name):
        """Get path to user's model file for a particular model name."""
        return os.path.join(self.data_dir, f'user_{user_id}_{model_name}.pkl')
    
    def _get_user_params_path(self, user_id):
        """Get path to user's current parameters file."""
        return os.path.join(self.data_dir, f'user_{user_id}_params.json')
    
    def record_feedback(self, user_id, schedule_data, feedback_data):
        """
        Record user feedback about a schedule to use for learning.
        
        Args:
            user_id: Unique identifier for the user
            schedule_data: Dictionary with schedule details including:
                - scheduled_tasks: List of tasks with {start, end, mandatory, priority, ...}
                - calendar_events: (optional) List of user events used in scheduling
                - constraints: includes work_hours and possibly max_continuous_work
            feedback_data: Dictionary with user feedback including:
                - mood_score: Overall mood rating (1-5)    
                - adjusted_tasks: Tasks the user moved or adjusted
                - completed_tasks: Tasks the user completed
        """
        # Extract features from the schedule
        features = self._extract_schedule_features(schedule_data)
        
        # Extract target values from feedback
        # Accessing as attributes instead of using .get()
        tasks_scheduled_count = max(1, len(schedule_data.get('scheduled_tasks', [])))
        mood_score = feedback_data.mood_score  # Changed from feedback_data.get('mood_score', 3)
        
        targets = {
            'mood_score': mood_score,
            'task_adjustments': len(feedback_data.adjusted_tasks),  # Changed from feedback_data.get('adjusted_tasks', [])
            'completion_rate': len(feedback_data.completed_tasks) / tasks_scheduled_count  # Changed from feedback_data.get('completed_tasks', [])
        }
        
        # Combine features and targets
        row_data = {**features, **targets, 'timestamp': datetime.now().isoformat()}
        
        # Load existing data or create new dataframe
        data_path = self._get_user_data_path(user_id)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            new_row = pd.DataFrame([row_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([row_data])
        
        # Save updated data
        df.to_csv(data_path, index=False)
        
        # Update models if we have enough data (at least 5 data points)
        if len(df) >= 5:
            self._update_models(user_id, df)
    
    def _extract_schedule_features(self, schedule_data):
        """
        Extract features from a schedule that might correlate with user satisfaction.
        
        Returns a dict of numeric features that get stored in the feedback CSV.
        """
        tasks = schedule_data.get('scheduled_tasks', [])
        constraints = schedule_data.get('constraints', {})
        calendar_events = schedule_data.get('calendar_events', [])
        
        # Prepare some default feature values
        features = {
            'avg_task_duration': 0.0,
            'total_work_minutes': 0.0,
            'actual_break_minutes': 0.0,
            'optional_tasks_scheduled': 0,
            'total_tasks_scheduled': len(tasks),
            'excess_work': 0.0,
            'work_start_time': 0,
            'work_end_time': 0,
            'high_priority_early': 0,
            'evening_work': 0,
            'longest_stretch': 0,
        }
        
        # If no tasks were scheduled, return all zeros
        if not tasks:
            return features
        
        # Parse user work hours
        wh = constraints.get('work_hours', {})
        start_str = wh.get('start', '09:00')
        end_str = wh.get('end', '17:00')
        
        # Convert to minutes
        def _time_to_minutes(tstr):
            hh, mm = map(int, tstr.split(':'))
            return hh * 60 + mm
        
        work_start_min = _time_to_minutes(start_str)
        work_end_min   = _time_to_minutes(end_str)
        workable_window = max(0, work_end_min - work_start_min)
        
        # Sum up total event durations
        total_event_duration = 0
        for evt in calendar_events:
            try:
                start_dt = datetime.fromisoformat(evt['start'].replace('Z', '+00:00'))
                end_dt   = datetime.fromisoformat(evt['end'].replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds() / 60
                if duration > 0:
                    total_event_duration += duration
            except:
                pass
        
        # So the maximum workable minutes for tasks:
        workable_minutes = max(0, workable_window - total_event_duration)
        
        # Build a list of tasks with start/end times
        task_times = []
        for tk in tasks:
            start_dt = datetime.fromisoformat(tk['start'].replace('Z', '+00:00'))
            end_dt   = datetime.fromisoformat(tk['end'].replace('Z', '+00:00'))
            priority = self._priority_to_value(tk.get('priority', 'Medium'))
            mandatory_flag = tk.get('mandatory', True)
            
            duration = (end_dt - start_dt).total_seconds() / 60
            task_times.append({
                'start': start_dt,
                'end': end_dt,
                'priority': priority,
                'mandatory': mandatory_flag,
                'duration': duration
            })
        
        task_times.sort(key=lambda x: x['start'])
        
        # total work minutes
        total_work_minutes = sum(x['duration'] for x in task_times)
        features['total_work_minutes'] = total_work_minutes
        
        # optional tasks scheduled
        opt_scheduled = [x for x in task_times if x['mandatory'] == False]
        features['optional_tasks_scheduled'] = len(opt_scheduled)
        
        # average task duration
        features['avg_task_duration'] = (total_work_minutes / len(task_times)) if task_times else 0
        
        # actual break minutes
        # = workable_minutes - total_work_minutes (clamp at 0)
        actual_break = max(0, workable_minutes - total_work_minutes)
        features['actual_break_minutes'] = actual_break
        
        # If the constraints have 'max_continuous_work', measure how
        # much we exceed that (approx).
        max_cont_work = constraints.get('max_continuous_work_min', 90)
        # If your ML param is stored in the user param file, you might not see it here, but let's just do best effort:
        excess = max(0, total_work_minutes - max_cont_work)
        features['excess_work'] = excess
        
        # measure earliest start and latest end among tasks
        earliest_start = task_times[0]['start']
        latest_end = task_times[-1]['end']
        
        features['work_start_time'] = earliest_start.hour * 60 + earliest_start.minute
        features['work_end_time']   = latest_end.hour * 60 + latest_end.minute
        
        # measure if high priority tasks are early
        # (like the old approach)
        high_priority_count = sum(1 for t in task_times if t['priority'] >= 3)
        if high_priority_count:
            half_index = len(task_times) // 2
            early_high = 0
            # count how many high priority tasks appear in first half of the day
            # by index
            for i, t in enumerate(task_times):
                if t['priority'] >= 3 and i < half_index:
                    early_high += 1
            features['high_priority_early'] = early_high / high_priority_count
        else:
            features['high_priority_early'] = 0
        
        # measure "evening work" as fraction of tasks ending after (e.g.) 17:00
        # or you could do after "work_end_min - 60"
        # We'll keep your existing approach (5 PM)
        evening_threshold = 17 * 60
        evening_work_count = sum(1 for t in task_times if (t['end'].hour * 60 + t['end'].minute) > evening_threshold)
        features['evening_work'] = evening_work_count / len(task_times)
        
        # measure longest continuous stretch
        # e.g. tasks with <15-min gap are considered continuous
        longest_stretch = 0
        current_stretch = 0
        last_end = None
        
        for i, t in enumerate(task_times):
            if i == 0:
                current_stretch = t['duration']
                last_end = t['end']
            else:
                gap = (t['start'] - last_end).total_seconds() / 60
                if gap < 15:
                    current_stretch += t['duration']
                else:
                    longest_stretch = max(longest_stretch, current_stretch)
                    current_stretch = t['duration']
                last_end = t['end']
        
        longest_stretch = max(longest_stretch, current_stretch)
        features['longest_stretch'] = longest_stretch
        
        return features
    
    def _update_models(self, user_id, df):
        """
        Update ML models based on collected data.
        
        Args:
            user_id: User identifier
            df: DataFrame with features and targets
        """
        # We define a set of features that likely influence mood
        # You can add or remove as you see fit, matching the columns we collect.
        features = [
            'avg_task_duration',
            'total_work_minutes',
            'actual_break_minutes',
            'optional_tasks_scheduled',
            'excess_work',
            'work_start_time',
            'work_end_time',
            'high_priority_early',
            'evening_work',
            'longest_stretch'
        ]
        
        # Our main target is mood score
        target_col = 'mood_score'
        
        # Make sure these columns exist in df
        for col in features:
            if col not in df.columns:
                df[col] = 0  # fill missing if older rows lacked it
        
        if target_col not in df.columns:
            # No mood data, skip
            return
        
        X = df[features]
        y = df[target_col]
        
        # We'll use a simple random forest for interpretability
        model = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Save the model
        model_path = self._get_user_model_path(user_id, 'mood_predictor')
        joblib.dump(model, model_path)
        
        # Interpret the model to update parameters
        feature_importances = model.feature_importances_
        importances = dict(zip(features, feature_importances))
        
        params = self.default_params.copy()
        
        # Example: if 'actual_break_minutes' is important, boost break_importance
        if importances.get('actual_break_minutes', 0) > 0.1:
            corr_break = df[['actual_break_minutes', target_col]].corr().iloc[0,1]
            if corr_break > 0.2:
                # more break => better mood
                params['break_importance'] = 1.5
            else:
                # no or negative correlation => keep it or reduce
                params['break_importance'] = 1.0
        
        # Example: if 'excess_work' is quite important, we adjust max_continuous_work or penalty
        if importances.get('excess_work', 0) > 0.1:
            corr_excess = df[['excess_work', target_col]].corr().iloc[0,1]
            if corr_excess < -0.2:
                # more excess => significantly worse mood => raise penalty
                params['continuous_work_penalty'] = 3.0
            else:
                # not strongly negative => keep or reduce
                params['continuous_work_penalty'] = 2.0
        
        # If evening_work is important
        if importances.get('evening_work', 0) > 0.1:
            corr_evening = df[['evening_work', target_col]].corr().iloc[0,1]
            if corr_evening < -0.2:
                # tasks in evening => user mood is worse
                params['evening_work_penalty'] = 4.0
            else:
                params['evening_work_penalty'] = 2.0
        
        # If high_priority_early is important
        if importances.get('high_priority_early', 0) > 0.1:
            corr_hp_early = df[['high_priority_early', target_col]].corr().iloc[0,1]
            if corr_hp_early > 0.2:
                params['early_completion_bonus'] = 3.0
            else:
                params['early_completion_bonus'] = 2.0
        
        # If break_percentage is still in your older data, handle it similarly if you like
        if 'break_percentage' in df.columns and importances.get('break_percentage', 0) > 0.1:
            corr_bp = df[['break_percentage', target_col]].corr().iloc[0,1]
            if corr_bp > 0.2:
                params['break_importance'] += 0.5
        
        # (Optional) You can do more advanced logic to modify max_continuous_work if you want
        # For instance, if 'longest_stretch' or 'excess_work' strongly correlate, adjust that param.
        
        # Save updated parameters
        params_path = self._get_user_params_path(user_id)
        with open(params_path, 'w') as f:
            json.dump(params, f)
    
    def get_user_parameters(self, user_id):
        """
        Get learned parameters for a user, or defaults if not enough data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of parameter values
        """
        params_path = self._get_user_params_path(user_id)
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                return json.load(f)
        return self.default_params.copy()
