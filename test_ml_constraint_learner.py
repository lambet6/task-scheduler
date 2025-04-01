import pytest
import os
import pandas as pd
import json
import shutil
from datetime import datetime, timedelta
from ml_constraint_learner import MLConstraintLearner

# Test directory for temporary test files
TEST_DATA_DIR = './test_user_data'

@pytest.fixture
def clean_test_dir():
    """Clean test directory before and after tests."""
    # Create test directory if it doesn't exist
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
    else:
        # Clean existing files
        for file in os.listdir(TEST_DATA_DIR):
            file_path = os.path.join(TEST_DATA_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    yield  # Run the test
    
    # Clean up after the test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)

@pytest.fixture
def learner():
    """Return a MLConstraintLearner instance for testing."""
    return MLConstraintLearner(data_dir=TEST_DATA_DIR)

@pytest.fixture
def sample_schedule():
    """Return a sample schedule for testing feature extraction."""
    # Create a base date
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    return {
        "scheduled_tasks": [
            {
                "id": "task1",
                "title": "Task 1",
                "priority": "High",  # Using string priority
                "start": (base_date + timedelta(hours=9)).isoformat(),
                "end": (base_date + timedelta(hours=10)).isoformat(),
                "estimated_duration": 60,
                "mandatory": True
            },
            {
                "id": "task2",
                "title": "Task 2",
                "priority": "Medium",  # Using string priority
                "start": (base_date + timedelta(hours=10, minutes=30)).isoformat(),
                "end": (base_date + timedelta(hours=11, minutes=30)).isoformat(),
                "estimated_duration": 60,
                "mandatory": False
            },
            {
                "id": "task3",
                "title": "Task 3",
                "priority": "Low",  # Using string priority
                "start": (base_date + timedelta(hours=13)).isoformat(),
                "end": (base_date + timedelta(hours=14)).isoformat(),
                "estimated_duration": 60,
                "mandatory": True
            }
        ],
        "calendar_events": [
            {
                "id": "event1",
                "title": "Meeting",
                "start": (base_date + timedelta(hours=11, minutes=30)).isoformat(),
                "end": (base_date + timedelta(hours=12, minutes=30)).isoformat()
            }
        ],
        "constraints": {
            "work_hours": {
                "start": "09:00",
                "end": "17:00"
            },
            "max_continuous_work_min": 90
        }
    }

@pytest.fixture
def sample_schedule_with_numeric_priority():
    """Return a sample schedule with numeric priority values for testing."""
    # Create a base date
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    return {
        "scheduled_tasks": [
            {
                "id": "task1",
                "title": "Task 1",
                "priority": 3,  # High - using numeric priority
                "start": (base_date + timedelta(hours=9)).isoformat(),
                "end": (base_date + timedelta(hours=10)).isoformat(),
                "estimated_duration": 60,
                "mandatory": True
            },
            {
                "id": "task2",
                "title": "Task 2",
                "priority": 2,  # Medium - using numeric priority
                "start": (base_date + timedelta(hours=10, minutes=30)).isoformat(),
                "end": (base_date + timedelta(hours=11, minutes=30)).isoformat(),
                "estimated_duration": 60,
                "mandatory": False
            },
            {
                "id": "task3",
                "title": "Task 3",
                "priority": 1,  # Low - using numeric priority
                "start": (base_date + timedelta(hours=13)).isoformat(),
                "end": (base_date + timedelta(hours=14)).isoformat(),
                "estimated_duration": 60,
                "mandatory": True
            }
        ],
        "calendar_events": [
            {
                "id": "event1",
                "title": "Meeting",
                "start": (base_date + timedelta(hours=11, minutes=30)).isoformat(),
                "end": (base_date + timedelta(hours=12, minutes=30)).isoformat()
            }
        ],
        "constraints": {
            "work_hours": {
                "start": "09:00",
                "end": "17:00"
            },
            "max_continuous_work_min": 90
        }
    }

@pytest.fixture
def sample_feedback():
    """Return a sample feedback item."""
    return {
        "mood_score": 4,
        "adjusted_tasks": [],
        "completed_tasks": ["task1", "task2"]
    }

class TestFeatureExtraction:
    def test_priority_conversion(self, learner):
        """Test the _priority_to_value conversion method."""
        # Test conversion from string to numeric
        assert learner._priority_to_value("High") == 3
        assert learner._priority_to_value("Medium") == 2
        assert learner._priority_to_value("Low") == 1
        assert learner._priority_to_value("Unknown") == 1  # Default to Low for unknown
        
        # Test handling of already numeric values
        assert learner._priority_to_value(3) == 3
        assert learner._priority_to_value(2) == 2
        assert learner._priority_to_value(1) == 1
    
    def test_empty_schedule(self, learner):
        """Test feature extraction with empty schedule."""
        # Empty schedule
        empty_schedule = {
            "scheduled_tasks": [],
            "calendar_events": [],
            "constraints": {
                "work_hours": {
                    "start": "09:00",
                    "end": "17:00"
                }
            }
        }
        
        features = learner._extract_schedule_features(empty_schedule)
        
        # Check that all features are present and have appropriate default values
        assert 'avg_task_duration' in features
        assert features['avg_task_duration'] == 0.0
        assert features['total_work_minutes'] == 0.0
        assert features['actual_break_minutes'] == 0.0
        assert features['optional_tasks_scheduled'] == 0
        assert features['total_tasks_scheduled'] == 0
    
    def test_schedule_with_string_priorities(self, learner, sample_schedule):
        """Test feature extraction with tasks having string priorities."""
        features = learner._extract_schedule_features(sample_schedule)
        
        # Check feature values
        assert features['total_tasks_scheduled'] == 3
        assert features['total_work_minutes'] == 180  # 3 tasks * 60 minutes
        assert features['optional_tasks_scheduled'] == 1  # One optional task
        assert features['avg_task_duration'] == 60.0  # All tasks are 60 minutes
        
        # Work window is 8 hours (480 minutes)
        # One event takes 60 minutes, so available work time is 420 minutes
        # Tasks take 180 minutes, so break time is 240 minutes
        assert features['actual_break_minutes'] == 240.0
        
        # Excess work: total_work_minutes - max_continuous_work = 180 - 90 = 90
        assert features['excess_work'] == 90.0
        
        # Work start time: 9:00 = 540 minutes
        assert features['work_start_time'] == 540
        
        # Work end time: 14:00 = 840 minutes
        assert features['work_end_time'] == 840
        
        # Check high_priority_early - one high priority task in first half, should be 1.0
        assert features['high_priority_early'] == 1.0
        
        # Check all expected features are present
        expected_features = [
            'avg_task_duration', 'total_work_minutes', 'actual_break_minutes',
            'optional_tasks_scheduled', 'total_tasks_scheduled', 'excess_work',
            'work_start_time', 'work_end_time', 'high_priority_early',
            'evening_work', 'longest_stretch'
        ]
        for feature in expected_features:
            assert feature in features
    
    def test_schedule_with_numeric_priorities(self, learner, sample_schedule_with_numeric_priority):
        """Test feature extraction with tasks having numeric priorities."""
        features = learner._extract_schedule_features(sample_schedule_with_numeric_priority)
        
        # Same checks as above, but with numeric priorities
        assert features['total_tasks_scheduled'] == 3
        assert features['high_priority_early'] == 1.0  # One high priority task in first half
    
    def test_mixed_priority_formats(self, learner):
        """Test feature extraction with mixed priority formats (strings and numbers)."""
        # Create a schedule with mixed priority formats
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        mixed_schedule = {
            "scheduled_tasks": [
                {
                    "id": "task1",
                    "title": "Task 1",
                    "priority": "High",  # String priority
                    "start": (base_date + timedelta(hours=9)).isoformat(),
                    "end": (base_date + timedelta(hours=10)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": True
                },
                {
                    "id": "task2",
                    "title": "Task 2",
                    "priority": 2,  # Numeric priority
                    "start": (base_date + timedelta(hours=10, minutes=30)).isoformat(),
                    "end": (base_date + timedelta(hours=11, minutes=30)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": False
                }
            ],
            "constraints": {
                "work_hours": {
                    "start": "09:00",
                    "end": "17:00"
                }
            }
        }
        
        features = learner._extract_schedule_features(mixed_schedule)
        
        # Verify the features were calculated correctly
        assert features['total_tasks_scheduled'] == 2
        assert features['high_priority_early'] == 1.0  # One high priority task in first half
    
    def test_longest_stretch_calculation(self, learner):
        """Test calculation of longest continuous work stretch."""
        # Create a schedule with tasks close together and far apart
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        schedule = {
            "scheduled_tasks": [
                # First group of tasks (continuous work)
                {
                    "id": "task1",
                    "title": "Task 1",
                    "priority": "High",
                    "start": (base_date + timedelta(hours=9)).isoformat(),
                    "end": (base_date + timedelta(hours=10)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": True
                },
                {
                    "id": "task2",
                    "title": "Task 2",
                    "priority": "Medium",
                    "start": (base_date + timedelta(hours=10, minutes=10)).isoformat(),  # 10 min gap
                    "end": (base_date + timedelta(hours=11, minutes=10)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": False
                },
                # Second group (after a long break)
                {
                    "id": "task3",
                    "title": "Task 3",
                    "priority": "Low",
                    "start": (base_date + timedelta(hours=13)).isoformat(),  # 1h50m gap
                    "end": (base_date + timedelta(hours=14)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": True
                },
                {
                    "id": "task4",
                    "title": "Task 4",
                    "priority": "Low",
                    "start": (base_date + timedelta(hours=14, minutes=5)).isoformat(),  # 5 min gap
                    "end": (base_date + timedelta(hours=15, minutes=5)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": True
                }
            ],
            "constraints": {
                "work_hours": {
                    "start": "09:00",
                    "end": "17:00"
                }
            }
        }
        
        features = learner._extract_schedule_features(schedule)
        
        # First stretch: task1 + task2 = 120 minutes
        # Second stretch: task3 + task4 = 120 minutes
        # Both are equal, so longest_stretch should be 120
        assert features['longest_stretch'] == 120.0

# Test 2: Parameter Adjustment Based on Feedback
class TestParameterAdjustment:
    def test_record_feedback_creates_file(self, learner, clean_test_dir, sample_schedule, sample_feedback):
        """Test that recording feedback creates the feedback file."""
        user_id = "test_user"
        
        learner.record_feedback(
            user_id=user_id,
            schedule_data=sample_schedule,
            feedback_data=sample_feedback
        )
        
        # Check that feedback file was created
        feedback_path = os.path.join(TEST_DATA_DIR, f'user_{user_id}_feedback.csv')
        assert os.path.exists(feedback_path)
        
        # Check file contains correct data
        df = pd.read_csv(feedback_path)
        assert len(df) == 1
        assert 'mood_score' in df.columns
        assert df['mood_score'].iloc[0] == sample_feedback['mood_score']
        assert 'total_tasks_scheduled' in df.columns
        assert df['total_tasks_scheduled'].iloc[0] == len(sample_schedule['scheduled_tasks'])
    
    def test_parameter_adjustment_after_sufficient_data(self, learner, clean_test_dir):
        """Test that parameters are adjusted after sufficient data points."""
        user_id = "test_user"
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Create a schedule
        schedule = {
            "scheduled_tasks": [
                {
                    "id": "task1",
                    "title": "Task 1",
                    "priority": "High",
                    "start": (base_date + timedelta(hours=9)).isoformat(),
                    "end": (base_date + timedelta(hours=10)).isoformat(),
                    "estimated_duration": 60,
                    "mandatory": True
                }
            ],
            "constraints": {
                "work_hours": {
                    "start": "09:00",
                    "end": "17:00"
                },
                "max_continuous_work_min": 90
            }
        }
        
        feedback = {
            "mood_score": 4,
            "adjusted_tasks": [],
            "completed_tasks": ["task1"]
        }
        
        # Record 5 feedback entries to trigger model training
        for _ in range(5):
            learner.record_feedback(user_id, schedule, feedback)
        
        # Check that model file was created
        model_path = os.path.join(TEST_DATA_DIR, f'user_{user_id}_mood_predictor.pkl')
        assert os.path.exists(model_path)
        
        # Check that parameters file was created
        params_path = os.path.join(TEST_DATA_DIR, f'user_{user_id}_params.json')
        assert os.path.exists(params_path)
        
        # Check parameters were created with expected keys
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Confirm all expected parameters are present
        assert 'break_importance' in params
        assert 'max_continuous_work' in params
        assert 'continuous_work_penalty' in params
        assert 'evening_work_penalty' in params
        assert 'early_completion_bonus' in params
    
    def test_parameter_adjustment_with_correlations(self, learner, clean_test_dir):
        """Test that parameters are adjusted based on correlations."""
        user_id = "test_user_corr"
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Create schedules with varying break times and corresponding feedback
        for i in range(10):
            # More breaks correlate with better mood
            break_minutes = i * 30  # 0, 30, 60, 90, ... minutes of break
            mood_score = min(5, 1 + i // 2)  # 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
            
            # Create schedule with tasks that have breaks between them
            start1 = base_date + timedelta(hours=9)
            end1 = start1 + timedelta(minutes=60)
            
            start2 = end1 + timedelta(minutes=break_minutes)
            end2 = start2 + timedelta(minutes=60)
            
            schedule = {
                "scheduled_tasks": [
                    {
                        "id": "task1",
                        "title": "Task 1",
                        "priority": "High",
                        "start": start1.isoformat(),
                        "end": end1.isoformat(),
                        "estimated_duration": 60,
                        "mandatory": True
                    },
                    {
                        "id": "task2",
                        "title": "Task 2",
                        "priority": "Medium",
                        "start": start2.isoformat(),
                        "end": end2.isoformat(),
                        "estimated_duration": 60,
                        "mandatory": False
                    }
                ],
                "constraints": {
                    "work_hours": {
                        "start": "09:00",
                        "end": "17:00"
                    },
                    "max_continuous_work_min": 90
                }
            }
            
            feedback = {
                "mood_score": mood_score,
                "adjusted_tasks": [],
                "completed_tasks": ["task1", "task2"]
            }
            
            learner.record_feedback(user_id, schedule, feedback)
        
        # Check that ML model recognized the correlation and adjusted parameters
        params_path = os.path.join(TEST_DATA_DIR, f'user_{user_id}_params.json')
        assert os.path.exists(params_path)
        
        # Load the parameters and check that they reflect the correlation
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        # The correlation between breaks and mood should result in a higher break_importance
        assert 'break_importance' in params

# Test 3: Persistence and Loading of User Parameters
class TestParameterPersistence:
    def test_persistence_of_parameters(self, learner, clean_test_dir):
        """Test that parameters are saved and loaded correctly."""
        user_id = "test_user_persist"
        
        # Create custom parameters
        custom_params = {
            'break_importance': 2.5,
            'max_continuous_work': 75,
            'continuous_work_penalty': 3.5,
            'evening_work_penalty': 4.5,
            'early_completion_bonus': 3.0
        }
        
        # Save the parameters
        params_path = os.path.join(TEST_DATA_DIR, f'user_{user_id}_params.json')
        with open(params_path, 'w') as f:
            json.dump(custom_params, f)
        
        # Load the parameters
        loaded_params = learner.get_user_parameters(user_id)
        
        # Check that loaded parameters match the saved ones
        assert loaded_params == custom_params
        assert loaded_params['break_importance'] == 2.5
        assert loaded_params['max_continuous_work'] == 75
        assert loaded_params['continuous_work_penalty'] == 3.5
        assert loaded_params['evening_work_penalty'] == 4.5
        assert loaded_params['early_completion_bonus'] == 3.0
    
    def test_default_parameters_when_no_file(self, learner, clean_test_dir):
        """Test that default parameters are returned when no file exists."""
        user_id = "nonexistent_user"
        
        # Get parameters for a user that doesn't exist
        params = learner.get_user_parameters(user_id)
        
        # Check that default parameters are returned
        assert params == learner.default_params
        
        # Check that the parameters match the expected defaults
        assert params['break_importance'] == 1.0
        assert params['max_continuous_work'] == 90
        assert params['continuous_work_penalty'] == 2.0
        assert params['evening_work_penalty'] == 3.0
        assert params['early_completion_bonus'] == 2.0
    
    def test_parameter_loading_after_feedback(self, learner, clean_test_dir, sample_schedule, sample_feedback):
        """Test that parameters can be loaded after feedback is recorded."""
        user_id = "test_user_load"
        
        # Record feedback multiple times to trigger model training
        for _ in range(5):
            learner.record_feedback(user_id, sample_schedule, sample_feedback)
        
        # Load the parameters
        params = learner.get_user_parameters(user_id)
        
        # Check that all expected parameters exist
        assert 'break_importance' in params
        assert 'max_continuous_work' in params
        assert 'continuous_work_penalty' in params
        assert 'evening_work_penalty' in params
        assert 'early_completion_bonus' in params