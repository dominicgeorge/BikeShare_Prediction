import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split
import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_raw_dataset


# @pytest.fixture is a decorator in the pytest testing framework used to define a fixture. 
# Fixtures are a way to provide setup and teardown functionality for your test functions or test classes. 
# They allow you to create reusable components that can be shared across multiple test functions, 
# improving code modularity and maintainability.

@pytest.fixture
def sample_input_data():
    data = load_raw_dataset(file_name=config.app_config_.training_data_file)

    X = data.drop(config.model_config_.target, axis=1)       # predictors
    y = data[config.model_config_.target]                    # target

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,  # target
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state,
    )

    return X_test, y_test

# Define fixtures for testing
@pytest.fixture
def sample_mapper_data():
    """Create sample data for testing the Mapper."""
    data = {
        'season': ['spring', 'summer', 'fall', 'winter'],
        'other_column': [1, 2, 3, 4]
    }
    return pd.DataFrame(data)

@pytest.fixture
def season_mappings():
    """Create sample mappings for seasons."""
    return {
        'winter': 4,
        'spring': 1,
        'summer': 2,
        'fall': 3
    }