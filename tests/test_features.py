
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer,Mapper
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest

# Import your WeekdayImputer class
# from your_module import WeekdayImputer

def test_weekday_imputer_transform():
    # Create test data
    data = {
        'dteday': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'weekday': [None, None, None, None, None]  # Initially all None
    }
    df = pd.DataFrame(data)
    
    # Expected output after transformation
    expected_data = {
        'dteday': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']),
        'weekday': ['Sat', 'Sun', 'Mon', 'Tue', 'Wed']  # Expected weekday values
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Initialize and apply transformer
    imputer = WeekdayImputer()
    result_df = imputer.fit_transform(df)
    
    # Assert the transformation was correct
    assert_frame_equal(result_df, expected_df)

def test_weekday_imputer(sample_input_data):
    # Given
    transformer = WeekdayImputer()
    
    # Create a test case where we know the expected result
    # Assuming sample_input_data has a row with index 3 where dteday is '2022-01-03' 
    # and weekday is currently None/NaN
    # assert pd.isna(sample_input_data[0].loc[335, 'weekday'])
    assert sample_input_data[0].loc[431, 'dteday'] == '2012-07-31'  # Monday
    
    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])
    
    # Then
    assert subject.loc[431, 'weekday'] == 'Tue'

def test_weekday_imputer_without_dteday_column():
    # Create test data without 'dteday' column
    data = {
        'other_column': [1, 2, 3],
        'weekday': [None, None, None]
    }
    df = pd.DataFrame(data)
    
    # Initialize transformer
    imputer = WeekdayImputer()
    
    # Test should raise ValueError
    with pytest.raises(ValueError, match="'dteday' column is missing from the dataset"):
        imputer.transform(df)

def test_mapper_initialization():
    # Given
    variables = 'season'
    mappings = {'winter': 4, 'spring': 1, 'summer': 2, 'fall': 3}
    
    # When
    mapper = Mapper(variables=variables, mappings=mappings)
    
    # Then
    assert mapper.variables == 'season'
    assert mapper.mappings == mappings
    
def test_mapper_initialization_error():
    # Given
    variables = ['season'] 
    mappings = {'winter': 4, 'spring': 1, 'summer': 2, 'fall': 3}
    
    # When/Then
    with pytest.raises(ValueError, match="variables should be a str"):
        Mapper(variables=variables, mappings=mappings)

def test_mapper_transform(sample_mapper_data, season_mappings):
    # Given
    mapper = Mapper(variables='season', mappings=season_mappings)
    
    # When
    transformed_df = mapper.fit_transform(sample_mapper_data)
    
    # Then
    # Check transformed values match the mapping
    expected_season_values = [1, 2, 3, 4]  # [spring, summer, fall, winter]
    assert transformed_df['season'].tolist() == expected_season_values
    
    # Check the type is int
    assert pd.api.types.is_integer_dtype(transformed_df['season'])