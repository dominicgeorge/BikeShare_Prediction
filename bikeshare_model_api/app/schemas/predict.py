from typing import Any, List, Optional

from pydantic import BaseModel, field_validator
from datetime import datetime
from bikeshare_model import config


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]

class DataInputSchema(BaseModel):
    dteday: Optional[str]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str] 
    weekday: Optional[str] #*
    workingday: Optional[str] #*
    weathersit: Optional[str] #*
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    casual: Optional[int]
    registered: Optional[int]
    

    @field_validator("dteday", mode="before")
    def validate_dteday_format(cls, value):
        if value is None:
            return value
        try:
            # Ensure the date is in the '%Y-%m-%d' format
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"dteday must be in the format '%Y-%m-%d' eg: '2013-11-05'. Got: {value}")
        return value

    # Validate the season field
    @field_validator("season", mode="before")
    def validate_season(cls, value):
        if value is None:
            return value
        if value not in config.model_config_.season_mappings:
            raise ValueError(f"Invalid season: '{value}'. Allowed values are {list(config.model_config_.season_mappings.keys())}.")
        return value
    
    
    # Validate the hr field
    @field_validator("hr", mode="before")
    def validate_hr(cls, value):
        if value is None:
            return value
        if value not in config.model_config_.hour_mapping:
            raise ValueError(f"Invalid hr: '{value}'. Allowed values are {list(config.model_config_.hour_mapping.keys())}.")
        return value
    
    # Validate the holiday field
    @field_validator("holiday", mode="before")
    def validate_holiday(cls, value):
        if value is None:
            return value
        if value not in config.model_config_.holiday_mapping:
            raise ValueError(f"Invalid holiday: '{value}'. Allowed values are {list(config.model_config_.holiday_mapping.keys())}.")
        return value        

    # # Validate the weekday field
    # @field_validator("weekday", mode="before")
    # def validate_weekday(cls, value):
    #     if value is None:
    #         return value
    #     if value not in config.model_config_.hour_mapping:
    #         raise ValueError(f"Invalid weekday: '{value}'. Allowed values are {list(config.model_config_.hour_mapping.keys())}.")
    #     return value

    # Validate the workingday field
    @field_validator("workingday", mode="before")
    def validate_workingday(cls, value):
        if value is None:
            return value
        if value not in config.model_config_.workingday_mapping:
            raise ValueError(f"Invalid workingday: '{value}'. Allowed values are {list(config.model_config_.workingday_mapping.keys())}.")
        return value
    
    # Validate the weathersit field
    @field_validator("weathersit", mode="before")
    def validate_weathersit(cls, value):
        if value is None:
            return value
        if value not in config.model_config_.weathersit_mappings:
            raise ValueError(f"Invalid weathersit: '{value}'. Allowed values are {list(config.model_config_.weathersit_mappings.keys())}.")
        return value



class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

