import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import ColumnDropper

bikeshare_pipe=Pipeline([
    ('weekdayimputer', WeekdayImputer()),
    
    ('weathersitimputer',WeathersitImputer(config.model_config_.weathersit_var)),
    ('drop_columns', ColumnDropper(columns_to_drop=config.model_config_.unused_fields)),
    ##==========Mapper======##
    ('year_mapper',Mapper(config.model_config_.year_var, config.model_config_.year_mappings)),
    ('month_mapper',Mapper(config.model_config_.month_var, config.model_config_.month_mapping)),
    ('season_mapper',Mapper(config.model_config_.season_var, config.model_config_.season_mappings)),
    ('weather_mapper',Mapper(config.model_config_.weathersit_var, config.model_config_.weathersit_mappings)),
    ('holiday_mapper',Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mapping)),
    ('workingday_mapper',Mapper(config.model_config_.workingday_var, config.model_config_.workingday_mapping)),
    ('hour_mapper',Mapper(config.model_config_.hour_var, config.model_config_.hour_mapping)),
    ('outlierhandler',OutlierHandler(method='IQR', factor=1.5)),
    ('weekdayoneHotencoder', WeekdayOneHotEncoder()),
     
    
  

   
   
     
    # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config_.n_estimators, 
                                         random_state=config.model_config_.random_state))
    
          
     ])
