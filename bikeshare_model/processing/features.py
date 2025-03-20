from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'dteday' in X.columns:
            X['dteday'] = pd.to_datetime(X['dteday'], errors='coerce')
            # Extract day names (Weekdays) from the dteday column
            X['weekday'] = X['dteday'].dt.strftime('%a')  # Extract short weekday name like 'Mon', 'Tue'
        else:
            raise ValueError("'dteday' column is missing from the dataset")
        
        print(X)
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        X[self.variables]=X[self.variables].fillna(self.fill_value)
        return X

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - To upper-bound if the value is higher than the upper-bound
        - To lower-bound if the value is lower than the lower-bound
    """

    def __init__(self, method='IQR', factor=1.5):
        """
        Initialize the OutlierHandler with the method to calculate bounds.

        :param method: Method to calculate bounds ('IQR' or 'z-score')
        :param factor: The factor to use for calculating the bounds (default is 1.5 for IQR)
        """
        self.method = method
        self.factor = factor
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        """
        Compute the lower and upper bounds for each numerical column.
        """
        # Select only numerical columns
        X_num = X.select_dtypes(include=['number'])

        if self.method == 'IQR':
            # Using IQR method to compute bounds
            Q1 = X_num.quantile(0.25)
            Q3 = X_num.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds = Q1 - self.factor * IQR
            self.upper_bounds = Q3 + self.factor * IQR
        elif self.method == 'z-score':
            # Using Z-score method to compute bounds
            mean = X_num.mean()
            std = X_num.std()
            self.lower_bounds = mean - self.factor * std
            self.upper_bounds = mean + self.factor * std
        else:
            raise ValueError("Unsupported method. Choose 'IQR' or 'z-score'.")

        return self

    def transform(self, X):
        """
        Replace the outliers with upper and lower bounds in numerical columns.
        """
        # Select only numerical columns
        X_copy = X.copy()
        X_num = X_copy.select_dtypes(include=['number'])

        # Iterate over all numerical columns
        for col in X_num.columns:
            # Apply upper and lower bounds to each numerical column
            X_copy[col] = X_copy[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col])

        return X_copy
    

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self):
        self.weekday_categories_ = None  # Store the categories for fitting

    def fit(self, X, y=None):
        # Extract unique weekday categories, handling NaNs
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[['weekday']])

        return self

    def transform(self, X):
        X = X.copy()  # Create a copy to avoid modifying the original
        encoded_weekday = self.encoder.transform(X[['weekday']])
        enc_wkday_features = self.encoder.get_feature_names_out(['weekday'])
        X = X.drop(columns=['weekday'])
        X[enc_wkday_features] = encoded_weekday
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors='ignore')  # Avoids KeyError
