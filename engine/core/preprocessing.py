from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BasePreprocessor(ABC):
    """Base class for all preprocessors."""
    
    @abstractmethod
    def fit(self, data: Any) -> 'BasePreprocessor':
        """Fit preprocessor to data."""
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform data."""
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """Fit and transform data."""
        return self.fit(data).transform(data)

class NumericPreprocessor(BasePreprocessor):
    """Preprocessor for numeric data."""
    
    def __init__(self, 
                 scaling: str = 'standard',
                 handle_missing: str = 'mean'):
        self.scaling = scaling
        self.handle_missing = handle_missing
        self.scaler: Optional[BaseEstimator] = None
        self.fill_values: Dict[str, float] = {}
        
    def fit(self, data: pd.DataFrame) -> 'NumericPreprocessor':
        # Handle missing values
        if self.handle_missing == 'mean':
            self.fill_values = data.mean().to_dict()
        elif self.handle_missing == 'median':
            self.fill_values = data.median().to_dict()
        
        # Initialize scaler
        if self.scaling == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif self.scaling == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            
        if self.scaler:
            self.scaler.fit(data.fillna(self.fill_values))
            
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_filled = data.fillna(self.fill_values)
        if self.scaler:
            return pd.DataFrame(
                self.scaler.transform(data_filled),
                columns=data.columns,
                index=data.index
            )
        return data_filled

class PreprocessingPipeline:
    """Pipeline for data preprocessing."""
    
    def __init__(self, steps: List[Dict[str, Any]]):
        self.steps = []
        for step in steps:
            preprocessor_class = step.pop('preprocessor')
            columns = step.pop('columns', None)
            self.steps.append({
                'preprocessor': preprocessor_class(**step),
                'columns': columns
            })
            
    def fit(self, data: pd.DataFrame) -> 'PreprocessingPipeline':
        for step in self.steps:
            cols = step['columns'] or data.columns
            step['preprocessor'].fit(data[cols])
        return self
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for step in self.steps:
            cols = step['columns'] or data.columns
            result[cols] = step['preprocessor'].transform(result[cols])
        return result
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data) 