import pytest
import numpy as np
import pandas as pd
from engine.core.preprocessing import NumericPreprocessor, PreprocessingPipeline

def test_numeric_preprocessor():
    """Test numeric data preprocessor."""
    # Create sample data with missing values
    data = pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0],
        'b': [np.nan, 2.0, 3.0, 4.0]
    })
    
    # Test mean imputation
    preprocessor = NumericPreprocessor(
        scaling='standard',
        handle_missing='mean'
    )
    
    # Test fit
    preprocessor.fit(data)
    assert 'a' in preprocessor.fill_values
    assert 'b' in preprocessor.fill_values
    assert np.isclose(preprocessor.fill_values['a'], data['a'].mean())
    
    # Test transform
    transformed = preprocessor.transform(data)
    assert not transformed.isna().any().any()  # No missing values
    assert transformed.shape == data.shape
    
    # Test with different scaling
    preprocessor = NumericPreprocessor(
        scaling='minmax',
        handle_missing='median'
    )
    transformed = preprocessor.fit_transform(data)
    assert (transformed >= 0).all().all()  # MinMax scaling
    assert (transformed <= 1).all().all()

def test_preprocessing_pipeline():
    """Test preprocessing pipeline."""
    data = pd.DataFrame({
        'numeric': [1.0, 2.0, np.nan, 4.0],
        'category': ['A', 'B', 'A', 'C']
    })
    
    pipeline = PreprocessingPipeline([
        {
            'preprocessor': NumericPreprocessor,
            'columns': ['numeric'],
            'scaling': 'standard'
        }
    ])
    
    # Test fit
    pipeline.fit(data)
    
    # Test transform
    transformed = pipeline.transform(data)
    assert not transformed['numeric'].isna().any()  # No missing values in numeric
    assert transformed['category'].equals(data['category'])  # Category unchanged
    
    # Test fit_transform
    result = pipeline.fit_transform(data)
    assert result.shape == data.shape 