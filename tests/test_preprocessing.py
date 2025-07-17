import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import convert_timestamps, clean_user_ids

def test_convert_timestamps():
    # Create test data
    df = pd.DataFrame({
        'ts': [1538352117000],
        'registration': [1538173000000]
    })
    
    result = convert_timestamps(df)
    
    assert pd.api.types.is_datetime64_any_dtype(result['ts'])
    assert pd.api.types.is_datetime64_any_dtype(result['registration'])

def test_clean_user_ids():
    df = pd.DataFrame({
        'userId': ['123', '', '456', 'abc']
    })
    
    result = clean_user_ids(df)
    
    assert result['userId'].dtype == 'float64'
    assert result['userId'].isna().sum() == 2  # Empty string and 'abc'