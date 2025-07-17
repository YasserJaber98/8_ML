import pandas as pd
import numpy as np
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from JSON file"""
    return pd.read_json(filepath, lines=True)

def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp columns to datetime"""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['registration'] = pd.to_datetime(df['registration'], unit='ms')
    return df

def clean_user_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert user IDs"""
    df = df.copy()
    df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
    return df

def impute_missing_userids(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing user IDs using session logic"""

    def is_valid_sequence_assignment(user_items_array, item):
        """Check if assigning an item to a user creates a valid consecutive sequence."""
        if len(user_items_array) == 0:
            return True
        
        min_item = user_items_array[0]
        max_item = user_items_array[-1]
        
        if item == max_item + 1 or item == min_item - 1:
            return True
        elif min_item < item < max_item:
            for existing_item in user_items_array:
                if abs(existing_item - item) == 1:
                    return True
            return False
        else:
            return False
    
    def map_user_attributes(df):
        """Map user attributes based on imputed userId."""
        df = df.copy()
        user_cols = ['location', 'userAgent', 'lastName', 'firstName', 'registration', 'gender']
        
        user_map = df[df['userId'].notna()].groupby('userId')[user_cols].first()
        
        for col in user_cols:
            df[col] = df['userId'].map(user_map[col]).fillna(df[col])
        
        return df

    df = df.copy()
    df['imputed'] = False
    df['ts'] = pd.to_datetime(df['ts'])
    
    userId_col = df.columns.get_loc('userId')
    imputed_col = df.columns.get_loc('imputed')

    for session in df[df['userId'].isna()]['sessionId'].unique():
        mask = df['sessionId'] == session
        session_df = df[mask].copy()
        
        userId_array = session_df['userId'].values
        itemInSession_array = session_df['itemInSession'].values
        ts_array = pd.to_datetime(session_df['ts']).values
        
        max_iterations = len(session_df)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            filled_any = False
            user_items_cache = {}
            
            missing_mask = pd.isna(userId_array)
            missing_indices = np.where(missing_mask)[0]
            
            if len(missing_indices) == 0:
                break
            
            for i in missing_indices:
                missing_ts = ts_array[i]
                missing_item = itemInSession_array[i]
                
                forward_user = None
                backward_user = None
                forward_ts = None
                backward_ts = None
                
                # Look forward
                for j in range(i + 1, len(userId_array)):
                    candidate_user = userId_array[j]
                    if not pd.isna(candidate_user):
                        if candidate_user not in user_items_cache:
                            user_mask = userId_array == candidate_user
                            user_items_cache[candidate_user] = np.sort(itemInSession_array[user_mask])
                        
                        user_items = user_items_cache[candidate_user]
                        
                        if missing_item not in user_items:
                            if is_valid_sequence_assignment(user_items, missing_item):
                                forward_user = candidate_user
                                forward_ts = ts_array[j]
                                break
                
                # Look backward
                for j in range(i - 1, -1, -1):
                    candidate_user = userId_array[j]
                    if not pd.isna(candidate_user):
                        if candidate_user not in user_items_cache:
                            user_mask = userId_array == candidate_user
                            user_items_cache[candidate_user] = np.sort(itemInSession_array[user_mask])
                        
                        user_items = user_items_cache[candidate_user]
                        
                        if missing_item not in user_items:
                            if is_valid_sequence_assignment(user_items, missing_item):
                                backward_user = candidate_user
                                backward_ts = ts_array[j]
                                break
                
                # Choose user based on timestamp proximity
                chosen_user = None
                
                if forward_user is not None and backward_user is not None:
                    forward_diff = abs((forward_ts - missing_ts).astype('timedelta64[s]').astype(int))
                    backward_diff = abs((backward_ts - missing_ts).astype('timedelta64[s]').astype(int))
                    
                    if forward_diff <= backward_diff:
                        chosen_user = forward_user
                    else:
                        chosen_user = backward_user
                elif forward_user is not None:
                    chosen_user = forward_user
                elif backward_user is not None:
                    chosen_user = backward_user
                
                if chosen_user is not None:
                    userId_array[i] = chosen_user
                    session_df.iloc[i, imputed_col] = True
                    filled_any = True
                    if chosen_user in user_items_cache:
                        del user_items_cache[chosen_user]
            
            if not filled_any:
                break
        
        session_df.iloc[:, userId_col] = userId_array
        df.loc[mask, 'userId'] = session_df['userId'].values
        df.loc[mask, 'imputed'] = session_df['imputed'].values

    df = map_user_attributes(df)
    return df

def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract city and state from location"""
    df = df.copy()
    df['city'] = df['location'].str.split(',').str[0]
    df['state'] = df['location'].str.split(',').str[1].str.strip()
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Complete preprocessing pipeline"""
    df = convert_timestamps(df)
    df = clean_user_ids(df)
    df = impute_missing_userids(df)
    df = create_location_features(df)
    df = df.dropna(subset=['userId']).reset_index(drop=True)
    df['userId'] = df['userId'].astype(int)
    return df