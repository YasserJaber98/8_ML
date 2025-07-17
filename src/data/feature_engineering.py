import pandas as pd
import numpy as np

def create_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create activity-based features"""
    features = df.groupby('userId').agg({
        'ts': 'count',
        'sessionId': 'nunique',
        'itemInSession': 'sum'
    }).fillna(0)
    
    features.columns = ['total_events', 'num_sessions', 'total_interactions']
    features['events_per_session'] = (
        features['total_events'] / features['num_sessions']
    ).fillna(0)
    
    return features

def create_listening_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create music listening features"""
    songs_df = df[df['song'].notna()]
    
    features = songs_df.groupby('userId').agg({
        'ts': 'count',
        'length': ['sum', 'mean'],
        'artist': 'nunique',
        'song': 'nunique',
        'registration': 'first'
    }).fillna(0)
    
    # Flatten column names and process
    features.columns = ['songs_played', 'total_listening_time', 'avg_song_length',
                       'unique_artists', 'unique_songs', 'registration_date']
    
    # Additional calculations
    max_date = df['ts'].max()
    features['days_since_registration'] = (
        (max_date - pd.to_datetime(features['registration_date'])).dt.total_seconds() / (24 * 3600)
    )
    
    features['avg_daily_listening_time'] = (
        features['total_listening_time'] / features['days_since_registration']
    ).fillna(0)
    
    features['avg_daily_songs'] = (
        features['songs_played'] / features['days_since_registration']
    ).fillna(0)
    
    features['artist_diversity'] = (
        features['unique_artists'] / features['songs_played']
    ).fillna(0)
    
    return features.drop('registration_date', axis=1)

def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user engagement features"""
    # Engagement Features
    thumbs_up = df.query("page == 'Thumbs Up'").groupby('userId').size()
    thumbs_down = df.query("page == 'Thumbs Down'").groupby('userId').size()

    engagement_features = pd.DataFrame({
        'thumbs_up': thumbs_up,
        'thumbs_down': thumbs_down
    }).fillna(0)

    engagement_features['total_feedback'] = engagement_features['thumbs_up'] + engagement_features['thumbs_down']
    engagement_features['positive_feedback_ratio'] = (
        engagement_features['thumbs_up'] / engagement_features['total_feedback']
    ).fillna(0.5)

    # Other engagement metrics
    engagement_features['playlist_adds'] = df[df['page'] == 'Add to Playlist'].groupby('userId').size()
    engagement_features['add_friend'] = df[df['page'] == 'Add Friend'].groupby('userId').size()
    engagement_features['advert_roll'] = df[df['page'] == 'Roll Advert'].groupby('userId').size()
    engagement_features = engagement_features.fillna(0)
    return engagement_features

def create_subscription_features(df: pd.DataFrame) -> pd.DataFrame:
    # Subscription Features
    latest_level = df.groupby('userId')['level'].last()
    subscription_features = pd.DataFrame({
        'is_paid': (latest_level == 'paid').astype(int)
    })

    level_changes = df.groupby('userId')['level'].nunique()
    subscription_features['subscription_changes'] = (level_changes > 1).astype(int)

    subscription_features['downgrades'] = df.query("page == 'Submit Downgrade'").groupby('userId').size()
    subscription_features['upgrades'] = df.query("page == 'Submit Upgrade'").groupby('userId').size()
    subscription_features = subscription_features.fillna(0)
    return subscription_features

def create_issues_features(df: pd.DataFrame) -> pd.DataFrame:
    issue_features = pd.DataFrame({
        'error_count': df.query("page == 'Error'").groupby('userId').size(),
        'help_visits': df.query("page == 'Help'").groupby('userId').size(),
        'settings_visits': df.query("page == 'Settings'").groupby('userId').size(),
        'logout_count': df.query("page == 'Logout'").groupby('userId').size()
    }).fillna(0)

    issue_features['has_issues'] = (
        (issue_features['error_count'] > 0) | 
        (issue_features['help_visits'] > 0)
    ).astype(int)

    return issue_features

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    last_activity = df.groupby('userId')['ts'].max()
    max_date = df['ts'].max()
    days_since_last_activity = (max_date - last_activity).dt.total_seconds() / (24 * 3600)

    user_registration = df.groupby('userId')['registration'].first()
    user_registration = pd.to_datetime(user_registration)

    days_available = []
    for user_id in user_registration.index:
        reg_date = user_registration[user_id]
        days_available.append((max_date.date() - reg_date.date()).days + 1)

    days_used = df.groupby('userId')['ts'].apply(lambda x: x.dt.date.nunique())

    temporal_features = pd.DataFrame({
        'days_since_last_activity': days_since_last_activity,
        'days_used_in_period': days_used,
        'days_available_in_period': days_available,
        'usage_frequency': days_used / pd.Series(days_available, index=days_used.index)
    }, index=user_registration.index).fillna(0)

    return temporal_features

def create_session_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    # Session Pattern Features
    session_lengths = df.groupby(['userId', 'sessionId'])['itemInSession'].max()
    session_length_stats = session_lengths.groupby('userId').agg(['mean', 'std', 'max']).fillna(0)
    session_length_stats.columns = ['avg_session_length', 'session_length_std', 'max_session_length']

    session_durations = df.groupby(['userId', 'sessionId'])['ts'].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    session_duration_stats = session_durations.groupby('userId').agg(['mean', 'std', 'max']).fillna(0)
    session_duration_stats.columns = ['avg_session_duration_mins', 'session_duration_std_mins', 'max_session_duration_mins']

    session_features = pd.concat([session_length_stats, session_duration_stats], axis=1)
    session_features['session_consistency'] = 1 / (1 + session_features['session_length_std'])
    return session_features

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features from preprocessed data"""
    all_users = df['userId'].dropna().unique()
    features = pd.DataFrame(index=all_users)
    
    # Add all feature groups
    features = features.join(create_activity_features(df), how='left')
    features = features.join(create_listening_features(df), how='left')
    features = features.join(create_engagement_features(df), how='left')
    features = features.join(create_subscription_features(df), how='left')
    features = features.join(create_issues_features(df), how='left')
    features = features.join(create_temporal_features(df), how='left')
    features = features.join(create_session_pattern_features(df), how='left')

    # Add target variable
    churned_users = df.query("page == 'Cancellation Confirmation'")['userId'].unique()
    features['is_churned'] = features.index.isin(churned_users).astype(int)

    return features.fillna(0)