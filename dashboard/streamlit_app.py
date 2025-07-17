import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import numpy as np

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Title
st.title("Customer Churn Prediction Dashboard")

# Sidebar
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 30)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Users", "12,543", "+123")
    
with col2:
    st.metric("Churn Rate", "23.1%", "+2.3%")
    
with col3:
    st.metric("High Risk Users", "1,234", "+45")
    
with col4:
    st.metric("Model F1 Score", "0.824", "-0.012")

# Churn predictions over time
st.header("Churn Predictions Trend")

# Mock data - replace with real data
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
churn_trend = pd.DataFrame({
    'date': dates,
    'predicted_churn': np.random.randint(50, 150, 30),
    'actual_churn': np.random.randint(40, 140, 30)
})

fig = px.line(churn_trend, x='date', y=['predicted_churn', 'actual_churn'],
              title='Predicted vs Actual Churn (30 days)')
st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.header("Feature Importance")

col1, col2 = st.columns(2)

with col1:
    # Top features
    features_df = pd.DataFrame({
        'feature': ['days_since_last_activity', 'usage_frequency', 'error_count', 
                   'positive_feedback_ratio', 'artist_diversity'],
        'importance': [0.25, 0.18, 0.15, 0.12, 0.10]
    })
    
    fig = px.bar(features_df, x='importance', y='feature', orientation='h',
                 title='Top 5 Important Features')
    st.plotly_chart(fig)

with col2:
    # Risk distribution
    risk_dist = pd.DataFrame({
        'risk_level': ['Low', 'Medium', 'High'],
        'count': [8234, 3456, 853]
    })
    
    fig = px.pie(risk_dist, values='count', names='risk_level',
                 title='User Risk Distribution')
    st.plotly_chart(fig)

# Model performance monitoring
st.header("Model Performance Monitoring")

col1, col2 = st.columns(2)

with col1:
    # Performance metrics over time
    perf_df = pd.DataFrame({
        'date': dates,
        'f1_score': np.random.uniform(0.8, 0.85, 30),
        'precision': np.random.uniform(0.75, 0.82, 30),
        'recall': np.random.uniform(0.82, 0.88, 30)
    })
    
    fig = px.line(perf_df, x='date', y=['f1_score', 'precision', 'recall'],
                  title='Model Performance Metrics')
    st.plotly_chart(fig)

with col2:
    # Drift detection
    st.subheader("Data Drift Detection")
    
    drift_features = ['days_since_last_activity', 'usage_frequency', 'songs_played']
    drift_scores = [0.02, 0.08, 0.15]
    
    colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' 
              for x in drift_scores]
    
    fig = go.Figure(data=[
        go.Bar(x=drift_features, y=drift_scores, marker_color=colors)
    ])
    fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                  annotation_text="Drift threshold")
    fig.update_layout(title="Feature Drift Scores")
    st.plotly_chart(fig)

# Real-time predictions
st.header("Recent Predictions")

# Mock recent predictions
recent_preds = pd.DataFrame({
    'User ID': [1234, 5678, 9012, 3456, 7890],
    'Churn Probability': [0.89, 0.23, 0.67, 0.45, 0.91],
    'Risk Level': ['High', 'Low', 'Medium', 'Medium', 'High'],
    'Last Activity': ['2 days ago', '1 hour ago', '5 days ago', '3 days ago', '7 days ago']
})

st.dataframe(recent_preds, use_container_width=True)

# Auto-refresh
if st.sidebar.button("Manual Refresh"):
    st.experimental_rerun()