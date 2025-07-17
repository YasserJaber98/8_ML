Customer Churn Prediction System
A comprehensive machine learning system for predicting customer churn in a music streaming service, featuring automated retraining, drift detection, and real-time monitoring.

Project Overview
This project implements an end-to-end solution for customer churn prediction using event log data from a music streaming platform. The system addresses key challenges including:

Class imbalance (23.1% churn rate)
Missing user IDs recovery
Data leakage prevention
Real-time monitoring and drift detection

Key Results
Model Performance: F1 Score of 82.4% on minority class (churned users)
Data Recovery: Successfully recovered 98% of missing user IDs using custom algorithm
Stability: Model shows excellent stability across different random seeds (std: 0.0058)

Architecture
customer-churn-prediction/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model training and prediction
│   ├── api/               # FastAPI implementation
│   ├── monitoring/        # Drift detection and monitoring
│   └── utils/             # Utilities and configuration
├── dashboard/             # Streamlit dashboard
├── notebooks/             # Jupyter notebooks for EDA
├── tests/                 # Unit tests
└── docker/                # Docker configuration

🚀 Features
Data Processing

Custom algorithm for imputing missing user IDs based on session sequences
Comprehensive feature engineering (40+ features across 7 categories)
Handling of reused session IDs across multiple users

Model Development

Multiple models tested (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
Hyperparameter tuning with cross-validation
Class imbalance handling through appropriate weights
Feature selection to avoid overfitting

MLOps Implementation

MLflow Integration: Experiment tracking and model versioning
FastAPI: RESTful API for model serving
Streamlit Dashboard: Real-time monitoring and visualization
Drift Detection: Both data drift and concept drift monitoring
Automated Retraining: Scheduled retraining based on multiple criteria

Monitoring System

Real-time performance tracking
Feature drift detection using Kolmogorov-Smirnov test
Concept drift detection through performance windowing
Business metrics dashboard

Installation
Using Conda (Recommended)
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate customer-churn


Quick Start

Run the API:
uvicorn src.api.main:app --reload
Then go to 
http://localhost:8000/docs

Launch dashboard:
streamlit run dashboard/streamlit_app.py


API Endpoints

POST /predict: Single user churn prediction
POST /batch_predict: Batch predictions
GET /model/info: Current model information
POST /update_user_events: Update user events for real-time features


Model Performance
Best Model: Logistic Regression

F1 Score: 0.824 (±0.068)
Precision: 0.843 (minority class)
Recall: 0.827 (minority class)
Feature Selection: Top 25 features using SelectKBest

Top Predictive Features

Days since last activity (strongest predictor)
Session consistency
Artist diversity
Session length standard deviation
Add friend actions

Technical Challenges & Solutions
1. Missing User IDs
Challenge: 8,346 records without user IDs
Solution: Developed custom imputation algorithm using:

Session ID and item sequence patterns
Temporal proximity
Sequence validation logic

2. Session ID Reuse
Challenge: Session IDs shared across multiple users
Solution: Combined sessionId + itemInSession for accurate user tracking
3. Churn Definition
Challenge: Defining accurate churn criteria
Solution: Used explicit churn signal (Cancellation Confirmation page visit)
🔄 Retraining Strategy
The system supports automated retraining based on:

Time-based: Every 30 days
Performance-based: When F1 score drops below 0.75
Drift-based: When significant feature drift is detected

Monitoring & Alerting

Real-time performance metrics
Feature drift scores with threshold alerts
Concept drift detection through windowed performance analysis
Business KPI tracking

⚠️ Current Limitations

Data Storage: Currently using local file system (production should use cloud storage)
Model Storage: Models stored locally (production should use model registry)
Scheduling: Retraining scheduler written but not active (needs production scheduler like Airflow)
Docker: Configuration complete but may need environment-specific adjustments
Database: Using mock data (production needs real database connection)

Future Improvements

Technical Enhancements:

Migrate to cloud infrastructure (AWS/Azure/GCP)
Implement Kubernetes for container orchestration
Add CI/CD pipeline
Implement A/B testing framework

Model Improvements:

Explore deep learning for sequence modeling
Implement SMOTE for better class balance
Add ensemble methods
Include external features (seasonality, holidays)


Note: This is a demonstration project. Some components (Docker, scheduling, cloud storage) are implemented but not fully operational in the current development environment.