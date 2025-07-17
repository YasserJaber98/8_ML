import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
from datetime import datetime

def train_with_mlflow(X, y, model, model_name, param_grid=None):
    """Train model with MLflow tracking"""
    
    mlflow.set_experiment("customer_churn_prediction")
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "model_type": model_name,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "churn_rate": y.mean()
        })
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        # Log metrics
        mlflow.log_metrics({
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "cv_f1_min": cv_scores.min(),
            "cv_f1_max": cv_scores.max()
        })
        
        # Train final model
        model.fit(X, y)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"churn_predictor_{model_name}"
        )
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
        return model, cv_scores.mean()