from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from typing import List
import mlflow
import mlflow.sklearn

from .schemas import PredictionRequest, PredictionResponse, UserEvent
from ..data.preprocessing import preprocess_pipeline
from ..data.feature_engineering import create_all_features
from ..utils.config import settings
import os
import joblib
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
features_path = os.path.join(os.path.dirname(__file__), "..", "data", "user_features.json")
features_path = os.path.abspath(features_path)
features_df = pd.read_json(features_path).set_index("user_id")

@app.on_event("startup")
def load_model():

    global model
    try:
        # Try to load from MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        # model = mlflow.sklearn.load_model(settings.model_uri)
        
        # For now, load from file
        model_path = "models/lg_churn.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("WARNING: No model found. API will not work properly.")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(request: PredictionRequest):
    """Predict churn for a single user"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded")

        user_features = get_user_features(request.user_id)
        if user_features is None:
            raise HTTPException(status_code=404, detail="User not found")

        X = pd.DataFrame([user_features])
        print("User features:", X.columns.tolist())

        churn_prob = round(model.predict_proba(X)[0, 1], 2)
        churn_pred = model.predict(X)[0]

        if churn_prob < 0.3:
            risk_level = "low"
        elif churn_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        log_prediction(request.user_id, churn_prob, churn_pred)

        return PredictionResponse(
            user_id=request.user_id,
            churn_probability=float(churn_prob),
            churn_prediction=bool(churn_pred),
            risk_level=risk_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
def batch_predict(user_ids: List[int]):
    """Predict churn for multiple users"""
    predictions = []
    for user_id in user_ids:
        try:
            pred = predict_churn(PredictionRequest(user_id=user_id))
            predictions.append(pred)
        except:
            continue
    return predictions

@app.post("/update_user_events")
def update_user_events(events: List[UserEvent]):
    """Update user events for real-time feature computation"""
    # Store events in database/cache
    # Trigger feature recomputation
    return {"message": f"Processed {len(events)} events"}

@app.get("/model/info")
def model_info():
    """Get current model information"""
    return {
        "model_version": settings.MODEL_VERSION,
        "model_uri": settings.MODEL_URI,
        "last_updated": settings.MODEL_UPDATED,
        "features": list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
    }

def get_user_features(user_id: int):
    """Fetch pre-computed features for a user"""
    # In production, this would query a feature store
    try:
        user_row = features_df.loc[user_id]
        return user_row.to_dict()
    except KeyError:
        return None

def log_prediction(user_id: int, probability: float, prediction: bool):
    """Log predictions for monitoring"""
    pass