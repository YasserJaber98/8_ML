from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class UserEvent(BaseModel):
    userId: int
    sessionId: int
    page: str
    auth: str
    ts: datetime
    itemInSession: int
    length: Optional[float] = None
    artist: Optional[str] = None
    song: Optional[str] = None
    
class PredictionRequest(BaseModel):
    user_id: int
    
class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    churn_prediction: bool
    risk_level: str  # 'low', 'medium', 'high'
    
class UserFeatures(BaseModel):
    features: Dict[str, float]