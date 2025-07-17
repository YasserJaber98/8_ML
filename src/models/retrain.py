import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
from .train import train_with_mlflow
from ..monitoring.drift_detection import DriftDetector

class AutoRetrainer:
    def __init__(self, data_source, model_registry, monitoring_config):
        self.data_source = data_source
        self.model_registry = model_registry
        self.monitoring_config = monitoring_config
        self.drift_detector = None
        
    def check_retraining_criteria(self) -> bool:
        """Check if retraining is needed"""
        # Criteria:
        # 1. Time-based: Every 30 days
        # 2. Performance-based: F1 score drops below threshold
        # 3. Drift-based: Significant data drift detected
        
        last_training = self.get_last_training_date()
        
        # Time-based
        if (datetime.now() - last_training).days > 30:
            return True
            
        # Performance-based
        current_performance = self.evaluate_current_model()
        if current_performance['f1_score'] < 0.75:
            return True
            
        # Drift-based
        drift_results = self.check_drift()
        if drift_results['overall_drift']:
            return True
            
        return False
    
    def retrain_model(self):
        """Retrain the model with new data"""
        print(f"Starting retraining at {datetime.now()}")
        
        # Load new data
        new_data = self.load_recent_data()
        
        # Preprocess and engineer features
        X, y = self.prepare_training_data(new_data)
        
        # Train model with MLflow tracking
        model, score = train_with_mlflow(
            X, y, 
            self.create_model_pipeline(),
            f"auto_retrain_{datetime.now().strftime('%Y%m%d')}"
        )
        
        # Validate on holdout
        if self.validate_new_model(model, score):
            self.deploy_model(model)
            print(f"Model deployed successfully. F1 Score: {score:.4f}")
        else:
            print("New model did not meet performance criteria")
    
    def schedule_retraining(self):
        """Schedule periodic retraining checks"""
        schedule.every().day.at("02:00").do(self.check_and_retrain)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    def check_and_retrain(self):
        """Check criteria and retrain if needed"""
        if self.check_retraining_criteria():
            self.retrain_model()