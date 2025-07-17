import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import json
from datetime import datetime

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """Initialize with reference data"""
        self.reference_data = reference_data
        self.threshold = threshold
        self.feature_stats = self._calculate_stats(reference_data)
        
    def _calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for each feature"""
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift in current data compared to reference"""
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'features_drifted': [],
            'drift_scores': {},
            'overall_drift': False
        }
        
        for col in self.feature_stats.keys():
            if col in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[col], 
                    current_data[col]
                )
                
                drift_results['drift_scores'][col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drifted': p_value < self.threshold
                }
                
                if p_value < self.threshold:
                    drift_results['features_drifted'].append(col)
        
        drift_results['overall_drift'] = len(drift_results['features_drifted']) > 0
        
        return drift_results
    
    def detect_concept_drift(self, predictions: np.array, actuals: np.array) -> Dict:
        """Detect concept drift based on model performance"""
        # Calculate performance metrics over time windows
        window_size = len(predictions) // 10  # 10 windows
        
        performance_windows = []
        for i in range(0, len(predictions), window_size):
            window_preds = predictions[i:i+window_size]
            window_actuals = actuals[i:i+window_size]
            
            if len(window_preds) > 0:
                accuracy = (window_preds == window_actuals).mean()
                performance_windows.append(accuracy)
        
        # Detect trend
        if len(performance_windows) > 1:
            trend = np.polyfit(range(len(performance_windows)), performance_windows, 1)[0]
            
            return {
                'performance_trend': trend,
                'degrading': trend < -0.01,  # Performance dropping
                'window_performances': performance_windows
            }
        
        return {'degrading': False}