"""
Visual Intelligence ML Tuner for Bot 2
======================================
Specialized ML system that learns from visual forecast accuracy and chart pattern outcomes.
Evolves Bot 2's visual intelligence through experience-based learning.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pickle
import joblib

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

@dataclass
class ForecastSnapshot:
    """Captures a forecast at the moment of prediction"""
    timestamp: datetime
    symbol: str
    timeframe: str
    forecast_id: str
    
    # Visual analysis data
    regime_type: str  # 'bullish', 'bearish', 'sideways'
    pattern_detected: str  # 'breakout', 'reversal', 'continuation', etc.
    confidence_score: float
    
    # Forecast predictions
    predicted_direction: str  # 'LONG', 'SHORT'
    predicted_price_target: float
    predicted_timeframe_hours: int
    
    # Chart visual features
    volatility_level: str  # 'low', 'medium', 'high'
    trend_strength: float  # 0-1
    support_resistance_clarity: float  # 0-1
    volume_confirmation: bool
    
    # Strategy context
    strategy_name: str
    entry_price: float
    stop_loss: float
    profit_target: float

@dataclass
class ForecastOutcome:
    """Records the actual market outcome for a forecast"""
    forecast_id: str
    timestamp_evaluated: datetime
    
    # Actual market behavior
    actual_direction: str  # 'LONG', 'SHORT', 'SIDEWAYS'
    actual_price_change: float
    actual_max_favorable: float  # Best price in predicted direction
    actual_max_adverse: float   # Worst price against prediction
    
    # Accuracy metrics
    direction_correct: bool
    target_reached: bool
    stop_hit: bool
    forecast_accuracy_score: float  # 0-1 overall accuracy
    
    # Visual validation
    regime_transition_occurred: bool
    pattern_completed_as_expected: bool
    confidence_justified: bool

@dataclass
class VisualTuningRecommendation:
    """Recommendation for visual intelligence parameter adjustments"""
    component: str  # 'confidence_scoring', 'pattern_detection', 'regime_analysis'
    parameter: str
    current_value: float
    recommended_value: float
    confidence: float
    reasoning: str
    expected_improvement: float
    risk_level: str  # 'low', 'medium', 'high'
    
    # Visual-specific data
    pattern_success_rate: float
    confidence_zone_accuracy: float
    regime_prediction_accuracy: float

@dataclass
class VisualTuningSession:
    """Complete visual tuning session results"""
    session_id: str
    timestamp: datetime
    recommendations: List[VisualTuningRecommendation]
    model_performance: Dict[str, float]
    forecast_accuracy_metrics: Dict[str, float]
    total_forecasts_analyzed: int
    lookback_days: int
    
    # Visual intelligence metrics
    pattern_success_rates: Dict[str, float]
    confidence_zone_performance: Dict[str, float]
    regime_prediction_accuracy: float

class VisualMLTuner:
    """
    ML-powered visual intelligence optimization system that learns from forecast accuracy
    and chart pattern outcomes to continuously improve Bot 2's visual analysis.
    """
    
    def __init__(self, data_dir: str = "data/visual_ml_tuning"):
        """Initialize the visual ML tuner.
        
        Args:
            data_dir (str): Directory for visual ML tuning data and models
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Data storage
        self.forecast_snapshots: List[ForecastSnapshot] = []
        self.forecast_outcomes: List[ForecastOutcome] = []
        
        # ML models for visual intelligence optimization
        self.models = {
            'confidence_calibrator': None,      # Calibrates confidence scores
            'pattern_success_predictor': None,  # Predicts pattern success likelihood
            'regime_transition_predictor': None, # Predicts regime changes
            'visual_feature_optimizer': None,   # Optimizes visual feature weights
        }
        
        # Model scalers
        self.scalers = {
            'confidence_features': StandardScaler(),
            'pattern_features': StandardScaler(),
            'regime_features': StandardScaler(),
            'visual_features': StandardScaler(),
        }
        
        # Load existing data and models
        self._load_data()
        self._load_models()
        
        logger.info("Visual ML Tuner initialized")
    
    def log_forecast_snapshot(self, forecast_data: Dict) -> str:
        """
        Log a forecast snapshot at the moment of prediction
        
        Args:
            forecast_data: Dictionary containing forecast information
            
        Returns:
            str: Unique forecast ID for tracking
        """
        try:
            forecast_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{forecast_data.get('symbol', 'UNKNOWN')}"
            
            snapshot = ForecastSnapshot(
                timestamp=datetime.now(),
                symbol=forecast_data.get('symbol', 'UNKNOWN'),
                timeframe=forecast_data.get('timeframe', '1h'),
                forecast_id=forecast_id,
                
                # Visual analysis
                regime_type=forecast_data.get('regime_type', 'sideways'),
                pattern_detected=forecast_data.get('pattern_detected', 'unknown'),
                confidence_score=forecast_data.get('confidence_score', 0.5),
                
                # Predictions
                predicted_direction=forecast_data.get('predicted_direction', 'LONG'),
                predicted_price_target=forecast_data.get('predicted_price_target', 0.0),
                predicted_timeframe_hours=forecast_data.get('predicted_timeframe_hours', 24),
                
                # Visual features
                volatility_level=forecast_data.get('volatility_level', 'medium'),
                trend_strength=forecast_data.get('trend_strength', 0.5),
                support_resistance_clarity=forecast_data.get('support_resistance_clarity', 0.5),
                volume_confirmation=forecast_data.get('volume_confirmation', False),
                
                # Strategy context
                strategy_name=forecast_data.get('strategy_name', 'Unknown'),
                entry_price=forecast_data.get('entry_price', 0.0),
                stop_loss=forecast_data.get('stop_loss', 0.0),
                profit_target=forecast_data.get('profit_target', 0.0)
            )
            
            self.forecast_snapshots.append(snapshot)
            self._save_data()
            
            logger.info(f"Logged forecast snapshot: {forecast_id}")
            return forecast_id
            
        except Exception as e:
            logger.error(f"Error logging forecast snapshot: {e}")
            return ""
    
    def log_forecast_outcome(self, forecast_id: str, outcome_data: Dict) -> bool:
        """
        Log the actual market outcome for a forecast
        
        Args:
            forecast_id: ID of the original forecast
            outcome_data: Dictionary containing outcome information
            
        Returns:
            bool: Success status
        """
        try:
            # Find the original forecast
            original_forecast = None
            for snapshot in self.forecast_snapshots:
                if snapshot.forecast_id == forecast_id:
                    original_forecast = snapshot
                    break
            
            if not original_forecast:
                logger.warning(f"Original forecast not found for ID: {forecast_id}")
                return False
            
            # Calculate accuracy metrics
            direction_correct = (
                original_forecast.predicted_direction == outcome_data.get('actual_direction', 'SIDEWAYS')
            )
            
            target_reached = outcome_data.get('target_reached', False)
            stop_hit = outcome_data.get('stop_hit', False)
            
            # Calculate overall accuracy score
            accuracy_components = [
                1.0 if direction_correct else 0.0,
                1.0 if target_reached else 0.0,
                1.0 if not stop_hit else 0.0,
                outcome_data.get('pattern_accuracy', 0.5),
                outcome_data.get('timing_accuracy', 0.5)
            ]
            forecast_accuracy_score = np.mean(accuracy_components)
            
            outcome = ForecastOutcome(
                forecast_id=forecast_id,
                timestamp_evaluated=datetime.now(),
                
                # Actual results
                actual_direction=outcome_data.get('actual_direction', 'SIDEWAYS'),
                actual_price_change=outcome_data.get('actual_price_change', 0.0),
                actual_max_favorable=outcome_data.get('actual_max_favorable', 0.0),
                actual_max_adverse=outcome_data.get('actual_max_adverse', 0.0),
                
                # Accuracy
                direction_correct=direction_correct,
                target_reached=target_reached,
                stop_hit=stop_hit,
                forecast_accuracy_score=forecast_accuracy_score,
                
                # Visual validation
                regime_transition_occurred=outcome_data.get('regime_transition_occurred', False),
                pattern_completed_as_expected=outcome_data.get('pattern_completed_as_expected', False),
                confidence_justified=outcome_data.get('confidence_justified', False)
            )
            
            self.forecast_outcomes.append(outcome)
            self._save_data()
            
            logger.info(f"Logged forecast outcome: {forecast_id} (Accuracy: {forecast_accuracy_score:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error logging forecast outcome: {e}")
            return False
    
    def analyze_visual_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze visual intelligence performance over the specified period
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Dict containing performance metrics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get recent forecasts with outcomes
            recent_forecasts = []
            for snapshot in self.forecast_snapshots:
                if snapshot.timestamp >= cutoff_date:
                    # Find corresponding outcome
                    outcome = None
                    for out in self.forecast_outcomes:
                        if out.forecast_id == snapshot.forecast_id:
                            outcome = out
                            break
                    
                    if outcome:
                        recent_forecasts.append((snapshot, outcome))
            
            if not recent_forecasts:
                return {"error": "No recent forecasts with outcomes found"}
            
            # Calculate performance metrics
            total_forecasts = len(recent_forecasts)
            direction_accuracy = sum(1 for _, outcome in recent_forecasts if outcome.direction_correct) / total_forecasts
            target_hit_rate = sum(1 for _, outcome in recent_forecasts if outcome.target_reached) / total_forecasts
            stop_hit_rate = sum(1 for _, outcome in recent_forecasts if outcome.stop_hit) / total_forecasts
            
            # Pattern success rates
            pattern_performance = {}
            for snapshot, outcome in recent_forecasts:
                pattern = snapshot.pattern_detected
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {'total': 0, 'successful': 0}
                pattern_performance[pattern]['total'] += 1
                if outcome.pattern_completed_as_expected:
                    pattern_performance[pattern]['successful'] += 1
            
            # Convert to success rates
            pattern_success_rates = {}
            for pattern, data in pattern_performance.items():
                pattern_success_rates[pattern] = data['successful'] / data['total'] if data['total'] > 0 else 0
            
            # Confidence zone analysis
            confidence_zones = {'high': [], 'medium': [], 'low': []}
            for snapshot, outcome in recent_forecasts:
                if snapshot.confidence_score >= 0.8:
                    confidence_zones['high'].append(outcome.forecast_accuracy_score)
                elif snapshot.confidence_score >= 0.6:
                    confidence_zones['medium'].append(outcome.forecast_accuracy_score)
                else:
                    confidence_zones['low'].append(outcome.forecast_accuracy_score)
            
            confidence_zone_performance = {}
            for zone, scores in confidence_zones.items():
                confidence_zone_performance[zone] = np.mean(scores) if scores else 0
            
            # Regime prediction accuracy
            regime_correct = sum(1 for _, outcome in recent_forecasts if outcome.regime_transition_occurred) / total_forecasts
            
            return {
                'total_forecasts': total_forecasts,
                'direction_accuracy': direction_accuracy,
                'target_hit_rate': target_hit_rate,
                'stop_hit_rate': stop_hit_rate,
                'pattern_success_rates': pattern_success_rates,
                'confidence_zone_performance': confidence_zone_performance,
                'regime_prediction_accuracy': regime_correct,
                'overall_accuracy': np.mean([outcome.forecast_accuracy_score for _, outcome in recent_forecasts])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing visual performance: {e}")
            return {"error": str(e)}
    
    def generate_tuning_recommendations(self, lookback_days: int = 30) -> VisualTuningSession:
        """
        Generate ML-based recommendations for visual intelligence improvements
        
        Args:
            lookback_days: Number of days of data to analyze
            
        Returns:
            VisualTuningSession with recommendations
        """
        try:
            session_id = f"visual_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Analyze current performance
            performance = self.analyze_visual_performance(lookback_days)
            
            if 'error' in performance:
                logger.warning(f"Cannot generate recommendations: {performance['error']}")
                return None
            
            recommendations = []
            
            # Confidence scoring recommendations
            if performance['confidence_zone_performance'].get('high', 0) < 0.8:
                recommendations.append(VisualTuningRecommendation(
                    component='confidence_scoring',
                    parameter='high_confidence_threshold',
                    current_value=0.8,
                    recommended_value=0.85,
                    confidence=0.75,
                    reasoning="High confidence forecasts underperforming - raise threshold",
                    expected_improvement=0.1,
                    risk_level='low',
                    pattern_success_rate=performance.get('overall_accuracy', 0),
                    confidence_zone_accuracy=performance['confidence_zone_performance'].get('high', 0),
                    regime_prediction_accuracy=performance.get('regime_prediction_accuracy', 0)
                ))
            
            # Pattern detection recommendations
            worst_pattern = min(performance['pattern_success_rates'].items(), key=lambda x: x[1], default=(None, 1))
            if worst_pattern[0] and worst_pattern[1] < 0.5:
                recommendations.append(VisualTuningRecommendation(
                    component='pattern_detection',
                    parameter=f'{worst_pattern[0]}_sensitivity',
                    current_value=0.5,
                    recommended_value=0.3,
                    confidence=0.6,
                    reasoning=f"Pattern '{worst_pattern[0]}' has low success rate - reduce sensitivity",
                    expected_improvement=0.15,
                    risk_level='medium',
                    pattern_success_rate=worst_pattern[1],
                    confidence_zone_accuracy=performance.get('overall_accuracy', 0),
                    regime_prediction_accuracy=performance.get('regime_prediction_accuracy', 0)
                ))
            
            # Create tuning session
            session = VisualTuningSession(
                session_id=session_id,
                timestamp=datetime.now(),
                recommendations=recommendations,
                model_performance={'overall_accuracy': performance.get('overall_accuracy', 0)},
                forecast_accuracy_metrics=performance,
                total_forecasts_analyzed=performance.get('total_forecasts', 0),
                lookback_days=lookback_days,
                pattern_success_rates=performance.get('pattern_success_rates', {}),
                confidence_zone_performance=performance.get('confidence_zone_performance', {}),
                regime_prediction_accuracy=performance.get('regime_prediction_accuracy', 0)
            )
            
            logger.info(f"Generated {len(recommendations)} visual tuning recommendations")
            return session
            
        except Exception as e:
            logger.error(f"Error generating tuning recommendations: {e}")
            return None
    
    def _save_data(self):
        """Save forecast data to disk"""
        try:
            # Save snapshots
            snapshots_file = os.path.join(self.data_dir, 'forecast_snapshots.json')
            with open(snapshots_file, 'w') as f:
                json.dump([asdict(snapshot) for snapshot in self.forecast_snapshots], f, default=str)
            
            # Save outcomes
            outcomes_file = os.path.join(self.data_dir, 'forecast_outcomes.json')
            with open(outcomes_file, 'w') as f:
                json.dump([asdict(outcome) for outcome in self.forecast_outcomes], f, default=str)
                
        except Exception as e:
            logger.error(f"Error saving visual ML data: {e}")
    
    def _load_data(self):
        """Load forecast data from disk"""
        try:
            # Load snapshots
            snapshots_file = os.path.join(self.data_dir, 'forecast_snapshots.json')
            if os.path.exists(snapshots_file):
                with open(snapshots_file, 'r') as f:
                    data = json.load(f)
                    self.forecast_snapshots = [
                        ForecastSnapshot(**{k: datetime.fromisoformat(v) if k == 'timestamp' else v for k, v in item.items()})
                        for item in data
                    ]
            
            # Load outcomes
            outcomes_file = os.path.join(self.data_dir, 'forecast_outcomes.json')
            if os.path.exists(outcomes_file):
                with open(outcomes_file, 'r') as f:
                    data = json.load(f)
                    self.forecast_outcomes = [
                        ForecastOutcome(**{k: datetime.fromisoformat(v) if 'timestamp' in k else v for k, v in item.items()})
                        for item in data
                    ]
                    
        except Exception as e:
            logger.error(f"Error loading visual ML data: {e}")
    
    def _load_models(self):
        """Load trained ML models from disk"""
        try:
            models_dir = os.path.join(self.data_dir, 'models')
            if os.path.exists(models_dir):
                for model_name in self.models.keys():
                    model_file = os.path.join(models_dir, f'{model_name}.pkl')
                    if os.path.exists(model_file):
                        self.models[model_name] = joblib.load(model_file)
                        logger.info(f"Loaded model: {model_name}")
                        
        except Exception as e:
            logger.error(f"Error loading visual ML models: {e}")


# Global instance
visual_ml_tuner = None

def get_visual_ml_tuner() -> VisualMLTuner:
    """Get global visual ML tuner instance."""
    global visual_ml_tuner
    if visual_ml_tuner is None:
        visual_ml_tuner = VisualMLTuner()
    return visual_ml_tuner

def initialize_visual_ml_tuner(data_dir: str = "data/visual_ml_tuning") -> VisualMLTuner:
    """Initialize global visual ML tuner instance."""
    global visual_ml_tuner
    visual_ml_tuner = VisualMLTuner(data_dir)
    return visual_ml_tuner
