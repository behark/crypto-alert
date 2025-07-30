"""
Custom Pattern Learning System
=============================
Train the system to recognize user's favorite visual setups and patterns
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

@dataclass
class PatternAnnotation:
    """User annotation of a pattern"""
    annotation_id: str
    user_id: str
    symbol: str
    timeframe: str
    pattern_name: str
    chart_data: Dict[str, Any]
    pattern_features: Dict[str, float]
    success_outcome: Optional[bool]
    outcome_data: Optional[Dict[str, Any]]
    timestamp: datetime
    confidence: float = 50.0

@dataclass
class CustomPattern:
    """User's custom pattern definition"""
    pattern_id: str
    user_id: str
    pattern_name: str
    description: str
    feature_weights: Dict[str, float]
    success_rate: float
    occurrence_count: int
    last_seen: datetime
    training_samples: int

@dataclass
class PatternPrediction:
    """Pattern recognition prediction"""
    pattern_name: str
    confidence: float
    feature_match_score: float
    historical_success_rate: float
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'

class CustomPatternLearner:
    """
    Custom Pattern Learning System
    
    Enables users to train the system to recognize their favorite visual setups
    and patterns through interactive annotation and machine learning.
    """
    
    def __init__(self, data_dir: str = "data/pattern_learning"):
        """Initialize the custom pattern learner"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern storage
        self.pattern_annotations: Dict[str, PatternAnnotation] = {}
        self.custom_patterns: Dict[str, CustomPattern] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # ML models for each user and pattern
        self.pattern_classifiers: Dict[str, RandomForestClassifier] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_pattern_data()
        
        logger.info("Custom Pattern Learning System initialized")
    
    def annotate_pattern(self, user_id: str, symbol: str, timeframe: str,
                        pattern_name: str, chart_data: Dict[str, Any],
                        success_outcome: bool = None) -> str:
        """
        Annotate a pattern for learning
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            pattern_name: Name of the pattern
            chart_data: Chart data including OHLCV
            success_outcome: Whether pattern was successful (None if unknown)
            
        Returns:
            str: Annotation ID
        """
        try:
            with self._lock:
                annotation_id = f"pattern_{user_id}_{symbol}_{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Extract pattern features from chart data
                pattern_features = self._extract_pattern_features(chart_data)
                
                annotation = PatternAnnotation(
                    annotation_id=annotation_id,
                    user_id=user_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name=pattern_name,
                    chart_data=chart_data,
                    pattern_features=pattern_features,
                    success_outcome=success_outcome,
                    outcome_data=None,
                    timestamp=datetime.now()
                )
                
                self.pattern_annotations[annotation_id] = annotation
                
                # Update or create custom pattern
                self._update_custom_pattern(user_id, pattern_name, annotation)
                
                # Save annotation
                self._save_annotation(annotation)
                
                logger.info(f"Annotated pattern: {pattern_name} for user {user_id} on {symbol} {timeframe}")
                return annotation_id
                
        except Exception as e:
            logger.error(f"Failed to annotate pattern: {e}")
            return ""
    
    def update_pattern_outcome(self, annotation_id: str, success: bool,
                              outcome_data: Dict[str, Any] = None) -> bool:
        """
        Update the outcome of a previously annotated pattern
        
        Args:
            annotation_id: Annotation identifier
            success: Whether the pattern was successful
            outcome_data: Additional outcome data
            
        Returns:
            bool: Success status
        """
        try:
            with self._lock:
                if annotation_id not in self.pattern_annotations:
                    return False
                
                annotation = self.pattern_annotations[annotation_id]
                annotation.success_outcome = success
                annotation.outcome_data = outcome_data
                
                # Update custom pattern success rate
                pattern_key = f"{annotation.user_id}_{annotation.pattern_name}"
                if pattern_key in self.custom_patterns:
                    pattern = self.custom_patterns[pattern_key]
                    
                    # Recalculate success rate
                    successful_annotations = [
                        ann for ann in self.pattern_annotations.values()
                        if (ann.user_id == annotation.user_id and
                            ann.pattern_name == annotation.pattern_name and
                            ann.success_outcome is True)
                    ]
                    
                    total_annotations = [
                        ann for ann in self.pattern_annotations.values()
                        if (ann.user_id == annotation.user_id and
                            ann.pattern_name == annotation.pattern_name and
                            ann.success_outcome is not None)
                    ]
                    
                    if total_annotations:
                        pattern.success_rate = (len(successful_annotations) / len(total_annotations)) * 100
                    
                    # Retrain model if enough data
                    if len(total_annotations) >= 5:
                        self._train_pattern_model(annotation.user_id, annotation.pattern_name)
                
                # Save updated annotation
                self._save_annotation(annotation)
                
                logger.info(f"Updated pattern outcome: {annotation_id} -> {'success' if success else 'failure'}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update pattern outcome: {e}")
            return False
    
    def recognize_user_patterns(self, user_id: str, symbol: str, timeframe: str,
                               chart_data: Dict[str, Any]) -> List[PatternPrediction]:
        """
        Recognize user's custom patterns in chart data
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            chart_data: Chart data to analyze
            
        Returns:
            List of pattern predictions
        """
        try:
            with self._lock:
                predictions = []
                
                # Get user's custom patterns
                user_patterns = [
                    pattern for pattern in self.custom_patterns.values()
                    if pattern.user_id == user_id
                ]
                
                if not user_patterns:
                    return predictions
                
                # Extract features from current chart data
                current_features = self._extract_pattern_features(chart_data)
                
                for pattern in user_patterns:
                    try:
                        # Use ML model if available
                        model_key = f"{user_id}_{pattern.pattern_name}"
                        if model_key in self.pattern_classifiers:
                            classifier = self.pattern_classifiers[model_key]
                            scaler = self.feature_scalers.get(model_key)
                            
                            # Prepare features for prediction
                            feature_vector = self._prepare_feature_vector(current_features)
                            if scaler:
                                feature_vector = scaler.transform([feature_vector])
                            else:
                                feature_vector = [feature_vector]
                            
                            # Get prediction
                            prediction_proba = classifier.predict_proba(feature_vector)[0]
                            confidence = max(prediction_proba) * 100
                            
                            # Determine recommendation based on pattern success rate and confidence
                            if confidence >= 80 and pattern.success_rate >= 70:
                                recommendation = 'strong_buy'
                            elif confidence >= 60 and pattern.success_rate >= 60:
                                recommendation = 'buy'
                            elif confidence >= 40:
                                recommendation = 'hold'
                            else:
                                recommendation = 'neutral'
                        
                        else:
                            # Use feature similarity if no ML model
                            similarity = self._calculate_feature_similarity(
                                current_features, pattern.feature_weights
                            )
                            confidence = similarity * 100
                            
                            if confidence >= 70 and pattern.success_rate >= 60:
                                recommendation = 'buy'
                            elif confidence >= 50:
                                recommendation = 'hold'
                            else:
                                recommendation = 'neutral'
                        
                        prediction = PatternPrediction(
                            pattern_name=pattern.pattern_name,
                            confidence=confidence,
                            feature_match_score=confidence,
                            historical_success_rate=pattern.success_rate,
                            recommendation=recommendation
                        )
                        
                        predictions.append(prediction)
                        
                    except Exception as e:
                        logger.warning(f"Failed to predict pattern {pattern.pattern_name}: {e}")
                
                # Sort by confidence
                predictions.sort(key=lambda x: x.confidence, reverse=True)
                
                logger.info(f"Recognized {len(predictions)} patterns for user {user_id} on {symbol} {timeframe}")
                return predictions
                
        except Exception as e:
            logger.error(f"Failed to recognize user patterns: {e}")
            return []
    
    def get_pattern_success_rates(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get success rates for user's custom patterns
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict mapping pattern names to success statistics
        """
        try:
            with self._lock:
                pattern_stats = {}
                
                user_patterns = [
                    pattern for pattern in self.custom_patterns.values()
                    if pattern.user_id == user_id
                ]
                
                for pattern in user_patterns:
                    # Get annotations for this pattern
                    pattern_annotations = [
                        ann for ann in self.pattern_annotations.values()
                        if (ann.user_id == user_id and
                            ann.pattern_name == pattern.pattern_name)
                    ]
                    
                    # Calculate detailed statistics
                    total_annotations = len(pattern_annotations)
                    successful_outcomes = len([
                        ann for ann in pattern_annotations
                        if ann.success_outcome is True
                    ])
                    failed_outcomes = len([
                        ann for ann in pattern_annotations
                        if ann.success_outcome is False
                    ])
                    pending_outcomes = total_annotations - successful_outcomes - failed_outcomes
                    
                    pattern_stats[pattern.pattern_name] = {
                        'success_rate': pattern.success_rate,
                        'total_annotations': total_annotations,
                        'successful_outcomes': successful_outcomes,
                        'failed_outcomes': failed_outcomes,
                        'pending_outcomes': pending_outcomes,
                        'occurrence_count': pattern.occurrence_count,
                        'last_seen': pattern.last_seen,
                        'training_samples': pattern.training_samples
                    }
                
                return pattern_stats
                
        except Exception as e:
            logger.error(f"Failed to get pattern success rates: {e}")
            return {}
    
    def train_custom_patterns(self, user_id: str) -> Dict[str, bool]:
        """
        Train ML models for user's custom patterns
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict mapping pattern names to training success status
        """
        try:
            with self._lock:
                results = {}
                
                user_patterns = [
                    pattern for pattern in self.custom_patterns.values()
                    if pattern.user_id == user_id
                ]
                
                for pattern in user_patterns:
                    success = self._train_pattern_model(user_id, pattern.pattern_name)
                    results[pattern.pattern_name] = success
                
                logger.info(f"Trained {sum(results.values())} patterns for user {user_id}")
                return results
                
        except Exception as e:
            logger.error(f"Failed to train custom patterns: {e}")
            return {}
    
    def get_user_pattern_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of user's pattern learning
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing pattern learning summary
        """
        try:
            with self._lock:
                user_annotations = [
                    ann for ann in self.pattern_annotations.values()
                    if ann.user_id == user_id
                ]
                
                user_patterns = [
                    pattern for pattern in self.custom_patterns.values()
                    if pattern.user_id == user_id
                ]
                
                # Calculate overall statistics
                total_annotations = len(user_annotations)
                successful_patterns = len([
                    ann for ann in user_annotations
                    if ann.success_outcome is True
                ])
                
                overall_success_rate = 0.0
                if total_annotations > 0:
                    overall_success_rate = (successful_patterns / total_annotations) * 100
                
                # Get top performing patterns
                top_patterns = sorted(
                    user_patterns,
                    key=lambda x: (x.success_rate, x.occurrence_count),
                    reverse=True
                )[:5]
                
                return {
                    'user_id': user_id,
                    'timestamp': datetime.now(),
                    'total_annotations': total_annotations,
                    'total_patterns': len(user_patterns),
                    'overall_success_rate': overall_success_rate,
                    'trained_models': len([
                        key for key in self.pattern_classifiers.keys()
                        if key.startswith(f"{user_id}_")
                    ]),
                    'top_patterns': [
                        {
                            'name': pattern.pattern_name,
                            'success_rate': pattern.success_rate,
                            'occurrence_count': pattern.occurrence_count
                        }
                        for pattern in top_patterns
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get user pattern summary: {e}")
            return {'error': str(e)}
    
    def _extract_pattern_features(self, chart_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from chart data for pattern recognition"""
        try:
            features = {}
            
            if 'ohlcv' in chart_data:
                ohlcv = chart_data['ohlcv']
                if len(ohlcv) >= 20:  # Need minimum data for features
                    
                    # Price features
                    closes = [candle[4] for candle in ohlcv[-20:]]  # Last 20 closes
                    highs = [candle[2] for candle in ohlcv[-20:]]
                    lows = [candle[3] for candle in ohlcv[-20:]]
                    volumes = [candle[5] for candle in ohlcv[-20:]]
                    
                    # Technical indicators
                    features['price_change_5'] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
                    features['price_change_10'] = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0
                    features['price_change_20'] = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
                    
                    # Volatility
                    if len(closes) >= 10:
                        price_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                        features['volatility'] = np.std(price_changes)
                    
                    # Volume features
                    features['volume_avg'] = np.mean(volumes)
                    features['volume_ratio'] = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1.0
                    
                    # Range features
                    features['high_low_ratio'] = (max(highs) - min(lows)) / min(lows) if min(lows) > 0 else 0
                    features['close_position'] = (closes[-1] - min(lows)) / (max(highs) - min(lows)) if max(highs) != min(lows) else 0.5
                    
                    # Trend features
                    if len(closes) >= 5:
                        recent_trend = np.polyfit(range(5), closes[-5:], 1)[0]
                        features['recent_trend'] = recent_trend / closes[-1] if closes[-1] != 0 else 0
            
            # Additional features from forecast data if available
            if 'regime' in chart_data:
                regime_map = {'bullish': 1.0, 'bearish': -1.0, 'sideways': 0.0}
                features['regime'] = regime_map.get(chart_data['regime'], 0.0)
            
            if 'confidence' in chart_data:
                features['confidence'] = chart_data['confidence'] / 100.0
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract pattern features: {e}")
            return {}
    
    def _update_custom_pattern(self, user_id: str, pattern_name: str, annotation: PatternAnnotation):
        """Update or create custom pattern from annotation"""
        try:
            pattern_key = f"{user_id}_{pattern_name}"
            
            if pattern_key in self.custom_patterns:
                pattern = self.custom_patterns[pattern_key]
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now()
                
                # Update feature weights (simple averaging)
                for feature, value in annotation.pattern_features.items():
                    if feature in pattern.feature_weights:
                        pattern.feature_weights[feature] = (pattern.feature_weights[feature] + value) / 2
                    else:
                        pattern.feature_weights[feature] = value
            
            else:
                pattern = CustomPattern(
                    pattern_id=pattern_key,
                    user_id=user_id,
                    pattern_name=pattern_name,
                    description=f"Custom pattern: {pattern_name}",
                    feature_weights=annotation.pattern_features.copy(),
                    success_rate=0.0,
                    occurrence_count=1,
                    last_seen=datetime.now(),
                    training_samples=0
                )
                self.custom_patterns[pattern_key] = pattern
            
        except Exception as e:
            logger.error(f"Failed to update custom pattern: {e}")
    
    def _train_pattern_model(self, user_id: str, pattern_name: str) -> bool:
        """Train ML model for a specific pattern"""
        try:
            # Get training data
            pattern_annotations = [
                ann for ann in self.pattern_annotations.values()
                if (ann.user_id == user_id and
                    ann.pattern_name == pattern_name and
                    ann.success_outcome is not None)
            ]
            
            if len(pattern_annotations) < 5:
                logger.warning(f"Not enough training data for pattern {pattern_name} (need 5, have {len(pattern_annotations)})")
                return False
            
            # Prepare training data
            X = []
            y = []
            
            for ann in pattern_annotations:
                feature_vector = self._prepare_feature_vector(ann.pattern_features)
                X.append(feature_vector)
                y.append(1 if ann.success_outcome else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            model_key = f"{user_id}_{pattern_name}"
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train classifier
            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            classifier.fit(X_scaled, y)
            
            # Store model and scaler
            self.pattern_classifiers[model_key] = classifier
            self.feature_scalers[model_key] = scaler
            
            # Update pattern training samples count
            pattern_key = f"{user_id}_{pattern_name}"
            if pattern_key in self.custom_patterns:
                self.custom_patterns[pattern_key].training_samples = len(pattern_annotations)
            
            # Save model
            self._save_pattern_model(model_key, classifier, scaler)
            
            logger.info(f"Trained pattern model: {pattern_name} for user {user_id} with {len(pattern_annotations)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train pattern model: {e}")
            return False
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector for ML model"""
        # Define standard feature order
        standard_features = [
            'price_change_5', 'price_change_10', 'price_change_20',
            'volatility', 'volume_avg', 'volume_ratio',
            'high_low_ratio', 'close_position', 'recent_trend',
            'regime', 'confidence'
        ]
        
        vector = []
        for feature in standard_features:
            vector.append(features.get(feature, 0.0))
        
        return vector
    
    def _calculate_feature_similarity(self, current_features: Dict[str, float],
                                    pattern_weights: Dict[str, float]) -> float:
        """Calculate similarity between current features and pattern weights"""
        try:
            if not pattern_weights:
                return 0.0
            
            similarities = []
            for feature, weight in pattern_weights.items():
                if feature in current_features:
                    # Calculate normalized similarity
                    current_value = current_features[feature]
                    similarity = 1.0 - abs(current_value - weight) / (abs(current_value) + abs(weight) + 1e-8)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate feature similarity: {e}")
            return 0.0
    
    def _save_annotation(self, annotation: PatternAnnotation):
        """Save annotation to disk"""
        try:
            file_path = self.data_dir / f"annotation_{annotation.annotation_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(annotation)
                data['timestamp'] = annotation.timestamp.isoformat()
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save annotation: {e}")
    
    def _save_pattern_model(self, model_key: str, classifier: RandomForestClassifier, scaler: StandardScaler):
        """Save trained model to disk"""
        try:
            model_dir = self.data_dir / "models"
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(classifier, model_dir / f"{model_key}_classifier.pkl")
            joblib.dump(scaler, model_dir / f"{model_key}_scaler.pkl")
            
        except Exception as e:
            logger.error(f"Failed to save pattern model: {e}")
    
    def _load_pattern_data(self):
        """Load pattern data from disk"""
        try:
            if not self.data_dir.exists():
                return
            
            # Load annotations
            annotation_count = 0
            for file_path in self.data_dir.glob("annotation_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    annotation = PatternAnnotation(**data)
                    self.pattern_annotations[annotation.annotation_id] = annotation
                    annotation_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load annotation from {file_path}: {e}")
            
            # Rebuild custom patterns from annotations
            self._rebuild_custom_patterns()
            
            # Load trained models
            model_dir = self.data_dir / "models"
            if model_dir.exists():
                for classifier_file in model_dir.glob("*_classifier.pkl"):
                    try:
                        model_key = classifier_file.stem.replace("_classifier", "")
                        scaler_file = model_dir / f"{model_key}_scaler.pkl"
                        
                        if scaler_file.exists():
                            classifier = joblib.load(classifier_file)
                            scaler = joblib.load(scaler_file)
                            
                            self.pattern_classifiers[model_key] = classifier
                            self.feature_scalers[model_key] = scaler
                            
                    except Exception as e:
                        logger.warning(f"Failed to load model from {classifier_file}: {e}")
            
            logger.info(f"Loaded {annotation_count} annotations and {len(self.pattern_classifiers)} trained models")
            
        except Exception as e:
            logger.error(f"Failed to load pattern data: {e}")
    
    def _rebuild_custom_patterns(self):
        """Rebuild custom patterns from loaded annotations"""
        try:
            pattern_groups = {}
            
            # Group annotations by user and pattern name
            for annotation in self.pattern_annotations.values():
                key = f"{annotation.user_id}_{annotation.pattern_name}"
                if key not in pattern_groups:
                    pattern_groups[key] = []
                pattern_groups[key].append(annotation)
            
            # Create custom patterns
            for pattern_key, annotations in pattern_groups.items():
                user_id, pattern_name = pattern_key.split("_", 1)
                
                # Calculate success rate
                successful = len([ann for ann in annotations if ann.success_outcome is True])
                total_with_outcome = len([ann for ann in annotations if ann.success_outcome is not None])
                success_rate = (successful / total_with_outcome * 100) if total_with_outcome > 0 else 0.0
                
                # Average feature weights
                feature_weights = {}
                for annotation in annotations:
                    for feature, value in annotation.pattern_features.items():
                        if feature not in feature_weights:
                            feature_weights[feature] = []
                        feature_weights[feature].append(value)
                
                # Calculate averages
                avg_feature_weights = {
                    feature: np.mean(values)
                    for feature, values in feature_weights.items()
                }
                
                custom_pattern = CustomPattern(
                    pattern_id=pattern_key,
                    user_id=user_id,
                    pattern_name=pattern_name,
                    description=f"Custom pattern: {pattern_name}",
                    feature_weights=avg_feature_weights,
                    success_rate=success_rate,
                    occurrence_count=len(annotations),
                    last_seen=max(ann.timestamp for ann in annotations),
                    training_samples=total_with_outcome
                )
                
                self.custom_patterns[pattern_key] = custom_pattern
            
        except Exception as e:
            logger.error(f"Failed to rebuild custom patterns: {e}")


# Global instance
_pattern_learner = None

def get_pattern_learner() -> CustomPatternLearner:
    """Get global pattern learner instance"""
    global _pattern_learner
    if _pattern_learner is None:
        _pattern_learner = CustomPatternLearner()
    return _pattern_learner
