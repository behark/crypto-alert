"""
Regime Change Prediction Engine
==============================
Forecast potential regime shifts before they confirm using advanced pattern analysis
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Market regime enumeration"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    BULL_CONSOLIDATION = "bull_consolidation"
    BEAR_CONSOLIDATION = "bear_consolidation"
    SIDEWAYS = "sideways"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_COMPRESSION = "volatility_compression"

class RegimeShiftType(Enum):
    """Regime shift type enumeration"""
    TREND_REVERSAL = "trend_reversal"
    TREND_CONTINUATION = "trend_continuation"
    CONSOLIDATION_BREAK = "consolidation_break"
    VOLATILITY_SPIKE = "volatility_spike"
    MOMENTUM_SHIFT = "momentum_shift"

@dataclass
class RegimeState:
    """Current regime state"""
    symbol: str
    current_regime: RegimeType
    regime_strength: float
    regime_duration: timedelta
    regime_confidence: float
    key_characteristics: List[str]
    supporting_indicators: Dict[str, float]
    timestamp: datetime

@dataclass
class RegimeShiftSignal:
    """Regime shift prediction signal"""
    signal_id: str
    symbol: str
    current_regime: RegimeType
    predicted_regime: RegimeType
    shift_type: RegimeShiftType
    probability: float
    confidence: float
    time_horizon: timedelta
    trigger_conditions: List[str]
    recommended_actions: List[str]
    created_at: datetime

class RegimePredictor:
    """
    Regime Change Prediction Engine
    
    Forecasts potential regime shifts before they confirm using volatility compression,
    pattern exhaustion, macro divergence analysis, and behavioral pattern recognition.
    """
    
    def __init__(self, data_dir: str = "data/regime_prediction"):
        """Initialize the regime predictor"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Current regime states
        self.regime_states: Dict[str, RegimeState] = {}
        self.regime_history: Dict[str, List[RegimeState]] = {}
        
        # Shift predictions
        self.shift_signals: Dict[str, List[RegimeShiftSignal]] = {}
        
        # Detection parameters
        self.regime_thresholds = {
            'trend_strength_min': 0.3,
            'volatility_expansion_min': 0.05,
            'momentum_threshold': 0.4
        }
        
        self.shift_detection = {
            'volatility_compression_threshold': 0.7,
            'pattern_exhaustion_threshold': 0.8,
            'divergence_significance_min': 0.6
        }
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Regime Predictor initialized")
    
    def analyze_regime_state(self, symbol: str, price_data: pd.DataFrame) -> RegimeState:
        """
        Analyze current regime state for symbol
        
        Args:
            symbol: Trading symbol
            price_data: Price data with OHLCV
            
        Returns:
            RegimeState: Current regime analysis
        """
        try:
            if len(price_data) < 50:
                logger.warning(f"Insufficient data for regime analysis: {symbol}")
                return None
            
            # Calculate regime indicators
            indicators = self._calculate_regime_indicators(price_data)
            
            # Determine current regime
            current_regime = self._classify_regime(indicators)
            
            # Calculate regime strength and confidence
            regime_strength = self._calculate_regime_strength(indicators, current_regime)
            regime_confidence = self._calculate_regime_confidence(indicators, price_data)
            
            # Calculate regime duration
            regime_duration = self._calculate_regime_duration(symbol, current_regime)
            
            # Identify key characteristics
            key_characteristics = self._identify_regime_characteristics(indicators, current_regime)
            
            # Supporting indicators
            supporting_indicators = {
                'trend_strength': indicators.get('trend_strength', 0.0),
                'volatility_regime': indicators.get('volatility_regime', 0.0),
                'momentum_persistence': indicators.get('momentum_persistence', 0.0),
                'volume_confirmation': indicators.get('volume_confirmation', 0.0)
            }
            
            # Create regime state
            regime_state = RegimeState(
                symbol=symbol,
                current_regime=current_regime,
                regime_strength=regime_strength,
                regime_duration=regime_duration,
                regime_confidence=regime_confidence,
                key_characteristics=key_characteristics,
                supporting_indicators=supporting_indicators,
                timestamp=datetime.now()
            )
            
            # Store regime state
            self.regime_states[symbol] = regime_state
            
            # Update regime history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            
            self.regime_history[symbol].append(regime_state)
            
            # Keep only recent history
            if len(self.regime_history[symbol]) > 1000:
                self.regime_history[symbol] = self.regime_history[symbol][-1000:]
            
            logger.info(f"Analyzed regime for {symbol}: {current_regime.value} (strength: {regime_strength:.2f})")
            return regime_state
            
        except Exception as e:
            logger.error(f"Failed to analyze regime state: {e}")
            return None
    
    def predict_regime_shift(self, symbol: str, price_data: pd.DataFrame,
                           sentiment_data: Dict[str, Any] = None) -> List[RegimeShiftSignal]:
        """
        Predict potential regime shifts
        
        Args:
            symbol: Trading symbol
            price_data: Price data for analysis
            sentiment_data: Sentiment analysis data
            
        Returns:
            List of regime shift signals
        """
        try:
            if symbol not in self.regime_states:
                logger.warning(f"No regime state available for {symbol}")
                return []
            
            current_regime = self.regime_states[symbol]
            shift_signals = []
            
            # 1. Volatility compression analysis
            compression_signal = self._detect_volatility_compression(symbol, price_data)
            if compression_signal:
                shift_signals.append(compression_signal)
            
            # 2. Pattern exhaustion analysis
            exhaustion_signal = self._detect_pattern_exhaustion(symbol, price_data, current_regime)
            if exhaustion_signal:
                shift_signals.append(exhaustion_signal)
            
            # 3. Momentum shift analysis
            momentum_signal = self._detect_momentum_shift(symbol, price_data, current_regime)
            if momentum_signal:
                shift_signals.append(momentum_signal)
            
            # Store shift signals
            self.shift_signals[symbol] = shift_signals
            
            # Filter and rank signals by probability
            significant_signals = [s for s in shift_signals if s.probability > 0.3]
            significant_signals.sort(key=lambda x: x.probability, reverse=True)
            
            logger.info(f"Generated {len(significant_signals)} regime shift signals for {symbol}")
            return significant_signals
            
        except Exception as e:
            logger.error(f"Failed to predict regime shift: {e}")
            return []
    
    def get_regime_forecast(self, symbol: str, forecast_horizon: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get regime forecast for symbol"""
        try:
            if symbol not in self.regime_states:
                return {'error': 'No regime state available'}
            
            current_regime = self.regime_states[symbol]
            shift_signals = self.shift_signals.get(symbol, [])
            
            # Filter signals within forecast horizon
            relevant_signals = [
                s for s in shift_signals 
                if s.time_horizon <= forecast_horizon and s.probability > 0.2
            ]
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(symbol, current_regime)
            
            # Predict most likely regime
            if relevant_signals:
                highest_prob_signal = max(relevant_signals, key=lambda x: x.probability)
                predicted_regime = highest_prob_signal.predicted_regime.value
                shift_probability = highest_prob_signal.probability
                shift_type = highest_prob_signal.shift_type.value
            else:
                predicted_regime = current_regime.current_regime.value
                shift_probability = 0.0
                shift_type = 'none'
            
            return {
                'current_regime': current_regime.current_regime.value,
                'regime_strength': current_regime.regime_strength,
                'regime_confidence': current_regime.regime_confidence,
                'regime_stability': regime_stability,
                'predicted_regime': predicted_regime,
                'shift_probability': shift_probability,
                'shift_type': shift_type,
                'forecast_horizon_days': forecast_horizon.days,
                'active_signals': len(relevant_signals),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get regime forecast: {e}")
            return {'error': str(e)}
    
    def get_regime_status(self) -> Dict[str, Any]:
        """Get regime predictor status"""
        try:
            analyzed_symbols = len(self.regime_states)
            total_signals = sum(len(signals) for signals in self.shift_signals.values())
            
            # Regime distribution
            regime_distribution = {}
            for state in self.regime_states.values():
                regime = state.current_regime.value
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
            
            return {
                'analyzed_symbols': analyzed_symbols,
                'total_shift_signals': total_signals,
                'regime_distribution': regime_distribution,
                'monitoring_active': not self._should_stop,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get regime status: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_indicators(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime classification indicators"""
        try:
            indicators = {}
            
            # Trend strength indicator
            sma_20 = price_data['close'].rolling(20).mean()
            sma_50 = price_data['close'].rolling(50).mean()
            current_price = price_data['close'].iloc[-1]
            
            trend_alignment = 0.0
            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend_alignment = 1.0
            elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                trend_alignment = -1.0
            
            price_change_20 = (current_price - price_data['close'].iloc[-21]) / price_data['close'].iloc[-21] if len(price_data) > 21 else 0
            indicators['trend_strength'] = abs(price_change_20) * abs(trend_alignment)
            
            # Volatility regime
            returns = price_data['close'].pct_change().dropna()
            current_vol = returns.iloc[-20:].std() * np.sqrt(252)
            avg_vol = returns.std() * np.sqrt(252)
            indicators['volatility_regime'] = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Momentum persistence
            momentum_periods = [5, 10, 20]
            momentum_scores = []
            for period in momentum_periods:
                if len(price_data) > period:
                    momentum = (price_data['close'].iloc[-1] - price_data['close'].iloc[-period-1]) / price_data['close'].iloc[-period-1]
                    momentum_scores.append(momentum)
            
            if momentum_scores:
                momentum_consistency = 1.0 - np.std(momentum_scores) if len(momentum_scores) > 1 else 1.0
                indicators['momentum_persistence'] = momentum_consistency * np.mean([abs(m) for m in momentum_scores])
            else:
                indicators['momentum_persistence'] = 0.0
            
            # Volume confirmation
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'].iloc[-10:].mean()
                avg_volume = price_data['volume'].mean()
                indicators['volume_confirmation'] = recent_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                indicators['volume_confirmation'] = 1.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate regime indicators: {e}")
            return {}
    
    def _classify_regime(self, indicators: Dict[str, float]) -> RegimeType:
        """Classify current market regime"""
        try:
            trend_strength = indicators.get('trend_strength', 0.0)
            volatility_regime = indicators.get('volatility_regime', 1.0)
            momentum_persistence = indicators.get('momentum_persistence', 0.0)
            
            # Volatility-based classification
            if volatility_regime > 2.0:
                return RegimeType.VOLATILITY_EXPANSION
            elif volatility_regime < 0.5:
                return RegimeType.VOLATILITY_COMPRESSION
            
            # Trend-based classification
            if trend_strength > self.regime_thresholds['trend_strength_min']:
                if momentum_persistence > 0:
                    return RegimeType.BULL_TREND
                else:
                    return RegimeType.BEAR_TREND
            
            # Consolidation classification
            if volatility_regime < 1.2 and trend_strength < 0.2:
                if momentum_persistence > 0:
                    return RegimeType.BULL_CONSOLIDATION
                elif momentum_persistence < 0:
                    return RegimeType.BEAR_CONSOLIDATION
                else:
                    return RegimeType.SIDEWAYS
            
            return RegimeType.SIDEWAYS
            
        except Exception as e:
            logger.error(f"Failed to classify regime: {e}")
            return RegimeType.SIDEWAYS
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], regime: RegimeType) -> float:
        """Calculate regime strength"""
        try:
            trend_strength = indicators.get('trend_strength', 0.0)
            volatility_regime = indicators.get('volatility_regime', 1.0)
            momentum_persistence = indicators.get('momentum_persistence', 0.0)
            
            base_strength = min(trend_strength * 2, 1.0)
            
            if regime in [RegimeType.VOLATILITY_EXPANSION, RegimeType.VOLATILITY_COMPRESSION]:
                vol_strength = abs(volatility_regime - 1.0)
                base_strength = max(base_strength, vol_strength)
            
            momentum_boost = min(abs(momentum_persistence), 0.3)
            regime_strength = base_strength + momentum_boost
            
            return float(np.clip(regime_strength, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate regime strength: {e}")
            return 0.5
    
    def _calculate_regime_confidence(self, indicators: Dict[str, float], price_data: pd.DataFrame) -> float:
        """Calculate confidence in regime classification"""
        try:
            data_quality = min(len(price_data) / 100, 1.0)
            
            indicator_values = list(indicators.values())
            if len(indicator_values) > 1:
                consistency = 1.0 - np.std(indicator_values) / (np.mean(indicator_values) + 0.1)
            else:
                consistency = 0.5
            
            confidence = (data_quality * 0.6 + consistency * 0.4)
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate regime confidence: {e}")
            return 0.5
    
    def _calculate_regime_duration(self, symbol: str, current_regime: RegimeType) -> timedelta:
        """Calculate how long current regime has been active"""
        try:
            if symbol not in self.regime_history:
                return timedelta(hours=1)
            
            history = self.regime_history[symbol]
            if not history:
                return timedelta(hours=1)
            
            regime_start = datetime.now()
            for i in range(len(history) - 1, -1, -1):
                if history[i].current_regime == current_regime:
                    regime_start = history[i].timestamp
                else:
                    break
            
            return datetime.now() - regime_start
            
        except Exception as e:
            logger.error(f"Failed to calculate regime duration: {e}")
            return timedelta(hours=1)
    
    def _identify_regime_characteristics(self, indicators: Dict[str, float], regime: RegimeType) -> List[str]:
        """Identify key characteristics of current regime"""
        try:
            characteristics = []
            
            trend_strength = indicators.get('trend_strength', 0.0)
            volatility_regime = indicators.get('volatility_regime', 1.0)
            
            if trend_strength > 0.5:
                characteristics.append("Strong directional trend")
            elif trend_strength > 0.2:
                characteristics.append("Moderate trend")
            else:
                characteristics.append("Weak trend")
            
            if volatility_regime > 1.5:
                characteristics.append("High volatility environment")
            elif volatility_regime < 0.7:
                characteristics.append("Low volatility environment")
            else:
                characteristics.append("Normal volatility")
            
            if regime == RegimeType.VOLATILITY_COMPRESSION:
                characteristics.append("Potential breakout setup")
            elif regime == RegimeType.VOLATILITY_EXPANSION:
                characteristics.append("High uncertainty period")
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Failed to identify regime characteristics: {e}")
            return ["Unknown characteristics"]
    
    def _detect_volatility_compression(self, symbol: str, price_data: pd.DataFrame) -> Optional[RegimeShiftSignal]:
        """Detect volatility compression leading to potential breakout"""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            vol_window = 20
            rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
            
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            
            compression_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            if compression_ratio < self.shift_detection['volatility_compression_threshold']:
                compression_periods = 0
                for i in range(len(rolling_vol) - 1, -1, -1):
                    if rolling_vol.iloc[i] / avg_vol < 0.8:
                        compression_periods += 1
                    else:
                        break
                
                probability = min(0.3 + (compression_periods / 50), 0.9)
                
                signal = RegimeShiftSignal(
                    signal_id=f"vol_compression_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    current_regime=self.regime_states[symbol].current_regime,
                    predicted_regime=RegimeType.VOLATILITY_EXPANSION,
                    shift_type=RegimeShiftType.VOLATILITY_SPIKE,
                    probability=probability,
                    confidence=0.8,
                    time_horizon=timedelta(days=3),
                    trigger_conditions=[f"Volatility compressed to {compression_ratio:.2f}x average"],
                    recommended_actions=["Prepare for breakout", "Set wider stops", "Reduce position size"],
                    created_at=datetime.now()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect volatility compression: {e}")
            return None
    
    def _detect_pattern_exhaustion(self, symbol: str, price_data: pd.DataFrame, 
                                 current_regime: RegimeState) -> Optional[RegimeShiftSignal]:
        """Detect pattern exhaustion suggesting regime change"""
        try:
            exhaustion_score = 0.0
            exhaustion_factors = []
            
            # Momentum divergence
            if len(price_data) > 30:
                recent_momentum = price_data['close'].pct_change(10).iloc[-5:].mean()
                previous_momentum = price_data['close'].pct_change(10).iloc[-25:-20].mean()
                
                if abs(recent_momentum) < abs(previous_momentum) * 0.5:
                    exhaustion_score += 0.3
                    exhaustion_factors.append("Momentum divergence detected")
            
            # Volume exhaustion
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'].iloc[-10:].mean()
                trend_volume = price_data['volume'].iloc[-30:-10].mean()
                
                if recent_volume < trend_volume * 0.7:
                    exhaustion_score += 0.2
                    exhaustion_factors.append("Volume exhaustion")
            
            # Regime duration factor
            if current_regime.regime_duration > timedelta(days=30):
                exhaustion_score += 0.2
                exhaustion_factors.append("Extended regime duration")
            
            # Trend strength weakening
            if current_regime.regime_strength < 0.4:
                exhaustion_score += 0.3
                exhaustion_factors.append("Weakening trend strength")
            
            if exhaustion_score > self.shift_detection['pattern_exhaustion_threshold']:
                if current_regime.current_regime in [RegimeType.BULL_TREND, RegimeType.BULL_CONSOLIDATION]:
                    predicted_regime = RegimeType.BEAR_TREND
                elif current_regime.current_regime in [RegimeType.BEAR_TREND, RegimeType.BEAR_CONSOLIDATION]:
                    predicted_regime = RegimeType.BULL_TREND
                else:
                    predicted_regime = RegimeType.VOLATILITY_EXPANSION
                
                signal = RegimeShiftSignal(
                    signal_id=f"pattern_exhaustion_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    current_regime=current_regime.current_regime,
                    predicted_regime=predicted_regime,
                    shift_type=RegimeShiftType.TREND_REVERSAL,
                    probability=exhaustion_score,
                    confidence=0.7,
                    time_horizon=timedelta(days=7),
                    trigger_conditions=exhaustion_factors,
                    recommended_actions=["Reduce trend exposure", "Prepare for reversal"],
                    created_at=datetime.now()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect pattern exhaustion: {e}")
            return None
    
    def _detect_momentum_shift(self, symbol: str, price_data: pd.DataFrame,
                             current_regime: RegimeState) -> Optional[RegimeShiftSignal]:
        """Detect momentum shifts"""
        try:
            if len(price_data) < 20:
                return None
            
            # Calculate momentum indicators
            short_momentum = price_data['close'].pct_change(5).iloc[-1]
            medium_momentum = price_data['close'].pct_change(10).iloc[-1]
            long_momentum = price_data['close'].pct_change(20).iloc[-1]
            
            # Check for momentum alignment changes
            momentum_alignment_score = 0.0
            if short_momentum * medium_momentum < 0:  # Short vs medium divergence
                momentum_alignment_score += 0.3
            if medium_momentum * long_momentum < 0:  # Medium vs long divergence
                momentum_alignment_score += 0.4
            
            if momentum_alignment_score > 0.5:
                signal = RegimeShiftSignal(
                    signal_id=f"momentum_shift_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    current_regime=current_regime.current_regime,
                    predicted_regime=RegimeType.SIDEWAYS,
                    shift_type=RegimeShiftType.MOMENTUM_SHIFT,
                    probability=momentum_alignment_score,
                    confidence=0.6,
                    time_horizon=timedelta(days=5),
                    trigger_conditions=["Momentum divergence across timeframes"],
                    recommended_actions=["Monitor for direction confirmation"],
                    created_at=datetime.now()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect momentum shift: {e}")
            return None
    
    def _calculate_regime_stability(self, symbol: str, current_regime: RegimeState) -> float:
        """Calculate regime stability"""
        try:
            # Base stability from regime strength
            base_stability = current_regime.regime_strength
            
            # Duration factor (longer regimes are more stable)
            duration_days = current_regime.regime_duration.days
            duration_factor = min(duration_days / 30, 1.0)  # Max at 30 days
            
            # Confidence factor
            confidence_factor = current_regime.regime_confidence
            
            # Combined stability
            stability = (base_stability * 0.4 + duration_factor * 0.3 + confidence_factor * 0.3)
            
            return float(np.clip(stability, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate regime stability: {e}")
            return 0.5
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="RegimeMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started regime monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Clean up old signals
                current_time = datetime.now()
                for symbol in list(self.shift_signals.keys()):
                    self.shift_signals[symbol] = [
                        signal for signal in self.shift_signals[symbol]
                        if current_time - signal.created_at < timedelta(hours=24)
                    ]
                
                # Sleep
                threading.Event().wait(300.0)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in regime monitoring: {e}")
    
    def stop(self):
        """Stop the regime predictor"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Regime Predictor stopped")


# Global instance
_regime_predictor = None

def get_regime_predictor() -> RegimePredictor:
    """Get global regime predictor instance"""
    global _regime_predictor
    if _regime_predictor is None:
        _regime_predictor = RegimePredictor()
    return _regime_predictor
