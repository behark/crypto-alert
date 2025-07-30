"""
Multi-Timeframe Predictive Intelligence
======================================
Advanced multi-timeframe analysis with correlation engines and predictive modeling
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

class TimeframeType(Enum):
    """Timeframe enumeration"""
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class TrendStrength(Enum):
    """Trend strength enumeration"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TimeframeAnalysis:
    """Analysis for specific timeframe"""
    timeframe: TimeframeType
    timestamp: datetime
    trend_direction: str  # bullish/bearish/neutral
    trend_strength: TrendStrength
    volatility: float
    momentum_score: float
    mean_reversion_score: float
    breakout_probability: float
    confidence: float
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: Dict[str, float]

@dataclass
class CrossTimeframeSignal:
    """Cross-timeframe correlation signal"""
    signal_id: str
    primary_timeframe: TimeframeType
    supporting_timeframes: List[TimeframeType]
    signal_type: str  # confluence/divergence/breakout/reversal
    strength: float
    confidence: float
    bias_direction: str
    entry_zones: List[float]
    target_zones: List[float]
    reasoning: str
    created_at: datetime

class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Predictive Intelligence
    
    Analyzes market conditions across multiple timeframes to generate
    predictive signals with correlation engines and behavioral insights.
    """
    
    def __init__(self, data_dir: str = "data/multi_timeframe"):
        """Initialize the multi-timeframe analyzer"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported timeframes
        self.timeframes = [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4, TimeframeType.D1]
        
        # Current analysis cache
        self.timeframe_analysis: Dict[str, Dict[TimeframeType, TimeframeAnalysis]] = {}
        self.cross_timeframe_signals: Dict[str, List[CrossTimeframeSignal]] = {}
        
        # Analysis history
        self.analysis_history: Dict[str, List[TimeframeAnalysis]] = {}
        self.signal_history: List[CrossTimeframeSignal] = []
        
        # Configuration
        self.correlation_threshold = 0.7
        self.confluence_min_timeframes = 2
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Multi-Timeframe Analyzer initialized")
    
    def analyze_symbol(self, symbol: str, price_data: Dict[TimeframeType, pd.DataFrame]) -> Dict[TimeframeType, TimeframeAnalysis]:
        """
        Analyze symbol across multiple timeframes
        
        Args:
            symbol: Trading symbol
            price_data: Price data for each timeframe
            
        Returns:
            Dict of timeframe analyses
        """
        try:
            analyses = {}
            
            for timeframe in self.timeframes:
                if timeframe not in price_data or price_data[timeframe].empty:
                    continue
                
                # Perform timeframe-specific analysis
                analysis = self._analyze_timeframe(symbol, timeframe, price_data[timeframe])
                if analysis:
                    analyses[timeframe] = analysis
            
            # Store analyses
            self.timeframe_analysis[symbol] = analyses
            
            # Update analysis history
            if symbol not in self.analysis_history:
                self.analysis_history[symbol] = []
            
            for analysis in analyses.values():
                self.analysis_history[symbol].append(analysis)
                
            # Keep only recent history
            if len(self.analysis_history[symbol]) > 1000:
                self.analysis_history[symbol] = self.analysis_history[symbol][-1000:]
            
            # Generate cross-timeframe signals
            self._generate_cross_timeframe_signals(symbol, analyses)
            
            logger.info(f"Analyzed {symbol} across {len(analyses)} timeframes")
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol {symbol}: {e}")
            return {}
    
    def get_predictive_bias(self, symbol: str, target_timeframe: TimeframeType) -> Dict[str, Any]:
        """
        Get predictive bias for symbol and timeframe
        
        Args:
            symbol: Trading symbol
            target_timeframe: Target timeframe for prediction
            
        Returns:
            Dict containing predictive bias information
        """
        try:
            if symbol not in self.timeframe_analysis:
                return {'bias': 'neutral', 'confidence': 0.0, 'reasoning': 'No analysis available'}
            
            analyses = self.timeframe_analysis[symbol]
            
            # Get higher timeframe context
            higher_timeframes = self._get_higher_timeframes(target_timeframe)
            bias_scores = []
            confidence_scores = []
            reasoning_parts = []
            
            for htf in higher_timeframes:
                if htf in analyses:
                    analysis = analyses[htf]
                    
                    # Calculate bias score (-1 to 1)
                    if analysis.trend_direction == 'bullish':
                        bias_score = 0.5 + (analysis.momentum_score * 0.5)
                    elif analysis.trend_direction == 'bearish':
                        bias_score = -0.5 - (analysis.momentum_score * 0.5)
                    else:
                        bias_score = 0.0
                    
                    # Weight by trend strength
                    strength_weight = self._get_strength_weight(analysis.trend_strength)
                    weighted_bias = bias_score * strength_weight
                    
                    bias_scores.append(weighted_bias)
                    confidence_scores.append(analysis.confidence * strength_weight)
                    reasoning_parts.append(f"{htf.value}: {analysis.trend_direction} ({analysis.trend_strength.value})")
            
            # Calculate overall bias
            if bias_scores:
                overall_bias_score = np.mean(bias_scores)
                overall_confidence = np.mean(confidence_scores)
                
                if overall_bias_score > 0.3:
                    bias = 'bullish'
                elif overall_bias_score < -0.3:
                    bias = 'bearish'
                else:
                    bias = 'neutral'
            else:
                bias = 'neutral'
                overall_bias_score = 0.0
                overall_confidence = 0.0
                reasoning_parts = ['No higher timeframe data available']
            
            return {
                'bias': bias,
                'bias_score': overall_bias_score,
                'confidence': overall_confidence,
                'reasoning': '; '.join(reasoning_parts),
                'supporting_timeframes': len(bias_scores),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get predictive bias: {e}")
            return {'bias': 'neutral', 'confidence': 0.0, 'reasoning': f'Error: {str(e)}'}
    
    def detect_regime_shift_probability(self, symbol: str) -> Dict[str, Any]:
        """
        Detect probability of regime shift
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing regime shift analysis
        """
        try:
            if symbol not in self.timeframe_analysis:
                return {'probability': 0.0, 'type': 'none', 'confidence': 0.0}
            
            analyses = self.timeframe_analysis[symbol]
            
            # Check for regime shift indicators
            shift_indicators = []
            
            # 1. Volatility compression across timeframes
            volatility_compression = self._detect_volatility_compression(analyses)
            if volatility_compression > 0.5:
                shift_indicators.append({'type': 'volatility_compression', 'strength': volatility_compression})
            
            # 2. Cross-timeframe divergence
            divergence = self._detect_cross_timeframe_divergence(analyses)
            if divergence > 0.5:
                shift_indicators.append({'type': 'cross_timeframe_divergence', 'strength': divergence})
            
            # 3. Momentum exhaustion
            momentum_exhaustion = self._detect_momentum_exhaustion(analyses)
            if momentum_exhaustion > 0.5:
                shift_indicators.append({'type': 'momentum_exhaustion', 'strength': momentum_exhaustion})
            
            # Calculate overall probability
            if shift_indicators:
                total_strength = sum(indicator['strength'] for indicator in shift_indicators)
                probability = min(total_strength / len(shift_indicators), 1.0)
                
                # Determine shift type
                if any(i['type'] == 'momentum_exhaustion' for i in shift_indicators):
                    shift_type = 'trend_reversal'
                elif any(i['type'] == 'volatility_compression' for i in shift_indicators):
                    shift_type = 'breakout_pending'
                else:
                    shift_type = 'momentum_shift'
                
                confidence = probability * 0.8
            else:
                probability = 0.0
                shift_type = 'none'
                confidence = 0.0
            
            return {
                'probability': probability,
                'type': shift_type,
                'confidence': confidence,
                'indicators': shift_indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to detect regime shift probability: {e}")
            return {'probability': 0.0, 'type': 'error', 'confidence': 0.0}
    
    def get_multi_timeframe_status(self) -> Dict[str, Any]:
        """Get multi-timeframe analyzer status"""
        try:
            analyzed_symbols = len(self.timeframe_analysis)
            total_analyses = sum(len(analyses) for analyses in self.timeframe_analysis.values())
            total_signals = sum(len(signals) for signals in self.cross_timeframe_signals.values())
            
            return {
                'analyzed_symbols': analyzed_symbols,
                'total_analyses': total_analyses,
                'active_signals': total_signals,
                'supported_timeframes': [tf.value for tf in self.timeframes],
                'monitoring_active': not self._should_stop,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {'error': str(e)}
    
    def _analyze_timeframe(self, symbol: str, timeframe: TimeframeType, data: pd.DataFrame) -> Optional[TimeframeAnalysis]:
        """Analyze single timeframe"""
        try:
            if len(data) < 20:
                return None
            
            # Calculate technical indicators
            data = self._calculate_indicators(data)
            
            # Determine trend direction and strength
            trend_direction, trend_strength = self._analyze_trend(data)
            
            # Calculate volatility
            volatility = data['close'].pct_change().std() * np.sqrt(252)
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_key_levels(data)
            
            # Calculate scores
            momentum_score = self._calculate_momentum_score(data)
            mean_reversion_score = self._calculate_mean_reversion_score(data)
            breakout_probability = self._calculate_breakout_probability(data)
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(data, trend_strength)
            
            # Key levels
            key_levels = {
                'current_price': float(data['close'].iloc[-1]),
                'sma_20': float(data['sma_20'].iloc[-1]) if 'sma_20' in data.columns else float(data['close'].iloc[-1]),
                'sma_50': float(data['sma_50'].iloc[-1]) if 'sma_50' in data.columns else float(data['close'].iloc[-1])
            }
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility=volatility,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                breakout_probability=breakout_probability,
                confidence=confidence,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_levels=key_levels
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze timeframe {timeframe.value}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Simple moving averages
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return data
    
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[str, TrendStrength]:
        """Analyze trend direction and strength"""
        try:
            current_price = data['close'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1] if 'sma_20' in data.columns else current_price
            sma_50 = data['sma_50'].iloc[-1] if 'sma_50' in data.columns else current_price
            
            # Trend direction
            if current_price > sma_20 > sma_50:
                trend_direction = 'bullish'
            elif current_price < sma_20 < sma_50:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
            
            # Trend strength
            price_change_20 = (current_price - data['close'].iloc[-21]) / data['close'].iloc[-21] if len(data) > 21 else 0
            strength_score = abs(price_change_20)
            
            if strength_score > 0.15:
                trend_strength = TrendStrength.VERY_STRONG
            elif strength_score > 0.10:
                trend_strength = TrendStrength.STRONG
            elif strength_score > 0.05:
                trend_strength = TrendStrength.MODERATE
            elif strength_score > 0.02:
                trend_strength = TrendStrength.WEAK
            else:
                trend_strength = TrendStrength.VERY_WEAK
            
            return trend_direction, trend_strength
            
        except Exception as e:
            logger.error(f"Failed to analyze trend: {e}")
            return 'neutral', TrendStrength.WEAK
    
    def _find_key_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            highs = data['high'].iloc[-50:]
            lows = data['low'].iloc[-50:]
            
            resistance_levels = []
            support_levels = []
            
            # Simple pivot detection
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(float(highs.iloc[i]))
                
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(float(lows.iloc[i]))
            
            return sorted(support_levels)[-3:], sorted(resistance_levels, reverse=True)[:3]
            
        except Exception as e:
            logger.error(f"Failed to find key levels: {e}")
            return [], []
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
            rsi_score = (rsi - 50) / 50
            
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10] if len(data) > 10 else 0
            price_score = np.tanh(price_change * 10)
            
            momentum_score = (rsi_score * 0.6 + price_score * 0.4)
            return float(np.clip(momentum_score, -1, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate momentum score: {e}")
            return 0.0
    
    def _calculate_mean_reversion_score(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion score"""
        try:
            current_price = data['close'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1] if 'sma_20' in data.columns else current_price
            
            deviation = abs(current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            reversion_score = min(deviation * 5, 1.0)  # Scale and cap at 1
            
            return float(reversion_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate mean reversion score: {e}")
            return 0.0
    
    def _calculate_breakout_probability(self, data: pd.DataFrame) -> float:
        """Calculate breakout probability"""
        try:
            # Simple volatility-based breakout probability
            recent_volatility = data['close'].iloc[-10:].pct_change().std()
            avg_volatility = data['close'].pct_change().std()
            
            volatility_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1.0
            
            # Lower recent volatility suggests compression and potential breakout
            if volatility_ratio < 0.7:
                breakout_prob = (0.7 - volatility_ratio) / 0.7
            else:
                breakout_prob = 0.0
            
            return float(np.clip(breakout_prob, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate breakout probability: {e}")
            return 0.0
    
    def _calculate_analysis_confidence(self, data: pd.DataFrame, trend_strength: TrendStrength) -> float:
        """Calculate analysis confidence"""
        try:
            # Base confidence from data quality
            data_quality = min(len(data) / 100, 1.0)  # More data = higher confidence
            
            # Trend strength component
            strength_weights = {
                TrendStrength.VERY_STRONG: 1.0,
                TrendStrength.STRONG: 0.8,
                TrendStrength.MODERATE: 0.6,
                TrendStrength.WEAK: 0.4,
                TrendStrength.VERY_WEAK: 0.2
            }
            trend_confidence = strength_weights.get(trend_strength, 0.5)
            
            # Volume confirmation (if available)
            volume_confidence = 0.8  # Default if no volume data
            if 'volume' in data.columns:
                recent_volume = data['volume'].iloc[-10:].mean()
                avg_volume = data['volume'].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                volume_confidence = min(volume_ratio, 1.0)
            
            # Combined confidence
            confidence = (data_quality * 0.3 + trend_confidence * 0.4 + volume_confidence * 0.3)
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _get_higher_timeframes(self, target_timeframe: TimeframeType) -> List[TimeframeType]:
        """Get higher timeframes for context"""
        timeframe_hierarchy = [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4, TimeframeType.D1]
        
        try:
            target_index = timeframe_hierarchy.index(target_timeframe)
            return timeframe_hierarchy[target_index + 1:]
        except ValueError:
            return []
    
    def _get_strength_weight(self, strength: TrendStrength) -> float:
        """Get weight for trend strength"""
        weights = {
            TrendStrength.VERY_STRONG: 1.0,
            TrendStrength.STRONG: 0.8,
            TrendStrength.MODERATE: 0.6,
            TrendStrength.WEAK: 0.4,
            TrendStrength.VERY_WEAK: 0.2
        }
        return weights.get(strength, 0.5)
    
    def _generate_cross_timeframe_signals(self, symbol: str, analyses: Dict[TimeframeType, TimeframeAnalysis]):
        """Generate cross-timeframe signals"""
        try:
            signals = []
            
            # Look for confluence signals
            bullish_timeframes = [tf for tf, analysis in analyses.items() if analysis.trend_direction == 'bullish']
            bearish_timeframes = [tf for tf, analysis in analyses.items() if analysis.trend_direction == 'bearish']
            
            if len(bullish_timeframes) >= self.confluence_min_timeframes:
                signal = CrossTimeframeSignal(
                    signal_id=f"confluence_bullish_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    primary_timeframe=min(bullish_timeframes),
                    supporting_timeframes=bullish_timeframes[1:],
                    signal_type='confluence',
                    strength=len(bullish_timeframes) / len(analyses),
                    confidence=np.mean([analyses[tf].confidence for tf in bullish_timeframes]),
                    bias_direction='bullish',
                    entry_zones=[analyses[tf].key_levels['current_price'] for tf in bullish_timeframes],
                    target_zones=[],
                    reasoning=f"Bullish confluence across {len(bullish_timeframes)} timeframes",
                    created_at=datetime.now()
                )
                signals.append(signal)
            
            if len(bearish_timeframes) >= self.confluence_min_timeframes:
                signal = CrossTimeframeSignal(
                    signal_id=f"confluence_bearish_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    primary_timeframe=min(bearish_timeframes),
                    supporting_timeframes=bearish_timeframes[1:],
                    signal_type='confluence',
                    strength=len(bearish_timeframes) / len(analyses),
                    confidence=np.mean([analyses[tf].confidence for tf in bearish_timeframes]),
                    bias_direction='bearish',
                    entry_zones=[analyses[tf].key_levels['current_price'] for tf in bearish_timeframes],
                    target_zones=[],
                    reasoning=f"Bearish confluence across {len(bearish_timeframes)} timeframes",
                    created_at=datetime.now()
                )
                signals.append(signal)
            
            self.cross_timeframe_signals[symbol] = signals
            
        except Exception as e:
            logger.error(f"Failed to generate cross-timeframe signals: {e}")
    
    def _detect_volatility_compression(self, analyses: Dict[TimeframeType, TimeframeAnalysis]) -> float:
        """Detect volatility compression across timeframes"""
        try:
            volatilities = [analysis.volatility for analysis in analyses.values()]
            if len(volatilities) < 2:
                return 0.0
            
            # Check if volatilities are consistently low
            avg_volatility = np.mean(volatilities)
            compression_score = max(0, (0.02 - avg_volatility) / 0.02)  # Higher score for lower volatility
            
            return float(np.clip(compression_score, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to detect volatility compression: {e}")
            return 0.0
    
    def _detect_cross_timeframe_divergence(self, analyses: Dict[TimeframeType, TimeframeAnalysis]) -> float:
        """Detect divergence across timeframes"""
        try:
            momentum_scores = [analysis.momentum_score for analysis in analyses.values()]
            if len(momentum_scores) < 2:
                return 0.0
            
            # Calculate standard deviation of momentum scores
            momentum_std = np.std(momentum_scores)
            divergence_score = min(momentum_std * 2, 1.0)  # Scale to 0-1
            
            return float(divergence_score)
            
        except Exception as e:
            logger.error(f"Failed to detect cross-timeframe divergence: {e}")
            return 0.0
    
    def _detect_momentum_exhaustion(self, analyses: Dict[TimeframeType, TimeframeAnalysis]) -> float:
        """Detect momentum exhaustion"""
        try:
            momentum_scores = [abs(analysis.momentum_score) for analysis in analyses.values()]
            if not momentum_scores:
                return 0.0
            
            # High momentum with high mean reversion suggests exhaustion
            avg_momentum = np.mean(momentum_scores)
            avg_reversion = np.mean([analysis.mean_reversion_score for analysis in analyses.values()])
            
            exhaustion_score = min(avg_momentum * avg_reversion, 1.0)
            
            return float(exhaustion_score)
            
        except Exception as e:
            logger.error(f"Failed to detect momentum exhaustion: {e}")
            return 0.0
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="MultiTimeframeMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started multi-timeframe monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Clean up old signals
                current_time = datetime.now()
                for symbol in list(self.cross_timeframe_signals.keys()):
                    self.cross_timeframe_signals[symbol] = [
                        signal for signal in self.cross_timeframe_signals[symbol]
                        if current_time - signal.created_at < timedelta(hours=4)
                    ]
                
                # Sleep
                threading.Event().wait(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in multi-timeframe monitoring: {e}")
    
    def stop(self):
        """Stop the multi-timeframe analyzer"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Multi-Timeframe Analyzer stopped")


# Global instance
_multi_timeframe_analyzer = None

def get_multi_timeframe_analyzer() -> MultiTimeframeAnalyzer:
    """Get global multi-timeframe analyzer instance"""
    global _multi_timeframe_analyzer
    if _multi_timeframe_analyzer is None:
        _multi_timeframe_analyzer = MultiTimeframeAnalyzer()
    return _multi_timeframe_analyzer
