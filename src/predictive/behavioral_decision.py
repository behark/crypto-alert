"""
Behavioral Decision-Making Engine
=================================
Unified intelligence that fuses all predictive layers into context-aware trading decisions
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Trading decision types"""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    WAIT = "wait"
    REDUCE_EXPOSURE = "reduce_exposure"

class EntryType(Enum):
    """Entry execution types"""
    MARKET = "market"
    LIMIT = "limit"
    SCALED = "scaled"

class ConfidenceLevel(Enum):
    """Decision confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SignalInput:
    """Input signal from predictive engines"""
    source: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    time_horizon: timedelta
    supporting_factors: List[str]
    timestamp: datetime

@dataclass
class DecisionContext:
    """Complete decision context"""
    symbol: str
    current_price: float
    current_position: Optional[str]
    position_size: float
    account_balance: float
    risk_level: str
    timestamp: datetime

@dataclass
class TradingDecision:
    """Final trading decision output"""
    decision_id: str
    symbol: str
    decision_type: DecisionType
    entry_type: EntryType
    direction: Optional[str]
    confidence: ConfidenceLevel
    confidence_score: float
    position_size_pct: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    sl_tp_ratio: float
    decision_trail: List[str]
    supporting_signals: List[SignalInput]
    execution_priority: int
    valid_until: datetime
    created_at: datetime

class BehavioralDecisionEngine:
    """
    Unified Behavioral Decision-Making Engine
    
    Fuses signals from all predictive layers into context-aware trading decisions
    with human-readable reasoning trails and adaptive safety controls.
    """
    
    def __init__(self, data_dir: str = "data/behavioral_decisions"):
        """Initialize the behavioral decision engine"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Decision state
        self.recent_decisions: List[TradingDecision] = []
        self.decision_history: Dict[str, List[TradingDecision]] = {}
        self.last_decision_time: Dict[str, datetime] = {}
        
        # Signal weights
        self.signal_weights = {
            'regime_predictor': 0.3,
            'sentiment_integrator': 0.25,
            'multi_timeframe_analyzer': 0.25,
            'environmental_risk': 0.2
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            'entry_min_confidence': 0.6,
            'exit_min_confidence': 0.4,
            'position_size_max': 0.1,
            'cooldown_period': 300
        }
        
        # Safety rules
        self.safety_rules = {
            'red_risk_block_entries': True,
            'orange_risk_reduce_size': 0.5,
            'weak_consensus_reduce': 0.5
        }
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Behavioral Decision Engine initialized")
    
    async def make_trading_decision(self, 
                                  symbol: str,
                                  context: DecisionContext,
                                  signals: List[SignalInput]) -> TradingDecision:
        """Make unified trading decision based on all available signals"""
        try:
            decision_id = f"decision_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 1. Check cooldown period
            if self._is_in_cooldown(symbol):
                return self._create_wait_decision(decision_id, symbol, "Cooldown period active")
            
            # 2. Apply safety filters
            safety_check = self._apply_safety_filters(context, signals)
            if safety_check['blocked']:
                return self._create_wait_decision(decision_id, symbol, safety_check['reason'])
            
            # 3. Fuse signals into unified assessment
            signal_fusion = self._fuse_signals(signals)
            
            # 4. Determine decision type and direction
            decision_type, direction = self._determine_decision_type(context, signal_fusion)
            
            # 5. Calculate confidence and position sizing
            confidence_score = self._calculate_decision_confidence(signal_fusion, context)
            confidence_level = self._map_confidence_level(confidence_score)
            position_size = self._calculate_position_size(confidence_score, context, safety_check)
            
            # 6. Determine entry type and execution parameters
            entry_type = self._determine_entry_type(signal_fusion, context)
            entry_price, stop_loss, take_profit, sl_tp_ratio = self._calculate_execution_params(
                decision_type, direction, context, signal_fusion
            )
            
            # 7. Build decision trail
            decision_trail = self._build_decision_trail(
                decision_type, direction, signal_fusion, context, safety_check
            )
            
            # 8. Calculate execution priority
            execution_priority = self._calculate_execution_priority(
                confidence_score, signal_fusion, context
            )
            
            # 9. Create final decision
            decision = TradingDecision(
                decision_id=decision_id,
                symbol=symbol,
                decision_type=decision_type,
                entry_type=entry_type,
                direction=direction,
                confidence=confidence_level,
                confidence_score=confidence_score,
                position_size_pct=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                sl_tp_ratio=sl_tp_ratio,
                decision_trail=decision_trail,
                supporting_signals=signals,
                execution_priority=execution_priority,
                valid_until=datetime.now() + timedelta(minutes=30),
                created_at=datetime.now()
            )
            
            # 10. Store decision
            self._store_decision(decision)
            
            logger.info(f"Decision made for {symbol}: {decision_type.value} "
                       f"(confidence: {confidence_score:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make trading decision: {e}")
            return self._create_error_decision(symbol, str(e))
    
    def get_decision_trail(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision trail for symbol"""
        try:
            if symbol not in self.decision_history:
                return []
            
            recent_decisions = self.decision_history[symbol][-limit:]
            
            trail = []
            for decision in recent_decisions:
                trail.append({
                    'decision_id': decision.decision_id,
                    'timestamp': decision.created_at,
                    'decision_type': decision.decision_type.value,
                    'direction': decision.direction,
                    'confidence': decision.confidence.value,
                    'confidence_score': decision.confidence_score,
                    'position_size_pct': decision.position_size_pct,
                    'decision_trail': decision.decision_trail,
                    'execution_priority': decision.execution_priority
                })
            
            return trail
            
        except Exception as e:
            logger.error(f"Failed to get decision trail: {e}")
            return []
    
    def get_current_bias(self, symbol: str) -> Dict[str, Any]:
        """Get current directional bias with reasoning"""
        try:
            if symbol not in self.decision_history or not self.decision_history[symbol]:
                return {
                    'bias': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'reasons': ['No recent decisions available'],
                    'last_updated': None
                }
            
            # Analyze recent decisions for bias
            recent_decisions = self.decision_history[symbol][-5:]
            
            bullish_weight = 0.0
            bearish_weight = 0.0
            total_weight = 0.0
            reasons = []
            
            for decision in recent_decisions:
                # Weight by recency and confidence
                hours_old = (datetime.now() - decision.created_at).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - hours_old / 24)
                decision_weight = decision.confidence_score * recency_weight
                
                if decision.direction == 'long':
                    bullish_weight += decision_weight
                    reasons.extend([f"Recent long: {trail}" for trail in decision.decision_trail[:2]])
                elif decision.direction == 'short':
                    bearish_weight += decision_weight
                    reasons.extend([f"Recent short: {trail}" for trail in decision.decision_trail[:2]])
                
                total_weight += decision_weight
            
            # Determine bias
            if total_weight == 0:
                bias = 'neutral'
                strength = 0.0
            else:
                net_bias = (bullish_weight - bearish_weight) / total_weight
                if net_bias > 0.2:
                    bias = 'bullish'
                    strength = net_bias
                elif net_bias < -0.2:
                    bias = 'bearish'
                    strength = abs(net_bias)
                else:
                    bias = 'neutral'
                    strength = abs(net_bias)
            
            bias_confidence = min(total_weight, 1.0)
            
            return {
                'bias': bias,
                'strength': strength,
                'confidence': bias_confidence,
                'reasons': reasons[-5:],
                'last_updated': recent_decisions[-1].created_at,
                'decision_count': len(recent_decisions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get current bias: {e}")
            return {
                'bias': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'reasons': [f'Error: {str(e)}'],
                'last_updated': None
            }
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.last_decision_time:
            return False
        
        last_decision = self.last_decision_time[symbol]
        cooldown_period = timedelta(seconds=self.decision_thresholds['cooldown_period'])
        
        return datetime.now() - last_decision < cooldown_period
    
    def _apply_safety_filters(self, context: DecisionContext, signals: List[SignalInput]) -> Dict[str, Any]:
        """Apply safety filters to decision making"""
        try:
            safety_result = {'blocked': False, 'reason': '', 'adjustments': {}}
            
            # Risk level filter
            if context.risk_level == 'red' and self.safety_rules['red_risk_block_entries']:
                safety_result['blocked'] = True
                safety_result['reason'] = "Red risk level - entries blocked"
                return safety_result
            
            # Position size adjustments
            if context.risk_level == 'orange':
                safety_result['adjustments']['position_multiplier'] = self.safety_rules['orange_risk_reduce_size']
            
            # Weak consensus filter
            if len(signals) < 2:
                safety_result['adjustments']['position_multiplier'] = self.safety_rules['weak_consensus_reduce']
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Failed to apply safety filters: {e}")
            return {'blocked': True, 'reason': f'Safety filter error: {str(e)}'}
    
    def _fuse_signals(self, signals: List[SignalInput]) -> Dict[str, Any]:
        """Fuse multiple signals into unified assessment"""
        try:
            if not signals:
                return {
                    'overall_direction': 'neutral',
                    'overall_strength': 0.0,
                    'overall_confidence': 0.0,
                    'signal_count': 0,
                    'consensus_strength': 0.0,
                    'dominant_signals': []
                }
            
            # Categorize signals by direction
            bullish_signals = [s for s in signals if s.direction == 'bullish']
            bearish_signals = [s for s in signals if s.direction == 'bearish']
            
            # Calculate weighted scores
            bullish_score = sum(s.strength * s.confidence * self.signal_weights.get(s.source, 0.1) 
                              for s in bullish_signals)
            bearish_score = sum(s.strength * s.confidence * self.signal_weights.get(s.source, 0.1) 
                              for s in bearish_signals)
            
            total_weight = sum(self.signal_weights.get(s.source, 0.1) for s in signals)
            
            # Determine overall direction and strength
            if bullish_score > bearish_score * 1.2:
                overall_direction = 'bullish'
                overall_strength = bullish_score / total_weight if total_weight > 0 else 0
            elif bearish_score > bullish_score * 1.2:
                overall_direction = 'bearish'
                overall_strength = bearish_score / total_weight if total_weight > 0 else 0
            else:
                overall_direction = 'neutral'
                overall_strength = abs(bullish_score - bearish_score) / total_weight if total_weight > 0 else 0
            
            # Calculate consensus strength
            if len(signals) > 1:
                same_direction_signals = (bullish_signals if overall_direction == 'bullish' 
                                        else bearish_signals if overall_direction == 'bearish' 
                                        else [])
                consensus_strength = len(same_direction_signals) / len(signals)
            else:
                consensus_strength = 1.0
            
            # Overall confidence
            avg_confidence = np.mean([s.confidence for s in signals])
            overall_confidence = avg_confidence * consensus_strength
            
            # Identify dominant signals
            dominant_signals = sorted(signals, key=lambda s: s.strength * s.confidence, reverse=True)[:3]
            
            return {
                'overall_direction': overall_direction,
                'overall_strength': overall_strength,
                'overall_confidence': overall_confidence,
                'signal_count': len(signals),
                'consensus_strength': consensus_strength,
                'dominant_signals': [s.source for s in dominant_signals],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score
            }
            
        except Exception as e:
            logger.error(f"Failed to fuse signals: {e}")
            return {
                'overall_direction': 'neutral',
                'overall_strength': 0.0,
                'overall_confidence': 0.0,
                'signal_count': 0,
                'consensus_strength': 0.0,
                'dominant_signals': []
            }
    
    def _determine_decision_type(self, context: DecisionContext, signal_fusion: Dict[str, Any]) -> Tuple[DecisionType, Optional[str]]:
        """Determine decision type and direction"""
        try:
            overall_direction = signal_fusion['overall_direction']
            overall_strength = signal_fusion['overall_strength']
            overall_confidence = signal_fusion['overall_confidence']
            
            min_confidence = self.decision_thresholds['entry_min_confidence']
            current_position = context.current_position
            
            # Entry decisions
            if current_position is None:
                if overall_confidence >= min_confidence and overall_strength > 0.5:
                    if overall_direction == 'bullish':
                        return DecisionType.ENTER_LONG, 'long'
                    elif overall_direction == 'bearish':
                        return DecisionType.ENTER_SHORT, 'short'
                
                return DecisionType.WAIT, None
            
            # Exit decisions
            elif current_position == 'long':
                if overall_direction == 'bearish' and overall_confidence >= self.decision_thresholds['exit_min_confidence']:
                    return DecisionType.EXIT_LONG, None
                elif overall_confidence < 0.3:
                    return DecisionType.REDUCE_EXPOSURE, None
                else:
                    return DecisionType.HOLD, None
            
            elif current_position == 'short':
                if overall_direction == 'bullish' and overall_confidence >= self.decision_thresholds['exit_min_confidence']:
                    return DecisionType.EXIT_SHORT, None
                elif overall_confidence < 0.3:
                    return DecisionType.REDUCE_EXPOSURE, None
                else:
                    return DecisionType.HOLD, None
            
            return DecisionType.WAIT, None
            
        except Exception as e:
            logger.error(f"Failed to determine decision type: {e}")
            return DecisionType.WAIT, None
    
    def _calculate_decision_confidence(self, signal_fusion: Dict[str, Any], context: DecisionContext) -> float:
        """Calculate overall decision confidence"""
        try:
            base_confidence = signal_fusion['overall_confidence']
            consensus_strength = signal_fusion['consensus_strength']
            signal_count = signal_fusion['signal_count']
            
            # Signal count bonus
            signal_bonus = min(signal_count / 4, 0.2)
            
            # Consensus bonus
            consensus_bonus = consensus_strength * 0.15
            
            # Risk adjustment
            risk_adjustment = 1.0
            if context.risk_level == 'red':
                risk_adjustment = 0.7
            elif context.risk_level == 'orange':
                risk_adjustment = 0.85
            elif context.risk_level == 'yellow':
                risk_adjustment = 0.95
            
            final_confidence = (base_confidence + signal_bonus + consensus_bonus) * risk_adjustment
            
            return float(np.clip(final_confidence, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate decision confidence: {e}")
            return 0.5
    
    def _map_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Map confidence score to confidence level"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_position_size(self, confidence_score: float, context: DecisionContext, safety_check: Dict[str, Any]) -> float:
        """Calculate position size based on confidence and risk"""
        try:
            base_size = confidence_score * self.decision_thresholds['position_size_max']
            
            # Apply safety adjustments
            position_multiplier = safety_check.get('adjustments', {}).get('position_multiplier', 1.0)
            adjusted_size = base_size * position_multiplier
            
            # Risk level adjustments
            if context.risk_level == 'red':
                adjusted_size *= 0.3
            elif context.risk_level == 'orange':
                adjusted_size *= 0.6
            elif context.risk_level == 'yellow':
                adjusted_size *= 0.8
            
            return float(np.clip(adjusted_size, 0.001, self.decision_thresholds['position_size_max']))
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.01
    
    def _determine_entry_type(self, signal_fusion: Dict[str, Any], context: DecisionContext) -> EntryType:
        """Determine optimal entry type"""
        try:
            overall_strength = signal_fusion['overall_strength']
            consensus_strength = signal_fusion['consensus_strength']
            
            if overall_strength > 0.8 and consensus_strength > 0.8:
                return EntryType.MARKET
            elif overall_strength > 0.6:
                return EntryType.LIMIT
            else:
                return EntryType.SCALED
            
        except Exception as e:
            logger.error(f"Failed to determine entry type: {e}")
            return EntryType.LIMIT
    
    def _calculate_execution_params(self, decision_type: DecisionType, direction: Optional[str], 
                                  context: DecisionContext, signal_fusion: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        """Calculate execution parameters"""
        try:
            if decision_type in [DecisionType.WAIT, DecisionType.HOLD]:
                return None, None, None, 0.0
            
            current_price = context.current_price
            overall_strength = signal_fusion['overall_strength']
            
            entry_price = current_price if decision_type in [DecisionType.ENTER_LONG, DecisionType.ENTER_SHORT] else None
            
            # Stop loss and take profit
            if direction == 'long':
                stop_loss = current_price * (1 - 0.02 * overall_strength)
                take_profit = current_price * (1 + 0.04 * overall_strength)
            elif direction == 'short':
                stop_loss = current_price * (1 + 0.02 * overall_strength)
                take_profit = current_price * (1 - 0.04 * overall_strength)
            else:
                stop_loss = None
                take_profit = None
            
            # SL/TP ratio
            if stop_loss and take_profit and entry_price:
                if direction == 'long':
                    sl_distance = abs(entry_price - stop_loss)
                    tp_distance = abs(take_profit - entry_price)
                else:
                    sl_distance = abs(stop_loss - entry_price)
                    tp_distance = abs(entry_price - take_profit)
                
                sl_tp_ratio = tp_distance / sl_distance if sl_distance > 0 else 2.0
            else:
                sl_tp_ratio = 2.0
            
            return entry_price, stop_loss, take_profit, sl_tp_ratio
            
        except Exception as e:
            logger.error(f"Failed to calculate execution params: {e}")
            return None, None, None, 0.0
    
    def _build_decision_trail(self, decision_type: DecisionType, direction: Optional[str],
                            signal_fusion: Dict[str, Any], context: DecisionContext,
                            safety_check: Dict[str, Any]) -> List[str]:
        """Build human-readable decision trail"""
        try:
            trail = []
            
            # Primary decision reason
            if decision_type == DecisionType.ENTER_LONG:
                trail.append(f"Entering LONG: {signal_fusion['overall_direction']} signals "
                           f"(strength: {signal_fusion['overall_strength']:.2f})")
            elif decision_type == DecisionType.ENTER_SHORT:
                trail.append(f"Entering SHORT: {signal_fusion['overall_direction']} signals "
                           f"(strength: {signal_fusion['overall_strength']:.2f})")
            elif decision_type == DecisionType.HOLD:
                trail.append("Holding position - signals remain supportive")
            elif decision_type == DecisionType.WAIT:
                trail.append("Waiting - insufficient signal conviction")
            
            # Supporting signals
            dominant_signals = signal_fusion.get('dominant_signals', [])
            if dominant_signals:
                trail.append(f"Key signals: {', '.join(dominant_signals[:3])}")
            
            # Risk environment
            trail.append(f"Risk level: {context.risk_level.upper()}")
            
            # Signal consensus
            consensus = signal_fusion.get('consensus_strength', 0)
            trail.append(f"Signal consensus: {consensus:.1%}")
            
            # Safety adjustments
            if safety_check.get('adjustments'):
                trail.append("Safety adjustments applied")
            
            return trail
            
        except Exception as e:
            logger.error(f"Failed to build decision trail: {e}")
            return ["Decision trail unavailable"]
    
    def _calculate_execution_priority(self, confidence_score: float, signal_fusion: Dict[str, Any], context: DecisionContext) -> int:
        """Calculate execution priority (1-10)"""
        try:
            base_priority = int(confidence_score * 10)
            
            # Boost for strong consensus
            if signal_fusion.get('consensus_strength', 0) > 0.8:
                base_priority += 1
            
            # Boost for multiple signals
            if signal_fusion.get('signal_count', 0) >= 3:
                base_priority += 1
            
            # Reduce for high risk
            if context.risk_level in ['red', 'orange']:
                base_priority = max(1, base_priority - 2)
            
            return max(1, min(10, base_priority))
            
        except Exception as e:
            logger.error(f"Failed to calculate execution priority: {e}")
            return 5
    
    def _store_decision(self, decision: TradingDecision):
        """Store decision in history"""
        try:
            symbol = decision.symbol
            
            if symbol not in self.decision_history:
                self.decision_history[symbol] = []
            
            self.decision_history[symbol].append(decision)
            self.last_decision_time[symbol] = decision.created_at
            
            # Keep only recent decisions
            if len(self.decision_history[symbol]) > 100:
                self.decision_history[symbol] = self.decision_history[symbol][-100:]
            
            # Add to recent decisions
            self.recent_decisions.append(decision)
            if len(self.recent_decisions) > 50:
                self.recent_decisions = self.recent_decisions[-50:]
            
        except Exception as e:
            logger.error(f"Failed to store decision: {e}")
    
    def _create_wait_decision(self, decision_id: str, symbol: str, reason: str) -> TradingDecision:
        """Create a wait decision"""
        return TradingDecision(
            decision_id=decision_id,
            symbol=symbol,
            decision_type=DecisionType.WAIT,
            entry_type=EntryType.LIMIT,
            direction=None,
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.0,
            position_size_pct=0.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            sl_tp_ratio=0.0,
            decision_trail=[reason],
            supporting_signals=[],
            execution_priority=1,
            valid_until=datetime.now() + timedelta(minutes=5),
            created_at=datetime.now()
        )
    
    def _create_error_decision(self, symbol: str, error_msg: str) -> TradingDecision:
        """Create an error decision"""
        return TradingDecision(
            decision_id=f"error_{symbol}_{datetime.now().strftime('%H%M%S')}",
            symbol=symbol,
            decision_type=DecisionType.WAIT,
            entry_type=EntryType.LIMIT,
            direction=None,
            confidence=ConfidenceLevel.VERY_LOW,
            confidence_score=0.0,
            position_size_pct=0.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            sl_tp_ratio=0.0,
            decision_trail=[f"Error in decision making: {error_msg}"],
            supporting_signals=[],
            execution_priority=1,
            valid_until=datetime.now() + timedelta(minutes=1),
            created_at=datetime.now()
        )
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="BehavioralDecisionMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started behavioral decision monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Clean up expired decisions
                current_time = datetime.now()
                for symbol in list(self.decision_history.keys()):
                    self.decision_history[symbol] = [
                        d for d in self.decision_history[symbol]
                        if current_time - d.created_at < timedelta(hours=24)
                    ]
                
                # Sleep
                threading.Event().wait(60.0)
                
            except Exception as e:
                logger.error(f"Error in behavioral decision monitoring: {e}")
    
    def stop(self):
        """Stop the behavioral decision engine"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Behavioral Decision Engine stopped")


# Global instance
_behavioral_decision_engine = None

def get_behavioral_decision_engine() -> BehavioralDecisionEngine:
    """Get global behavioral decision engine instance"""
    global _behavioral_decision_engine
    if _behavioral_decision_engine is None:
        _behavioral_decision_engine = BehavioralDecisionEngine()
    return _behavioral_decision_engine
