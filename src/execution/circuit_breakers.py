"""
Circuit Breaker System
=====================
Black swan event detection and emergency protection protocols
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Market threat level enumeration"""
    GREEN = "green"      # Normal conditions
    YELLOW = "yellow"    # Elevated risk
    ORANGE = "orange"    # High risk
    RED = "red"          # Critical risk
    BLACK = "black"      # Black swan event

class CircuitBreakerType(Enum):
    """Circuit breaker type enumeration"""
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_GAP = "price_gap"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_EVENT = "external_event"

@dataclass
class MarketCondition:
    """Market condition snapshot"""
    timestamp: datetime
    volatility: float
    volume: float
    price_change: float
    correlation: float
    liquidity: float
    spread: float
    anomaly_score: float
    threat_level: ThreatLevel

@dataclass
class CircuitBreakerTrigger:
    """Circuit breaker trigger event"""
    trigger_id: str
    breaker_type: CircuitBreakerType
    threat_level: ThreatLevel
    description: str
    market_conditions: MarketCondition
    triggered_at: datetime
    auto_actions: List[str]
    manual_actions: List[str]
    estimated_impact: float
    confidence: float

class CircuitBreakerSystem:
    """
    Circuit Breaker System
    
    Monitors market conditions for black swan events and executes
    emergency protection protocols to safeguard the trading system.
    """
    
    def __init__(self, data_dir: str = "data/circuit_breakers"):
        """Initialize the circuit breaker system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Threat detection thresholds
        self.volatility_thresholds = {
            ThreatLevel.YELLOW: 0.05,   # 5% volatility
            ThreatLevel.ORANGE: 0.10,   # 10% volatility
            ThreatLevel.RED: 0.20,      # 20% volatility
            ThreatLevel.BLACK: 0.40     # 40% volatility (black swan)
        }
        
        self.price_change_thresholds = {
            ThreatLevel.YELLOW: 0.03,   # 3% price change
            ThreatLevel.ORANGE: 0.07,   # 7% price change
            ThreatLevel.RED: 0.15,      # 15% price change
            ThreatLevel.BLACK: 0.30     # 30% price change
        }
        
        # Current state
        self.current_threat_level = ThreatLevel.GREEN
        self.active_triggers: Dict[str, CircuitBreakerTrigger] = {}
        self.market_conditions: Dict[str, MarketCondition] = {}
        
        # Protection callbacks
        self.protection_callbacks: List[Callable] = []
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        # Configuration
        self.auto_protection_enabled = True
        
        self._start_monitoring()
        
        logger.info("Circuit Breaker System initialized")
    
    def update_market_conditions(self, symbol: str, price: float, volume: float,
                                volatility: float, spread: float = None) -> MarketCondition:
        """
        Update market conditions for symbol
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Current volume
            volatility: Current volatility
            spread: Bid-ask spread
            
        Returns:
            MarketCondition: Updated market condition
        """
        try:
            # Get previous condition for comparison
            prev_condition = self.market_conditions.get(symbol)
            
            # Calculate price change
            price_change = 0.0
            if prev_condition:
                prev_price = getattr(prev_condition, 'price', price)
                price_change = abs(price - prev_price) / prev_price if prev_price != 0 else 0.0
            
            # Calculate volume anomaly
            avg_volume = 5000.0  # Simplified
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate correlation and liquidity (simplified)
            correlation = np.random.uniform(0.3, 0.9)
            liquidity = min(volume / 10000, 1.0)
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(volatility, price_change, volume_ratio, correlation, liquidity)
            
            # Determine threat level
            threat_level = self._determine_threat_level(volatility, price_change, volume_ratio, anomaly_score)
            
            # Create market condition
            condition = MarketCondition(
                timestamp=datetime.now(),
                volatility=volatility,
                volume=volume,
                price_change=price_change,
                correlation=correlation,
                liquidity=liquidity,
                spread=spread or 0.001,
                anomaly_score=anomaly_score,
                threat_level=threat_level
            )
            
            # Update current conditions
            self.market_conditions[symbol] = condition
            
            # Check for circuit breaker triggers
            self._check_circuit_breakers(symbol, condition)
            
            return condition
            
        except Exception as e:
            logger.error(f"Failed to update market conditions: {e}")
            return None
    
    def trigger_circuit_breaker(self, breaker_type: CircuitBreakerType, 
                               description: str, threat_level: ThreatLevel = ThreatLevel.RED,
                               market_conditions: MarketCondition = None) -> str:
        """
        Trigger circuit breaker
        
        Args:
            breaker_type: Type of circuit breaker
            description: Trigger description
            threat_level: Threat level
            market_conditions: Current market conditions
            
        Returns:
            str: Trigger ID
        """
        try:
            trigger_id = f"trigger_{breaker_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Define automatic actions based on breaker type
            auto_actions = self._get_auto_actions(breaker_type, threat_level)
            manual_actions = self._get_manual_actions(breaker_type, threat_level)
            
            # Create trigger
            trigger = CircuitBreakerTrigger(
                trigger_id=trigger_id,
                breaker_type=breaker_type,
                threat_level=threat_level,
                description=description,
                market_conditions=market_conditions,
                triggered_at=datetime.now(),
                auto_actions=auto_actions,
                manual_actions=manual_actions,
                estimated_impact=self._estimate_impact(breaker_type, threat_level),
                confidence=0.85
            )
            
            # Add to active triggers
            self.active_triggers[trigger_id] = trigger
            
            # Update system threat level
            self._update_system_threat_level()
            
            # Execute automatic protection actions
            if self.auto_protection_enabled:
                self._execute_auto_protection(trigger)
            
            # Notify protection callbacks
            self._notify_protection_callbacks(trigger)
            
            logger.critical(f"Circuit breaker triggered: {trigger_id} ({breaker_type.value}) - {description}")
            return trigger_id
            
        except Exception as e:
            logger.error(f"Failed to trigger circuit breaker: {e}")
            return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get circuit breaker system status"""
        try:
            # Calculate threat distribution
            threat_distribution = {}
            for level in ThreatLevel:
                count = len([t for t in self.active_triggers.values() if t.threat_level == level])
                if count > 0:
                    threat_distribution[level.value] = count
            
            return {
                'current_threat_level': self.current_threat_level.value,
                'active_triggers': len(self.active_triggers),
                'threat_distribution': threat_distribution,
                'monitored_symbols': len(self.market_conditions),
                'auto_protection_enabled': self.auto_protection_enabled,
                'monitoring_active': not self._should_stop,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def get_active_triggers(self, threat_filter: Optional[ThreatLevel] = None) -> List[CircuitBreakerTrigger]:
        """Get active circuit breaker triggers"""
        try:
            triggers = list(self.active_triggers.values())
            
            if threat_filter:
                triggers = [t for t in triggers if t.threat_level == threat_filter]
            
            # Sort by threat level (highest first) then by trigger time
            threat_order = {ThreatLevel.BLACK: 5, ThreatLevel.RED: 4, ThreatLevel.ORANGE: 3, 
                          ThreatLevel.YELLOW: 2, ThreatLevel.GREEN: 1}
            triggers.sort(key=lambda x: (threat_order.get(x.threat_level, 0), x.triggered_at), reverse=True)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to get active triggers: {e}")
            return []
    
    def register_protection_callback(self, callback: Callable[[CircuitBreakerTrigger], None]):
        """Register callback for circuit breaker events"""
        self.protection_callbacks.append(callback)
        logger.info("Registered protection callback")
    
    def _check_circuit_breakers(self, symbol: str, condition: MarketCondition):
        """Check for circuit breaker conditions"""
        try:
            # Volatility spike detection
            if condition.volatility > self.volatility_thresholds[ThreatLevel.RED]:
                if condition.threat_level == ThreatLevel.BLACK:
                    self.trigger_circuit_breaker(
                        CircuitBreakerType.VOLATILITY_SPIKE,
                        f"Extreme volatility spike detected: {condition.volatility:.2%}",
                        ThreatLevel.BLACK,
                        condition
                    )
                elif condition.threat_level == ThreatLevel.RED:
                    self.trigger_circuit_breaker(
                        CircuitBreakerType.VOLATILITY_SPIKE,
                        f"High volatility detected: {condition.volatility:.2%}",
                        ThreatLevel.RED,
                        condition
                    )
            
            # Flash crash detection
            if condition.price_change > self.price_change_thresholds[ThreatLevel.RED]:
                self.trigger_circuit_breaker(
                    CircuitBreakerType.FLASH_CRASH,
                    f"Rapid price movement detected: {condition.price_change:.2%}",
                    condition.threat_level,
                    condition
                )
            
            # Liquidity crisis detection
            if condition.liquidity < 0.3:  # Low liquidity threshold
                self.trigger_circuit_breaker(
                    CircuitBreakerType.LIQUIDITY_CRISIS,
                    f"Low liquidity detected: {condition.liquidity:.2f}",
                    ThreatLevel.ORANGE,
                    condition
                )
            
        except Exception as e:
            logger.error(f"Failed to check circuit breakers: {e}")
    
    def _determine_threat_level(self, volatility: float, price_change: float, 
                              volume_ratio: float, anomaly_score: float) -> ThreatLevel:
        """Determine overall threat level"""
        try:
            # Check for black swan conditions
            if (volatility > self.volatility_thresholds[ThreatLevel.BLACK] or
                price_change > self.price_change_thresholds[ThreatLevel.BLACK] or
                anomaly_score > 0.9):
                return ThreatLevel.BLACK
            
            # Check for red conditions
            if (volatility > self.volatility_thresholds[ThreatLevel.RED] or
                price_change > self.price_change_thresholds[ThreatLevel.RED] or
                anomaly_score > 0.7):
                return ThreatLevel.RED
            
            # Check for orange conditions
            if (volatility > self.volatility_thresholds[ThreatLevel.ORANGE] or
                price_change > self.price_change_thresholds[ThreatLevel.ORANGE] or
                anomaly_score > 0.5):
                return ThreatLevel.ORANGE
            
            # Check for yellow conditions
            if (volatility > self.volatility_thresholds[ThreatLevel.YELLOW] or
                price_change > self.price_change_thresholds[ThreatLevel.YELLOW] or
                anomaly_score > 0.3):
                return ThreatLevel.YELLOW
            
            return ThreatLevel.GREEN
            
        except Exception as e:
            logger.error(f"Failed to determine threat level: {e}")
            return ThreatLevel.GREEN
    
    def _calculate_anomaly_score(self, volatility: float, price_change: float,
                               volume_ratio: float, correlation: float, liquidity: float) -> float:
        """Calculate composite anomaly score"""
        try:
            # Normalize individual scores
            vol_score = min(volatility / 0.5, 1.0)  # Normalize to 50% max
            price_score = min(price_change / 0.3, 1.0)  # Normalize to 30% max
            volume_score = min(volume_ratio / 20.0, 1.0)  # Normalize to 20x max
            corr_score = 1.0 - correlation  # Lower correlation = higher anomaly
            liq_score = 1.0 - liquidity  # Lower liquidity = higher anomaly
            
            # Weighted composite score
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # vol, price, volume, corr, liq
            scores = [vol_score, price_score, volume_score, corr_score, liq_score]
            
            anomaly_score = sum(w * s for w, s in zip(weights, scores))
            
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate anomaly score: {e}")
            return 0.0
    
    def _get_auto_actions(self, breaker_type: CircuitBreakerType, threat_level: ThreatLevel) -> List[str]:
        """Get automatic actions for breaker type and threat level"""
        try:
            actions = []
            
            if threat_level == ThreatLevel.BLACK:
                actions.extend([
                    "emergency_stop_all_trading",
                    "close_all_positions",
                    "cancel_all_orders",
                    "notify_administrators"
                ])
            elif threat_level == ThreatLevel.RED:
                actions.extend([
                    "halt_new_positions",
                    "reduce_position_sizes",
                    "tighten_stop_losses"
                ])
            elif threat_level == ThreatLevel.ORANGE:
                actions.extend([
                    "reduce_leverage",
                    "increase_cash_reserves"
                ])
            elif threat_level == ThreatLevel.YELLOW:
                actions.extend([
                    "increase_monitoring"
                ])
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to get auto actions: {e}")
            return []
    
    def _get_manual_actions(self, breaker_type: CircuitBreakerType, threat_level: ThreatLevel) -> List[str]:
        """Get manual actions for breaker type and threat level"""
        try:
            actions = [
                "analyze_root_cause",
                "update_risk_parameters",
                "review_strategy_performance"
            ]
            
            if threat_level in [ThreatLevel.BLACK, ThreatLevel.RED]:
                actions.extend([
                    "review_portfolio_exposure",
                    "assess_market_conditions",
                    "contact_brokers"
                ])
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to get manual actions: {e}")
            return []
    
    def _execute_auto_protection(self, trigger: CircuitBreakerTrigger):
        """Execute automatic protection actions"""
        try:
            for action in trigger.auto_actions:
                try:
                    # Simulate protection action execution
                    logger.info(f"Executing protection action: {action}")
                    # In real implementation, would execute actual protection logic
                except Exception as e:
                    logger.error(f"Failed to execute auto protection action {action}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to execute auto protection: {e}")
    
    def _estimate_impact(self, breaker_type: CircuitBreakerType, threat_level: ThreatLevel) -> float:
        """Estimate impact of circuit breaker trigger"""
        try:
            base_impact = {
                ThreatLevel.YELLOW: 0.1,
                ThreatLevel.ORANGE: 0.3,
                ThreatLevel.RED: 0.6,
                ThreatLevel.BLACK: 1.0
            }.get(threat_level, 0.1)
            
            return base_impact
            
        except Exception as e:
            logger.error(f"Failed to estimate impact: {e}")
            return 0.5
    
    def _update_system_threat_level(self):
        """Update overall system threat level"""
        try:
            if not self.active_triggers:
                self.current_threat_level = ThreatLevel.GREEN
                return
            
            # Find highest threat level among active triggers
            max_threat = ThreatLevel.GREEN
            threat_order = {ThreatLevel.GREEN: 1, ThreatLevel.YELLOW: 2, ThreatLevel.ORANGE: 3, 
                          ThreatLevel.RED: 4, ThreatLevel.BLACK: 5}
            
            for trigger in self.active_triggers.values():
                if threat_order.get(trigger.threat_level, 1) > threat_order.get(max_threat, 1):
                    max_threat = trigger.threat_level
            
            if max_threat != self.current_threat_level:
                old_level = self.current_threat_level
                self.current_threat_level = max_threat
                logger.warning(f"System threat level changed: {old_level.value} -> {max_threat.value}")
                
        except Exception as e:
            logger.error(f"Failed to update system threat level: {e}")
    
    def _notify_protection_callbacks(self, trigger: CircuitBreakerTrigger):
        """Notify protection callbacks"""
        try:
            for callback in self.protection_callbacks:
                try:
                    callback(trigger)
                except Exception as e:
                    logger.warning(f"Protection callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to notify protection callbacks: {e}")
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="CircuitBreakerMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started circuit breaker monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Update system threat level
                self._update_system_threat_level()
                
                # Sleep
                threading.Event().wait(5.0)
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring: {e}")
    
    def stop(self):
        """Stop the circuit breaker system"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Circuit Breaker System stopped")


# Global instance
_circuit_breaker_system = None

def get_circuit_breaker_system() -> CircuitBreakerSystem:
    """Get global circuit breaker system instance"""
    global _circuit_breaker_system
    if _circuit_breaker_system is None:
        _circuit_breaker_system = CircuitBreakerSystem()
    return _circuit_breaker_system
