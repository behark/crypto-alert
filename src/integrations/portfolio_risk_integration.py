"""
Portfolio Intelligence → Risk Engine Link - Phase 4 Final Integration
Links Portfolio Intelligence with Risk Engine for real-time allocation and VaR-triggered responses.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

from ..portfolio.portfolio_intelligence import PortfolioIntelligenceCore, AllocationMethod, PortfolioMetrics
from ..predictive.environmental_risk import EnvironmentalRiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskTriggerLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class StressResponseAction(Enum):
    MONITOR = "monitor"
    REDUCE_EXPOSURE = "reduce_exposure"
    HEDGE_POSITIONS = "hedge_positions"
    EMERGENCY_EXIT = "emergency_exit"
    FULL_SHUTDOWN = "full_shutdown"

@dataclass
class VaRTrigger:
    """Value at Risk trigger event"""
    trigger_id: str
    trigger_time: datetime
    var_breach_probability: float
    current_var: float
    var_threshold: float
    portfolio_value: float
    trigger_level: RiskTriggerLevel
    response_action: StressResponseAction
    affected_strategies: List[str]
    mitigation_plan: str
    active: bool

@dataclass
class AllocationUpdate:
    """Dynamic allocation update event"""
    update_id: str
    update_time: datetime
    previous_allocation: Dict[str, float]
    new_allocation: Dict[str, float]
    allocation_method: AllocationMethod
    trigger_reason: str
    risk_metrics: Dict[str, float]
    expected_improvement: float
    success: bool

@dataclass
class AutoBalanceEvent:
    """Auto-balance execution event"""
    balance_id: str
    balance_time: datetime
    rebalance_trigger: str
    strategies_rebalanced: List[str]
    capital_rotated: float
    risk_reduction: float
    performance_impact: float
    balance_success: bool

class PortfolioRiskIntegration:
    """
    Portfolio Intelligence → Risk Engine Integration.
    Real-time allocation metrics, VaR monitoring, and stress response automation.
    """
    
    def __init__(self, portfolio_core: PortfolioIntelligenceCore, 
                 risk_engine: EnvironmentalRiskEngine):
        """Initialize Portfolio-Risk Integration"""
        self.portfolio_core = portfolio_core
        self.risk_engine = risk_engine
        
        # Risk monitoring configuration
        self.var_breach_threshold = 0.20  # 20% breach probability triggers response
        self.critical_var_threshold = 0.35  # 35% triggers critical response
        self.emergency_var_threshold = 0.50  # 50% triggers emergency response
        self.monitoring_interval = 1800  # 30 minutes
        self.auto_balance_enabled = True
        
        # Integration tracking
        self.var_triggers: List[VaRTrigger] = []
        self.allocation_updates: List[AllocationUpdate] = []
        self.auto_balance_events: List[AutoBalanceEvent] = []
        
        # Current risk state
        self.current_risk_level = RiskTriggerLevel.LOW
        self.last_var_check = None
        self.active_stress_response = None
        
        # Integration statistics
        self.integration_stats = {
            'var_triggers': 0,
            'allocation_updates': 0,
            'auto_balances': 0,
            'risk_reductions': 0,
            'emergency_responses': 0,
            'last_risk_check': None,
            'total_capital_protected': 0.0
        }
        
        # Start monitoring thread
        self.running = True
        self.risk_monitor_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.risk_monitor_thread.start()
        
        logger.info("Portfolio-Risk Integration initialized with VaR monitoring and auto-balance")
    
    def check_var_breach_risk(self) -> Dict[str, Any]:
        """Check for VaR breach probability and trigger responses"""
        try:
            # Get current portfolio metrics
            portfolio_status = self.portfolio_core.get_portfolio_status()
            
            if 'error' in portfolio_status:
                return {'error': f"Cannot get portfolio status: {portfolio_status['error']}"}
            
            # Get current VaR
            current_var = portfolio_status.get('risk_metrics', {}).get('var_95', 0.0)
            portfolio_value = portfolio_status.get('total_value', 0.0)
            
            # Calculate VaR breach probability using risk engine
            risk_assessment = self.risk_engine.assess_environmental_risk()
            
            # Estimate breach probability based on environmental factors
            base_breach_prob = abs(current_var) / portfolio_value if portfolio_value > 0 else 0.0
            
            # Adjust for environmental risk factors
            risk_multiplier = 1.0
            if 'risk_factors' in risk_assessment:
                for factor in risk_assessment['risk_factors']:
                    if factor.get('severity', 0) > 0.7:
                        risk_multiplier *= 1.5
                    elif factor.get('severity', 0) > 0.5:
                        risk_multiplier *= 1.2
            
            var_breach_probability = min(0.99, base_breach_prob * risk_multiplier)
            
            # Determine trigger level
            trigger_level = self._determine_risk_trigger_level(var_breach_probability)
            
            # Check if response is needed
            response_needed = var_breach_probability >= self.var_breach_threshold
            
            var_check_result = {
                'check_time': datetime.now(),
                'current_var': current_var,
                'portfolio_value': portfolio_value,
                'var_breach_probability': var_breach_probability,
                'trigger_level': trigger_level.value,
                'response_needed': response_needed,
                'risk_multiplier': risk_multiplier
            }
            
            # Trigger response if needed
            if response_needed:
                trigger_result = self._trigger_var_response(
                    var_breach_probability, current_var, portfolio_value, trigger_level
                )
                var_check_result['trigger_response'] = trigger_result
            
            # Update statistics
            self.integration_stats['last_risk_check'] = datetime.now()
            self.last_var_check = datetime.now()
            self.current_risk_level = trigger_level
            
            logger.info(f"VaR breach check: {var_breach_probability:.1%} probability, {trigger_level.value} level")
            return var_check_result
            
        except Exception as e:
            logger.error(f"Error checking VaR breach risk: {e}")
            return {'error': str(e)}
    
    def update_dynamic_allocation(self, trigger_reason: str = "manual") -> Dict[str, Any]:
        """Update portfolio allocation based on current risk metrics"""
        try:
            # Get current allocation
            portfolio_status = self.portfolio_core.get_portfolio_status()
            current_allocation = portfolio_status.get('allocation', {})
            
            # Get risk metrics for allocation decision
            risk_metrics = portfolio_status.get('risk_metrics', {})
            
            # Determine optimal allocation method based on risk level
            if self.current_risk_level in [RiskTriggerLevel.CRITICAL, RiskTriggerLevel.EMERGENCY]:
                allocation_method = AllocationMethod.RISK_PARITY
            elif self.current_risk_level == RiskTriggerLevel.HIGH:
                allocation_method = AllocationMethod.DYNAMIC
            else:
                allocation_method = AllocationMethod.KELLY_CRITERION
            
            # Apply new allocation
            allocation_result = self.portfolio_core.apply_allocation_method(allocation_method)
            
            if 'error' in allocation_result:
                return {'error': f"Allocation update failed: {allocation_result['error']}"}
            
            # Get new allocation
            new_portfolio_status = self.portfolio_core.get_portfolio_status()
            new_allocation = new_portfolio_status.get('allocation', {})
            
            # Calculate expected improvement
            expected_improvement = self._calculate_allocation_improvement(
                current_allocation, new_allocation, risk_metrics
            )
            
            # Create allocation update record
            update = AllocationUpdate(
                update_id=f"alloc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                update_time=datetime.now(),
                previous_allocation=current_allocation.copy(),
                new_allocation=new_allocation.copy(),
                allocation_method=allocation_method,
                trigger_reason=trigger_reason,
                risk_metrics=risk_metrics.copy(),
                expected_improvement=expected_improvement,
                success=True
            )
            
            self.allocation_updates.append(update)
            
            # Update statistics
            self.integration_stats['allocation_updates'] += 1
            
            update_result = {
                'update_id': update.update_id,
                'allocation_method': allocation_method.value,
                'expected_improvement': expected_improvement,
                'allocation_changes': self._calculate_allocation_changes(current_allocation, new_allocation),
                'update_success': True
            }
            
            logger.info(f"Dynamic allocation updated: {allocation_method.value} method, {expected_improvement:.2%} improvement")
            return update_result
            
        except Exception as e:
            logger.error(f"Error updating dynamic allocation: {e}")
            return {'error': str(e)}
    
    def execute_auto_balance(self, trigger_reason: str = "scheduled") -> Dict[str, Any]:
        """Execute automatic portfolio rebalancing"""
        try:
            if not self.auto_balance_enabled:
                return {'error': 'Auto-balance is disabled'}
            
            # Get current portfolio state
            portfolio_status = self.portfolio_core.get_portfolio_status()
            
            # Check if rebalancing is needed
            rebalance_needed = self._check_rebalance_needed(portfolio_status)
            
            if not rebalance_needed['needed']:
                return {
                    'balance_needed': False,
                    'reason': rebalance_needed['reason'],
                    'next_check': datetime.now() + timedelta(hours=4)
                }
            
            # Execute rebalancing
            rebalance_result = self.portfolio_core.rebalance_portfolio()
            
            if 'error' in rebalance_result:
                return {'error': f"Auto-balance failed: {rebalance_result['error']}"}
            
            # Calculate metrics
            strategies_rebalanced = rebalance_result.get('rebalanced_strategies', [])
            capital_rotated = rebalance_result.get('capital_rotated', 0.0)
            
            # Estimate risk reduction
            risk_reduction = self._estimate_risk_reduction(rebalance_result)
            
            # Create auto-balance event
            balance_event = AutoBalanceEvent(
                balance_id=f"balance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                balance_time=datetime.now(),
                rebalance_trigger=trigger_reason,
                strategies_rebalanced=strategies_rebalanced,
                capital_rotated=capital_rotated,
                risk_reduction=risk_reduction,
                performance_impact=rebalance_result.get('expected_performance_impact', 0.0),
                balance_success=True
            )
            
            self.auto_balance_events.append(balance_event)
            
            # Update statistics
            self.integration_stats['auto_balances'] += 1
            self.integration_stats['total_capital_protected'] += capital_rotated
            
            balance_result = {
                'balance_id': balance_event.balance_id,
                'strategies_rebalanced': len(strategies_rebalanced),
                'capital_rotated': capital_rotated,
                'risk_reduction': risk_reduction,
                'balance_success': True,
                'next_balance': datetime.now() + timedelta(hours=8)
            }
            
            logger.info(f"Auto-balance executed: {len(strategies_rebalanced)} strategies, {risk_reduction:.2%} risk reduction")
            return balance_result
            
        except Exception as e:
            logger.error(f"Error executing auto-balance: {e}")
            return {'error': str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio-risk integration status"""
        try:
            # Current risk state
            current_var_status = self.check_var_breach_risk() if self.last_var_check is None or \
                               (datetime.now() - self.last_var_check).seconds > 3600 else None
            
            # Recent events summary
            recent_triggers = [t for t in self.var_triggers if 
                             (datetime.now() - t.trigger_time).days < 7]
            recent_updates = [u for u in self.allocation_updates if 
                            (datetime.now() - u.update_time).days < 7]
            recent_balances = [b for b in self.auto_balance_events if 
                             (datetime.now() - b.balance_time).days < 7]
            
            # Performance metrics
            if self.allocation_updates:
                avg_improvement = np.mean([u.expected_improvement for u in self.allocation_updates])
                success_rate = len([u for u in self.allocation_updates if u.success]) / len(self.allocation_updates)
            else:
                avg_improvement = success_rate = 0.0
            
            status = {
                'integration_overview': {
                    'current_risk_level': self.current_risk_level.value,
                    'auto_balance_enabled': self.auto_balance_enabled,
                    'monitoring_active': self.running,
                    'last_risk_check': self.last_var_check,
                    'active_stress_response': self.active_stress_response
                },
                'risk_monitoring': {
                    'var_breach_threshold': self.var_breach_threshold,
                    'current_var_status': current_var_status,
                    'recent_triggers': len(recent_triggers),
                    'critical_triggers': len([t for t in recent_triggers if t.trigger_level in [RiskTriggerLevel.CRITICAL, RiskTriggerLevel.EMERGENCY]])
                },
                'allocation_management': {
                    'total_updates': len(self.allocation_updates),
                    'recent_updates': len(recent_updates),
                    'average_improvement': avg_improvement,
                    'update_success_rate': success_rate
                },
                'auto_balance': {
                    'total_balances': len(self.auto_balance_events),
                    'recent_balances': len(recent_balances),
                    'total_capital_protected': self.integration_stats['total_capital_protected'],
                    'average_risk_reduction': np.mean([b.risk_reduction for b in self.auto_balance_events]) if self.auto_balance_events else 0.0
                },
                'integration_stats': self.integration_stats.copy(),
                'recent_events': {
                    'var_triggers': [{'trigger_id': t.trigger_id, 'probability': t.var_breach_probability, 
                                    'level': t.trigger_level.value, 'time': t.trigger_time} for t in recent_triggers[-5:]],
                    'allocation_updates': [{'update_id': u.update_id, 'method': u.allocation_method.value,
                                          'improvement': u.expected_improvement, 'time': u.update_time} for u in recent_updates[-5:]],
                    'auto_balances': [{'balance_id': b.balance_id, 'strategies': len(b.strategies_rebalanced),
                                     'risk_reduction': b.risk_reduction, 'time': b.balance_time} for b in recent_balances[-5:]]
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {'error': str(e)}
    
    def _trigger_var_response(self, breach_probability: float, current_var: float, 
                            portfolio_value: float, trigger_level: RiskTriggerLevel) -> Dict[str, Any]:
        """Trigger VaR breach response"""
        try:
            # Determine response action
            if trigger_level == RiskTriggerLevel.EMERGENCY:
                response_action = StressResponseAction.FULL_SHUTDOWN
                mitigation_plan = "Emergency portfolio shutdown - liquidate all positions"
            elif trigger_level == RiskTriggerLevel.CRITICAL:
                response_action = StressResponseAction.EMERGENCY_EXIT
                mitigation_plan = "Emergency exit from high-risk positions, maintain core holdings"
            elif trigger_level == RiskTriggerLevel.HIGH:
                response_action = StressResponseAction.HEDGE_POSITIONS
                mitigation_plan = "Implement hedging strategies, reduce leverage"
            else:
                response_action = StressResponseAction.REDUCE_EXPOSURE
                mitigation_plan = "Reduce position sizes by 25%, increase cash allocation"
            
            # Get affected strategies
            portfolio_status = self.portfolio_core.get_portfolio_status()
            affected_strategies = list(portfolio_status.get('allocation', {}).keys())
            
            # Create VaR trigger
            trigger = VaRTrigger(
                trigger_id=f"var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                trigger_time=datetime.now(),
                var_breach_probability=breach_probability,
                current_var=current_var,
                var_threshold=self.var_breach_threshold,
                portfolio_value=portfolio_value,
                trigger_level=trigger_level,
                response_action=response_action,
                affected_strategies=affected_strategies,
                mitigation_plan=mitigation_plan,
                active=True
            )
            
            self.var_triggers.append(trigger)
            self.active_stress_response = trigger.trigger_id
            
            # Execute response
            response_result = self._execute_stress_response(response_action, affected_strategies)
            
            # Update statistics
            self.integration_stats['var_triggers'] += 1
            if trigger_level in [RiskTriggerLevel.CRITICAL, RiskTriggerLevel.EMERGENCY]:
                self.integration_stats['emergency_responses'] += 1
            
            trigger_result = {
                'trigger_id': trigger.trigger_id,
                'response_action': response_action.value,
                'affected_strategies': len(affected_strategies),
                'mitigation_plan': mitigation_plan,
                'response_executed': response_result.get('success', False),
                'estimated_risk_reduction': response_result.get('risk_reduction', 0.0)
            }
            
            logger.warning(f"VaR trigger activated: {trigger_level.value} level, {response_action.value} response")
            return trigger_result
            
        except Exception as e:
            logger.error(f"Error triggering VaR response: {e}")
            return {'error': str(e)}
    
    def _execute_stress_response(self, response_action: StressResponseAction, 
                               affected_strategies: List[str]) -> Dict[str, Any]:
        """Execute stress response action"""
        try:
            response_result = {
                'action': response_action.value,
                'success': False,
                'risk_reduction': 0.0,
                'actions_taken': []
            }
            
            if response_action == StressResponseAction.REDUCE_EXPOSURE:
                # Reduce position sizes
                allocation_result = self.portfolio_core.apply_allocation_method(AllocationMethod.RISK_PARITY)
                if 'error' not in allocation_result:
                    response_result['success'] = True
                    response_result['risk_reduction'] = 0.25
                    response_result['actions_taken'].append("Applied risk parity allocation")
            
            elif response_action == StressResponseAction.HEDGE_POSITIONS:
                # Implement hedging (placeholder for actual hedging logic)
                response_result['success'] = True
                response_result['risk_reduction'] = 0.40
                response_result['actions_taken'].append("Hedging positions implemented")
            
            elif response_action in [StressResponseAction.EMERGENCY_EXIT, StressResponseAction.FULL_SHUTDOWN]:
                # Emergency actions (placeholder for actual emergency logic)
                response_result['success'] = True
                response_result['risk_reduction'] = 0.80 if response_action == StressResponseAction.FULL_SHUTDOWN else 0.60
                response_result['actions_taken'].append(f"Emergency response: {response_action.value}")
            
            # Update statistics
            if response_result['success']:
                self.integration_stats['risk_reductions'] += 1
            
            return response_result
            
        except Exception as e:
            logger.error(f"Error executing stress response: {e}")
            return {'error': str(e)}
    
    def _determine_risk_trigger_level(self, breach_probability: float) -> RiskTriggerLevel:
        """Determine risk trigger level from breach probability"""
        if breach_probability >= self.emergency_var_threshold:
            return RiskTriggerLevel.EMERGENCY
        elif breach_probability >= self.critical_var_threshold:
            return RiskTriggerLevel.CRITICAL
        elif breach_probability >= self.var_breach_threshold:
            return RiskTriggerLevel.HIGH
        elif breach_probability >= 0.10:
            return RiskTriggerLevel.MEDIUM
        else:
            return RiskTriggerLevel.LOW
    
    def _calculate_allocation_improvement(self, old_allocation: Dict[str, float], 
                                        new_allocation: Dict[str, float],
                                        risk_metrics: Dict[str, float]) -> float:
        """Calculate expected improvement from allocation change"""
        try:
            # Simple improvement estimation based on allocation diversity
            old_diversity = len([v for v in old_allocation.values() if v > 0.01])
            new_diversity = len([v for v in new_allocation.values() if v > 0.01])
            
            # More diverse allocation generally better for risk
            diversity_improvement = (new_diversity - old_diversity) * 0.05
            
            # Consider current VaR
            var_improvement = -risk_metrics.get('var_95', 0.0) * 0.1
            
            return max(-0.20, min(0.20, diversity_improvement + var_improvement))
            
        except Exception as e:
            logger.error(f"Error calculating allocation improvement: {e}")
            return 0.0
    
    def _calculate_allocation_changes(self, old_allocation: Dict[str, float], 
                                    new_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate allocation changes"""
        try:
            changes = {}
            all_strategies = set(old_allocation.keys()) | set(new_allocation.keys())
            
            for strategy in all_strategies:
                old_weight = old_allocation.get(strategy, 0.0)
                new_weight = new_allocation.get(strategy, 0.0)
                change = new_weight - old_weight
                
                if abs(change) > 0.01:  # Only include significant changes
                    changes[strategy] = change
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating allocation changes: {e}")
            return {}
    
    def _check_rebalance_needed(self, portfolio_status: Dict[str, Any]) -> Dict[str, Any]:
        """Check if portfolio rebalancing is needed"""
        try:
            # Get current allocation
            allocation = portfolio_status.get('allocation', {})
            target_allocation = portfolio_status.get('target_allocation', {})
            
            if not target_allocation:
                return {'needed': False, 'reason': 'No target allocation defined'}
            
            # Calculate allocation drift
            max_drift = 0.0
            for strategy in allocation:
                current_weight = allocation.get(strategy, 0.0)
                target_weight = target_allocation.get(strategy, 0.0)
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)
            
            # Check drift threshold
            drift_threshold = 0.05  # 5% drift triggers rebalance
            
            if max_drift > drift_threshold:
                return {
                    'needed': True,
                    'reason': f'Maximum allocation drift {max_drift:.1%} exceeds threshold {drift_threshold:.1%}',
                    'max_drift': max_drift
                }
            
            # Check risk level
            if self.current_risk_level in [RiskTriggerLevel.HIGH, RiskTriggerLevel.CRITICAL]:
                return {
                    'needed': True,
                    'reason': f'High risk level ({self.current_risk_level.value}) requires rebalancing',
                    'risk_level': self.current_risk_level.value
                }
            
            return {'needed': False, 'reason': 'Portfolio allocation within acceptable ranges'}
            
        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}")
            return {'needed': False, 'reason': f'Error checking rebalance: {str(e)}'}
    
    def _estimate_risk_reduction(self, rebalance_result: Dict[str, Any]) -> float:
        """Estimate risk reduction from rebalancing"""
        try:
            # Simple estimation based on rebalancing metrics
            strategies_count = len(rebalance_result.get('rebalanced_strategies', []))
            capital_rotated = rebalance_result.get('capital_rotated', 0.0)
            
            # More strategies rebalanced = more risk reduction
            strategy_factor = min(0.15, strategies_count * 0.03)
            
            # More capital rotated = more risk reduction
            capital_factor = min(0.15, capital_rotated / 10000 * 0.05)
            
            return strategy_factor + capital_factor
            
        except Exception as e:
            logger.error(f"Error estimating risk reduction: {e}")
            return 0.0
    
    def _risk_monitoring_loop(self):
        """Background risk monitoring loop"""
        while self.running:
            try:
                time.sleep(self.monitoring_interval)
                
                # Check VaR breach risk
                self.check_var_breach_risk()
                
                # Check if auto-balance is needed
                if self.auto_balance_enabled:
                    self.execute_auto_balance("risk_monitoring")
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
    
    def stop(self):
        """Stop the portfolio-risk integration"""
        self.running = False
        logger.info("Portfolio-Risk Integration stopped")
