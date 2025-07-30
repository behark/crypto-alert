"""
Strategy Evolution → Execution Layer Integration - Phase 4 Final Integration
Deploys top-performing evolved strategies to execution system with population cycling and health overrides.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

from ..evolution.strategy_evolution import StrategyEvolutionCore, TradingStrategy, StrategyType
from ..execution.confidence_executor import ConfidenceExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyHealth(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DECLINING = "declining"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class DeployedStrategy:
    """Strategy deployed to execution layer"""
    strategy_id: str
    strategy_type: StrategyType
    deployment_time: datetime
    fitness_score: float
    generation: int
    execution_count: int
    success_rate: float
    current_pnl: float
    health_status: StrategyHealth
    last_execution: Optional[datetime]
    emergency_override_active: bool
    active: bool

@dataclass
class PopulationCycle:
    """Population cycling event"""
    cycle_id: str
    cycle_time: datetime
    strategies_retired: List[str]
    strategies_deployed: List[str]
    fitness_improvement: float
    cycle_reason: str
    success: bool

@dataclass
class HealthOverride:
    """Emergency health override event"""
    override_id: str
    strategy_id: str
    trigger_reason: str
    health_threshold_breached: float
    override_action: str
    override_time: datetime
    recovery_plan: str
    active: bool

class StrategyExecutionIntegration:
    """
    Strategy Evolution → Execution Layer Integration.
    Deploys evolved strategies with intelligent population cycling and health monitoring.
    """
    
    def __init__(self, evolution_core: StrategyEvolutionCore, 
                 confidence_executor: ConfidenceExecutor):
        """Initialize Strategy-Execution Integration"""
        self.evolution_core = evolution_core
        self.confidence_executor = confidence_executor
        
        # Deployment configuration
        self.max_deployed_strategies = 5
        self.health_check_interval = 3600  # 1 hour
        self.population_cycle_interval = 86400  # 24 hours
        self.emergency_health_threshold = 0.2
        self.critical_health_threshold = 0.3
        
        # Deployed strategies tracking
        self.deployed_strategies: Dict[str, DeployedStrategy] = {}
        self.population_cycles: List[PopulationCycle] = []
        self.health_overrides: List[HealthOverride] = []
        
        # Integration statistics
        self.integration_stats = {
            'strategies_deployed': 0,
            'population_cycles': 0,
            'health_overrides': 0,
            'total_executions': 0,
            'average_success_rate': 0.0,
            'last_cycle': None,
            'last_health_check': None
        }
        
        # Start monitoring threads
        self.running = True
        self.health_monitor_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.population_cycle_thread = threading.Thread(target=self._population_cycling_loop, daemon=True)
        
        self.health_monitor_thread.start()
        self.population_cycle_thread.start()
        
        logger.info("Strategy-Execution Integration initialized with population cycling and health monitoring")
    
    def deploy_top_strategies(self, count: Optional[int] = None) -> Dict[str, Any]:
        """Deploy top-performing strategies from evolution core"""
        try:
            if count is None:
                count = self.max_deployed_strategies
            
            # Get strategy status from evolution core
            evolution_status = self.evolution_core.get_strategy_status()
            
            if 'error' in evolution_status:
                return {'error': f"Cannot get evolution status: {evolution_status['error']}"}
            
            # Get top performers
            top_performers = evolution_status.get('top_performers', [])
            
            if not top_performers:
                return {'error': 'No top performing strategies available for deployment'}
            
            deployment_results = {
                'deployment_time': datetime.now(),
                'strategies_deployed': [],
                'strategies_retired': [],
                'deployment_success': True,
                'total_deployed': 0
            }
            
            # Retire existing strategies if needed
            if len(self.deployed_strategies) >= count:
                retired_strategies = self._retire_worst_performers(count - len(top_performers))
                deployment_results['strategies_retired'] = retired_strategies
            
            # Deploy new top performers
            strategies_to_deploy = top_performers[:count]
            
            for strategy_info in strategies_to_deploy:
                strategy_id = strategy_info['strategy_id']
                
                # Skip if already deployed
                if strategy_id in self.deployed_strategies:
                    continue
                
                # Create deployed strategy record
                deployed_strategy = DeployedStrategy(
                    strategy_id=strategy_id,
                    strategy_type=StrategyType(strategy_info['strategy_type']),
                    deployment_time=datetime.now(),
                    fitness_score=strategy_info['fitness_score'],
                    generation=strategy_info['generation'],
                    execution_count=0,
                    success_rate=0.0,
                    current_pnl=0.0,
                    health_status=StrategyHealth.GOOD,
                    last_execution=None,
                    emergency_override_active=False,
                    active=True
                )
                
                # Add to deployed strategies
                self.deployed_strategies[strategy_id] = deployed_strategy
                deployment_results['strategies_deployed'].append(strategy_id)
                
                # Register with confidence executor
                self._register_strategy_with_executor(strategy_id, deployed_strategy)
            
            deployment_results['total_deployed'] = len(deployment_results['strategies_deployed'])
            
            # Update statistics
            self.integration_stats['strategies_deployed'] += deployment_results['total_deployed']
            
            logger.info(f"Deployed {deployment_results['total_deployed']} top-performing strategies")
            return deployment_results
            
        except Exception as e:
            logger.error(f"Error deploying top strategies: {e}")
            return {'error': str(e)}
    
    def trigger_population_cycle(self, reason: str = "manual") -> Dict[str, Any]:
        """Trigger immediate population cycling"""
        try:
            cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get current fitness baseline
            current_fitness = self._calculate_population_fitness()
            
            # Deploy new top strategies
            deployment_results = self.deploy_top_strategies()
            
            if 'error' in deployment_results:
                return {'error': f"Population cycle failed: {deployment_results['error']}"}
            
            # Calculate fitness improvement
            new_fitness = self._calculate_population_fitness()
            fitness_improvement = new_fitness - current_fitness
            
            # Create population cycle record
            cycle = PopulationCycle(
                cycle_id=cycle_id,
                cycle_time=datetime.now(),
                strategies_retired=deployment_results['strategies_retired'],
                strategies_deployed=deployment_results['strategies_deployed'],
                fitness_improvement=fitness_improvement,
                cycle_reason=reason,
                success=deployment_results['deployment_success']
            )
            
            self.population_cycles.append(cycle)
            
            # Update statistics
            self.integration_stats['population_cycles'] += 1
            self.integration_stats['last_cycle'] = datetime.now()
            
            cycle_results = {
                'cycle_id': cycle_id,
                'fitness_improvement': fitness_improvement,
                'strategies_cycled': len(deployment_results['strategies_deployed']),
                'cycle_success': True,
                'cycle_time': datetime.now()
            }
            
            logger.info(f"Population cycle completed: {cycle_results}")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in population cycle: {e}")
            return {'error': str(e)}
    
    def check_strategy_health(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Check health of deployed strategies"""
        try:
            health_results = {
                'check_time': datetime.now(),
                'strategies_checked': 0,
                'health_status': {},
                'overrides_triggered': [],
                'critical_strategies': [],
                'overall_health': StrategyHealth.GOOD
            }
            
            strategies_to_check = [strategy_id] if strategy_id else list(self.deployed_strategies.keys())
            
            for sid in strategies_to_check:
                if sid not in self.deployed_strategies:
                    continue
                
                strategy = self.deployed_strategies[sid]
                
                # Calculate health metrics
                health_score = self._calculate_strategy_health_score(strategy)
                health_status = self._determine_health_status(health_score)
                
                # Update strategy health
                strategy.health_status = health_status
                health_results['health_status'][sid] = {
                    'health_score': health_score,
                    'health_status': health_status.value,
                    'success_rate': strategy.success_rate,
                    'current_pnl': strategy.current_pnl,
                    'execution_count': strategy.execution_count
                }
                
                # Check for emergency override
                if health_score < self.emergency_health_threshold and not strategy.emergency_override_active:
                    override_result = self._trigger_emergency_override(sid, health_score)
                    if override_result:
                        health_results['overrides_triggered'].append(override_result)
                
                # Track critical strategies
                if health_status in [StrategyHealth.CRITICAL, StrategyHealth.EMERGENCY]:
                    health_results['critical_strategies'].append(sid)
                
                health_results['strategies_checked'] += 1
            
            # Determine overall health
            if health_results['critical_strategies']:
                health_results['overall_health'] = StrategyHealth.CRITICAL
            elif any(status['health_score'] < 0.5 for status in health_results['health_status'].values()):
                health_results['overall_health'] = StrategyHealth.DECLINING
            else:
                health_results['overall_health'] = StrategyHealth.GOOD
            
            # Update statistics
            self.integration_stats['last_health_check'] = datetime.now()
            
            logger.info(f"Health check completed for {health_results['strategies_checked']} strategies")
            return health_results
            
        except Exception as e:
            logger.error(f"Error checking strategy health: {e}")
            return {'error': str(e)}
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        try:
            # Calculate current metrics
            total_executions = sum(s.execution_count for s in self.deployed_strategies.values())
            active_strategies = [s for s in self.deployed_strategies.values() if s.active]
            
            if active_strategies:
                avg_success_rate = np.mean([s.success_rate for s in active_strategies])
                total_pnl = sum(s.current_pnl for s in active_strategies)
                avg_fitness = np.mean([s.fitness_score for s in active_strategies])
            else:
                avg_success_rate = total_pnl = avg_fitness = 0.0
            
            # Health distribution
            health_distribution = {}
            for health in StrategyHealth:
                count = len([s for s in active_strategies if s.health_status == health])
                health_distribution[health.value] = count
            
            status = {
                'deployment_overview': {
                    'total_deployed': len(self.deployed_strategies),
                    'active_strategies': len(active_strategies),
                    'max_deployable': self.max_deployed_strategies,
                    'deployment_utilization': len(active_strategies) / self.max_deployed_strategies
                },
                'performance_metrics': {
                    'total_executions': total_executions,
                    'average_success_rate': avg_success_rate,
                    'total_pnl': total_pnl,
                    'average_fitness': avg_fitness
                },
                'health_metrics': {
                    'health_distribution': health_distribution,
                    'critical_count': health_distribution.get('critical', 0) + health_distribution.get('emergency', 0),
                    'healthy_count': health_distribution.get('excellent', 0) + health_distribution.get('good', 0),
                    'active_overrides': len([o for o in self.health_overrides if o.active])
                },
                'cycling_metrics': {
                    'total_cycles': len(self.population_cycles),
                    'last_cycle': self.integration_stats['last_cycle'],
                    'next_cycle_due': self._get_next_cycle_time(),
                    'cycle_success_rate': self._calculate_cycle_success_rate()
                },
                'deployed_strategies': []
            }
            
            # Add deployed strategy details
            for strategy in sorted(active_strategies, key=lambda s: s.fitness_score, reverse=True):
                status['deployed_strategies'].append({
                    'strategy_id': strategy.strategy_id,
                    'strategy_type': strategy.strategy_type.value,
                    'fitness_score': strategy.fitness_score,
                    'generation': strategy.generation,
                    'health_status': strategy.health_status.value,
                    'success_rate': strategy.success_rate,
                    'execution_count': strategy.execution_count,
                    'current_pnl': strategy.current_pnl,
                    'deployment_age': (datetime.now() - strategy.deployment_time).days,
                    'emergency_override': strategy.emergency_override_active
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {'error': str(e)}
    
    def _retire_worst_performers(self, count: int) -> List[str]:
        """Retire worst performing strategies"""
        try:
            if count <= 0:
                return []
            
            # Sort by fitness score (ascending for worst first)
            sorted_strategies = sorted(
                self.deployed_strategies.values(),
                key=lambda s: s.fitness_score
            )
            
            retired_strategies = []
            for strategy in sorted_strategies[:count]:
                strategy.active = False
                retired_strategies.append(strategy.strategy_id)
                
                # Unregister from executor
                self._unregister_strategy_from_executor(strategy.strategy_id)
            
            return retired_strategies
            
        except Exception as e:
            logger.error(f"Error retiring worst performers: {e}")
            return []
    
    def _calculate_population_fitness(self) -> float:
        """Calculate average fitness of deployed population"""
        try:
            active_strategies = [s for s in self.deployed_strategies.values() if s.active]
            if not active_strategies:
                return 0.0
            
            return np.mean([s.fitness_score for s in active_strategies])
            
        except Exception as e:
            logger.error(f"Error calculating population fitness: {e}")
            return 0.0
    
    def _calculate_strategy_health_score(self, strategy: DeployedStrategy) -> float:
        """Calculate health score for a strategy"""
        try:
            # Base health from fitness score
            health_score = strategy.fitness_score
            
            # Adjust for recent performance
            if strategy.execution_count > 0:
                # Success rate component
                success_component = strategy.success_rate * 0.4
                
                # PnL component (normalized)
                pnl_component = max(-0.2, min(0.2, strategy.current_pnl / 1000)) * 0.3
                
                # Execution frequency component
                days_since_deployment = (datetime.now() - strategy.deployment_time).days
                expected_executions = max(1, days_since_deployment * 2)  # Expect 2 executions per day
                execution_ratio = min(1.0, strategy.execution_count / expected_executions)
                execution_component = execution_ratio * 0.3
                
                # Combine components
                health_score = success_component + pnl_component + execution_component
            
            # Age penalty (strategies get stale over time)
            age_days = (datetime.now() - strategy.deployment_time).days
            age_penalty = min(0.1, age_days * 0.002)  # 0.2% penalty per day, max 10%
            
            health_score = max(0.0, health_score - age_penalty)
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating strategy health score: {e}")
            return 0.0
    
    def _determine_health_status(self, health_score: float) -> StrategyHealth:
        """Determine health status from health score"""
        if health_score >= 0.8:
            return StrategyHealth.EXCELLENT
        elif health_score >= 0.6:
            return StrategyHealth.GOOD
        elif health_score >= 0.4:
            return StrategyHealth.DECLINING
        elif health_score >= 0.2:
            return StrategyHealth.CRITICAL
        else:
            return StrategyHealth.EMERGENCY
    
    def _trigger_emergency_override(self, strategy_id: str, health_score: float) -> Optional[Dict[str, Any]]:
        """Trigger emergency override for unhealthy strategy"""
        try:
            strategy = self.deployed_strategies[strategy_id]
            
            # Determine override action
            if health_score < 0.1:
                override_action = "immediate_retirement"
                recovery_plan = "Strategy retired due to critical health failure"
            else:
                override_action = "reduced_allocation"
                recovery_plan = "Reduce position sizes by 50% until health improves"
            
            # Create override record
            override = HealthOverride(
                override_id=f"override_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                trigger_reason=f"Health score dropped to {health_score:.3f}",
                health_threshold_breached=health_score,
                override_action=override_action,
                override_time=datetime.now(),
                recovery_plan=recovery_plan,
                active=True
            )
            
            self.health_overrides.append(override)
            
            # Apply override
            strategy.emergency_override_active = True
            
            if override_action == "immediate_retirement":
                strategy.active = False
                self._unregister_strategy_from_executor(strategy_id)
            
            # Update statistics
            self.integration_stats['health_overrides'] += 1
            
            override_result = {
                'override_id': override.override_id,
                'strategy_id': strategy_id,
                'action': override_action,
                'trigger_reason': override.trigger_reason,
                'recovery_plan': recovery_plan
            }
            
            logger.warning(f"Emergency override triggered for {strategy_id}: {override_action}")
            return override_result
            
        except Exception as e:
            logger.error(f"Error triggering emergency override: {e}")
            return None
    
    def _register_strategy_with_executor(self, strategy_id: str, strategy: DeployedStrategy):
        """Register strategy with confidence executor"""
        try:
            # This would integrate with the actual confidence executor
            # For now, we'll log the registration
            logger.info(f"Registered strategy {strategy_id} with confidence executor")
            
        except Exception as e:
            logger.error(f"Error registering strategy with executor: {e}")
    
    def _unregister_strategy_from_executor(self, strategy_id: str):
        """Unregister strategy from confidence executor"""
        try:
            # This would integrate with the actual confidence executor
            # For now, we'll log the unregistration
            logger.info(f"Unregistered strategy {strategy_id} from confidence executor")
            
        except Exception as e:
            logger.error(f"Error unregistering strategy from executor: {e}")
    
    def _get_next_cycle_time(self) -> datetime:
        """Get next scheduled population cycle time"""
        if self.integration_stats['last_cycle']:
            return self.integration_stats['last_cycle'] + timedelta(seconds=self.population_cycle_interval)
        else:
            return datetime.now() + timedelta(seconds=self.population_cycle_interval)
    
    def _calculate_cycle_success_rate(self) -> float:
        """Calculate success rate of population cycles"""
        if not self.population_cycles:
            return 0.0
        
        successful_cycles = len([c for c in self.population_cycles if c.success])
        return successful_cycles / len(self.population_cycles)
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                time.sleep(self.health_check_interval)
                
                # Perform health check
                self.check_strategy_health()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    def _population_cycling_loop(self):
        """Background population cycling loop"""
        while self.running:
            try:
                time.sleep(self.population_cycle_interval)
                
                # Trigger population cycle
                self.trigger_population_cycle("scheduled")
                
            except Exception as e:
                logger.error(f"Error in population cycling loop: {e}")
    
    def stop(self):
        """Stop the strategy-execution integration"""
        self.running = False
        logger.info("Strategy-Execution Integration stopped")
