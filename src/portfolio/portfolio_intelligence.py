"""
Portfolio Intelligence Core - Phase 4 Evolution Layer
Kelly Criterion capital allocation, cross-bot coordination, and portfolio-level risk management.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading
import time
import numpy as np
from collections import defaultdict, deque
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllocationMethod(Enum):
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MOMENTUM_BASED = "momentum_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    DYNAMIC_ALLOCATION = "dynamic_allocation"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class PortfolioPosition:
    """Individual position in the portfolio"""
    symbol: str
    strategy_id: str
    bot_id: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    allocation_weight: float
    risk_contribution: float
    entry_time: datetime
    last_update: datetime

@dataclass
class StrategyAllocation:
    """Allocation for a specific strategy"""
    strategy_id: str
    bot_id: str
    strategy_type: str
    allocated_capital: float
    current_positions: List[PortfolioPosition]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    kelly_fraction: float
    allocation_weight: float
    active: bool

@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics"""
    total_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    beta: float
    alpha: float
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_contribution: Dict[str, float]
    timestamp: datetime

@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    scenario_id: str
    name: str
    description: str
    market_shocks: Dict[str, float]  # symbol -> price change %
    volatility_multipliers: Dict[str, float]
    correlation_adjustments: Dict[str, Dict[str, float]]
    duration_days: int
    probability: float

@dataclass
class StressTestResult:
    """Results of portfolio stress testing"""
    scenario_id: str
    portfolio_pnl: float
    portfolio_return: float
    max_drawdown: float
    var_breach: bool
    positions_at_risk: List[str]
    recovery_time_days: int
    risk_adjusted_return: float
    timestamp: datetime

class PortfolioIntelligenceCore:
    """
    Portfolio Intelligence Core - Intelligent capital allocation and risk management.
    Implements Kelly Criterion, risk parity, and cross-bot coordination.
    """
    
    def __init__(self, data_dir: str = "data/portfolio"):
        """Initialize Portfolio Intelligence Core"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Portfolio configuration
        self.total_capital = 100000.0  # Base capital
        self.max_position_size = 0.2  # 20% max per position
        self.max_strategy_allocation = 0.3  # 30% max per strategy
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.rebalance_threshold = 0.05  # 5% drift threshold
        
        # Current portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.available_capital = self.total_capital
        
        # Risk management
        self.risk_limits = {
            'max_portfolio_var': 0.05,  # 5% daily VaR limit
            'max_correlation': 0.8,  # Max correlation between strategies
            'max_concentration': 0.25,  # Max 25% in single asset
            'max_leverage': 2.0  # Max 2x leverage
        }
        
        # Performance tracking
        self.portfolio_history: List[PortfolioMetrics] = []
        self.rebalance_history: List[Dict[str, Any]] = []
        
        # Stress testing scenarios
        self.stress_scenarios: Dict[str, StressTestScenario] = {}
        self.stress_results: List[StressTestResult] = []
        
        # Portfolio statistics
        self.portfolio_stats = {
            'total_positions': 0,
            'active_strategies': 0,
            'rebalances_performed': 0,
            'stress_tests_run': 0,
            'last_rebalance': None,
            'last_stress_test': None
        }
        
        # Initialize default stress scenarios
        self._initialize_stress_scenarios()
        
        # Load existing portfolio state
        self._load_portfolio_state()
        
        # Start portfolio monitoring thread
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._portfolio_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Portfolio Intelligence Core initialized with Kelly Criterion and risk management")
    
    def calculate_kelly_allocation(self, strategy_performance: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion optimal allocation for a strategy"""
        try:
            # Extract performance metrics
            win_rate = strategy_performance.get('win_rate', 0.5)
            avg_win = strategy_performance.get('avg_win', 0.02)
            avg_loss = strategy_performance.get('avg_loss', -0.01)
            
            # Avoid division by zero
            if avg_loss >= 0:
                avg_loss = -0.01
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win / |avg_loss|, p = win_rate, q = 1 - win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety constraints
            kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Apply confidence adjustment
            confidence = strategy_performance.get('confidence', 0.5)
            adjusted_kelly = kelly_fraction * confidence
            
            logger.info(f"Kelly allocation calculated: {adjusted_kelly:.4f} (raw: {kelly_fraction:.4f})")
            return adjusted_kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly allocation: {e}")
            return 0.05  # Conservative default
    
    def optimize_portfolio_allocation(self, allocation_method: AllocationMethod = AllocationMethod.KELLY_CRITERION) -> Dict[str, float]:
        """Optimize portfolio allocation across all strategies"""
        try:
            if not self.strategy_allocations:
                logger.warning("No strategies available for allocation optimization")
                return {}
            
            allocations = {}
            
            if allocation_method == AllocationMethod.KELLY_CRITERION:
                allocations = self._kelly_criterion_optimization()
            elif allocation_method == AllocationMethod.RISK_PARITY:
                allocations = self._risk_parity_optimization()
            elif allocation_method == AllocationMethod.EQUAL_WEIGHT:
                allocations = self._equal_weight_optimization()
            elif allocation_method == AllocationMethod.DYNAMIC_ALLOCATION:
                allocations = self._dynamic_allocation_optimization()
            else:
                allocations = self._kelly_criterion_optimization()  # Default
            
            # Normalize allocations to sum to 1.0
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                allocations = {k: v / total_allocation for k, v in allocations.items()}
            
            # Apply maximum allocation constraints
            for strategy_id in allocations:
                allocations[strategy_id] = min(allocations[strategy_id], self.max_strategy_allocation)
            
            logger.info(f"Portfolio allocation optimized using {allocation_method.value}")
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return {}
    
    def rebalance_portfolio(self, target_allocations: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Rebalance portfolio to target allocations"""
        try:
            if target_allocations is None:
                target_allocations = self.optimize_portfolio_allocation()
            
            if not target_allocations:
                return {'error': 'No target allocations provided'}
            
            rebalance_results = {
                'timestamp': datetime.now(),
                'total_capital': self.total_capital,
                'rebalance_actions': [],
                'allocation_changes': {},
                'risk_impact': {},
                'expected_improvement': 0.0
            }
            
            # Calculate current allocations
            current_allocations = {}
            for strategy_id, allocation in self.strategy_allocations.items():
                current_allocations[strategy_id] = allocation.allocated_capital / self.total_capital
            
            # Calculate rebalancing actions
            for strategy_id, target_weight in target_allocations.items():
                current_weight = current_allocations.get(strategy_id, 0.0)
                weight_change = target_weight - current_weight
                
                if abs(weight_change) > self.rebalance_threshold:
                    capital_change = weight_change * self.total_capital
                    
                    rebalance_action = {
                        'strategy_id': strategy_id,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_change': weight_change,
                        'capital_change': capital_change,
                        'action': 'increase' if capital_change > 0 else 'decrease'
                    }
                    
                    rebalance_results['rebalance_actions'].append(rebalance_action)
                    rebalance_results['allocation_changes'][strategy_id] = weight_change
                    
                    # Update strategy allocation
                    if strategy_id in self.strategy_allocations:
                        self.strategy_allocations[strategy_id].allocated_capital += capital_change
                        self.strategy_allocations[strategy_id].allocation_weight = target_weight
            
            # Update available capital
            total_allocated = sum(alloc.allocated_capital for alloc in self.strategy_allocations.values())
            self.available_capital = self.total_capital - total_allocated
            
            # Calculate expected improvement
            current_portfolio_metrics = self.calculate_portfolio_metrics()
            rebalance_results['expected_improvement'] = self._estimate_rebalance_benefit(target_allocations)
            
            # Store rebalance history
            self.rebalance_history.append(rebalance_results)
            self.portfolio_stats['rebalances_performed'] += 1
            self.portfolio_stats['last_rebalance'] = datetime.now()
            
            # Save portfolio state
            self._save_portfolio_state()
            
            logger.info(f"Portfolio rebalanced with {len(rebalance_results['rebalance_actions'])} actions")
            return rebalance_results
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {'error': str(e)}
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Calculate basic metrics
            total_value = sum(pos.current_price * pos.position_size for pos in self.positions.values())
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            total_return = total_pnl / self.total_capital if self.total_capital > 0 else 0.0
            
            # Calculate risk metrics
            returns = self._get_portfolio_returns()
            
            if len(returns) > 1:
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                sortino_ratio = self._calculate_sortino_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(returns)
                var_95 = np.percentile(returns, 5) if returns else 0.0
                var_99 = np.percentile(returns, 1) if returns else 0.0
            else:
                sharpe_ratio = sortino_ratio = max_drawdown = var_95 = var_99 = 0.0
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix()
            
            # Calculate risk contribution
            risk_contribution = self._calculate_risk_contribution()
            
            metrics = PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                beta=0.0,  # Would need market benchmark
                alpha=0.0,  # Would need market benchmark
                correlation_matrix=correlation_matrix,
                risk_contribution=risk_contribution,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.portfolio_history.append(metrics)
            
            # Keep only recent history (last 1000 entries)
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(
                total_value=0.0, total_pnl=0.0, unrealized_pnl=0.0, realized_pnl=0.0,
                total_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                var_95=0.0, var_99=0.0, beta=0.0, alpha=0.0,
                correlation_matrix={}, risk_contribution={}, timestamp=datetime.now()
            )
    
    def run_stress_test(self, scenario_ids: Optional[List[str]] = None) -> List[StressTestResult]:
        """Run portfolio stress tests"""
        try:
            if scenario_ids is None:
                scenario_ids = list(self.stress_scenarios.keys())
            
            stress_results = []
            
            for scenario_id in scenario_ids:
                if scenario_id not in self.stress_scenarios:
                    continue
                
                scenario = self.stress_scenarios[scenario_id]
                result = self._run_single_stress_test(scenario)
                stress_results.append(result)
            
            # Store results
            self.stress_results.extend(stress_results)
            self.portfolio_stats['stress_tests_run'] += len(stress_results)
            self.portfolio_stats['last_stress_test'] = datetime.now()
            
            # Keep only recent results (last 100)
            if len(self.stress_results) > 100:
                self.stress_results = self.stress_results[-100:]
            
            logger.info(f"Completed {len(stress_results)} stress tests")
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return []
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            current_metrics = self.calculate_portfolio_metrics()
            
            # Calculate allocation breakdown
            allocation_breakdown = {}
            for strategy_id, allocation in self.strategy_allocations.items():
                allocation_breakdown[strategy_id] = {
                    'allocated_capital': allocation.allocated_capital,
                    'allocation_weight': allocation.allocation_weight,
                    'strategy_type': allocation.strategy_type,
                    'bot_id': allocation.bot_id,
                    'position_count': len(allocation.current_positions),
                    'performance': allocation.performance_metrics,
                    'kelly_fraction': allocation.kelly_fraction
                }
            
            # Calculate risk metrics
            risk_metrics = {
                'portfolio_var_95': current_metrics.var_95,
                'portfolio_var_99': current_metrics.var_99,
                'max_drawdown': current_metrics.max_drawdown,
                'sharpe_ratio': current_metrics.sharpe_ratio,
                'total_risk_contribution': sum(current_metrics.risk_contribution.values())
            }
            
            # Recent performance
            recent_returns = self._get_portfolio_returns()[-30:] if self.portfolio_history else []
            recent_performance = {
                'avg_daily_return': np.mean(recent_returns) if recent_returns else 0.0,
                'volatility': np.std(recent_returns) if recent_returns else 0.0,
                'best_day': np.max(recent_returns) if recent_returns else 0.0,
                'worst_day': np.min(recent_returns) if recent_returns else 0.0
            }
            
            status = {
                'portfolio_overview': {
                    'total_capital': self.total_capital,
                    'available_capital': self.available_capital,
                    'total_value': current_metrics.total_value,
                    'total_return': current_metrics.total_return,
                    'total_positions': len(self.positions),
                    'active_strategies': len([a for a in self.strategy_allocations.values() if a.active])
                },
                'performance_metrics': {
                    'total_pnl': current_metrics.total_pnl,
                    'unrealized_pnl': current_metrics.unrealized_pnl,
                    'realized_pnl': current_metrics.realized_pnl,
                    'sharpe_ratio': current_metrics.sharpe_ratio,
                    'sortino_ratio': current_metrics.sortino_ratio,
                    'max_drawdown': current_metrics.max_drawdown
                },
                'risk_metrics': risk_metrics,
                'allocation_breakdown': allocation_breakdown,
                'recent_performance': recent_performance,
                'portfolio_health': self._assess_portfolio_health(current_metrics),
                'last_rebalance': self.portfolio_stats['last_rebalance'],
                'rebalance_needed': self._check_rebalance_needed()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {'error': str(e)}
    
    def _kelly_criterion_optimization(self) -> Dict[str, float]:
        """Optimize allocations using Kelly Criterion"""
        try:
            allocations = {}
            
            for strategy_id, allocation in self.strategy_allocations.items():
                if allocation.active:
                    kelly_fraction = self.calculate_kelly_allocation(allocation.performance_metrics)
                    allocations[strategy_id] = kelly_fraction
                    
                    # Update stored Kelly fraction
                    allocation.kelly_fraction = kelly_fraction
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in Kelly Criterion optimization: {e}")
            return {}
    
    def _risk_parity_optimization(self) -> Dict[str, float]:
        """Optimize allocations using Risk Parity"""
        try:
            allocations = {}
            
            # Calculate risk contributions
            risk_contributions = {}
            for strategy_id, allocation in self.strategy_allocations.items():
                if allocation.active:
                    volatility = allocation.risk_metrics.get('volatility', 0.02)
                    risk_contributions[strategy_id] = volatility
            
            # Inverse volatility weighting
            if risk_contributions:
                total_inv_vol = sum(1/vol for vol in risk_contributions.values())
                for strategy_id, vol in risk_contributions.items():
                    allocations[strategy_id] = (1/vol) / total_inv_vol
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in Risk Parity optimization: {e}")
            return {}
    
    def _equal_weight_optimization(self) -> Dict[str, float]:
        """Equal weight allocation"""
        try:
            active_strategies = [sid for sid, alloc in self.strategy_allocations.items() if alloc.active]
            
            if active_strategies:
                weight = 1.0 / len(active_strategies)
                return {sid: weight for sid in active_strategies}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in Equal Weight optimization: {e}")
            return {}
    
    def _dynamic_allocation_optimization(self) -> Dict[str, float]:
        """Dynamic allocation based on recent performance"""
        try:
            allocations = {}
            
            # Combine Kelly Criterion with momentum
            kelly_allocations = self._kelly_criterion_optimization()
            
            for strategy_id, kelly_weight in kelly_allocations.items():
                allocation = self.strategy_allocations[strategy_id]
                
                # Recent performance momentum
                recent_return = allocation.performance_metrics.get('recent_return', 0.0)
                momentum_multiplier = 1.0 + np.tanh(recent_return * 10)  # Scale momentum
                
                # Risk adjustment
                volatility = allocation.risk_metrics.get('volatility', 0.02)
                risk_adjustment = 1.0 / (1.0 + volatility * 5)
                
                # Combined allocation
                dynamic_weight = kelly_weight * momentum_multiplier * risk_adjustment
                allocations[strategy_id] = dynamic_weight
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in Dynamic allocation optimization: {e}")
            return {}
    
    def _initialize_stress_scenarios(self):
        """Initialize default stress test scenarios"""
        try:
            scenarios = [
                {
                    'id': 'market_crash',
                    'name': 'Market Crash',
                    'description': '20% market decline with increased volatility',
                    'market_shocks': {'BTCUSDT': -0.2, 'ETHUSDT': -0.25, 'ADAUSDT': -0.3},
                    'volatility_multipliers': {'BTCUSDT': 2.0, 'ETHUSDT': 2.2, 'ADAUSDT': 2.5},
                    'correlation_adjustments': {},
                    'duration_days': 7,
                    'probability': 0.05
                },
                {
                    'id': 'flash_crash',
                    'name': 'Flash Crash',
                    'description': 'Sudden 10% drop with quick recovery',
                    'market_shocks': {'BTCUSDT': -0.1, 'ETHUSDT': -0.12, 'ADAUSDT': -0.15},
                    'volatility_multipliers': {'BTCUSDT': 5.0, 'ETHUSDT': 5.0, 'ADAUSDT': 5.0},
                    'correlation_adjustments': {},
                    'duration_days': 1,
                    'probability': 0.1
                },
                {
                    'id': 'correlation_spike',
                    'name': 'Correlation Spike',
                    'description': 'All assets move together (correlation = 0.9)',
                    'market_shocks': {},
                    'volatility_multipliers': {},
                    'correlation_adjustments': {'BTCUSDT': {'ETHUSDT': 0.9, 'ADAUSDT': 0.9}},
                    'duration_days': 14,
                    'probability': 0.15
                }
            ]
            
            for scenario_config in scenarios:
                scenario = StressTestScenario(
                    scenario_id=scenario_config['id'],
                    name=scenario_config['name'],
                    description=scenario_config['description'],
                    market_shocks=scenario_config['market_shocks'],
                    volatility_multipliers=scenario_config['volatility_multipliers'],
                    correlation_adjustments=scenario_config['correlation_adjustments'],
                    duration_days=scenario_config['duration_days'],
                    probability=scenario_config['probability']
                )
                self.stress_scenarios[scenario.scenario_id] = scenario
            
            logger.info(f"Initialized {len(scenarios)} stress test scenarios")
            
        except Exception as e:
            logger.error(f"Error initializing stress scenarios: {e}")
    
    def _portfolio_monitoring_loop(self):
        """Background portfolio monitoring loop"""
        while self.running:
            try:
                time.sleep(3600)  # Run every hour
                
                # Check if rebalancing is needed
                if self._check_rebalance_needed():
                    logger.info("Portfolio drift detected - triggering rebalance")
                    self.rebalance_portfolio()
                
                # Run daily stress tests
                if datetime.now().hour == 1:  # Run at 1 AM
                    self.run_stress_test()
                
                # Calculate and store portfolio metrics
                self.calculate_portfolio_metrics()
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
    
    def _save_portfolio_state(self):
        """Save portfolio state to disk"""
        try:
            portfolio_data = {
                'total_capital': self.total_capital,
                'available_capital': self.available_capital,
                'positions': {pid: asdict(pos) for pid, pos in self.positions.items()},
                'strategy_allocations': {sid: asdict(alloc) for sid, alloc in self.strategy_allocations.items()},
                'portfolio_stats': self.portfolio_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            state_file = os.path.join(self.data_dir, 'portfolio_state.json')
            with open(state_file, 'w') as f:
                json.dump(portfolio_data, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    def _load_portfolio_state(self):
        """Load portfolio state from disk"""
        try:
            state_file = os.path.join(self.data_dir, 'portfolio_state.json')
            if not os.path.exists(state_file):
                return
            
            with open(state_file, 'r') as f:
                portfolio_data = json.load(f)
            
            self.total_capital = portfolio_data.get('total_capital', 100000.0)
            self.available_capital = portfolio_data.get('available_capital', self.total_capital)
            self.portfolio_stats.update(portfolio_data.get('portfolio_stats', {}))
            
            logger.info("Portfolio state loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
    
    def stop(self):
        """Stop the portfolio intelligence core"""
        self.running = False
        self._save_portfolio_state()
        logger.info("Portfolio Intelligence Core stopped")
