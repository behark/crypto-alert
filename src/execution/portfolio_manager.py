"""
Dynamic Portfolio Manager
========================
Intelligent portfolio balancing with cross-bot coordination and risk-adjusted allocation
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    """Portfolio allocation strategy"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM_BASED = "momentum_based"
    CORRELATION_AWARE = "correlation_aware"

class RebalanceFrequency(Enum):
    """Rebalancing frequency"""
    CONTINUOUS = "continuous"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    THRESHOLD_BASED = "threshold_based"

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    bot_id: str
    direction: str  # long/short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    confidence: float
    risk_score: float
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class AllocationTarget:
    """Target allocation for symbol"""
    symbol: str
    target_weight: float
    current_weight: float
    deviation: float
    rebalance_needed: bool
    target_size: float
    current_size: float
    confidence: float
    risk_adjusted_weight: float

@dataclass
class RebalanceAction:
    """Portfolio rebalancing action"""
    action_id: str
    symbol: str
    action_type: str  # buy/sell/hold
    current_size: float
    target_size: float
    size_change: float
    priority: int
    reasoning: str
    estimated_cost: float
    created_at: datetime

class DynamicPortfolioManager:
    """
    Dynamic Portfolio Manager
    
    Manages intelligent portfolio balancing with cross-bot coordination,
    risk-adjusted allocation, and dynamic rebalancing based on market conditions.
    """
    
    def __init__(self, data_dir: str = "data/portfolio"):
        """Initialize the portfolio manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Portfolio configuration
        self.total_capital = 100000.0  # Base capital
        self.max_position_size = 0.2   # 20% max per position
        self.max_correlation_exposure = 0.4  # 40% max in correlated assets
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
        # Current portfolio state
        self.positions: Dict[str, Position] = {}
        self.allocation_targets: Dict[str, AllocationTarget] = {}
        self.pending_rebalances: List[RebalanceAction] = []
        
        # Portfolio metrics
        self.total_value = self.total_capital
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Strategy settings
        self.allocation_strategy = AllocationStrategy.CONFIDENCE_WEIGHTED
        self.rebalance_frequency = RebalanceFrequency.THRESHOLD_BASED
        
        # Risk limits
        self.max_leverage = 2.0
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_portfolio_volatility = 0.15  # 15% max portfolio volatility
        
        # Cross-bot coordination
        self.bot_allocations: Dict[str, float] = {}  # Bot ID -> allocation
        self.bot_performance: Dict[str, Dict[str, float]] = {}
        
        # Monitoring
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Dynamic Portfolio Manager initialized")
    
    def add_position(self, symbol: str, bot_id: str, direction: str, 
                    size: float, entry_price: float, confidence: float,
                    metadata: Dict[str, Any] = None) -> bool:
        """
        Add new position to portfolio
        
        Args:
            symbol: Trading symbol
            bot_id: Bot that created the position
            direction: Position direction (long/short)
            size: Position size
            entry_price: Entry price
            confidence: Position confidence
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            position_key = f"{symbol}_{bot_id}_{direction}"
            
            # Check position limits
            if not self._check_position_limits(symbol, size):
                logger.warning(f"Position limits exceeded for {symbol}")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                bot_id=bot_id,
                direction=direction,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,  # Will be updated
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                confidence=confidence,
                risk_score=self._calculate_risk_score(symbol, size, confidence),
                last_updated=datetime.now(),
                metadata=metadata or {}
            )
            
            # Add to positions
            self.positions[position_key] = position
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check if rebalancing is needed
            self._check_rebalancing_needed()
            
            logger.info(f"Added position: {position_key} (size: {size}, confidence: {confidence})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    def update_position_price(self, symbol: str, bot_id: str, direction: str, 
                            current_price: float) -> bool:
        """
        Update position with current market price
        
        Args:
            symbol: Trading symbol
            bot_id: Bot ID
            direction: Position direction
            current_price: Current market price
            
        Returns:
            bool: Success status
        """
        try:
            position_key = f"{symbol}_{bot_id}_{direction}"
            
            if position_key not in self.positions:
                return False
            
            position = self.positions[position_key]
            position.current_price = current_price
            position.last_updated = datetime.now()
            
            # Calculate unrealized PnL
            if direction == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update position price: {e}")
            return False
    
    def close_position(self, symbol: str, bot_id: str, direction: str, 
                      exit_price: float, reason: str = "Manual close") -> bool:
        """
        Close position and realize PnL
        
        Args:
            symbol: Trading symbol
            bot_id: Bot ID
            direction: Position direction
            exit_price: Exit price
            reason: Closing reason
            
        Returns:
            bool: Success status
        """
        try:
            position_key = f"{symbol}_{bot_id}_{direction}"
            
            if position_key not in self.positions:
                return False
            
            position = self.positions[position_key]
            
            # Calculate realized PnL
            if direction == 'long':
                realized_pnl = (exit_price - position.entry_price) * position.size
            else:
                realized_pnl = (position.entry_price - exit_price) * position.size
            
            position.realized_pnl = realized_pnl
            self.realized_pnl += realized_pnl
            
            # Remove position
            del self.positions[position_key]
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Save closed position to history
            self._save_closed_position(position, exit_price, reason)
            
            logger.info(f"Closed position: {position_key} (PnL: {realized_pnl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def calculate_optimal_allocation(self, signals: List[Dict[str, Any]]) -> Dict[str, AllocationTarget]:
        """
        Calculate optimal portfolio allocation based on signals
        
        Args:
            signals: List of trading signals from different bots
            
        Returns:
            Dict of allocation targets by symbol
        """
        try:
            if not signals:
                return {}
            
            # Extract signal data
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'symbol': signal.get('symbol'),
                    'confidence': signal.get('confidence', 50),
                    'expected_return': signal.get('expected_return', 0),
                    'volatility': signal.get('volatility', 0.02),
                    'bot_id': signal.get('bot_id'),
                    'direction': signal.get('direction')
                })
            
            # Calculate allocation based on strategy
            if self.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
                allocations = self._calculate_equal_weight(signal_data)
            elif self.allocation_strategy == AllocationStrategy.CONFIDENCE_WEIGHTED:
                allocations = self._calculate_confidence_weighted(signal_data)
            elif self.allocation_strategy == AllocationStrategy.RISK_PARITY:
                allocations = self._calculate_risk_parity(signal_data)
            elif self.allocation_strategy == AllocationStrategy.VOLATILITY_ADJUSTED:
                allocations = self._calculate_volatility_adjusted(signal_data)
            else:
                allocations = self._calculate_confidence_weighted(signal_data)  # Default
            
            # Create allocation targets
            targets = {}
            for symbol, weight in allocations.items():
                current_weight = self._get_current_weight(symbol)
                deviation = abs(weight - current_weight)
                
                targets[symbol] = AllocationTarget(
                    symbol=symbol,
                    target_weight=weight,
                    current_weight=current_weight,
                    deviation=deviation,
                    rebalance_needed=deviation > self.rebalance_threshold,
                    target_size=weight * self.total_value,
                    current_size=current_weight * self.total_value,
                    confidence=self._get_symbol_confidence(symbol, signal_data),
                    risk_adjusted_weight=weight
                )
            
            self.allocation_targets = targets
            return targets
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal allocation: {e}")
            return {}
    
    def generate_rebalance_actions(self) -> List[RebalanceAction]:
        """
        Generate rebalancing actions based on allocation targets
        
        Returns:
            List of rebalancing actions
        """
        try:
            actions = []
            
            for symbol, target in self.allocation_targets.items():
                if not target.rebalance_needed:
                    continue
                
                size_change = target.target_size - target.current_size
                
                if abs(size_change) < 100:  # Minimum trade size
                    continue
                
                action_type = "buy" if size_change > 0 else "sell"
                priority = int(target.deviation * 100)  # Higher deviation = higher priority
                
                action = RebalanceAction(
                    action_id=f"rebalance_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    action_type=action_type,
                    current_size=target.current_size,
                    target_size=target.target_size,
                    size_change=abs(size_change),
                    priority=priority,
                    reasoning=f"Deviation: {target.deviation:.2%}, Target: {target.target_weight:.2%}",
                    estimated_cost=abs(size_change) * 0.001,  # Estimated transaction cost
                    created_at=datetime.now()
                )
                
                actions.append(action)
            
            # Sort by priority (highest first)
            actions.sort(key=lambda x: x.priority, reverse=True)
            
            self.pending_rebalances = actions
            return actions
            
        except Exception as e:
            logger.error(f"Failed to generate rebalance actions: {e}")
            return []
    
    def execute_rebalance_action(self, action_id: str) -> bool:
        """
        Execute rebalancing action
        
        Args:
            action_id: Action identifier
            
        Returns:
            bool: Success status
        """
        try:
            action = None
            for a in self.pending_rebalances:
                if a.action_id == action_id:
                    action = a
                    break
            
            if not action:
                return False
            
            # Execute the rebalancing action
            # In real implementation, this would place orders through broker
            logger.info(f"Executing rebalance: {action.action_type} {action.size_change} of {action.symbol}")
            
            # Remove from pending
            self.pending_rebalances.remove(action)
            
            # Save execution record
            self._save_rebalance_execution(action)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute rebalance action: {e}")
            return False
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status
        
        Returns:
            Dict containing portfolio status
        """
        try:
            position_count = len(self.positions)
            total_exposure = sum(abs(p.size * p.current_price) for p in self.positions.values())
            
            # Calculate diversification metrics
            symbol_count = len(set(p.symbol for p in self.positions.values()))
            bot_count = len(set(p.bot_id for p in self.positions.values()))
            
            # Risk metrics
            portfolio_volatility = self._calculate_portfolio_volatility()
            max_position_exposure = max([abs(p.size * p.current_price) / self.total_value 
                                       for p in self.positions.values()], default=0)
            
            return {
                'total_value': self.total_value,
                'total_capital': self.total_capital,
                'unrealized_pnl': self.unrealized_pnl,
                'realized_pnl': self.realized_pnl,
                'total_return': (self.total_value - self.total_capital) / self.total_capital * 100,
                'position_count': position_count,
                'symbol_count': symbol_count,
                'bot_count': bot_count,
                'total_exposure': total_exposure,
                'leverage': total_exposure / self.total_value if self.total_value > 0 else 0,
                'max_position_exposure': max_position_exposure,
                'portfolio_volatility': portfolio_volatility,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'pending_rebalances': len(self.pending_rebalances),
                'allocation_strategy': self.allocation_strategy.value,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio status: {e}")
            return {'error': str(e)}
    
    def get_positions(self, symbol_filter: Optional[str] = None, 
                     bot_filter: Optional[str] = None) -> List[Position]:
        """
        Get current positions
        
        Args:
            symbol_filter: Optional symbol filter
            bot_filter: Optional bot filter
            
        Returns:
            List of positions
        """
        try:
            positions = list(self.positions.values())
            
            if symbol_filter:
                positions = [p for p in positions if p.symbol == symbol_filter]
            
            if bot_filter:
                positions = [p for p in positions if p.bot_id == bot_filter]
            
            # Sort by unrealized PnL (highest first)
            positions.sort(key=lambda x: x.unrealized_pnl, reverse=True)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def _calculate_confidence_weighted(self, signal_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence-weighted allocation"""
        try:
            if not signal_data:
                return {}
            
            # Calculate confidence-weighted scores
            total_confidence = sum(s['confidence'] for s in signal_data)
            
            if total_confidence == 0:
                return self._calculate_equal_weight(signal_data)
            
            allocations = {}
            for signal in signal_data:
                weight = signal['confidence'] / total_confidence
                # Apply position size limits
                weight = min(weight, self.max_position_size)
                allocations[signal['symbol']] = weight
            
            # Normalize to ensure sum = 1
            total_weight = sum(allocations.values())
            if total_weight > 0:
                allocations = {k: v/total_weight for k, v in allocations.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence weighted allocation: {e}")
            return {}
    
    def _calculate_equal_weight(self, signal_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate equal weight allocation"""
        try:
            if not signal_data:
                return {}
            
            weight = 1.0 / len(signal_data)
            weight = min(weight, self.max_position_size)  # Apply limits
            
            return {signal['symbol']: weight for signal in signal_data}
            
        except Exception as e:
            logger.error(f"Failed to calculate equal weight allocation: {e}")
            return {}
    
    def _calculate_risk_parity(self, signal_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk parity allocation"""
        try:
            if not signal_data:
                return {}
            
            # Use inverse volatility weighting as proxy for risk parity
            inv_vol_weights = {}
            total_inv_vol = 0
            
            for signal in signal_data:
                volatility = max(signal['volatility'], 0.01)  # Minimum volatility
                inv_vol = 1.0 / volatility
                inv_vol_weights[signal['symbol']] = inv_vol
                total_inv_vol += inv_vol
            
            # Normalize
            allocations = {}
            for symbol, inv_vol in inv_vol_weights.items():
                weight = inv_vol / total_inv_vol
                weight = min(weight, self.max_position_size)
                allocations[symbol] = weight
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to calculate risk parity allocation: {e}")
            return {}
    
    def _calculate_volatility_adjusted(self, signal_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate volatility-adjusted allocation"""
        try:
            # Similar to risk parity but with confidence adjustment
            allocations = self._calculate_risk_parity(signal_data)
            
            # Adjust by confidence
            for signal in signal_data:
                symbol = signal['symbol']
                if symbol in allocations:
                    confidence_factor = signal['confidence'] / 100.0
                    allocations[symbol] *= confidence_factor
            
            # Renormalize
            total_weight = sum(allocations.values())
            if total_weight > 0:
                allocations = {k: v/total_weight for k, v in allocations.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility adjusted allocation: {e}")
            return {}
    
    def _check_position_limits(self, symbol: str, size: float) -> bool:
        """Check if position respects limits"""
        try:
            # Check individual position size limit
            position_value = size  # Simplified
            max_position_value = self.total_value * self.max_position_size
            
            if position_value > max_position_value:
                return False
            
            # Check total exposure
            current_exposure = sum(abs(p.size * p.current_price) for p in self.positions.values())
            total_exposure = current_exposure + position_value
            max_total_exposure = self.total_value * self.max_leverage
            
            if total_exposure > max_total_exposure:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check position limits: {e}")
            return False
    
    def _calculate_risk_score(self, symbol: str, size: float, confidence: float) -> float:
        """Calculate risk score for position"""
        try:
            # Base risk from position size
            position_risk = (size / self.total_value) * 100
            
            # Adjust by confidence (lower confidence = higher risk)
            confidence_risk = (100 - confidence) / 100 * 50
            
            # Combine risk factors
            total_risk = position_risk + confidence_risk
            
            return min(total_risk, 100.0)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Failed to calculate risk score: {e}")
            return 50.0  # Default medium risk
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            # Calculate total unrealized PnL
            self.unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
            
            # Calculate total portfolio value
            self.total_value = self.total_capital + self.realized_pnl + self.unrealized_pnl
            
            # Update max drawdown
            peak_value = max(self.total_value, getattr(self, '_peak_value', self.total_capital))
            self._peak_value = peak_value
            
            current_drawdown = (peak_value - self.total_value) / peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"Failed to update portfolio metrics: {e}")
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if not self.positions:
                return 0.0
            
            # Simplified volatility calculation
            # In real implementation, would use historical returns and correlation matrix
            
            weighted_volatilities = []
            total_value = max(self.total_value, 1.0)
            
            for position in self.positions.values():
                weight = abs(position.size * position.current_price) / total_value
                volatility = 0.02  # Default 2% daily volatility
                weighted_volatilities.append(weight * volatility)
            
            # Simple sum (ignoring correlations for now)
            portfolio_vol = sum(weighted_volatilities)
            
            return min(portfolio_vol, 1.0)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio volatility: {e}")
            return 0.02  # Default 2%
    
    def _get_current_weight(self, symbol: str) -> float:
        """Get current weight of symbol in portfolio"""
        try:
            if self.total_value <= 0:
                return 0.0
            
            symbol_value = 0.0
            for position in self.positions.values():
                if position.symbol == symbol:
                    symbol_value += abs(position.size * position.current_price)
            
            return symbol_value / self.total_value
            
        except Exception as e:
            logger.error(f"Failed to get current weight: {e}")
            return 0.0
    
    def _get_symbol_confidence(self, symbol: str, signal_data: List[Dict[str, Any]]) -> float:
        """Get average confidence for symbol"""
        try:
            confidences = [s['confidence'] for s in signal_data if s['symbol'] == symbol]
            return sum(confidences) / len(confidences) if confidences else 50.0
            
        except Exception as e:
            logger.error(f"Failed to get symbol confidence: {e}")
            return 50.0
    
    def _check_rebalancing_needed(self):
        """Check if portfolio rebalancing is needed"""
        try:
            if not self.allocation_targets:
                return
            
            needs_rebalancing = any(target.rebalance_needed for target in self.allocation_targets.values())
            
            if needs_rebalancing:
                logger.info("Portfolio rebalancing needed")
                self.generate_rebalance_actions()
                
        except Exception as e:
            logger.error(f"Failed to check rebalancing: {e}")
    
    def _start_monitoring(self):
        """Start portfolio monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="PortfolioMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started portfolio monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for portfolio monitoring"""
        while not self._should_stop:
            try:
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Check rebalancing needs
                if self.rebalance_frequency == RebalanceFrequency.THRESHOLD_BASED:
                    self._check_rebalancing_needed()
                
                # Sleep
                threading.Event().wait(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
    
    def _check_risk_limits(self):
        """Check portfolio risk limits"""
        try:
            # Check daily loss limit
            daily_pnl = self.unrealized_pnl  # Simplified
            daily_loss_pct = abs(daily_pnl) / self.total_capital
            
            if daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                # Could trigger emergency protocols here
            
            # Check leverage limit
            total_exposure = sum(abs(p.size * p.current_price) for p in self.positions.values())
            current_leverage = total_exposure / self.total_value if self.total_value > 0 else 0
            
            if current_leverage > self.max_leverage:
                logger.warning(f"Leverage limit exceeded: {current_leverage:.2f}x")
            
        except Exception as e:
            logger.error(f"Failed to check risk limits: {e}")
    
    def _save_closed_position(self, position: Position, exit_price: float, reason: str):
        """Save closed position to history"""
        try:
            file_path = self.data_dir / f"closed_position_{position.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(file_path, 'w') as f:
                data = asdict(position)
                data['exit_price'] = exit_price
                data['close_reason'] = reason
                data['last_updated'] = position.last_updated.isoformat()
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save closed position: {e}")
    
    def _save_rebalance_execution(self, action: RebalanceAction):
        """Save rebalance execution to history"""
        try:
            file_path = self.data_dir / f"rebalance_{action.action_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(action)
                data['created_at'] = action.created_at.isoformat()
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save rebalance execution: {e}")
    
    def stop(self):
        """Stop the portfolio manager"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Dynamic Portfolio Manager stopped")


# Global instance
_portfolio_manager = None

def get_portfolio_manager() -> DynamicPortfolioManager:
    """Get global portfolio manager instance"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = DynamicPortfolioManager()
    return _portfolio_manager
