"""
Smart Entry Manager
==================
Intelligent entry logic with multi-layer validation and dynamic timing
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

class EntryStrategy(Enum):
    """Entry strategy enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"

class EntryCondition(Enum):
    """Entry condition enumeration"""
    IMMEDIATE = "immediate"
    PRICE_LEVEL = "price_level"
    VOLUME_SPIKE = "volume_spike"
    MOMENTUM_CONFIRMATION = "momentum_confirmation"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN_COMPLETION = "pattern_completion"

@dataclass
class EntrySignal:
    """Smart entry signal"""
    signal_id: str
    symbol: str
    direction: str
    strategy: EntryStrategy
    target_price: float
    max_slippage: float
    position_size: float
    conditions: List[EntryCondition]
    confidence: float
    urgency: int  # 1-5
    valid_until: datetime
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class EntryExecution:
    """Entry execution record"""
    execution_id: str
    signal: EntrySignal
    actual_price: float
    actual_size: float
    slippage: float
    execution_time: datetime
    strategy_used: EntryStrategy
    conditions_met: List[EntryCondition]
    success: bool
    notes: str

class SmartEntryManager:
    """
    Smart Entry Manager
    
    Manages intelligent entry execution with multi-layer validation,
    dynamic timing, and adaptive strategies based on market conditions.
    """
    
    def __init__(self, data_dir: str = "data/smart_entry"):
        """Initialize the smart entry manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Entry configuration
        self.max_slippage_tolerance = {
            EntryStrategy.MARKET: 0.5,      # 0.5% max slippage for market orders
            EntryStrategy.LIMIT: 0.0,       # No slippage for limit orders
            EntryStrategy.STOP_LIMIT: 0.3,  # 0.3% max slippage for stop-limit
            EntryStrategy.TWAP: 0.2,        # 0.2% max slippage for TWAP
            EntryStrategy.VWAP: 0.2,        # 0.2% max slippage for VWAP
            EntryStrategy.ICEBERG: 0.1      # 0.1% max slippage for iceberg
        }
        
        # Active entries
        self.pending_entries: Dict[str, EntrySignal] = {}
        self.execution_history: List[EntryExecution] = []
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Entry monitoring
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Smart Entry Manager initialized")
    
    def create_entry_signal(self, symbol: str, direction: str, target_price: float,
                          position_size: float, confidence: float,
                          strategy: EntryStrategy = EntryStrategy.MARKET,
                          conditions: List[EntryCondition] = None,
                          max_slippage: float = None,
                          valid_minutes: int = 30,
                          urgency: int = 3,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Create smart entry signal
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (long/short)
            target_price: Target entry price
            position_size: Position size
            confidence: Entry confidence (0-100)
            strategy: Entry strategy
            conditions: Entry conditions to wait for
            max_slippage: Maximum acceptable slippage
            valid_minutes: Signal validity in minutes
            urgency: Urgency level (1-5)
            metadata: Additional metadata
            
        Returns:
            str: Signal ID
        """
        try:
            signal_id = f"entry_{symbol}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Set default values
            if conditions is None:
                conditions = [EntryCondition.IMMEDIATE]
            if max_slippage is None:
                max_slippage = self.max_slippage_tolerance.get(strategy, 0.5)
            if metadata is None:
                metadata = {}
            
            # Create entry signal
            signal = EntrySignal(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                target_price=target_price,
                max_slippage=max_slippage,
                position_size=position_size,
                conditions=conditions,
                confidence=confidence,
                urgency=urgency,
                valid_until=datetime.now() + timedelta(minutes=valid_minutes),
                created_at=datetime.now(),
                metadata=metadata
            )
            
            # Add to pending entries
            self.pending_entries[signal_id] = signal
            
            # Save signal
            self._save_entry_signal(signal)
            
            logger.info(f"Created entry signal: {signal_id} ({symbol} {direction} @ {target_price})")
            return signal_id
            
        except Exception as e:
            logger.error(f"Failed to create entry signal: {e}")
            return ""
    
    def execute_entry(self, signal_id: str, force_execution: bool = False) -> Optional[EntryExecution]:
        """
        Execute entry signal
        
        Args:
            signal_id: Signal identifier
            force_execution: Force execution regardless of conditions
            
        Returns:
            EntryExecution: Execution result or None if failed
        """
        try:
            if signal_id not in self.pending_entries:
                logger.warning(f"Entry signal {signal_id} not found")
                return None
            
            signal = self.pending_entries[signal_id]
            
            # Check if signal is still valid
            if datetime.now() > signal.valid_until:
                logger.warning(f"Entry signal {signal_id} has expired")
                self._remove_pending_entry(signal_id)
                return None
            
            # Check entry conditions (unless forced)
            if not force_execution:
                conditions_met = self._check_entry_conditions(signal)
                if not conditions_met:
                    logger.info(f"Entry conditions not met for {signal_id}")
                    return None
            else:
                conditions_met = signal.conditions  # Assume all conditions met when forced
            
            # Execute based on strategy
            execution = self._execute_by_strategy(signal, conditions_met)
            
            if execution:
                # Remove from pending and add to history
                self._remove_pending_entry(signal_id)
                self.execution_history.append(execution)
                
                # Save execution
                self._save_entry_execution(execution)
                
                logger.info(f"Successfully executed entry: {signal_id}")
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute entry: {e}")
            return None
    
    def get_optimal_entry_strategy(self, symbol: str, direction: str, 
                                 position_size: float, urgency: int) -> EntryStrategy:
        """
        Determine optimal entry strategy based on market conditions
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            position_size: Position size
            urgency: Urgency level
            
        Returns:
            EntryStrategy: Recommended strategy
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            
            if not market_data:
                return EntryStrategy.MARKET
            
            # Extract market metrics
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 0)
            spread = market_data.get('spread', 0.001)
            liquidity = market_data.get('liquidity', 1.0)
            
            # Decision logic based on conditions
            if urgency >= 4:
                return EntryStrategy.MARKET
            
            if position_size > 50000:  # Large position
                if volatility < 0.01:
                    return EntryStrategy.ICEBERG
                else:
                    return EntryStrategy.TWAP
            
            if volatility > 0.05:
                return EntryStrategy.LIMIT
            
            if spread > 0.002:
                return EntryStrategy.LIMIT
            
            if liquidity < 0.5:
                return EntryStrategy.VWAP
            
            if urgency >= 2:
                return EntryStrategy.STOP_LIMIT
            
            return EntryStrategy.LIMIT
            
        except Exception as e:
            logger.error(f"Failed to determine optimal strategy: {e}")
            return EntryStrategy.MARKET
    
    def get_entry_status(self) -> Dict[str, Any]:
        """Get current entry status"""
        try:
            pending_by_strategy = {}
            for signal in self.pending_entries.values():
                strategy = signal.strategy.value
                pending_by_strategy[strategy] = pending_by_strategy.get(strategy, 0) + 1
            
            recent_executions = [e for e in self.execution_history 
                               if e.execution_time > datetime.now() - timedelta(hours=24)]
            
            success_rate = 0.0
            if recent_executions:
                successful = len([e for e in recent_executions if e.success])
                success_rate = (successful / len(recent_executions)) * 100
            
            return {
                'pending_entries': len(self.pending_entries),
                'pending_by_strategy': pending_by_strategy,
                'total_executions': len(self.execution_history),
                'recent_executions_24h': len(recent_executions),
                'success_rate_24h': success_rate,
                'average_slippage': self._calculate_average_slippage(),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get entry status: {e}")
            return {'error': str(e)}
    
    def _check_entry_conditions(self, signal: EntrySignal) -> bool:
        """Check if entry conditions are met"""
        try:
            for condition in signal.conditions:
                if not self._check_single_condition(signal, condition):
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Failed to check entry conditions: {e}")
            return False
    
    def _check_single_condition(self, signal: EntrySignal, condition: EntryCondition) -> bool:
        """Check single entry condition"""
        try:
            if condition == EntryCondition.IMMEDIATE:
                return True
            
            market_data = self._get_market_data(signal.symbol)
            if not market_data:
                return False
            
            current_price = market_data.get('price', 0)
            
            if condition == EntryCondition.PRICE_LEVEL:
                price_diff = abs(current_price - signal.target_price) / signal.target_price
                return price_diff <= 0.002  # Within 0.2%
            
            elif condition == EntryCondition.VOLUME_SPIKE:
                current_volume = market_data.get('volume', 0)
                avg_volume = market_data.get('avg_volume', current_volume)
                return current_volume > avg_volume * 1.5
            
            elif condition == EntryCondition.MOMENTUM_CONFIRMATION:
                rsi = market_data.get('rsi', 50)
                if signal.direction == 'long':
                    return rsi > 60
                else:
                    return rsi < 40
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check condition {condition}: {e}")
            return False
    
    def _execute_by_strategy(self, signal: EntrySignal, conditions_met: List[EntryCondition]) -> Optional[EntryExecution]:
        """Execute entry using specified strategy"""
        try:
            execution_id = f"exec_{signal.signal_id}_{datetime.now().strftime('%H%M%S')}"
            
            # Get current market price
            market_data = self._get_market_data(signal.symbol)
            current_price = market_data.get('price', signal.target_price) if market_data else signal.target_price
            
            # Execute based on strategy (simplified simulation)
            if signal.strategy == EntryStrategy.MARKET:
                slippage_factor = np.random.uniform(0.0001, 0.002)
                if signal.direction == 'long':
                    actual_price = current_price * (1 + slippage_factor)
                else:
                    actual_price = current_price * (1 - slippage_factor)
                actual_size = signal.position_size
            
            else:  # All other strategies
                actual_price = signal.target_price
                actual_size = signal.position_size
            
            # Calculate slippage
            slippage = abs(actual_price - signal.target_price) / signal.target_price * 100
            
            # Check if execution was successful
            success = (actual_size > 0 and slippage <= signal.max_slippage)
            
            # Create execution record
            execution = EntryExecution(
                execution_id=execution_id,
                signal=signal,
                actual_price=actual_price,
                actual_size=actual_size,
                slippage=slippage,
                execution_time=datetime.now(),
                strategy_used=signal.strategy,
                conditions_met=conditions_met,
                success=success,
                notes=f"Executed via {signal.strategy.value}"
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute by strategy: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol"""
        try:
            if symbol not in self.market_data_cache:
                base_price = 100.0
                self.market_data_cache[symbol] = {
                    'price': base_price + np.random.uniform(-5, 5),
                    'volume': np.random.uniform(1000, 10000),
                    'avg_volume': 5000,
                    'volatility': np.random.uniform(0.01, 0.05),
                    'spread': np.random.uniform(0.001, 0.003),
                    'liquidity': np.random.uniform(0.3, 1.0),
                    'rsi': np.random.uniform(30, 70),
                    'timestamp': datetime.now()
                }
            
            return self.market_data_cache[symbol]
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def _calculate_average_slippage(self) -> float:
        """Calculate average slippage from recent executions"""
        try:
            recent_executions = [e for e in self.execution_history 
                               if e.execution_time > datetime.now() - timedelta(hours=24) and e.success]
            
            if not recent_executions:
                return 0.0
            
            total_slippage = sum(e.slippage for e in recent_executions)
            return total_slippage / len(recent_executions)
            
        except Exception as e:
            logger.error(f"Failed to calculate average slippage: {e}")
            return 0.0
    
    def _remove_pending_entry(self, signal_id: str):
        """Remove entry from pending list"""
        try:
            if signal_id in self.pending_entries:
                del self.pending_entries[signal_id]
                
        except Exception as e:
            logger.error(f"Failed to remove pending entry: {e}")
    
    def _start_monitoring(self):
        """Start entry monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="EntryMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started entry monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring entries"""
        while not self._should_stop:
            try:
                # Check for expired entries
                current_time = datetime.now()
                expired_entries = []
                
                for signal_id, signal in self.pending_entries.items():
                    if current_time > signal.valid_until:
                        expired_entries.append(signal_id)
                
                # Cancel expired entries
                for signal_id in expired_entries:
                    self._remove_pending_entry(signal_id)
                
                # Check for automatic execution opportunities
                for signal_id, signal in list(self.pending_entries.items()):
                    if signal.urgency >= 4:  # High urgency signals
                        conditions_met = self._check_entry_conditions(signal)
                        if conditions_met:
                            self.execute_entry(signal_id)
                
                # Sleep
                threading.Event().wait(5.0)
                
            except Exception as e:
                logger.error(f"Error in entry monitoring: {e}")
    
    def _save_entry_signal(self, signal: EntrySignal):
        """Save entry signal to disk"""
        try:
            file_path = self.data_dir / f"signal_{signal.signal_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(signal)
                data['strategy'] = signal.strategy.value
                data['conditions'] = [c.value for c in signal.conditions]
                data['valid_until'] = signal.valid_until.isoformat()
                data['created_at'] = signal.created_at.isoformat()
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save entry signal: {e}")
    
    def _save_entry_execution(self, execution: EntryExecution):
        """Save entry execution to disk"""
        try:
            file_path = self.data_dir / f"execution_{execution.execution_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(execution)
                data['execution_time'] = execution.execution_time.isoformat()
                data['strategy_used'] = execution.strategy_used.value
                data['conditions_met'] = [c.value for c in execution.conditions_met]
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save entry execution: {e}")
    
    def stop(self):
        """Stop the smart entry manager"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Smart Entry Manager stopped")


# Global instance
_smart_entry_manager = None

def get_smart_entry_manager() -> SmartEntryManager:
    """Get global smart entry manager instance"""
    global _smart_entry_manager
    if _smart_entry_manager is None:
        _smart_entry_manager = SmartEntryManager()
    return _smart_entry_manager
