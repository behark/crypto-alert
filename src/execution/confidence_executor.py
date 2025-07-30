"""
Confidence-Based Execution System
=================================
Execute trades based on forecast confidence levels and cross-bot validation
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution mode enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MANUAL_ONLY = "manual_only"

class TradeDirection(Enum):
    """Trade direction enumeration"""
    LONG = "long"
    SHORT = "short"

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING_VALIDATION = "pending_validation"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

@dataclass
class ExecutionSignal:
    """Execution signal from forecast analysis"""
    signal_id: str
    bot_id: str
    symbol: str
    timeframe: str
    direction: TradeDirection
    confidence: float
    entry_price: float
    target_prices: List[float]
    stop_loss: float
    position_size: float
    reasoning: str
    timestamp: datetime
    forecast_data: Dict[str, Any]

@dataclass
class ValidationResult:
    """Result from signal validation"""
    validator_name: str
    passed: bool
    score: float
    reason: str
    timestamp: datetime

@dataclass
class ExecutionOrder:
    """Order ready for execution"""
    order_id: str
    signal: ExecutionSignal
    validation_results: List[ValidationResult]
    consensus_data: Optional[Dict[str, Any]]
    final_confidence: float
    execution_mode: ExecutionMode
    requires_approval: bool
    created_at: datetime
    status: TradeStatus

class ConfidenceExecutor:
    """
    Confidence-Based Execution System
    
    Executes trades based on forecast confidence levels, cross-bot consensus,
    and comprehensive validation with human oversight capabilities.
    """
    
    def __init__(self, data_dir: str = "data/execution"):
        """Initialize the confidence executor"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution configuration
        self.execution_thresholds = {
            ExecutionMode.CONSERVATIVE: 85.0,    # High confidence required
            ExecutionMode.MODERATE: 75.0,        # Medium confidence
            ExecutionMode.AGGRESSIVE: 65.0,      # Lower confidence (smaller positions)
            ExecutionMode.MANUAL_ONLY: 100.0     # No automatic execution
        }
        
        self.consensus_requirements = {
            ExecutionMode.CONSERVATIVE: 3,       # Require 3+ bot consensus
            ExecutionMode.MODERATE: 2,           # Require 2+ bot consensus
            ExecutionMode.AGGRESSIVE: 1,         # Single bot sufficient
            ExecutionMode.MANUAL_ONLY: 1         # Manual approval required
        }
        
        self.position_size_multipliers = {
            ExecutionMode.CONSERVATIVE: 0.8,     # Smaller positions for safety
            ExecutionMode.MODERATE: 1.0,         # Standard position sizing
            ExecutionMode.AGGRESSIVE: 1.2,       # Larger positions for higher returns
            ExecutionMode.MANUAL_ONLY: 1.0       # Standard sizing
        }
        
        # Current state
        self.current_mode = ExecutionMode.MANUAL_ONLY
        self.pending_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history: List[ExecutionOrder] = []
        
        # Validation components
        self.signal_validators = []
        self._initialize_validators()
        
        # Execution thread
        self._execution_thread = None
        self._should_stop = False
        
        logger.info("Confidence Executor initialized")
    
    def set_execution_mode(self, mode: ExecutionMode) -> bool:
        """
        Set execution mode
        
        Args:
            mode: New execution mode
            
        Returns:
            bool: Success status
        """
        try:
            old_mode = self.current_mode
            self.current_mode = mode
            
            # Start/stop execution thread based on mode
            if mode != ExecutionMode.MANUAL_ONLY:
                if not self._execution_thread or not self._execution_thread.is_alive():
                    self._start_execution_thread()
            else:
                self._should_stop = True
            
            logger.info(f"Execution mode changed: {old_mode.value} -> {mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set execution mode: {e}")
            return False
    
    def evaluate_execution_signal(self, signal: ExecutionSignal, 
                                consensus_data: Optional[Dict[str, Any]] = None) -> ExecutionOrder:
        """
        Evaluate execution signal and create order if validated
        
        Args:
            signal: Execution signal to evaluate
            consensus_data: Optional cross-bot consensus data
            
        Returns:
            ExecutionOrder: Order ready for execution or rejection
        """
        try:
            logger.info(f"Evaluating execution signal: {signal.symbol} {signal.direction.value}")
            
            # Run validation checks
            validation_results = []
            for validator in self.signal_validators:
                try:
                    result = validator.validate(signal, consensus_data)
                    validation_results.append(result)
                except Exception as e:
                    logger.warning(f"Validator {validator.__class__.__name__} failed: {e}")
                    validation_results.append(ValidationResult(
                        validator_name=validator.__class__.__name__,
                        passed=False,
                        score=0.0,
                        reason=f"Validation error: {str(e)}",
                        timestamp=datetime.now()
                    ))
            
            # Calculate final confidence based on validations
            final_confidence = self._calculate_final_confidence(signal, validation_results, consensus_data)
            
            # Determine if order requires approval
            requires_approval = self._requires_human_approval(signal, final_confidence, validation_results)
            
            # Create execution order
            order_id = f"order_{signal.symbol}_{signal.direction.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            execution_order = ExecutionOrder(
                order_id=order_id,
                signal=signal,
                validation_results=validation_results,
                consensus_data=consensus_data,
                final_confidence=final_confidence,
                execution_mode=self.current_mode,
                requires_approval=requires_approval,
                created_at=datetime.now(),
                status=TradeStatus.PENDING_VALIDATION
            )
            
            # Add to pending orders
            self.pending_orders[order_id] = execution_order
            
            # Save order
            self._save_execution_order(execution_order)
            
            logger.info(f"Created execution order: {order_id} (confidence: {final_confidence:.1f}%)")
            return execution_order
            
        except Exception as e:
            logger.error(f"Failed to evaluate execution signal: {e}")
            return None
    
    def approve_order(self, order_id: str, approved_by: str = "system") -> bool:
        """
        Approve pending order for execution
        
        Args:
            order_id: Order identifier
            approved_by: Who approved the order
            
        Returns:
            bool: Success status
        """
        try:
            if order_id not in self.pending_orders:
                logger.warning(f"Order {order_id} not found in pending orders")
                return False
            
            order = self.pending_orders[order_id]
            
            if order.status != TradeStatus.PENDING_VALIDATION:
                logger.warning(f"Order {order_id} is not pending validation (status: {order.status.value})")
                return False
            
            order.status = TradeStatus.APPROVED
            
            logger.info(f"Order {order_id} approved by {approved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve order: {e}")
            return False
    
    def reject_order(self, order_id: str, reason: str = "Manual rejection") -> bool:
        """
        Reject pending order
        
        Args:
            order_id: Order identifier
            reason: Rejection reason
            
        Returns:
            bool: Success status
        """
        try:
            if order_id not in self.pending_orders:
                return False
            
            order = self.pending_orders[order_id]
            order.status = TradeStatus.REJECTED
            
            # Move to history
            self.execution_history.append(order)
            del self.pending_orders[order_id]
            
            logger.info(f"Order {order_id} rejected: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reject order: {e}")
            return False
    
    def execute_approved_orders(self) -> List[str]:
        """
        Execute all approved orders
        
        Returns:
            List[str]: List of executed order IDs
        """
        try:
            executed_orders = []
            
            approved_orders = [
                order for order in self.pending_orders.values()
                if order.status == TradeStatus.APPROVED
            ]
            
            for order in approved_orders:
                try:
                    success = self._execute_single_order(order)
                    if success:
                        order.status = TradeStatus.EXECUTED
                        executed_orders.append(order.order_id)
                        
                        # Move to history
                        self.execution_history.append(order)
                        del self.pending_orders[order.order_id]
                        
                        logger.info(f"Successfully executed order: {order.order_id}")
                    else:
                        logger.error(f"Failed to execute order: {order.order_id}")
                        
                except Exception as e:
                    logger.error(f"Error executing order {order.order_id}: {e}")
            
            return executed_orders
            
        except Exception as e:
            logger.error(f"Failed to execute approved orders: {e}")
            return []
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution status
        
        Returns:
            Dict containing execution status information
        """
        try:
            pending_by_status = {}
            for status in TradeStatus:
                count = len([o for o in self.pending_orders.values() if o.status == status])
                if count > 0:
                    pending_by_status[status.value] = count
            
            return {
                'execution_mode': self.current_mode.value,
                'pending_orders': len(self.pending_orders),
                'pending_by_status': pending_by_status,
                'execution_history': len(self.execution_history),
                'thresholds': {mode.value: threshold for mode, threshold in self.execution_thresholds.items()},
                'last_execution': self.execution_history[-1].created_at if self.execution_history else None,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution status: {e}")
            return {'error': str(e)}
    
    def get_pending_orders(self, status_filter: Optional[TradeStatus] = None) -> List[ExecutionOrder]:
        """
        Get pending orders
        
        Args:
            status_filter: Optional status filter
            
        Returns:
            List of pending orders
        """
        try:
            orders = list(self.pending_orders.values())
            
            if status_filter:
                orders = [order for order in orders if order.status == status_filter]
            
            # Sort by confidence (highest first)
            orders.sort(key=lambda x: x.final_confidence, reverse=True)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return []
    
    def calculate_position_size(self, signal: ExecutionSignal, account_balance: float, 
                              risk_percentage: float = 2.0) -> float:
        """
        Calculate optimal position size based on confidence and risk parameters
        
        Args:
            signal: Execution signal
            account_balance: Current account balance
            risk_percentage: Risk percentage per trade
            
        Returns:
            float: Calculated position size
        """
        try:
            # Base position size calculation
            risk_amount = account_balance * (risk_percentage / 100)
            
            # Calculate risk per unit
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            
            if entry_price <= 0 or stop_loss <= 0:
                logger.warning("Invalid entry price or stop loss for position sizing")
                return 0.0
            
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit <= 0:
                logger.warning("Invalid risk per unit calculation")
                return 0.0
            
            # Base position size
            base_position_size = risk_amount / risk_per_unit
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence / 100.0
            confidence_adjusted_size = base_position_size * confidence_multiplier
            
            # Adjust based on execution mode
            mode_multiplier = self.position_size_multipliers.get(self.current_mode, 1.0)
            final_position_size = confidence_adjusted_size * mode_multiplier
            
            # Ensure minimum and maximum limits
            min_position_size = account_balance * 0.001  # 0.1% minimum
            max_position_size = account_balance * 0.1    # 10% maximum
            
            final_position_size = max(min_position_size, min(final_position_size, max_position_size))
            
            logger.info(f"Calculated position size: {final_position_size:.2f} (confidence: {signal.confidence:.1f}%)")
            return final_position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    def _calculate_final_confidence(self, signal: ExecutionSignal, 
                                  validation_results: List[ValidationResult],
                                  consensus_data: Optional[Dict[str, Any]]) -> float:
        """Calculate final confidence score"""
        try:
            base_confidence = signal.confidence
            
            # Apply validation score adjustments
            validation_adjustment = 0.0
            total_validators = len(validation_results)
            passed_validators = len([v for v in validation_results if v.passed])
            
            if total_validators > 0:
                validation_score = passed_validators / total_validators
                validation_adjustment = (validation_score - 0.5) * 20  # ±10% adjustment
            
            # Apply consensus adjustment
            consensus_adjustment = 0.0
            if consensus_data:
                consensus_strength = consensus_data.get('strength', 0.5)
                consensus_adjustment = (consensus_strength - 0.5) * 30  # ±15% adjustment
            
            # Calculate final confidence
            final_confidence = base_confidence + validation_adjustment + consensus_adjustment
            final_confidence = max(0.0, min(100.0, final_confidence))  # Clamp to 0-100%
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate final confidence: {e}")
            return signal.confidence
    
    def _requires_human_approval(self, signal: ExecutionSignal, final_confidence: float,
                               validation_results: List[ValidationResult]) -> bool:
        """Determine if order requires human approval"""
        try:
            # Always require approval in manual mode
            if self.current_mode == ExecutionMode.MANUAL_ONLY:
                return True
            
            # Require approval if confidence is below threshold
            threshold = self.execution_thresholds.get(self.current_mode, 100.0)
            if final_confidence < threshold:
                return True
            
            # Require approval if any critical validator failed
            critical_failures = [v for v in validation_results if not v.passed and v.score < 0.3]
            if critical_failures:
                return True
            
            # Require approval for large position sizes
            if signal.position_size > 10000:  # Configurable threshold
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to determine approval requirement: {e}")
            return True  # Default to requiring approval
    
    def _execute_single_order(self, order: ExecutionOrder) -> bool:
        """Execute a single order"""
        try:
            # This is where actual broker/exchange integration would happen
            # For now, simulate successful execution
            
            signal = order.signal
            
            logger.info(f"Executing order: {signal.symbol} {signal.direction.value} @ {signal.entry_price}")
            
            # Simulate execution delay
            import time
            time.sleep(0.1)
            
            # In real implementation, this would:
            # 1. Connect to broker/exchange API
            # 2. Place market/limit order
            # 3. Set stop loss and take profit orders
            # 4. Monitor order status
            # 5. Return execution result
            
            # For now, return success
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute single order: {e}")
            return False
    
    def _initialize_validators(self):
        """Initialize signal validators"""
        try:
            # Import and initialize validators
            from .validators import (
                ConfidenceValidator,
                TechnicalValidator,
                VolumeValidator,
                RiskValidator
            )
            
            self.signal_validators = [
                ConfidenceValidator(),
                TechnicalValidator(),
                VolumeValidator(),
                RiskValidator()
            ]
            
        except ImportError:
            logger.warning("Validators not available, using mock validators")
            self.signal_validators = []
    
    def _start_execution_thread(self):
        """Start the execution monitoring thread"""
        self._should_stop = False
        self._execution_thread = threading.Thread(
            target=self._execution_worker,
            daemon=True,
            name="ExecutionWorker"
        )
        self._execution_thread.start()
        logger.info("Started execution monitoring thread")
    
    def _execution_worker(self):
        """Worker thread for automatic execution"""
        while not self._should_stop:
            try:
                # Execute approved orders
                executed = self.execute_approved_orders()
                if executed:
                    logger.info(f"Executed {len(executed)} orders automatically")
                
                # Sleep briefly
                threading.Event().wait(1.0)
                
            except Exception as e:
                logger.error(f"Error in execution worker: {e}")
    
    def _save_execution_order(self, order: ExecutionOrder):
        """Save execution order to disk"""
        try:
            file_path = self.data_dir / f"order_{order.order_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(order)
                # Convert datetime and enum objects to strings
                data['created_at'] = order.created_at.isoformat()
                data['status'] = order.status.value
                data['execution_mode'] = order.execution_mode.value
                data['signal']['direction'] = order.signal.direction.value
                data['signal']['timestamp'] = order.signal.timestamp.isoformat()
                
                for result in data['validation_results']:
                    result['timestamp'] = result['timestamp'].isoformat() if isinstance(result['timestamp'], datetime) else result['timestamp']
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save execution order: {e}")
    
    def stop(self):
        """Stop the confidence executor"""
        self._should_stop = True
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=2.0)
        logger.info("Confidence Executor stopped")


# Global instance
_confidence_executor = None

def get_confidence_executor() -> ConfidenceExecutor:
    """Get global confidence executor instance"""
    global _confidence_executor
    if _confidence_executor is None:
        _confidence_executor = ConfidenceExecutor()
    return _confidence_executor
