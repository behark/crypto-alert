"""
Human Override System
====================
Maintain human control over all autonomous decisions with real-time notifications and emergency controls
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OverrideType(Enum):
    """Override type enumeration"""
    EMERGENCY_STOP = "emergency_stop"
    TRADE_APPROVAL = "trade_approval"
    RISK_OVERRIDE = "risk_override"
    MANUAL_EXECUTION = "manual_execution"
    SYSTEM_PAUSE = "system_pause"

class OverrideStatus(Enum):
    """Override status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"

@dataclass
class OverrideRequest:
    """Human override request"""
    request_id: str
    override_type: OverrideType
    requester: str
    description: str
    trade_data: Optional[Dict[str, Any]]
    urgency_level: int  # 1-5 (5 = critical)
    expires_at: datetime
    created_at: datetime
    status: OverrideStatus
    response_data: Optional[Dict[str, Any]] = None
    responded_by: Optional[str] = None
    responded_at: Optional[datetime] = None

@dataclass
class EmergencyProtocol:
    """Emergency protocol configuration"""
    protocol_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    actions: List[str]
    auto_execute: bool
    requires_confirmation: bool

class HumanOverrideSystem:
    """
    Human Override System
    
    Maintains human control over all autonomous decisions with real-time notifications,
    emergency stop functionality, and comprehensive approval workflows.
    """
    
    def __init__(self, data_dir: str = "data/human_override"):
        """Initialize the human override system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Override requests
        self.pending_requests: Dict[str, OverrideRequest] = {}
        self.request_history: List[OverrideRequest] = []
        
        # Notification callbacks
        self.notification_callbacks: List[Callable] = []
        
        # Emergency protocols
        self.emergency_protocols: Dict[str, EmergencyProtocol] = {}
        self._initialize_emergency_protocols()
        
        # System state
        self.emergency_stop_active = False
        self.system_paused = False
        self.last_heartbeat = datetime.now()
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop_monitoring = False
        
        # Configuration
        self.default_approval_timeout = timedelta(minutes=5)
        self.emergency_response_timeout = timedelta(seconds=30)
        
        self._start_monitoring()
        
        logger.info("Human Override System initialized")
    
    def request_trade_approval(self, trade_data: Dict[str, Any], 
                             urgency: int = 3, timeout_minutes: int = 5) -> str:
        """
        Request human approval for a trade
        
        Args:
            trade_data: Trade information requiring approval
            urgency: Urgency level (1-5)
            timeout_minutes: Approval timeout in minutes
            
        Returns:
            str: Request ID for tracking
        """
        try:
            request_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trade_data.get('symbol', 'unknown')}"
            
            # Create override request
            request = OverrideRequest(
                request_id=request_id,
                override_type=OverrideType.TRADE_APPROVAL,
                requester="autonomous_system",
                description=f"Trade approval required: {trade_data.get('symbol')} {trade_data.get('direction')}",
                trade_data=trade_data,
                urgency_level=urgency,
                expires_at=datetime.now() + timedelta(minutes=timeout_minutes),
                created_at=datetime.now(),
                status=OverrideStatus.PENDING
            )
            
            # Add to pending requests
            self.pending_requests[request_id] = request
            
            # Send notifications
            self._send_notifications(request)
            
            # Save request
            self._save_override_request(request)
            
            logger.info(f"Trade approval requested: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to request trade approval: {e}")
            return ""
    
    def emergency_stop(self, reason: str = "Manual emergency stop", 
                      triggered_by: str = "user") -> bool:
        """
        Trigger emergency stop of all autonomous operations
        
        Args:
            reason: Reason for emergency stop
            triggered_by: Who triggered the stop
            
        Returns:
            bool: Success status
        """
        try:
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason} (by: {triggered_by})")
            
            self.emergency_stop_active = True
            
            # Create emergency override request
            request_id = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            request = OverrideRequest(
                request_id=request_id,
                override_type=OverrideType.EMERGENCY_STOP,
                requester=triggered_by,
                description=f"Emergency stop: {reason}",
                trade_data=None,
                urgency_level=5,  # Critical
                expires_at=datetime.now() + self.emergency_response_timeout,
                created_at=datetime.now(),
                status=OverrideStatus.EXECUTED
            )
            
            # Execute emergency protocols
            self._execute_emergency_protocols(reason)
            
            # Notify all systems
            self._send_critical_notifications(request)
            
            # Save request
            self._save_override_request(request)
            self.request_history.append(request)
            
            logger.critical("Emergency stop executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False
    
    def approve_request(self, request_id: str, approved_by: str, 
                       response_data: Dict[str, Any] = None) -> bool:
        """
        Approve pending override request
        
        Args:
            request_id: Request identifier
            approved_by: Who approved the request
            response_data: Optional response data
            
        Returns:
            bool: Success status
        """
        try:
            if request_id not in self.pending_requests:
                logger.warning(f"Request {request_id} not found in pending requests")
                return False
            
            request = self.pending_requests[request_id]
            
            if request.status != OverrideStatus.PENDING:
                logger.warning(f"Request {request_id} is not pending (status: {request.status.value})")
                return False
            
            # Check if request has expired
            if datetime.now() > request.expires_at:
                request.status = OverrideStatus.EXPIRED
                self._move_to_history(request_id)
                logger.warning(f"Request {request_id} has expired")
                return False
            
            # Approve request
            request.status = OverrideStatus.APPROVED
            request.responded_by = approved_by
            request.responded_at = datetime.now()
            request.response_data = response_data or {}
            
            # Move to history
            self._move_to_history(request_id)
            
            logger.info(f"Request {request_id} approved by {approved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve request: {e}")
            return False
    
    def reject_request(self, request_id: str, rejected_by: str, 
                      reason: str = "Manual rejection") -> bool:
        """
        Reject pending override request
        
        Args:
            request_id: Request identifier
            rejected_by: Who rejected the request
            reason: Rejection reason
            
        Returns:
            bool: Success status
        """
        try:
            if request_id not in self.pending_requests:
                return False
            
            request = self.pending_requests[request_id]
            request.status = OverrideStatus.REJECTED
            request.responded_by = rejected_by
            request.responded_at = datetime.now()
            request.response_data = {'rejection_reason': reason}
            
            # Move to history
            self._move_to_history(request_id)
            
            logger.info(f"Request {request_id} rejected by {rejected_by}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reject request: {e}")
            return False
    
    def pause_system(self, reason: str = "Manual pause", paused_by: str = "user") -> bool:
        """
        Pause autonomous system operations
        
        Args:
            reason: Reason for pause
            paused_by: Who paused the system
            
        Returns:
            bool: Success status
        """
        try:
            self.system_paused = True
            
            request_id = f"pause_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            request = OverrideRequest(
                request_id=request_id,
                override_type=OverrideType.SYSTEM_PAUSE,
                requester=paused_by,
                description=f"System pause: {reason}",
                trade_data=None,
                urgency_level=3,
                expires_at=datetime.now() + timedelta(hours=1),  # Auto-resume after 1 hour
                created_at=datetime.now(),
                status=OverrideStatus.EXECUTED
            )
            
            self.request_history.append(request)
            self._save_override_request(request)
            
            logger.warning(f"System paused by {paused_by}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False
    
    def resume_system(self, resumed_by: str = "user") -> bool:
        """
        Resume autonomous system operations
        
        Args:
            resumed_by: Who resumed the system
            
        Returns:
            bool: Success status
        """
        try:
            self.system_paused = False
            self.emergency_stop_active = False
            
            logger.info(f"System resumed by {resumed_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False
    
    def register_notification_callback(self, callback: Callable[[OverrideRequest], None]):
        """
        Register callback for override notifications
        
        Args:
            callback: Notification callback function
        """
        self.notification_callbacks.append(callback)
        logger.info("Registered notification callback")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system override status
        
        Returns:
            Dict containing system status
        """
        try:
            return {
                'emergency_stop_active': self.emergency_stop_active,
                'system_paused': self.system_paused,
                'pending_requests': len(self.pending_requests),
                'pending_by_type': self._get_pending_by_type(),
                'last_heartbeat': self.last_heartbeat,
                'emergency_protocols': len(self.emergency_protocols),
                'notification_callbacks': len(self.notification_callbacks),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def get_pending_requests(self, urgency_filter: Optional[int] = None) -> List[OverrideRequest]:
        """
        Get pending override requests
        
        Args:
            urgency_filter: Optional urgency level filter
            
        Returns:
            List of pending requests
        """
        try:
            requests = list(self.pending_requests.values())
            
            if urgency_filter:
                requests = [r for r in requests if r.urgency_level >= urgency_filter]
            
            # Sort by urgency (highest first) then by creation time
            requests.sort(key=lambda x: (x.urgency_level, x.created_at), reverse=True)
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to get pending requests: {e}")
            return []
    
    def heartbeat(self) -> bool:
        """
        Update system heartbeat
        
        Returns:
            bool: System health status
        """
        try:
            self.last_heartbeat = datetime.now()
            
            # Check for system health issues
            health_issues = []
            
            # Check for expired requests
            expired_count = len([r for r in self.pending_requests.values() 
                               if datetime.now() > r.expires_at])
            if expired_count > 0:
                health_issues.append(f"{expired_count} expired requests")
            
            # Check for old pending requests
            old_requests = len([r for r in self.pending_requests.values() 
                              if datetime.now() - r.created_at > timedelta(minutes=30)])
            if old_requests > 0:
                health_issues.append(f"{old_requests} old pending requests")
            
            if health_issues:
                logger.warning(f"System health issues: {', '.join(health_issues)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return False
    
    def _initialize_emergency_protocols(self):
        """Initialize emergency protocols"""
        try:
            protocols = [
                EmergencyProtocol(
                    protocol_id="market_crash",
                    name="Market Crash Protection",
                    description="Protect against sudden market crashes",
                    trigger_conditions=["volatility_spike", "correlation_breakdown"],
                    actions=["close_all_positions", "halt_new_trades"],
                    auto_execute=True,
                    requires_confirmation=False
                ),
                EmergencyProtocol(
                    protocol_id="liquidity_crisis",
                    name="Liquidity Crisis Protection",
                    description="Protect during liquidity crises",
                    trigger_conditions=["low_liquidity", "high_slippage"],
                    actions=["reduce_position_sizes", "increase_spread_tolerance"],
                    auto_execute=True,
                    requires_confirmation=False
                ),
                EmergencyProtocol(
                    protocol_id="system_malfunction",
                    name="System Malfunction Protocol",
                    description="Handle system malfunctions",
                    trigger_conditions=["execution_errors", "data_feed_failure"],
                    actions=["pause_trading", "notify_administrators"],
                    auto_execute=False,
                    requires_confirmation=True
                )
            ]
            
            for protocol in protocols:
                self.emergency_protocols[protocol.protocol_id] = protocol
            
            logger.info(f"Initialized {len(protocols)} emergency protocols")
            
        except Exception as e:
            logger.error(f"Failed to initialize emergency protocols: {e}")
    
    def _execute_emergency_protocols(self, reason: str):
        """Execute emergency protocols"""
        try:
            executed_protocols = []
            
            for protocol in self.emergency_protocols.values():
                if protocol.auto_execute:
                    try:
                        # Execute protocol actions
                        for action in protocol.actions:
                            self._execute_emergency_action(action, reason)
                        
                        executed_protocols.append(protocol.name)
                        
                    except Exception as e:
                        logger.error(f"Failed to execute protocol {protocol.name}: {e}")
            
            logger.info(f"Executed emergency protocols: {', '.join(executed_protocols)}")
            
        except Exception as e:
            logger.error(f"Failed to execute emergency protocols: {e}")
    
    def _execute_emergency_action(self, action: str, reason: str):
        """Execute emergency action"""
        try:
            if action == "close_all_positions":
                logger.critical(f"Emergency action: Closing all positions - {reason}")
                # This would integrate with position management system
                
            elif action == "halt_new_trades":
                logger.critical(f"Emergency action: Halting new trades - {reason}")
                # This would stop new trade execution
                
            elif action == "pause_trading":
                logger.critical(f"Emergency action: Pausing trading - {reason}")
                self.system_paused = True
                
            elif action == "notify_administrators":
                logger.critical(f"Emergency action: Notifying administrators - {reason}")
                # This would send critical notifications
                
            else:
                logger.warning(f"Unknown emergency action: {action}")
                
        except Exception as e:
            logger.error(f"Failed to execute emergency action {action}: {e}")
    
    def _send_notifications(self, request: OverrideRequest):
        """Send notifications for override request"""
        try:
            for callback in self.notification_callbacks:
                try:
                    callback(request)
                except Exception as e:
                    logger.warning(f"Notification callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    def _send_critical_notifications(self, request: OverrideRequest):
        """Send critical notifications"""
        try:
            # This would integrate with Telegram, email, SMS, etc.
            logger.critical(f"CRITICAL NOTIFICATION: {request.description}")
            
            # Send to all notification callbacks
            self._send_notifications(request)
            
        except Exception as e:
            logger.error(f"Failed to send critical notifications: {e}")
    
    def _get_pending_by_type(self) -> Dict[str, int]:
        """Get pending requests count by type"""
        try:
            by_type = {}
            for request in self.pending_requests.values():
                request_type = request.override_type.value
                by_type[request_type] = by_type.get(request_type, 0) + 1
            return by_type
            
        except Exception as e:
            logger.error(f"Failed to get pending by type: {e}")
            return {}
    
    def _move_to_history(self, request_id: str):
        """Move request from pending to history"""
        try:
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                self.request_history.append(request)
                del self.pending_requests[request_id]
                
                # Save updated request
                self._save_override_request(request)
                
        except Exception as e:
            logger.error(f"Failed to move request to history: {e}")
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop_monitoring = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="OverrideMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started override monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring override requests"""
        while not self._should_stop_monitoring:
            try:
                # Check for expired requests
                current_time = datetime.now()
                expired_requests = []
                
                for request_id, request in self.pending_requests.items():
                    if current_time > request.expires_at:
                        request.status = OverrideStatus.EXPIRED
                        expired_requests.append(request_id)
                
                # Move expired requests to history
                for request_id in expired_requests:
                    self._move_to_history(request_id)
                    logger.warning(f"Request {request_id} expired")
                
                # Update heartbeat
                self.heartbeat()
                
                # Sleep
                threading.Event().wait(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in override monitoring: {e}")
    
    def _save_override_request(self, request: OverrideRequest):
        """Save override request to disk"""
        try:
            file_path = self.data_dir / f"override_{request.request_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(request)
                # Convert datetime and enum objects
                data['override_type'] = request.override_type.value
                data['status'] = request.status.value
                data['expires_at'] = request.expires_at.isoformat()
                data['created_at'] = request.created_at.isoformat()
                if request.responded_at:
                    data['responded_at'] = request.responded_at.isoformat()
                
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save override request: {e}")
    
    def stop(self):
        """Stop the human override system"""
        self._should_stop_monitoring = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Human Override System stopped")


# Global instance
_human_override_system = None

def get_human_override_system() -> HumanOverrideSystem:
    """Get global human override system instance"""
    global _human_override_system
    if _human_override_system is None:
        _human_override_system = HumanOverrideSystem()
    return _human_override_system
