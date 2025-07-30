"""
Unified Command Center - Core Orchestration Hub
==============================================
Central hub for controlling all trading bots from one interface with cross-bot intelligence
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class BotStatus(Enum):
    """Bot status enumeration"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class BotCapability(Enum):
    """Bot capability enumeration"""
    FORECAST = "forecast"
    TRADE = "trade"
    ANALYZE = "analyze"
    VISUAL = "visual"
    ML_TUNING = "ml_tuning"

@dataclass
class BotInfo:
    """Information about a registered bot"""
    bot_id: str
    name: str
    description: str
    capabilities: List[BotCapability]
    status: BotStatus
    last_heartbeat: datetime
    performance_metrics: Dict[str, Any]
    instance: Any  # Reference to actual bot instance

@dataclass
class CrossBotCommand:
    """Cross-bot command execution request"""
    command_id: str
    command: str
    args: List[str]
    target_bots: List[str]  # Empty list means all capable bots
    timestamp: datetime
    requester: str
    priority: int = 1  # 1=low, 5=high

@dataclass
class CommandResult:
    """Result from cross-bot command execution"""
    command_id: str
    bot_id: str
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time: float
    timestamp: datetime

class UnifiedCommandCenter:
    """
    Central orchestration hub for the Unified Trading Intelligence Platform
    
    Manages multiple trading bots, coordinates cross-bot commands, and enables
    intelligent collaboration between different bot instances.
    """
    
    def __init__(self):
        """Initialize the Unified Command Center"""
        self.registered_bots: Dict[str, BotInfo] = {}
        self.command_queue: List[CrossBotCommand] = []
        self.command_results: Dict[str, List[CommandResult]] = {}
        self.cross_bot_memory: Dict[str, Any] = {}
        
        # Initialize sub-components
        from .bot_registry import get_bot_registry
        from .intelligence_sharing import get_intelligence_sharing
        from .risk_coordinator import get_risk_coordinator
        
        self.bot_registry = get_bot_registry()
        self.intelligence_sharing = get_intelligence_sharing()
        self.risk_coordinator = get_risk_coordinator()
        
        # Command execution thread
        self._command_thread = None
        self._should_stop = False
        
        logger.info("Unified Command Center initialized")
    
    def register_bot(self, bot_id: str, bot_instance: Any, name: str, 
                    description: str, capabilities: List[BotCapability]) -> bool:
        """
        Register a new bot with the command center
        
        Args:
            bot_id: Unique identifier for the bot
            bot_instance: Reference to the bot instance
            name: Human-readable bot name
            description: Bot description
            capabilities: List of bot capabilities
            
        Returns:
            bool: Success status
        """
        try:
            bot_info = BotInfo(
                bot_id=bot_id,
                name=name,
                description=description,
                capabilities=capabilities,
                status=BotStatus.ONLINE,
                last_heartbeat=datetime.now(),
                performance_metrics={},
                instance=bot_instance
            )
            
            self.registered_bots[bot_id] = bot_info
            self.bot_registry.register_bot(bot_info)
            
            logger.info(f"Registered bot: {name} ({bot_id}) with capabilities: {[c.value for c in capabilities]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register bot {bot_id}: {e}")
            return False
    
    def unregister_bot(self, bot_id: str) -> bool:
        """
        Unregister a bot from the command center
        
        Args:
            bot_id: Bot identifier to unregister
            
        Returns:
            bool: Success status
        """
        try:
            if bot_id in self.registered_bots:
                bot_info = self.registered_bots[bot_id]
                del self.registered_bots[bot_id]
                self.bot_registry.unregister_bot(bot_id)
                
                logger.info(f"Unregistered bot: {bot_info.name} ({bot_id})")
                return True
            else:
                logger.warning(f"Bot {bot_id} not found for unregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister bot {bot_id}: {e}")
            return False
    
    def execute_cross_bot_command(self, command: str, args: List[str], 
                                target_bots: List[str] = None, 
                                requester: str = "system") -> str:
        """
        Execute a command across multiple bots
        
        Args:
            command: Command to execute (e.g., 'forecast', 'plan', 'tune')
            args: Command arguments
            target_bots: List of bot IDs to target (None = all capable bots)
            requester: Who requested the command
            
        Returns:
            str: Command ID for tracking results
        """
        try:
            command_id = f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{command}"
            
            # Determine target bots based on capability
            if target_bots is None:
                target_bots = self._get_capable_bots(command)
            
            if not target_bots:
                return f"❌ **No Capable Bots**\\n\\nNo bots found with capability for '{command}' command."
            
            # Create cross-bot command
            cross_command = CrossBotCommand(
                command_id=command_id,
                command=command,
                args=args,
                target_bots=target_bots,
                timestamp=datetime.now(),
                requester=requester
            )
            
            # Add to queue for execution
            self.command_queue.append(cross_command)
            
            # Start command execution thread if not running
            if not self._command_thread or not self._command_thread.is_alive():
                self._start_command_execution_thread()
            
            logger.info(f"Queued cross-bot command: {command} for bots: {target_bots}")
            return command_id
            
        except Exception as e:
            logger.error(f"Failed to execute cross-bot command: {e}")
            return f"❌ **Command Error:** {str(e)}"
    
    def get_command_results(self, command_id: str) -> List[CommandResult]:
        """
        Get results for a specific command
        
        Args:
            command_id: Command ID to get results for
            
        Returns:
            List[CommandResult]: Results from all bots
        """
        return self.command_results.get(command_id, [])
    
    def get_unified_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for all registered bots
        
        Returns:
            Dict containing unified dashboard information
        """
        try:
            dashboard_data = {
                'timestamp': datetime.now(),
                'total_bots': len(self.registered_bots),
                'online_bots': len([b for b in self.registered_bots.values() if b.status == BotStatus.ONLINE]),
                'bots': [],
                'cross_bot_intelligence': self.intelligence_sharing.get_shared_intelligence(),
                'risk_status': self.risk_coordinator.get_portfolio_risk_status(),
                'recent_commands': len(self.command_queue),
                'capabilities_summary': self._get_capabilities_summary()
            }
            
            # Add individual bot data
            for bot_id, bot_info in self.registered_bots.items():
                bot_data = {
                    'bot_id': bot_id,
                    'name': bot_info.name,
                    'status': bot_info.status.value,
                    'capabilities': [c.value for c in bot_info.capabilities],
                    'last_heartbeat': bot_info.last_heartbeat,
                    'performance': bot_info.performance_metrics
                }
                
                # Get bot-specific metrics if available
                if hasattr(bot_info.instance, 'get_performance_metrics'):
                    try:
                        bot_data['performance'].update(bot_info.instance.get_performance_metrics())
                    except Exception as e:
                        logger.warning(f"Failed to get performance metrics for {bot_id}: {e}")
                
                dashboard_data['bots'].append(bot_data)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {'error': str(e)}
    
    def get_bot_status(self, bot_id: str = None) -> Dict[str, Any]:
        """
        Get status information for specific bot or all bots
        
        Args:
            bot_id: Specific bot ID (None for all bots)
            
        Returns:
            Dict containing bot status information
        """
        try:
            if bot_id:
                if bot_id in self.registered_bots:
                    bot_info = self.registered_bots[bot_id]
                    return {
                        'bot_id': bot_id,
                        'name': bot_info.name,
                        'status': bot_info.status.value,
                        'capabilities': [c.value for c in bot_info.capabilities],
                        'last_heartbeat': bot_info.last_heartbeat,
                        'uptime': (datetime.now() - bot_info.last_heartbeat).total_seconds()
                    }
                else:
                    return {'error': f'Bot {bot_id} not found'}
            else:
                # Return all bots status
                return {
                    'total_bots': len(self.registered_bots),
                    'bots': {
                        bot_id: {
                            'name': info.name,
                            'status': info.status.value,
                            'capabilities': [c.value for c in info.capabilities]
                        }
                        for bot_id, info in self.registered_bots.items()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get bot status: {e}")
            return {'error': str(e)}
    
    def coordinate_risk_distribution(self, total_exposure: float) -> Dict[str, float]:
        """
        Coordinate risk distribution across all trading bots
        
        Args:
            total_exposure: Total portfolio exposure to distribute
            
        Returns:
            Dict mapping bot_id to allocated exposure
        """
        try:
            trading_bots = [
                bot_id for bot_id, bot_info in self.registered_bots.items()
                if BotCapability.TRADE in bot_info.capabilities and bot_info.status == BotStatus.ONLINE
            ]
            
            if not trading_bots:
                return {}
            
            # Use risk coordinator for intelligent distribution
            return self.risk_coordinator.distribute_risk(trading_bots, total_exposure)
            
        except Exception as e:
            logger.error(f"Failed to coordinate risk distribution: {e}")
            return {}
    
    def emergency_shutdown(self, reason: str = "Emergency shutdown") -> bool:
        """
        Emergency shutdown of all trading bots
        
        Args:
            reason: Reason for shutdown
            
        Returns:
            bool: Success status
        """
        try:
            logger.warning(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
            
            shutdown_results = []
            for bot_id, bot_info in self.registered_bots.items():
                try:
                    if hasattr(bot_info.instance, 'emergency_shutdown'):
                        result = bot_info.instance.emergency_shutdown(reason)
                        shutdown_results.append((bot_id, result))
                    else:
                        logger.warning(f"Bot {bot_id} does not support emergency shutdown")
                        shutdown_results.append((bot_id, False))
                except Exception as e:
                    logger.error(f"Failed to shutdown bot {bot_id}: {e}")
                    shutdown_results.append((bot_id, False))
            
            success_count = sum(1 for _, success in shutdown_results if success)
            total_count = len(shutdown_results)
            
            logger.info(f"Emergency shutdown completed: {success_count}/{total_count} bots shutdown successfully")
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"Failed emergency shutdown: {e}")
            return False
    
    def _get_capable_bots(self, command: str) -> List[str]:
        """Get list of bots capable of executing a specific command"""
        capability_map = {
            'forecast': BotCapability.FORECAST,
            'plan': BotCapability.FORECAST,
            'tune': BotCapability.ML_TUNING,
            'trade': BotCapability.TRADE,
            'analyze': BotCapability.ANALYZE
        }
        
        required_capability = capability_map.get(command)
        if not required_capability:
            return []
        
        return [
            bot_id for bot_id, bot_info in self.registered_bots.items()
            if required_capability in bot_info.capabilities and bot_info.status == BotStatus.ONLINE
        ]
    
    def _get_capabilities_summary(self) -> Dict[str, int]:
        """Get summary of capabilities across all bots"""
        capabilities_count = {}
        for capability in BotCapability:
            capabilities_count[capability.value] = len([
                bot for bot in self.registered_bots.values()
                if capability in bot.capabilities and bot.status == BotStatus.ONLINE
            ])
        return capabilities_count
    
    def _start_command_execution_thread(self):
        """Start the command execution thread"""
        self._should_stop = False
        self._command_thread = threading.Thread(
            target=self._command_execution_worker,
            daemon=True,
            name="CommandExecutionWorker"
        )
        self._command_thread.start()
        logger.info("Started command execution thread")
    
    def _command_execution_worker(self):
        """Worker thread for executing cross-bot commands"""
        while not self._should_stop:
            try:
                if self.command_queue:
                    command = self.command_queue.pop(0)
                    self._execute_command(command)
                else:
                    # Sleep briefly if no commands
                    threading.Event().wait(0.1)
                    
            except Exception as e:
                logger.error(f"Error in command execution worker: {e}")
    
    def _execute_command(self, command: CrossBotCommand):
        """Execute a single cross-bot command"""
        try:
            logger.info(f"Executing cross-bot command: {command.command} on bots: {command.target_bots}")
            
            results = []
            for bot_id in command.target_bots:
                if bot_id not in self.registered_bots:
                    continue
                
                bot_info = self.registered_bots[bot_id]
                if bot_info.status != BotStatus.ONLINE:
                    continue
                
                start_time = datetime.now()
                try:
                    # Execute command on bot instance
                    if hasattr(bot_info.instance, 'execute_command'):
                        result = bot_info.instance.execute_command(command.command, command.args)
                        success = True
                        error_message = None
                    else:
                        result = f"Bot {bot_id} does not support command execution"
                        success = False
                        error_message = "Command execution not supported"
                    
                except Exception as e:
                    result = None
                    success = False
                    error_message = str(e)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                command_result = CommandResult(
                    command_id=command.command_id,
                    bot_id=bot_id,
                    success=success,
                    result=result,
                    error_message=error_message,
                    execution_time=execution_time,
                    timestamp=datetime.now()
                )
                
                results.append(command_result)
            
            # Store results
            self.command_results[command.command_id] = results
            
            # Share intelligence if applicable
            if command.command in ['forecast', 'analyze']:
                self.intelligence_sharing.share_command_results(command, results)
            
            logger.info(f"Completed cross-bot command: {command.command_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute command {command.command_id}: {e}")
    
    def stop(self):
        """Stop the command center"""
        self._should_stop = True
        if self._command_thread and self._command_thread.is_alive():
            self._command_thread.join(timeout=2.0)
        logger.info("Unified Command Center stopped")


# Global instance
_command_center = None

def get_command_center() -> UnifiedCommandCenter:
    """Get global command center instance"""
    global _command_center
    if _command_center is None:
        _command_center = UnifiedCommandCenter()
    return _command_center

def initialize_command_center() -> UnifiedCommandCenter:
    """Initialize global command center instance"""
    global _command_center
    _command_center = UnifiedCommandCenter()
    return _command_center
