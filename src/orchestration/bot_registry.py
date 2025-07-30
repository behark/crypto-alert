"""
Bot Registry & Discovery System
==============================
Automatic discovery and management of available trading bots
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time

from .command_center import BotInfo, BotStatus, BotCapability

logger = logging.getLogger(__name__)

@dataclass
class BotHealthMetrics:
    """Health metrics for a registered bot"""
    last_heartbeat: datetime
    response_time_avg: float
    success_rate: float
    error_count: int
    uptime_percentage: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

class BotRegistry:
    """
    Bot Registry and Discovery System
    
    Manages automatic discovery, health monitoring, and capability tracking
    of all registered trading bots in the unified platform.
    """
    
    def __init__(self, registry_file: str = "data/bot_registry.json"):
        """Initialize the bot registry"""
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.registered_bots: Dict[str, BotInfo] = {}
        self.bot_health: Dict[str, BotHealthMetrics] = {}
        self.capability_index: Dict[BotCapability, Set[str]] = {
            capability: set() for capability in BotCapability
        }
        
        # Health monitoring
        self._health_monitor_thread = None
        self._should_stop_monitoring = False
        self._monitoring_interval = 30  # seconds
        
        # Load existing registry
        self._load_registry()
        
        logger.info("Bot Registry initialized")
    
    def register_bot(self, bot_info: BotInfo) -> bool:
        """
        Register a new bot in the registry
        
        Args:
            bot_info: Bot information to register
            
        Returns:
            bool: Success status
        """
        try:
            # Add to registry
            self.registered_bots[bot_info.bot_id] = bot_info
            
            # Initialize health metrics
            self.bot_health[bot_info.bot_id] = BotHealthMetrics(
                last_heartbeat=datetime.now(),
                response_time_avg=0.0,
                success_rate=100.0,
                error_count=0,
                uptime_percentage=100.0
            )
            
            # Update capability index
            for capability in bot_info.capabilities:
                self.capability_index[capability].add(bot_info.bot_id)
            
            # Save registry
            self._save_registry()
            
            # Start health monitoring if not running
            if not self._health_monitor_thread or not self._health_monitor_thread.is_alive():
                self._start_health_monitoring()
            
            logger.info(f"Registered bot in registry: {bot_info.name} ({bot_info.bot_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register bot {bot_info.bot_id}: {e}")
            return False
    
    def unregister_bot(self, bot_id: str) -> bool:
        """
        Unregister a bot from the registry
        
        Args:
            bot_id: Bot ID to unregister
            
        Returns:
            bool: Success status
        """
        try:
            if bot_id not in self.registered_bots:
                logger.warning(f"Bot {bot_id} not found in registry")
                return False
            
            bot_info = self.registered_bots[bot_id]
            
            # Remove from registry
            del self.registered_bots[bot_id]
            
            # Remove health metrics
            if bot_id in self.bot_health:
                del self.bot_health[bot_id]
            
            # Update capability index
            for capability in bot_info.capabilities:
                self.capability_index[capability].discard(bot_id)
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Unregistered bot from registry: {bot_info.name} ({bot_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister bot {bot_id}: {e}")
            return False
    
    def get_bots_by_capability(self, capability: BotCapability, 
                              online_only: bool = True) -> List[BotInfo]:
        """
        Get all bots with a specific capability
        
        Args:
            capability: Required capability
            online_only: Only return online bots
            
        Returns:
            List of bot info objects
        """
        try:
            bot_ids = self.capability_index.get(capability, set())
            bots = []
            
            for bot_id in bot_ids:
                if bot_id in self.registered_bots:
                    bot_info = self.registered_bots[bot_id]
                    if not online_only or bot_info.status == BotStatus.ONLINE:
                        bots.append(bot_info)
            
            return bots
            
        except Exception as e:
            logger.error(f"Failed to get bots by capability {capability}: {e}")
            return []
    
    def get_bot_info(self, bot_id: str) -> Optional[BotInfo]:
        """
        Get information for a specific bot
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            BotInfo object or None if not found
        """
        return self.registered_bots.get(bot_id)
    
    def get_all_bots(self, status_filter: Optional[BotStatus] = None) -> List[BotInfo]:
        """
        Get all registered bots
        
        Args:
            status_filter: Optional status filter
            
        Returns:
            List of all bot info objects
        """
        bots = list(self.registered_bots.values())
        
        if status_filter:
            bots = [bot for bot in bots if bot.status == status_filter]
        
        return bots
    
    def get_bot_health(self, bot_id: str) -> Optional[BotHealthMetrics]:
        """
        Get health metrics for a specific bot
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            BotHealthMetrics or None if not found
        """
        return self.bot_health.get(bot_id)
    
    def update_bot_status(self, bot_id: str, status: BotStatus, 
                         error_message: Optional[str] = None) -> bool:
        """
        Update bot status
        
        Args:
            bot_id: Bot identifier
            status: New status
            error_message: Optional error message
            
        Returns:
            bool: Success status
        """
        try:
            if bot_id not in self.registered_bots:
                logger.warning(f"Bot {bot_id} not found for status update")
                return False
            
            old_status = self.registered_bots[bot_id].status
            self.registered_bots[bot_id].status = status
            self.registered_bots[bot_id].last_heartbeat = datetime.now()
            
            # Update health metrics
            if bot_id in self.bot_health:
                self.bot_health[bot_id].last_heartbeat = datetime.now()
                
                if status == BotStatus.ERROR:
                    self.bot_health[bot_id].error_count += 1
            
            # Save registry if status changed
            if old_status != status:
                self._save_registry()
                logger.info(f"Bot {bot_id} status changed: {old_status.value} -> {status.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update bot status for {bot_id}: {e}")
            return False
    
    def heartbeat(self, bot_id: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record heartbeat from a bot
        
        Args:
            bot_id: Bot identifier
            metrics: Optional performance metrics
            
        Returns:
            bool: Success status
        """
        try:
            if bot_id not in self.registered_bots:
                return False
            
            now = datetime.now()
            self.registered_bots[bot_id].last_heartbeat = now
            
            # Update health metrics
            if bot_id in self.bot_health:
                health = self.bot_health[bot_id]
                health.last_heartbeat = now
                
                # Update metrics if provided
                if metrics:
                    if 'response_time' in metrics:
                        # Update average response time
                        old_avg = health.response_time_avg
                        new_time = metrics['response_time']
                        health.response_time_avg = (old_avg * 0.9) + (new_time * 0.1)
                    
                    if 'success_rate' in metrics:
                        health.success_rate = metrics['success_rate']
                    
                    if 'memory_usage' in metrics:
                        health.memory_usage = metrics['memory_usage']
                    
                    if 'cpu_usage' in metrics:
                        health.cpu_usage = metrics['cpu_usage']
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record heartbeat for {bot_id}: {e}")
            return False
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive registry summary
        
        Returns:
            Dict containing registry summary information
        """
        try:
            total_bots = len(self.registered_bots)
            online_bots = len([b for b in self.registered_bots.values() if b.status == BotStatus.ONLINE])
            
            capability_summary = {}
            for capability, bot_ids in self.capability_index.items():
                online_count = len([
                    bot_id for bot_id in bot_ids 
                    if bot_id in self.registered_bots and 
                    self.registered_bots[bot_id].status == BotStatus.ONLINE
                ])
                capability_summary[capability.value] = {
                    'total': len(bot_ids),
                    'online': online_count
                }
            
            return {
                'timestamp': datetime.now(),
                'total_bots': total_bots,
                'online_bots': online_bots,
                'offline_bots': total_bots - online_bots,
                'capabilities': capability_summary,
                'health_summary': self._get_health_summary()
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry summary: {e}")
            return {'error': str(e)}
    
    def discover_bots(self, discovery_paths: List[str] = None) -> List[str]:
        """
        Auto-discover bots in specified paths
        
        Args:
            discovery_paths: Paths to search for bots
            
        Returns:
            List of discovered bot IDs
        """
        # This would implement auto-discovery logic
        # For now, return empty list as manual registration is preferred
        logger.info("Bot auto-discovery not yet implemented")
        return []
    
    def _start_health_monitoring(self):
        """Start the health monitoring thread"""
        self._should_stop_monitoring = False
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitoring_worker,
            daemon=True,
            name="BotHealthMonitor"
        )
        self._health_monitor_thread.start()
        logger.info("Started bot health monitoring")
    
    def _health_monitoring_worker(self):
        """Worker thread for monitoring bot health"""
        while not self._should_stop_monitoring:
            try:
                self._check_bot_health()
                time.sleep(self._monitoring_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    def _check_bot_health(self):
        """Check health of all registered bots"""
        try:
            now = datetime.now()
            timeout_threshold = timedelta(minutes=5)  # 5 minutes timeout
            
            for bot_id, bot_info in self.registered_bots.items():
                # Check if bot has timed out
                if now - bot_info.last_heartbeat > timeout_threshold:
                    if bot_info.status == BotStatus.ONLINE:
                        logger.warning(f"Bot {bot_id} appears to be offline (no heartbeat)")
                        self.update_bot_status(bot_id, BotStatus.OFFLINE)
                
                # Update uptime percentage
                if bot_id in self.bot_health:
                    health = self.bot_health[bot_id]
                    # Calculate uptime based on heartbeat frequency
                    time_since_heartbeat = (now - health.last_heartbeat).total_seconds()
                    if time_since_heartbeat < self._monitoring_interval * 2:
                        # Bot is responsive
                        health.uptime_percentage = min(100.0, health.uptime_percentage + 1.0)
                    else:
                        # Bot is not responsive
                        health.uptime_percentage = max(0.0, health.uptime_percentage - 5.0)
            
        except Exception as e:
            logger.error(f"Failed to check bot health: {e}")
    
    def _get_health_summary(self) -> Dict[str, Any]:
        """Get summary of bot health metrics"""
        try:
            if not self.bot_health:
                return {}
            
            total_bots = len(self.bot_health)
            avg_uptime = sum(h.uptime_percentage for h in self.bot_health.values()) / total_bots
            avg_response_time = sum(h.response_time_avg for h in self.bot_health.values()) / total_bots
            total_errors = sum(h.error_count for h in self.bot_health.values())
            
            return {
                'average_uptime': round(avg_uptime, 2),
                'average_response_time': round(avg_response_time, 3),
                'total_errors': total_errors,
                'healthy_bots': len([h for h in self.bot_health.values() if h.uptime_percentage > 90])
            }
            
        except Exception as e:
            logger.error(f"Failed to get health summary: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            registry_data = {
                'timestamp': datetime.now().isoformat(),
                'bots': {}
            }
            
            for bot_id, bot_info in self.registered_bots.items():
                registry_data['bots'][bot_id] = {
                    'name': bot_info.name,
                    'description': bot_info.description,
                    'capabilities': [c.value for c in bot_info.capabilities],
                    'status': bot_info.status.value,
                    'last_heartbeat': bot_info.last_heartbeat.isoformat(),
                    'performance_metrics': bot_info.performance_metrics
                }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_registry(self):
        """Load registry from file"""
        try:
            if not self.registry_file.exists():
                return
            
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Note: We only load metadata, not bot instances
            # Bot instances need to be re-registered on startup
            logger.info(f"Loaded registry with {len(registry_data.get('bots', {}))} bot entries")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def stop(self):
        """Stop the bot registry"""
        self._should_stop_monitoring = True
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=2.0)
        logger.info("Bot Registry stopped")


# Global instance
_bot_registry = None

def get_bot_registry() -> BotRegistry:
    """Get global bot registry instance"""
    global _bot_registry
    if _bot_registry is None:
        _bot_registry = BotRegistry()
    return _bot_registry
