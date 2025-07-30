"""
Risk Coordination Engine
=======================
Intelligent risk distribution and portfolio-level risk management across multiple bots
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BotRiskProfile:
    """Risk profile for a specific bot"""
    bot_id: str
    max_exposure: float
    current_exposure: float
    risk_level: RiskLevel
    correlation_risk: float
    performance_score: float
    last_updated: datetime

@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    total_exposure: float
    max_exposure: float
    risk_utilization: float  # Percentage of max risk used
    correlation_risk: float
    concentration_risk: float
    volatility_risk: float
    overall_risk_level: RiskLevel
    timestamp: datetime

class RiskCoordinator:
    """
    Risk Coordination Engine
    
    Manages intelligent risk distribution across multiple trading bots,
    monitors portfolio-level risk, and implements emergency risk protocols.
    """
    
    def __init__(self, config_file: str = "config/risk_config.json"):
        """Initialize the risk coordinator"""
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Risk profiles for each bot
        self.bot_risk_profiles: Dict[str, BotRiskProfile] = {}
        
        # Portfolio risk settings
        self.max_portfolio_exposure = 100000.0  # Default max exposure
        self.max_correlation_risk = 0.7  # Max correlation between bots
        self.max_concentration_risk = 0.4  # Max exposure in single bot
        self.emergency_shutdown_threshold = 0.9  # Risk utilization threshold
        
        # Risk monitoring
        self.current_portfolio_risk: Optional[PortfolioRisk] = None
        self._risk_monitor_thread = None
        self._should_stop_monitoring = False
        self._monitoring_interval = 60  # seconds
        
        # Load configuration
        self._load_risk_config()
        
        logger.info("Risk Coordinator initialized")
    
    def register_bot_risk_profile(self, bot_id: str, max_exposure: float, 
                                 performance_score: float = 50.0) -> bool:
        """
        Register risk profile for a bot
        
        Args:
            bot_id: Bot identifier
            max_exposure: Maximum exposure allowed for this bot
            performance_score: Performance score (0-100)
            
        Returns:
            bool: Success status
        """
        try:
            risk_profile = BotRiskProfile(
                bot_id=bot_id,
                max_exposure=max_exposure,
                current_exposure=0.0,
                risk_level=RiskLevel.LOW,
                correlation_risk=0.0,
                performance_score=performance_score,
                last_updated=datetime.now()
            )
            
            self.bot_risk_profiles[bot_id] = risk_profile
            
            # Start risk monitoring if not running
            if not self._risk_monitor_thread or not self._risk_monitor_thread.is_alive():
                self._start_risk_monitoring()
            
            logger.info(f"Registered risk profile for bot {bot_id}: max_exposure={max_exposure}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register risk profile for {bot_id}: {e}")
            return False
    
    def update_bot_exposure(self, bot_id: str, current_exposure: float) -> bool:
        """
        Update current exposure for a bot
        
        Args:
            bot_id: Bot identifier
            current_exposure: Current exposure amount
            
        Returns:
            bool: Success status
        """
        try:
            if bot_id not in self.bot_risk_profiles:
                logger.warning(f"Bot {bot_id} not found in risk profiles")
                return False
            
            profile = self.bot_risk_profiles[bot_id]
            old_exposure = profile.current_exposure
            profile.current_exposure = current_exposure
            profile.last_updated = datetime.now()
            
            # Update risk level based on exposure
            exposure_ratio = current_exposure / profile.max_exposure
            if exposure_ratio >= 0.9:
                profile.risk_level = RiskLevel.CRITICAL
            elif exposure_ratio >= 0.7:
                profile.risk_level = RiskLevel.HIGH
            elif exposure_ratio >= 0.4:
                profile.risk_level = RiskLevel.MEDIUM
            else:
                profile.risk_level = RiskLevel.LOW
            
            # Update portfolio risk
            self._update_portfolio_risk()
            
            logger.info(f"Updated exposure for {bot_id}: {old_exposure} -> {current_exposure}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update exposure for {bot_id}: {e}")
            return False
    
    def distribute_risk(self, trading_bots: List[str], total_exposure: float) -> Dict[str, float]:
        """
        Intelligently distribute risk across trading bots
        
        Args:
            trading_bots: List of bot IDs that can trade
            total_exposure: Total exposure to distribute
            
        Returns:
            Dict mapping bot_id to allocated exposure
        """
        try:
            if not trading_bots:
                return {}
            
            # Check if total exposure exceeds portfolio limit
            if total_exposure > self.max_portfolio_exposure:
                logger.warning(f"Requested exposure {total_exposure} exceeds portfolio limit {self.max_portfolio_exposure}")
                total_exposure = self.max_portfolio_exposure
            
            allocation = {}
            
            # Get available bots with risk profiles
            available_bots = [
                bot_id for bot_id in trading_bots
                if bot_id in self.bot_risk_profiles
            ]
            
            if not available_bots:
                # Equal distribution if no risk profiles
                exposure_per_bot = total_exposure / len(trading_bots)
                return {bot_id: exposure_per_bot for bot_id in trading_bots}
            
            # Calculate allocation weights based on:
            # 1. Available capacity (max_exposure - current_exposure)
            # 2. Performance score
            # 3. Current risk level
            
            weights = {}
            total_weight = 0.0
            
            for bot_id in available_bots:
                profile = self.bot_risk_profiles[bot_id]
                
                # Available capacity weight
                available_capacity = max(0, profile.max_exposure - profile.current_exposure)
                capacity_weight = available_capacity / profile.max_exposure if profile.max_exposure > 0 else 0
                
                # Performance weight (0-1)
                performance_weight = profile.performance_score / 100.0
                
                # Risk level weight (lower risk = higher weight)
                risk_weights = {
                    RiskLevel.LOW: 1.0,
                    RiskLevel.MEDIUM: 0.7,
                    RiskLevel.HIGH: 0.4,
                    RiskLevel.CRITICAL: 0.1
                }
                risk_weight = risk_weights.get(profile.risk_level, 0.5)
                
                # Combined weight
                combined_weight = capacity_weight * 0.5 + performance_weight * 0.3 + risk_weight * 0.2
                weights[bot_id] = combined_weight
                total_weight += combined_weight
            
            # Distribute exposure based on weights
            if total_weight > 0:
                for bot_id in available_bots:
                    weight_ratio = weights[bot_id] / total_weight
                    allocated_exposure = total_exposure * weight_ratio
                    
                    # Ensure allocation doesn't exceed bot's max exposure
                    profile = self.bot_risk_profiles[bot_id]
                    max_additional = profile.max_exposure - profile.current_exposure
                    allocated_exposure = min(allocated_exposure, max_additional)
                    
                    allocation[bot_id] = allocated_exposure
            
            # Handle any remaining bots without risk profiles
            remaining_bots = [bot_id for bot_id in trading_bots if bot_id not in available_bots]
            if remaining_bots:
                remaining_exposure = total_exposure - sum(allocation.values())
                if remaining_exposure > 0:
                    exposure_per_remaining = remaining_exposure / len(remaining_bots)
                    for bot_id in remaining_bots:
                        allocation[bot_id] = exposure_per_remaining
            
            logger.info(f"Distributed {total_exposure} exposure across {len(allocation)} bots")
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to distribute risk: {e}")
            return {}
    
    def check_risk_limits(self, bot_id: str, additional_exposure: float) -> Tuple[bool, str]:
        """
        Check if additional exposure would violate risk limits
        
        Args:
            bot_id: Bot identifier
            additional_exposure: Additional exposure to check
            
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Check bot-specific limits
            if bot_id in self.bot_risk_profiles:
                profile = self.bot_risk_profiles[bot_id]
                new_exposure = profile.current_exposure + additional_exposure
                
                if new_exposure > profile.max_exposure:
                    return False, f"Would exceed bot max exposure ({profile.max_exposure})"
                
                if profile.risk_level == RiskLevel.CRITICAL:
                    return False, f"Bot is at critical risk level"
            
            # Check portfolio-level limits
            current_total = sum(profile.current_exposure for profile in self.bot_risk_profiles.values())
            new_total = current_total + additional_exposure
            
            if new_total > self.max_portfolio_exposure:
                return False, f"Would exceed portfolio max exposure ({self.max_portfolio_exposure})"
            
            # Check concentration risk
            if bot_id in self.bot_risk_profiles:
                profile = self.bot_risk_profiles[bot_id]
                new_concentration = (profile.current_exposure + additional_exposure) / new_total
                
                if new_concentration > self.max_concentration_risk:
                    return False, f"Would exceed concentration risk limit ({self.max_concentration_risk})"
            
            return True, "Risk limits OK"
            
        except Exception as e:
            logger.error(f"Failed to check risk limits: {e}")
            return False, f"Risk check error: {str(e)}"
    
    def get_portfolio_risk_status(self) -> Dict[str, Any]:
        """
        Get current portfolio risk status
        
        Returns:
            Dict containing portfolio risk information
        """
        try:
            self._update_portfolio_risk()
            
            if not self.current_portfolio_risk:
                return {'error': 'No portfolio risk data available'}
            
            risk = self.current_portfolio_risk
            
            # Get individual bot risk statuses
            bot_risks = []
            for bot_id, profile in self.bot_risk_profiles.items():
                bot_risks.append({
                    'bot_id': bot_id,
                    'current_exposure': profile.current_exposure,
                    'max_exposure': profile.max_exposure,
                    'utilization': (profile.current_exposure / profile.max_exposure * 100) if profile.max_exposure > 0 else 0,
                    'risk_level': profile.risk_level.value,
                    'performance_score': profile.performance_score
                })
            
            return {
                'timestamp': risk.timestamp,
                'total_exposure': risk.total_exposure,
                'max_exposure': risk.max_exposure,
                'risk_utilization': risk.risk_utilization,
                'overall_risk_level': risk.overall_risk_level.value,
                'correlation_risk': risk.correlation_risk,
                'concentration_risk': risk.concentration_risk,
                'volatility_risk': risk.volatility_risk,
                'bot_count': len(self.bot_risk_profiles),
                'bot_risks': bot_risks,
                'emergency_threshold': self.emergency_shutdown_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio risk status: {e}")
            return {'error': str(e)}
    
    def emergency_risk_shutdown(self, reason: str = "Emergency risk limit exceeded") -> bool:
        """
        Emergency shutdown due to risk limits
        
        Args:
            reason: Reason for emergency shutdown
            
        Returns:
            bool: Success status
        """
        try:
            logger.critical(f"EMERGENCY RISK SHUTDOWN: {reason}")
            
            # Set all bots to critical risk level
            for profile in self.bot_risk_profiles.values():
                profile.risk_level = RiskLevel.CRITICAL
            
            # This would trigger emergency protocols in the command center
            # For now, just log the event
            logger.critical("All bots set to CRITICAL risk level - manual intervention required")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed emergency risk shutdown: {e}")
            return False
    
    def _update_portfolio_risk(self):
        """Update portfolio-level risk assessment"""
        try:
            if not self.bot_risk_profiles:
                return
            
            # Calculate total exposure
            total_exposure = sum(profile.current_exposure for profile in self.bot_risk_profiles.values())
            
            # Calculate risk utilization
            risk_utilization = total_exposure / self.max_portfolio_exposure if self.max_portfolio_exposure > 0 else 0
            
            # Calculate concentration risk (max single bot exposure ratio)
            concentration_risk = 0.0
            if total_exposure > 0:
                max_bot_exposure = max(profile.current_exposure for profile in self.bot_risk_profiles.values())
                concentration_risk = max_bot_exposure / total_exposure
            
            # Calculate correlation risk (simplified - would need actual correlation data)
            correlation_risk = sum(profile.correlation_risk for profile in self.bot_risk_profiles.values()) / len(self.bot_risk_profiles)
            
            # Calculate volatility risk (simplified)
            volatility_risk = risk_utilization * 0.5 + concentration_risk * 0.3 + correlation_risk * 0.2
            
            # Determine overall risk level
            if risk_utilization >= 0.9 or concentration_risk >= 0.8:
                overall_risk_level = RiskLevel.CRITICAL
            elif risk_utilization >= 0.7 or concentration_risk >= 0.6:
                overall_risk_level = RiskLevel.HIGH
            elif risk_utilization >= 0.4 or concentration_risk >= 0.4:
                overall_risk_level = RiskLevel.MEDIUM
            else:
                overall_risk_level = RiskLevel.LOW
            
            # Create portfolio risk object
            self.current_portfolio_risk = PortfolioRisk(
                total_exposure=total_exposure,
                max_exposure=self.max_portfolio_exposure,
                risk_utilization=risk_utilization,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                overall_risk_level=overall_risk_level,
                timestamp=datetime.now()
            )
            
            # Check for emergency shutdown
            if risk_utilization >= self.emergency_shutdown_threshold:
                self.emergency_risk_shutdown(f"Risk utilization exceeded threshold: {risk_utilization:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to update portfolio risk: {e}")
    
    def _start_risk_monitoring(self):
        """Start the risk monitoring thread"""
        self._should_stop_monitoring = False
        self._risk_monitor_thread = threading.Thread(
            target=self._risk_monitoring_worker,
            daemon=True,
            name="RiskMonitor"
        )
        self._risk_monitor_thread.start()
        logger.info("Started risk monitoring")
    
    def _risk_monitoring_worker(self):
        """Worker thread for risk monitoring"""
        while not self._should_stop_monitoring:
            try:
                self._update_portfolio_risk()
                threading.Event().wait(self._monitoring_interval)
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
    
    def _load_risk_config(self):
        """Load risk configuration from file"""
        try:
            if not self.config_file.exists():
                self._save_risk_config()  # Create default config
                return
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.max_portfolio_exposure = config.get('max_portfolio_exposure', 100000.0)
            self.max_correlation_risk = config.get('max_correlation_risk', 0.7)
            self.max_concentration_risk = config.get('max_concentration_risk', 0.4)
            self.emergency_shutdown_threshold = config.get('emergency_shutdown_threshold', 0.9)
            
            logger.info("Loaded risk configuration")
            
        except Exception as e:
            logger.error(f"Failed to load risk config: {e}")
    
    def _save_risk_config(self):
        """Save risk configuration to file"""
        try:
            config = {
                'max_portfolio_exposure': self.max_portfolio_exposure,
                'max_correlation_risk': self.max_correlation_risk,
                'max_concentration_risk': self.max_concentration_risk,
                'emergency_shutdown_threshold': self.emergency_shutdown_threshold
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save risk config: {e}")
    
    def stop(self):
        """Stop the risk coordinator"""
        self._should_stop_monitoring = True
        if self._risk_monitor_thread and self._risk_monitor_thread.is_alive():
            self._risk_monitor_thread.join(timeout=2.0)
        logger.info("Risk Coordinator stopped")


# Global instance
_risk_coordinator = None

def get_risk_coordinator() -> RiskCoordinator:
    """Get global risk coordinator instance"""
    global _risk_coordinator
    if _risk_coordinator is None:
        _risk_coordinator = RiskCoordinator()
    return _risk_coordinator
