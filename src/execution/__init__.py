"""
Autonomous Execution Engine
==========================
Real-time execution logic, safe auto-entry/exit, portfolio balancing, and circuit breaker protection
"""

from .confidence_executor import ConfidenceExecutor, get_confidence_executor
from .market_analyzer import MarketAnalyzer, get_market_analyzer
from .smart_entry import SmartEntryManager, get_smart_entry_manager
from .smart_exit import SmartExitManager, get_smart_exit_manager
from .human_override import HumanOverrideSystem, get_human_override_system
from .portfolio_manager import DynamicPortfolioManager, get_portfolio_manager
from .cross_bot_coordinator import CrossBotCoordinator, get_cross_bot_coordinator
from .circuit_breakers import CircuitBreakerSystem, get_circuit_breaker_system
from .risk_enforcement import RiskEnforcementEngine, get_risk_enforcement_engine

__all__ = [
    'ConfidenceExecutor',
    'get_confidence_executor',
    'MarketAnalyzer',
    'get_market_analyzer',
    'SmartEntryManager',
    'get_smart_entry_manager',
    'SmartExitManager',
    'get_smart_exit_manager',
    'HumanOverrideSystem',
    'get_human_override_system',
    'DynamicPortfolioManager',
    'get_portfolio_manager',
    'CrossBotCoordinator',
    'get_cross_bot_coordinator',
    'CircuitBreakerSystem',
    'get_circuit_breaker_system',
    'RiskEnforcementEngine',
    'get_risk_enforcement_engine'
]
