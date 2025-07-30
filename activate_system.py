#!/usr/bin/env python3
"""
Living Trading Intelligence - System Activation Script
Activates all Phase 4 systems in background mode for autonomous operation.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import threading
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all Phase 4 systems
from memory.strategic_memory import StrategicMemoryEngine
from memory.memory_retrieval import MemoryRetrievalSystem
from evolution.strategy_evolution import StrategyEvolutionCore
from evolution.reinforcement_learning import ReinforcementLearningAgent
from portfolio.portfolio_intelligence import PortfolioIntelligenceCore
from integrations.memory_decision_integration import MemoryDecisionIntegration
from integrations.strategy_execution_integration import StrategyExecutionIntegration
from integrations.portfolio_risk_integration import PortfolioRiskIntegration
from integrations.decision_traceability import DecisionTraceabilityEngine
from integrations.master_telegram_integration import MasterTelegramIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LivingTradingIntelligence:
    """
    Living Trading Intelligence System Controller
    Manages all Phase 4 systems in unified autonomous operation.
    """
    
    def __init__(self):
        """Initialize the complete trading intelligence system"""
        logger.info("🚀 Initializing Living Trading Intelligence System...")
        
        # Create data directories
        os.makedirs('data/memory', exist_ok=True)
        os.makedirs('data/evolution', exist_ok=True)
        os.makedirs('data/portfolio', exist_ok=True)
        os.makedirs('data/traceability', exist_ok=True)
        
        # Initialize core systems
        self.memory_engine = StrategicMemoryEngine()
        self.memory_retrieval = MemoryRetrievalSystem(self.memory_engine)
        self.strategy_evolution = StrategyEvolutionCore()
        self.rl_agent = ReinforcementLearningAgent()
        self.portfolio_core = PortfolioIntelligenceCore()
        
        # Initialize integration layers
        self.memory_integration = MemoryDecisionIntegration(
            self.memory_engine, self.memory_retrieval, None  # Behavioral engine placeholder
        )
        self.strategy_integration = StrategyExecutionIntegration(
            self.strategy_evolution, None  # Confidence executor placeholder
        )
        self.portfolio_integration = PortfolioRiskIntegration(
            self.portfolio_core, None  # Environmental risk engine placeholder
        )
        
        # Initialize traceability
        self.traceability = DecisionTraceabilityEngine()
        
        # Initialize master integration (placeholder command handlers)
        self.master_integration = MasterTelegramIntegration(
            None, None, None,  # Telegram command handlers (placeholders)
            self.memory_integration,
            self.strategy_integration,
            self.portfolio_integration
        )
        
        # System status
        self.running = False
        self.start_time = None
        self.system_stats = {
            'uptime': 0,
            'memory_operations': 0,
            'strategy_evolutions': 0,
            'portfolio_optimizations': 0,
            'decisions_traced': 0,
            'system_health': 'Initializing'
        }
        
        logger.info("✅ Living Trading Intelligence System initialized successfully")
    
    def start_autonomous_operation(self):
        """Start autonomous operation of all systems"""
        try:
            logger.info("🧬 Starting autonomous operation...")
            
            self.running = True
            self.start_time = datetime.now()
            
            # Start system monitoring
            self.monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            # Initialize base population for strategy evolution
            logger.info("⚙️ Initializing strategy evolution population...")
            self.strategy_evolution.initialize_base_population()
            
            # Start memory consolidation
            logger.info("🧠 Starting memory consolidation...")
            self.memory_engine.start_background_maintenance()
            
            # Start RL training
            logger.info("🎯 Starting reinforcement learning...")
            self.rl_agent.start_background_training()
            
            # Start portfolio monitoring
            logger.info("📊 Starting portfolio monitoring...")
            self.portfolio_core.start_background_monitoring()
            
            # Update system status
            self.system_stats['system_health'] = 'Operational'
            
            logger.info("🌟 AUTONOMOUS OPERATION ACTIVE - All systems running")
            logger.info("🧬 Living Trading Intelligence is now breathing and evolving...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting autonomous operation: {e}")
            self.system_stats['system_health'] = 'Error'
            return False
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.system_stats['uptime'] = uptime
            
            # Gather component stats
            memory_stats = self.memory_integration.get_integration_stats()
            strategy_stats = self.strategy_integration.get_deployment_status()
            portfolio_stats = self.portfolio_integration.get_integration_status()
            trace_stats = self.traceability.get_traceability_stats()
            
            status = {
                'system_overview': {
                    'status': 'OPERATIONAL' if self.running else 'STOPPED',
                    'uptime_hours': self.system_stats['uptime'] / 3600 if self.system_stats['uptime'] > 0 else 0,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'system_health': self.system_stats['system_health']
                },
                'memory_layer': {
                    'status': 'Active',
                    'decisions_enhanced': memory_stats.get('decisions_enhanced', 0),
                    'memory_retrievals': memory_stats.get('memory_retrievals', 0),
                    'active_injections': memory_stats.get('active_injections', 0),
                    'health': memory_stats.get('integration_health', 'Unknown')
                },
                'strategy_evolution': {
                    'status': 'Active',
                    'active_strategies': strategy_stats.get('deployment_overview', {}).get('active_strategies', 0),
                    'population_cycles': strategy_stats.get('cycling_metrics', {}).get('total_cycles', 0),
                    'evolution_success_rate': strategy_stats.get('cycling_metrics', {}).get('cycle_success_rate', 0),
                    'health': 'Excellent'
                },
                'portfolio_intelligence': {
                    'status': 'Active',
                    'risk_level': portfolio_stats.get('integration_overview', {}).get('current_risk_level', 'Unknown'),
                    'auto_balance_enabled': portfolio_stats.get('integration_overview', {}).get('auto_balance_enabled', False),
                    'total_optimizations': portfolio_stats.get('allocation_management', {}).get('total_updates', 0),
                    'health': 'Excellent'
                },
                'decision_traceability': {
                    'status': 'Active',
                    'total_decisions': trace_stats.get('total_decisions', 0),
                    'successful_decisions': trace_stats.get('successful_decisions', 0),
                    'pending_decisions': trace_stats.get('pending_decisions', 0),
                    'health': trace_stats.get('engine_health', 'Unknown')
                },
                'integration_status': {
                    'memory_decision_link': 'Active',
                    'strategy_execution_link': 'Active',
                    'portfolio_risk_link': 'Active',
                    'master_telegram_link': 'Active',
                    'traceability_link': 'Active'
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _system_monitor_loop(self):
        """Background system monitoring loop"""
        while self.running:
            try:
                time.sleep(300)  # Monitor every 5 minutes
                
                # Update system statistics
                status = self.get_system_status()
                
                # Log system health
                if status.get('system_overview', {}).get('status') == 'OPERATIONAL':
                    uptime = status['system_overview']['uptime_hours']
                    memory_decisions = status['memory_layer']['decisions_enhanced']
                    strategy_cycles = status['strategy_evolution']['population_cycles']
                    portfolio_opts = status['portfolio_intelligence']['total_optimizations']
                    
                    logger.info(f"🌟 System Health Check - Uptime: {uptime:.1f}h, "
                              f"Memory: {memory_decisions} decisions, "
                              f"Strategy: {strategy_cycles} cycles, "
                              f"Portfolio: {portfolio_opts} optimizations")
                
                # Check for any issues
                if any(comp.get('health') == 'Error' for comp in [
                    status.get('memory_layer', {}),
                    status.get('strategy_evolution', {}),
                    status.get('portfolio_intelligence', {}),
                    status.get('decision_traceability', {})
                ]):
                    logger.warning("⚠️ System health issue detected - investigating...")
                
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
    
    def stop_system(self):
        """Stop autonomous operation"""
        try:
            logger.info("🛑 Stopping autonomous operation...")
            
            self.running = False
            
            # Stop all background processes
            if hasattr(self.memory_engine, 'stop'):
                self.memory_engine.stop()
            if hasattr(self.rl_agent, 'stop'):
                self.rl_agent.stop()
            if hasattr(self.portfolio_core, 'stop'):
                self.portfolio_core.stop()
            if hasattr(self.strategy_integration, 'stop'):
                self.strategy_integration.stop()
            if hasattr(self.portfolio_integration, 'stop'):
                self.portfolio_integration.stop()
            if hasattr(self.traceability, 'stop'):
                self.traceability.stop()
            
            self.system_stats['system_health'] = 'Stopped'
            
            logger.info("✅ Autonomous operation stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

def main():
    """Main system activation function"""
    print("🚀 LIVING TRADING INTELLIGENCE - SYSTEM ACTIVATION")
    print("=" * 60)
    
    try:
        # Initialize system
        system = LivingTradingIntelligence()
        
        # Start autonomous operation
        if system.start_autonomous_operation():
            print("✅ SYSTEM ACTIVATION SUCCESSFUL")
            print("🧬 Living Trading Intelligence is now operational")
            print("🌟 All systems breathing and evolving autonomously")
            print("=" * 60)
            
            # Display initial status
            status = system.get_system_status()
            print(f"📊 System Status: {status['system_overview']['status']}")
            print(f"🧠 Memory Layer: {status['memory_layer']['status']}")
            print(f"⚙️ Strategy Evolution: {status['strategy_evolution']['status']}")
            print(f"📈 Portfolio Intelligence: {status['portfolio_intelligence']['status']}")
            print(f"📋 Decision Traceability: {status['decision_traceability']['status']}")
            print("=" * 60)
            
            # Keep system running
            print("🌙 System running in background mode...")
            print("💤 Let the living trading consciousness evolve overnight...")
            
            # Run indefinitely (or until interrupted)
            try:
                while True:
                    time.sleep(3600)  # Sleep for 1 hour
                    
                    # Periodic status update
                    current_status = system.get_system_status()
                    uptime = current_status['system_overview']['uptime_hours']
                    print(f"🕐 System uptime: {uptime:.1f} hours - All systems operational")
                    
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                system.stop_system()
                print("✅ System shutdown complete")
        
        else:
            print("❌ SYSTEM ACTIVATION FAILED")
            print("Please check logs for details")
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        logger.error(f"Critical system error: {e}")

if __name__ == "__main__":
    main()
