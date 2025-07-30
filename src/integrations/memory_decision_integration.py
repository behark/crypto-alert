"""
Memory-Driven Decision Logic Integration - Phase 4 Final Integration
Connects Strategic Memory Engine to Behavioral Decision Engine for memory-enhanced trading decisions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import numpy as np
from dataclasses import dataclass, asdict

from ..memory.strategic_memory import StrategicMemoryEngine, TradeMemory, MarketCondition
from ..memory.memory_retrieval import MemoryRetrievalSystem, MemoryQuery, MemoryMatch
from ..predictive.behavioral_decision import BehavioralDecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEnhancedDecision:
    """Decision enhanced with memory insights"""
    original_decision: Dict[str, Any]
    memory_insights: List[str]
    similar_patterns: List[MemoryMatch]
    success_probability: float
    memory_confidence: float
    recommended_adjustments: Dict[str, Any]
    historical_context: str
    risk_warnings: List[str]
    timestamp: datetime

@dataclass
class PatternInjection:
    """Manual pattern injection for decision enhancement"""
    pattern_id: str
    symbol: str
    market_condition: MarketCondition
    confidence_override: Optional[float]
    position_size_multiplier: float
    notes: str
    injected_by: str
    injection_time: datetime
    active: bool

class MemoryDecisionIntegration:
    """
    Memory-Driven Decision Logic Integration.
    Enhances every trading decision with comprehensive memory insights.
    """
    
    def __init__(self, memory_engine: StrategicMemoryEngine, 
                 retrieval_system: MemoryRetrievalSystem,
                 behavioral_engine: BehavioralDecisionEngine):
        """Initialize Memory-Decision Integration"""
        self.memory_engine = memory_engine
        self.retrieval_system = retrieval_system
        self.behavioral_engine = behavioral_engine
        
        # Pattern injection system
        self.active_injections: Dict[str, PatternInjection] = {}
        
        # Integration statistics
        self.integration_stats = {
            'decisions_enhanced': 0,
            'memory_retrievals': 0,
            'pattern_injections': 0,
            'success_rate_improvement': 0.0,
            'last_enhancement': None
        }
        
        logger.info("Memory-Decision Integration initialized - every decision now memory-enhanced")
    
    def enhance_decision_with_memory(self, decision_context: Dict[str, Any]) -> MemoryEnhancedDecision:
        """Enhance trading decision with comprehensive memory insights"""
        try:
            symbol = decision_context.get('symbol', 'UNKNOWN')
            market_condition = decision_context.get('market_condition', MarketCondition.SIDEWAYS)
            confidence = decision_context.get('confidence', 0.5)
            
            # Create memory query
            query = MemoryQuery(
                symbol=symbol,
                market_condition=market_condition,
                confidence_range=(confidence - 0.1, confidence + 0.1),
                time_range=(datetime.now() - timedelta(days=90), datetime.now()),
                limit=10
            )
            
            # Retrieve contextual memories
            similar_patterns = self.retrieval_system.contextual_recall(query)
            
            # Extract memory insights
            memory_insights = []
            success_probability = 0.5  # Default
            memory_confidence = 0.5
            risk_warnings = []
            
            if similar_patterns:
                # Analyze successful patterns
                successful_patterns = [p for p in similar_patterns if hasattr(p.memory, 'success') and p.memory.success]
                
                if successful_patterns:
                    success_probability = len(successful_patterns) / len(similar_patterns)
                    memory_confidence = np.mean([p.confidence for p in successful_patterns])
                    
                    memory_insights.append(f"Found {len(successful_patterns)}/{len(similar_patterns)} successful similar patterns")
                    memory_insights.append(f"Historical success rate: {success_probability:.1%}")
                    memory_insights.append(f"Memory confidence: {memory_confidence:.2f}")
                    
                    # Extract key success factors
                    if len(successful_patterns) >= 3:
                        avg_profit = np.mean([p.memory.profit_loss for p in successful_patterns if hasattr(p.memory, 'profit_loss')])
                        memory_insights.append(f"Average profit from similar trades: {avg_profit:.4f}")
                
                # Analyze failure patterns for warnings
                failed_patterns = [p for p in similar_patterns if hasattr(p.memory, 'success') and not p.memory.success]
                if failed_patterns:
                    failure_rate = len(failed_patterns) / len(similar_patterns)
                    if failure_rate > 0.4:
                        risk_warnings.append(f"High failure rate ({failure_rate:.1%}) in similar conditions")
                    
                    # Extract common failure reasons
                    for pattern in failed_patterns[:3]:
                        if hasattr(pattern.memory, 'lessons_learned'):
                            risk_warnings.append(f"Past failure: {pattern.memory.lessons_learned[:100]}")
            
            # Check for active pattern injections
            injection_key = f"{symbol}_{market_condition.value}"
            if injection_key in self.active_injections:
                injection = self.active_injections[injection_key]
                memory_insights.append(f"Active pattern injection: {injection.pattern_id}")
                memory_insights.append(f"Injection notes: {injection.notes}")
                if injection.confidence_override:
                    memory_confidence = injection.confidence_override
            
            # Generate recommended adjustments
            recommended_adjustments = self._generate_memory_adjustments(
                decision_context, similar_patterns, success_probability, memory_confidence
            )
            
            # Create historical context summary
            historical_context = self._create_historical_context(symbol, market_condition, similar_patterns)
            
            # Create enhanced decision
            enhanced_decision = MemoryEnhancedDecision(
                original_decision=decision_context.copy(),
                memory_insights=memory_insights,
                similar_patterns=similar_patterns,
                success_probability=success_probability,
                memory_confidence=memory_confidence,
                recommended_adjustments=recommended_adjustments,
                historical_context=historical_context,
                risk_warnings=risk_warnings,
                timestamp=datetime.now()
            )
            
            # Update statistics
            self.integration_stats['decisions_enhanced'] += 1
            self.integration_stats['memory_retrievals'] += len(similar_patterns)
            self.integration_stats['last_enhancement'] = datetime.now()
            
            logger.info(f"Decision enhanced with {len(similar_patterns)} memory patterns for {symbol}")
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Error enhancing decision with memory: {e}")
            # Return minimal enhanced decision
            return MemoryEnhancedDecision(
                original_decision=decision_context.copy(),
                memory_insights=[f"Memory enhancement failed: {str(e)}"],
                similar_patterns=[],
                success_probability=0.5,
                memory_confidence=0.5,
                recommended_adjustments={},
                historical_context="No historical context available",
                risk_warnings=["Memory system unavailable"],
                timestamp=datetime.now()
            )
    
    def inject_memory_pattern(self, symbol: str, market_condition: MarketCondition,
                             pattern_id: str, confidence_override: Optional[float] = None,
                             position_size_multiplier: float = 1.0, notes: str = "",
                             injected_by: str = "manual") -> str:
        """Inject specific memory pattern to influence future decisions"""
        try:
            injection_key = f"{symbol}_{market_condition.value}"
            
            injection = PatternInjection(
                pattern_id=pattern_id,
                symbol=symbol,
                market_condition=market_condition,
                confidence_override=confidence_override,
                position_size_multiplier=position_size_multiplier,
                notes=notes,
                injected_by=injected_by,
                injection_time=datetime.now(),
                active=True
            )
            
            self.active_injections[injection_key] = injection
            self.integration_stats['pattern_injections'] += 1
            
            logger.info(f"Pattern injection activated: {pattern_id} for {symbol} in {market_condition.value}")
            return injection_key
            
        except Exception as e:
            logger.error(f"Error injecting memory pattern: {e}")
            return ""
    
    def remove_pattern_injection(self, injection_key: str) -> bool:
        """Remove active pattern injection"""
        try:
            if injection_key in self.active_injections:
                self.active_injections[injection_key].active = False
                del self.active_injections[injection_key]
                logger.info(f"Pattern injection removed: {injection_key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing pattern injection: {e}")
            return False
    
    def get_active_injections(self) -> List[PatternInjection]:
        """Get all active pattern injections"""
        return list(self.active_injections.values())
    
    def apply_memory_enhanced_decision(self, enhanced_decision: MemoryEnhancedDecision) -> Dict[str, Any]:
        """Apply memory enhancements to create final trading decision"""
        try:
            # Start with original decision
            final_decision = enhanced_decision.original_decision.copy()
            
            # Apply memory-based adjustments
            adjustments = enhanced_decision.recommended_adjustments
            
            # Adjust confidence based on memory
            if 'confidence_adjustment' in adjustments:
                original_confidence = final_decision.get('confidence', 0.5)
                memory_adjustment = adjustments['confidence_adjustment']
                final_decision['confidence'] = max(0.1, min(0.9, original_confidence + memory_adjustment))
            
            # Adjust position size based on memory
            if 'position_size_multiplier' in adjustments:
                original_size = final_decision.get('position_size', 0.1)
                multiplier = adjustments['position_size_multiplier']
                final_decision['position_size'] = max(0.01, min(0.3, original_size * multiplier))
            
            # Adjust stop loss based on memory
            if 'stop_loss_adjustment' in adjustments:
                original_sl = final_decision.get('stop_loss', 0.02)
                sl_adjustment = adjustments['stop_loss_adjustment']
                final_decision['stop_loss'] = max(0.005, min(0.1, original_sl + sl_adjustment))
            
            # Add memory metadata
            final_decision['memory_enhanced'] = True
            final_decision['memory_success_probability'] = enhanced_decision.success_probability
            final_decision['memory_confidence'] = enhanced_decision.memory_confidence
            final_decision['memory_insights_count'] = len(enhanced_decision.memory_insights)
            final_decision['similar_patterns_count'] = len(enhanced_decision.similar_patterns)
            final_decision['risk_warnings_count'] = len(enhanced_decision.risk_warnings)
            
            # Add decision trail for audit
            final_decision['decision_trail'] = {
                'memory_insights': enhanced_decision.memory_insights,
                'historical_context': enhanced_decision.historical_context,
                'risk_warnings': enhanced_decision.risk_warnings,
                'adjustments_applied': list(adjustments.keys()),
                'enhancement_timestamp': enhanced_decision.timestamp.isoformat()
            }
            
            logger.info(f"Memory-enhanced decision applied with {len(adjustments)} adjustments")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error applying memory-enhanced decision: {e}")
            return enhanced_decision.original_decision
    
    def _generate_memory_adjustments(self, decision_context: Dict[str, Any], 
                                   similar_patterns: List[MemoryMatch],
                                   success_probability: float,
                                   memory_confidence: float) -> Dict[str, Any]:
        """Generate recommended adjustments based on memory analysis"""
        try:
            adjustments = {}
            
            # Confidence adjustment based on success probability
            if success_probability > 0.7:
                adjustments['confidence_adjustment'] = +0.1
            elif success_probability < 0.3:
                adjustments['confidence_adjustment'] = -0.15
            
            # Position size adjustment based on memory confidence
            if memory_confidence > 0.8 and success_probability > 0.6:
                adjustments['position_size_multiplier'] = 1.2
            elif memory_confidence < 0.4 or success_probability < 0.4:
                adjustments['position_size_multiplier'] = 0.7
            
            # Stop loss adjustment based on historical volatility
            if similar_patterns:
                successful_trades = [p.memory for p in similar_patterns 
                                   if hasattr(p.memory, 'success') and p.memory.success]
                if successful_trades:
                    # Analyze historical stop loss effectiveness
                    avg_volatility = np.mean([getattr(trade, 'volatility', 0.02) for trade in successful_trades])
                    if avg_volatility > 0.03:
                        adjustments['stop_loss_adjustment'] = +0.005  # Wider stops in volatile conditions
            
            # Check for active pattern injections
            symbol = decision_context.get('symbol', '')
            market_condition = decision_context.get('market_condition', MarketCondition.SIDEWAYS)
            injection_key = f"{symbol}_{market_condition.value}"
            
            if injection_key in self.active_injections:
                injection = self.active_injections[injection_key]
                if injection.position_size_multiplier != 1.0:
                    adjustments['position_size_multiplier'] = injection.position_size_multiplier
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error generating memory adjustments: {e}")
            return {}
    
    def _create_historical_context(self, symbol: str, market_condition: MarketCondition,
                                 similar_patterns: List[MemoryMatch]) -> str:
        """Create human-readable historical context summary"""
        try:
            if not similar_patterns:
                return f"No historical context available for {symbol} in {market_condition.value} conditions"
            
            successful_count = len([p for p in similar_patterns if hasattr(p.memory, 'success') and p.memory.success])
            total_count = len(similar_patterns)
            
            context = f"Historical analysis for {symbol} in {market_condition.value.replace('_', ' ')} conditions:\n"
            context += f"• Found {total_count} similar historical situations\n"
            context += f"• {successful_count} were successful ({successful_count/total_count:.1%} success rate)\n"
            
            if successful_count > 0:
                successful_trades = [p.memory for p in similar_patterns 
                                   if hasattr(p.memory, 'success') and p.memory.success]
                
                if hasattr(successful_trades[0], 'profit_loss'):
                    avg_profit = np.mean([trade.profit_loss for trade in successful_trades])
                    context += f"• Average profit from successful trades: {avg_profit:.4f}\n"
                
                # Most recent successful trade
                recent_success = max(successful_trades, key=lambda x: x.entry_time if hasattr(x, 'entry_time') else datetime.min)
                if hasattr(recent_success, 'entry_time'):
                    days_ago = (datetime.now() - recent_success.entry_time).days
                    context += f"• Most recent success was {days_ago} days ago\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error creating historical context: {e}")
            return f"Error creating historical context: {str(e)}"
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get memory-decision integration statistics"""
        return {
            **self.integration_stats,
            'active_injections': len(self.active_injections),
            'integration_health': 'Excellent' if self.integration_stats['decisions_enhanced'] > 100 else 'Good'
        }
