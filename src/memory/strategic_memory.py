"""
Strategic Trading Memory Engine - Phase 4 Evolution Layer
Implements long-term memory for successful trades, decision logic, market conditions, and environmental context.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading
import time
from collections import defaultdict, deque
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    TRADE_SUCCESS = "trade_success"
    TRADE_FAILURE = "trade_failure"
    MARKET_REGIME = "market_regime"
    ENVIRONMENTAL_CONTEXT = "environmental_context"
    DECISION_PATTERN = "decision_pattern"
    BEHAVIORAL_PREFERENCE = "behavioral_preference"

class MarketCondition(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class TradeMemory:
    """Memory of a completed trade with full context"""
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    profit_loss: float
    entry_time: datetime
    exit_time: datetime
    decision_logic: str
    market_condition: MarketCondition
    environmental_context: Dict[str, Any]
    confidence_score: float
    success: bool
    lessons_learned: str

@dataclass
class MarketRegimeMemory:
    """Memory of market regime performance"""
    regime_id: str
    regime_type: str
    start_time: datetime
    end_time: Optional[datetime]
    market_conditions: List[MarketCondition]
    successful_strategies: List[str]
    failed_strategies: List[str]
    average_performance: float
    volatility_profile: Dict[str, float]
    optimal_parameters: Dict[str, Any]

@dataclass
class DecisionPatternMemory:
    """Memory of decision patterns and their outcomes"""
    pattern_id: str
    decision_type: str
    input_signals: Dict[str, Any]
    decision_logic: str
    outcome_quality: float
    frequency_used: int
    success_rate: float
    average_profit: float
    market_conditions: List[MarketCondition]
    timestamp: datetime

@dataclass
class EnvironmentalContextMemory:
    """Memory of environmental conditions and their impact"""
    context_id: str
    macro_indicators: Dict[str, float]
    on_chain_metrics: Dict[str, float]
    sentiment_scores: Dict[str, float]
    news_events: List[str]
    market_impact: float
    trading_adjustments: Dict[str, Any]
    timestamp: datetime

class StrategicMemoryEngine:
    """
    Strategic Trading Memory Engine - The brain's long-term memory system.
    Stores, organizes, and retrieves trading experiences for intelligent decision making.
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        """Initialize the Strategic Memory Engine"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Memory storage
        self.trade_memories: Dict[str, TradeMemory] = {}
        self.regime_memories: Dict[str, MarketRegimeMemory] = {}
        self.decision_patterns: Dict[str, DecisionPatternMemory] = {}
        self.environmental_contexts: Dict[str, EnvironmentalContextMemory] = {}
        
        # Memory indices for fast retrieval
        self.symbol_index: Dict[str, List[str]] = defaultdict(list)
        self.condition_index: Dict[MarketCondition, List[str]] = defaultdict(list)
        self.success_index: Dict[bool, List[str]] = defaultdict(list)
        self.time_index: Dict[str, List[str]] = defaultdict(list)  # YYYY-MM format
        
        # Memory statistics
        self.memory_stats = {
            'total_memories': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'regime_changes': 0,
            'pattern_discoveries': 0,
            'last_consolidation': None
        }
        
        # Load existing memories
        self._load_memories()
        
        # Start memory maintenance thread
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._memory_maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
        logger.info("Strategic Memory Engine initialized with comprehensive memory systems")
    
    def store_trade_memory(self, trade_memory: TradeMemory) -> str:
        """Store a completed trade memory with full context"""
        try:
            memory_id = f"trade_{trade_memory.symbol}_{trade_memory.entry_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Store in main memory
            self.trade_memories[memory_id] = trade_memory
            
            # Update indices
            self.symbol_index[trade_memory.symbol].append(memory_id)
            self.condition_index[trade_memory.market_condition].append(memory_id)
            self.success_index[trade_memory.success].append(memory_id)
            self.time_index[trade_memory.entry_time.strftime('%Y-%m')].append(memory_id)
            
            # Update statistics
            self.memory_stats['total_memories'] += 1
            if trade_memory.success:
                self.memory_stats['successful_trades'] += 1
            else:
                self.memory_stats['failed_trades'] += 1
            
            # Persist to disk
            self._save_memory(memory_id, trade_memory, MemoryType.TRADE_SUCCESS if trade_memory.success else MemoryType.TRADE_FAILURE)
            
            logger.info(f"Stored trade memory: {memory_id} (Success: {trade_memory.success})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing trade memory: {e}")
            return ""
    
    def store_regime_memory(self, regime_memory: MarketRegimeMemory) -> str:
        """Store market regime performance memory"""
        try:
            memory_id = f"regime_{regime_memory.regime_type}_{regime_memory.start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Store in main memory
            self.regime_memories[memory_id] = regime_memory
            
            # Update statistics
            self.memory_stats['regime_changes'] += 1
            
            # Persist to disk
            self._save_memory(memory_id, regime_memory, MemoryType.MARKET_REGIME)
            
            logger.info(f"Stored regime memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing regime memory: {e}")
            return ""
    
    def store_decision_pattern(self, pattern_memory: DecisionPatternMemory) -> str:
        """Store decision pattern memory"""
        try:
            memory_id = f"pattern_{pattern_memory.decision_type}_{pattern_memory.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Store in main memory
            self.decision_patterns[memory_id] = pattern_memory
            
            # Update statistics
            self.memory_stats['pattern_discoveries'] += 1
            
            # Persist to disk
            self._save_memory(memory_id, pattern_memory, MemoryType.DECISION_PATTERN)
            
            logger.info(f"Stored decision pattern: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing decision pattern: {e}")
            return ""
    
    def store_environmental_context(self, context_memory: EnvironmentalContextMemory) -> str:
        """Store environmental context memory"""
        try:
            memory_id = f"context_{context_memory.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Store in main memory
            self.environmental_contexts[memory_id] = context_memory
            
            # Persist to disk
            self._save_memory(memory_id, context_memory, MemoryType.ENVIRONMENTAL_CONTEXT)
            
            logger.info(f"Stored environmental context: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing environmental context: {e}")
            return ""
    
    def recall_by_symbol(self, symbol: str, limit: int = 10) -> List[TradeMemory]:
        """Recall trade memories for a specific symbol"""
        try:
            memory_ids = self.symbol_index.get(symbol, [])
            memories = []
            
            for memory_id in memory_ids[-limit:]:  # Get most recent
                if memory_id in self.trade_memories:
                    memories.append(self.trade_memories[memory_id])
            
            # Sort by entry time (most recent first)
            memories.sort(key=lambda x: x.entry_time, reverse=True)
            
            logger.info(f"Recalled {len(memories)} memories for symbol: {symbol}")
            return memories
            
        except Exception as e:
            logger.error(f"Error recalling memories by symbol: {e}")
            return []
    
    def recall_by_condition(self, condition: MarketCondition, limit: int = 10) -> List[TradeMemory]:
        """Recall trade memories for a specific market condition"""
        try:
            memory_ids = self.condition_index.get(condition, [])
            memories = []
            
            for memory_id in memory_ids[-limit:]:  # Get most recent
                if memory_id in self.trade_memories:
                    memories.append(self.trade_memories[memory_id])
            
            # Sort by entry time (most recent first)
            memories.sort(key=lambda x: x.entry_time, reverse=True)
            
            logger.info(f"Recalled {len(memories)} memories for condition: {condition.value}")
            return memories
            
        except Exception as e:
            logger.error(f"Error recalling memories by condition: {e}")
            return []
    
    def recall_successful_patterns(self, min_success_rate: float = 0.7, limit: int = 10) -> List[DecisionPatternMemory]:
        """Recall successful decision patterns"""
        try:
            successful_patterns = []
            
            for pattern in self.decision_patterns.values():
                if pattern.success_rate >= min_success_rate and pattern.frequency_used >= 3:
                    successful_patterns.append(pattern)
            
            # Sort by success rate and average profit
            successful_patterns.sort(key=lambda x: (x.success_rate, x.average_profit), reverse=True)
            
            logger.info(f"Recalled {len(successful_patterns[:limit])} successful patterns")
            return successful_patterns[:limit]
            
        except Exception as e:
            logger.error(f"Error recalling successful patterns: {e}")
            return []
    
    def recall_similar_context(self, current_context: Dict[str, Any], similarity_threshold: float = 0.8) -> List[EnvironmentalContextMemory]:
        """Recall environmental contexts similar to current conditions"""
        try:
            similar_contexts = []
            
            for context in self.environmental_contexts.values():
                similarity = self._calculate_context_similarity(current_context, context)
                if similarity >= similarity_threshold:
                    similar_contexts.append((context, similarity))
            
            # Sort by similarity
            similar_contexts.sort(key=lambda x: x[1], reverse=True)
            
            result = [ctx for ctx, _ in similar_contexts[:10]]
            logger.info(f"Recalled {len(result)} similar contexts")
            return result
            
        except Exception as e:
            logger.error(f"Error recalling similar contexts: {e}")
            return []
    
    def get_memory_insights(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive memory insights and statistics"""
        try:
            insights = {
                'total_memories': len(self.trade_memories),
                'success_rate': 0.0,
                'average_profit': 0.0,
                'top_performing_conditions': [],
                'most_common_failures': [],
                'regime_performance': {},
                'pattern_effectiveness': {},
                'memory_health': 'Good'
            }
            
            # Filter by symbol if specified
            memories = list(self.trade_memories.values())
            if symbol:
                memories = [m for m in memories if m.symbol == symbol]
            
            if not memories:
                return insights
            
            # Calculate success rate
            successful = [m for m in memories if m.success]
            insights['success_rate'] = len(successful) / len(memories)
            
            # Calculate average profit
            profits = [m.profit_loss for m in memories]
            insights['average_profit'] = np.mean(profits) if profits else 0.0
            
            # Analyze conditions
            condition_performance = defaultdict(list)
            for memory in memories:
                condition_performance[memory.market_condition].append(memory.success)
            
            condition_stats = []
            for condition, successes in condition_performance.items():
                success_rate = np.mean(successes)
                condition_stats.append((condition.value, success_rate, len(successes)))
            
            condition_stats.sort(key=lambda x: x[1], reverse=True)
            insights['top_performing_conditions'] = condition_stats[:5]
            
            # Analyze patterns
            pattern_stats = []
            for pattern in self.decision_patterns.values():
                pattern_stats.append({
                    'type': pattern.decision_type,
                    'success_rate': pattern.success_rate,
                    'frequency': pattern.frequency_used,
                    'avg_profit': pattern.average_profit
                })
            
            pattern_stats.sort(key=lambda x: x['success_rate'], reverse=True)
            insights['pattern_effectiveness'] = pattern_stats[:5]
            
            logger.info(f"Generated memory insights for {len(memories)} memories")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating memory insights: {e}")
            return {}
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories into higher-level patterns and insights"""
        try:
            consolidation_results = {
                'patterns_discovered': 0,
                'insights_generated': 0,
                'memories_pruned': 0,
                'knowledge_distilled': [],
                'consolidation_time': datetime.now()
            }
            
            # Discover new patterns from trade memories
            new_patterns = self._discover_trading_patterns()
            consolidation_results['patterns_discovered'] = len(new_patterns)
            
            # Generate strategic insights
            insights = self._generate_strategic_insights()
            consolidation_results['insights_generated'] = len(insights)
            consolidation_results['knowledge_distilled'] = insights
            
            # Prune outdated or contradictory memories
            pruned_count = self._prune_memories()
            consolidation_results['memories_pruned'] = pruned_count
            
            # Update consolidation timestamp
            self.memory_stats['last_consolidation'] = datetime.now()
            
            logger.info(f"Memory consolidation complete: {consolidation_results}")
            return consolidation_results
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            return {}
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: EnvironmentalContextMemory) -> float:
        """Calculate similarity between two environmental contexts"""
        try:
            # Simple similarity based on macro indicators
            similarity_scores = []
            
            for key in ['vix', 'dxy', 'us10y', 'gold', 'oil']:
                if key in context1 and key in context2.macro_indicators:
                    val1 = context1[key]
                    val2 = context2.macro_indicators[key]
                    # Normalize difference to 0-1 similarity
                    diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                    similarity_scores.append(1 - min(diff, 1))
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    def _discover_trading_patterns(self) -> List[DecisionPatternMemory]:
        """Discover new trading patterns from existing memories"""
        try:
            new_patterns = []
            
            # Group trades by similar characteristics
            trade_groups = defaultdict(list)
            for memory in self.trade_memories.values():
                # Group by market condition and decision logic similarity
                group_key = f"{memory.market_condition.value}_{memory.confidence_score:.1f}"
                trade_groups[group_key].append(memory)
            
            # Analyze groups for patterns
            for group_key, trades in trade_groups.items():
                if len(trades) >= 5:  # Minimum trades for pattern
                    success_rate = np.mean([t.success for t in trades])
                    avg_profit = np.mean([t.profit_loss for t in trades])
                    
                    if success_rate > 0.6:  # Potentially successful pattern
                        pattern = DecisionPatternMemory(
                            pattern_id=f"discovered_{group_key}_{datetime.now().strftime('%Y%m%d')}",
                            decision_type=f"pattern_{group_key}",
                            input_signals={'group_key': group_key},
                            decision_logic=f"Pattern discovered from {len(trades)} similar trades",
                            outcome_quality=success_rate,
                            frequency_used=len(trades),
                            success_rate=success_rate,
                            average_profit=avg_profit,
                            market_conditions=[trades[0].market_condition],
                            timestamp=datetime.now()
                        )
                        new_patterns.append(pattern)
                        self.store_decision_pattern(pattern)
            
            return new_patterns
            
        except Exception as e:
            logger.error(f"Error discovering trading patterns: {e}")
            return []
    
    def _generate_strategic_insights(self) -> List[str]:
        """Generate strategic insights from consolidated memories"""
        try:
            insights = []
            
            # Analyze success patterns
            if self.memory_stats['successful_trades'] > 10:
                success_rate = self.memory_stats['successful_trades'] / self.memory_stats['total_memories']
                insights.append(f"Overall success rate: {success_rate:.2%}")
                
                if success_rate > 0.7:
                    insights.append("High success rate indicates strong strategy performance")
                elif success_rate < 0.4:
                    insights.append("Low success rate suggests strategy refinement needed")
            
            # Analyze regime performance
            regime_performance = defaultdict(list)
            for regime in self.regime_memories.values():
                regime_performance[regime.regime_type].append(regime.average_performance)
            
            for regime_type, performances in regime_performance.items():
                avg_performance = np.mean(performances)
                insights.append(f"{regime_type} regime average performance: {avg_performance:.2f}")
            
            # Analyze pattern effectiveness
            effective_patterns = [p for p in self.decision_patterns.values() if p.success_rate > 0.8]
            if effective_patterns:
                insights.append(f"Discovered {len(effective_patterns)} highly effective patterns")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating strategic insights: {e}")
            return []
    
    def _prune_memories(self) -> int:
        """Prune outdated or contradictory memories"""
        try:
            pruned_count = 0
            cutoff_date = datetime.now() - timedelta(days=365)  # Keep 1 year of memories
            
            # Prune old trade memories (keep successful ones longer)
            to_remove = []
            for memory_id, memory in self.trade_memories.items():
                if memory.entry_time < cutoff_date and not memory.success:
                    to_remove.append(memory_id)
            
            for memory_id in to_remove:
                del self.trade_memories[memory_id]
                pruned_count += 1
            
            # Update indices
            self._rebuild_indices()
            
            return pruned_count
            
        except Exception as e:
            logger.error(f"Error pruning memories: {e}")
            return 0
    
    def _rebuild_indices(self):
        """Rebuild memory indices after pruning"""
        try:
            # Clear existing indices
            self.symbol_index.clear()
            self.condition_index.clear()
            self.success_index.clear()
            self.time_index.clear()
            
            # Rebuild from current memories
            for memory_id, memory in self.trade_memories.items():
                self.symbol_index[memory.symbol].append(memory_id)
                self.condition_index[memory.market_condition].append(memory_id)
                self.success_index[memory.success].append(memory_id)
                self.time_index[memory.entry_time.strftime('%Y-%m')].append(memory_id)
            
        except Exception as e:
            logger.error(f"Error rebuilding indices: {e}")
    
    def _save_memory(self, memory_id: str, memory_obj: Any, memory_type: MemoryType):
        """Save memory to disk"""
        try:
            file_path = os.path.join(self.data_dir, f"{memory_type.value}_{memory_id}.json")
            with open(file_path, 'w') as f:
                json.dump(asdict(memory_obj), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")
    
    def _load_memories(self):
        """Load existing memories from disk"""
        try:
            if not os.path.exists(self.data_dir):
                return
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.data_dir, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Determine memory type and load accordingly
                    if filename.startswith('trade_'):
                        # Convert back to TradeMemory object
                        memory = TradeMemory(**data)
                        memory_id = filename.replace('.json', '').replace('trade_success_', '').replace('trade_failure_', '')
                        self.trade_memories[memory_id] = memory
                    
                    # Add other memory type loading as needed
            
            logger.info(f"Loaded {len(self.trade_memories)} memories from disk")
            
        except Exception as e:
            logger.error(f"Error loading memories from disk: {e}")
    
    def _memory_maintenance_loop(self):
        """Background thread for memory maintenance"""
        while self.running:
            try:
                time.sleep(3600)  # Run every hour
                
                # Perform periodic consolidation
                if datetime.now().hour == 2:  # Run at 2 AM
                    self.consolidate_memories()
                
            except Exception as e:
                logger.error(f"Error in memory maintenance loop: {e}")
    
    def stop(self):
        """Stop the memory engine"""
        self.running = False
        logger.info("Strategic Memory Engine stopped")
