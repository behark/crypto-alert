"""
Memory Retrieval System - Phase 4 Evolution Layer
Advanced contextual recall, pattern matching, and memory-driven decision enhancement.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import threading
import time

from .strategic_memory import (
    StrategicMemoryEngine, TradeMemory, MarketRegimeMemory, 
    DecisionPatternMemory, EnvironmentalContextMemory, MarketCondition
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryQuery:
    """Query structure for memory retrieval"""
    symbol: Optional[str] = None
    market_condition: Optional[MarketCondition] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    success_filter: Optional[bool] = None
    confidence_range: Optional[Tuple[float, float]] = None
    profit_range: Optional[Tuple[float, float]] = None
    environmental_context: Optional[Dict[str, Any]] = None
    limit: int = 10

@dataclass
class MemoryMatch:
    """Memory match with relevance scoring"""
    memory: Any
    relevance_score: float
    match_reasons: List[str]
    confidence: float

@dataclass
class PatternAnalysis:
    """Analysis of memory patterns"""
    pattern_type: str
    frequency: int
    success_rate: float
    average_profit: float
    conditions: List[MarketCondition]
    key_factors: List[str]
    recommendation: str

class MemoryRetrievalSystem:
    """
    Advanced Memory Retrieval System - Contextual recall and pattern matching.
    Provides intelligent memory retrieval for decision enhancement.
    """
    
    def __init__(self, memory_engine: StrategicMemoryEngine):
        """Initialize the Memory Retrieval System"""
        self.memory_engine = memory_engine
        
        # Retrieval statistics
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'pattern_matches': 0,
            'context_matches': 0,
            'last_query_time': None
        }
        
        # Caching for performance
        self.query_cache: Dict[str, List[MemoryMatch]] = {}
        self.cache_ttl = timedelta(minutes=30)
        self.last_cache_clear = datetime.now()
        
        logger.info("Memory Retrieval System initialized with advanced pattern matching")
    
    def contextual_recall(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Perform contextual memory recall based on current market conditions"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                cached_time = self.query_cache[cache_key][0].memory.entry_time if self.query_cache[cache_key] else datetime.now()
                if datetime.now() - cached_time < self.cache_ttl:
                    logger.info(f"Retrieved {len(self.query_cache[cache_key])} memories from cache")
                    return self.query_cache[cache_key]
            
            matches = []
            
            # Retrieve relevant trade memories
            trade_matches = self._retrieve_trade_memories(query)
            matches.extend(trade_matches)
            
            # Retrieve relevant decision patterns
            pattern_matches = self._retrieve_decision_patterns(query)
            matches.extend(pattern_matches)
            
            # Retrieve environmental context matches
            if query.environmental_context:
                context_matches = self._retrieve_environmental_contexts(query)
                matches.extend(context_matches)
            
            # Sort by relevance score
            matches.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            matches = matches[:query.limit]
            
            # Cache results
            self.query_cache[cache_key] = matches
            
            # Update statistics
            self.retrieval_stats['total_queries'] += 1
            if matches:
                self.retrieval_stats['successful_retrievals'] += 1
            self.retrieval_stats['last_query_time'] = datetime.now()
            
            logger.info(f"Contextual recall found {len(matches)} relevant memories")
            return matches
            
        except Exception as e:
            logger.error(f"Error in contextual recall: {e}")
            return []
    
    def find_similar_patterns(self, current_signals: Dict[str, Any], min_similarity: float = 0.7) -> List[MemoryMatch]:
        """Find historical patterns similar to current market signals"""
        try:
            similar_patterns = []
            
            # Compare with stored decision patterns
            for pattern in self.memory_engine.decision_patterns.values():
                similarity = self._calculate_signal_similarity(current_signals, pattern.input_signals)
                
                if similarity >= min_similarity:
                    match_reasons = [
                        f"Signal similarity: {similarity:.2f}",
                        f"Success rate: {pattern.success_rate:.2f}",
                        f"Used {pattern.frequency_used} times"
                    ]
                    
                    match = MemoryMatch(
                        memory=pattern,
                        relevance_score=similarity * pattern.success_rate,
                        match_reasons=match_reasons,
                        confidence=similarity
                    )
                    similar_patterns.append(match)
            
            # Sort by relevance
            similar_patterns.sort(key=lambda x: x.relevance_score, reverse=True)
            
            self.retrieval_stats['pattern_matches'] += len(similar_patterns)
            
            logger.info(f"Found {len(similar_patterns)} similar patterns")
            return similar_patterns[:10]
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def extract_success_patterns(self, symbol: Optional[str] = None, lookback_days: int = 90) -> List[PatternAnalysis]:
        """Extract successful trading patterns from memory"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get relevant memories
            memories = []
            for memory in self.memory_engine.trade_memories.values():
                if memory.entry_time >= cutoff_date and memory.success:
                    if symbol is None or memory.symbol == symbol:
                        memories.append(memory)
            
            if not memories:
                return []
            
            # Group by patterns
            pattern_groups = defaultdict(list)
            
            for memory in memories:
                # Create pattern key based on market condition and confidence
                pattern_key = f"{memory.market_condition.value}_{int(memory.confidence_score * 10)}"
                pattern_groups[pattern_key].append(memory)
            
            # Analyze patterns
            pattern_analyses = []
            
            for pattern_key, group_memories in pattern_groups.items():
                if len(group_memories) >= 3:  # Minimum occurrences
                    analysis = self._analyze_pattern_group(pattern_key, group_memories)
                    if analysis.success_rate > 0.6:  # Only successful patterns
                        pattern_analyses.append(analysis)
            
            # Sort by success rate and frequency
            pattern_analyses.sort(key=lambda x: (x.success_rate, x.frequency), reverse=True)
            
            logger.info(f"Extracted {len(pattern_analyses)} success patterns")
            return pattern_analyses[:10]
            
        except Exception as e:
            logger.error(f"Error extracting success patterns: {e}")
            return []
    
    def get_failure_avoidance_insights(self, symbol: Optional[str] = None, lookback_days: int = 90) -> List[Dict[str, Any]]:
        """Get insights on patterns to avoid based on historical failures"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get failed trades
            failed_memories = []
            for memory in self.memory_engine.trade_memories.values():
                if memory.entry_time >= cutoff_date and not memory.success:
                    if symbol is None or memory.symbol == symbol:
                        failed_memories.append(memory)
            
            if not failed_memories:
                return []
            
            # Analyze failure patterns
            failure_patterns = defaultdict(list)
            
            for memory in failed_memories:
                # Group by various characteristics
                patterns = [
                    f"condition_{memory.market_condition.value}",
                    f"confidence_{int(memory.confidence_score * 10)}",
                    f"hour_{memory.entry_time.hour}"
                ]
                
                for pattern in patterns:
                    failure_patterns[pattern].append(memory)
            
            # Generate insights
            insights = []
            
            for pattern, failures in failure_patterns.items():
                if len(failures) >= 3:  # Significant failure count
                    avg_loss = np.mean([abs(m.profit_loss) for m in failures])
                    failure_rate = len(failures) / len(failed_memories)
                    
                    insight = {
                        'pattern': pattern,
                        'failure_count': len(failures),
                        'failure_rate': failure_rate,
                        'average_loss': avg_loss,
                        'recommendation': f"Avoid trading when {pattern.replace('_', ' ')}"
                    }
                    insights.append(insight)
            
            # Sort by failure impact
            insights.sort(key=lambda x: x['failure_count'] * x['average_loss'], reverse=True)
            
            logger.info(f"Generated {len(insights)} failure avoidance insights")
            return insights[:10]
            
        except Exception as e:
            logger.error(f"Error generating failure avoidance insights: {e}")
            return []
    
    def memory_guided_decision_enhancement(self, current_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance current trading decision using memory insights"""
        try:
            enhanced_decision = current_decision.copy()
            
            # Find similar historical decisions
            query = MemoryQuery(
                symbol=current_decision.get('symbol'),
                market_condition=current_decision.get('market_condition'),
                confidence_range=(current_decision.get('confidence', 0.5) - 0.1, 
                                current_decision.get('confidence', 0.5) + 0.1),
                limit=5
            )
            
            similar_memories = self.contextual_recall(query)
            
            if similar_memories:
                # Calculate memory-based adjustments
                successful_memories = [m for m in similar_memories if hasattr(m.memory, 'success') and m.memory.success]
                
                if successful_memories:
                    # Adjust position size based on historical success
                    avg_success_rate = np.mean([m.relevance_score for m in successful_memories])
                    size_multiplier = min(1.2, max(0.8, avg_success_rate * 1.1))
                    
                    if 'position_size' in enhanced_decision:
                        enhanced_decision['position_size'] *= size_multiplier
                    
                    # Adjust confidence based on memory
                    memory_confidence = np.mean([m.confidence for m in successful_memories])
                    enhanced_decision['memory_confidence'] = memory_confidence
                    
                    # Add memory insights
                    enhanced_decision['memory_insights'] = [
                        f"Found {len(successful_memories)} similar successful patterns",
                        f"Memory-based size adjustment: {size_multiplier:.2f}x",
                        f"Historical success rate: {avg_success_rate:.2f}"
                    ]
                    
                    # Add specific recommendations
                    if avg_success_rate > 0.8:
                        enhanced_decision['memory_recommendation'] = "Strong historical performance - consider increased position"
                    elif avg_success_rate < 0.4:
                        enhanced_decision['memory_recommendation'] = "Weak historical performance - consider reduced position"
                    else:
                        enhanced_decision['memory_recommendation'] = "Moderate historical performance - standard position"
            
            logger.info("Decision enhanced with memory insights")
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Error enhancing decision with memory: {e}")
            return current_decision
    
    def _retrieve_trade_memories(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Retrieve trade memories matching query criteria"""
        try:
            matches = []
            
            for memory in self.memory_engine.trade_memories.values():
                relevance_score = 0.0
                match_reasons = []
                
                # Symbol match
                if query.symbol and memory.symbol == query.symbol:
                    relevance_score += 0.3
                    match_reasons.append(f"Symbol match: {query.symbol}")
                
                # Market condition match
                if query.market_condition and memory.market_condition == query.market_condition:
                    relevance_score += 0.3
                    match_reasons.append(f"Market condition match: {query.market_condition.value}")
                
                # Success filter
                if query.success_filter is not None and memory.success == query.success_filter:
                    relevance_score += 0.2
                    match_reasons.append(f"Success filter match: {query.success_filter}")
                
                # Confidence range
                if query.confidence_range:
                    min_conf, max_conf = query.confidence_range
                    if min_conf <= memory.confidence_score <= max_conf:
                        relevance_score += 0.1
                        match_reasons.append(f"Confidence in range: {memory.confidence_score:.2f}")
                
                # Time relevance (more recent = higher score)
                if query.time_range:
                    start_time, end_time = query.time_range
                    if start_time <= memory.entry_time <= end_time:
                        relevance_score += 0.1
                        match_reasons.append("Time range match")
                
                # Only include if minimum relevance
                if relevance_score > 0.2:
                    match = MemoryMatch(
                        memory=memory,
                        relevance_score=relevance_score,
                        match_reasons=match_reasons,
                        confidence=relevance_score
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error retrieving trade memories: {e}")
            return []
    
    def _retrieve_decision_patterns(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Retrieve decision patterns matching query criteria"""
        try:
            matches = []
            
            for pattern in self.memory_engine.decision_patterns.values():
                relevance_score = 0.0
                match_reasons = []
                
                # Market condition match
                if query.market_condition and query.market_condition in pattern.market_conditions:
                    relevance_score += 0.4
                    match_reasons.append(f"Market condition in pattern: {query.market_condition.value}")
                
                # Success rate relevance
                if pattern.success_rate > 0.7:
                    relevance_score += 0.3
                    match_reasons.append(f"High success rate: {pattern.success_rate:.2f}")
                
                # Frequency relevance
                if pattern.frequency_used > 5:
                    relevance_score += 0.2
                    match_reasons.append(f"Frequently used: {pattern.frequency_used} times")
                
                # Recent usage
                time_diff = datetime.now() - pattern.timestamp
                if time_diff.days < 30:
                    relevance_score += 0.1
                    match_reasons.append("Recently used pattern")
                
                if relevance_score > 0.3:
                    match = MemoryMatch(
                        memory=pattern,
                        relevance_score=relevance_score,
                        match_reasons=match_reasons,
                        confidence=pattern.success_rate
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error retrieving decision patterns: {e}")
            return []
    
    def _retrieve_environmental_contexts(self, query: MemoryQuery) -> List[MemoryMatch]:
        """Retrieve environmental contexts similar to query context"""
        try:
            matches = []
            
            if not query.environmental_context:
                return matches
            
            similar_contexts = self.memory_engine.recall_similar_context(
                query.environmental_context, 
                similarity_threshold=0.6
            )
            
            for context in similar_contexts:
                relevance_score = 0.5  # Base score for environmental match
                match_reasons = ["Environmental context similarity"]
                
                # Higher score for positive market impact
                if context.market_impact > 0:
                    relevance_score += 0.2
                    match_reasons.append(f"Positive market impact: {context.market_impact:.2f}")
                
                match = MemoryMatch(
                    memory=context,
                    relevance_score=relevance_score,
                    match_reasons=match_reasons,
                    confidence=0.7
                )
                matches.append(match)
            
            self.retrieval_stats['context_matches'] += len(matches)
            return matches
            
        except Exception as e:
            logger.error(f"Error retrieving environmental contexts: {e}")
            return []
    
    def _calculate_signal_similarity(self, signals1: Dict[str, Any], signals2: Dict[str, Any]) -> float:
        """Calculate similarity between two signal sets"""
        try:
            if not signals1 or not signals2:
                return 0.0
            
            common_keys = set(signals1.keys()) & set(signals2.keys())
            if not common_keys:
                return 0.0
            
            similarities = []
            
            for key in common_keys:
                val1, val2 = signals1[key], signals2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    if val1 == 0 and val2 == 0:
                        similarities.append(1.0)
                    else:
                        diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                        similarities.append(1 - min(diff, 1))
                elif val1 == val2:
                    # Exact match
                    similarities.append(1.0)
                else:
                    # No match
                    similarities.append(0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating signal similarity: {e}")
            return 0.0
    
    def _analyze_pattern_group(self, pattern_key: str, memories: List[TradeMemory]) -> PatternAnalysis:
        """Analyze a group of memories to extract pattern insights"""
        try:
            success_rate = np.mean([m.success for m in memories])
            average_profit = np.mean([m.profit_loss for m in memories])
            conditions = list(set([m.market_condition for m in memories]))
            
            # Extract key factors
            key_factors = []
            
            # Confidence analysis
            avg_confidence = np.mean([m.confidence_score for m in memories])
            key_factors.append(f"Average confidence: {avg_confidence:.2f}")
            
            # Time analysis
            hours = [m.entry_time.hour for m in memories]
            most_common_hour = Counter(hours).most_common(1)[0][0]
            key_factors.append(f"Most common hour: {most_common_hour}:00")
            
            # Generate recommendation
            if success_rate > 0.8:
                recommendation = f"Strong pattern - consider increased allocation"
            elif success_rate > 0.6:
                recommendation = f"Moderate pattern - standard allocation"
            else:
                recommendation = f"Weak pattern - consider reduced allocation"
            
            return PatternAnalysis(
                pattern_type=pattern_key,
                frequency=len(memories),
                success_rate=success_rate,
                average_profit=average_profit,
                conditions=conditions,
                key_factors=key_factors,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pattern group: {e}")
            return PatternAnalysis(
                pattern_type=pattern_key,
                frequency=0,
                success_rate=0.0,
                average_profit=0.0,
                conditions=[],
                key_factors=[],
                recommendation="Analysis failed"
            )
    
    def _generate_cache_key(self, query: MemoryQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            str(query.symbol) if query.symbol else "None",
            str(query.market_condition.value) if query.market_condition else "None",
            str(query.success_filter) if query.success_filter is not None else "None",
            str(query.limit)
        ]
        return "_".join(key_parts)
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        self.last_cache_clear = datetime.now()
        logger.info("Memory retrieval cache cleared")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            **self.retrieval_stats,
            'cache_size': len(self.query_cache),
            'cache_hit_rate': self.retrieval_stats['successful_retrievals'] / max(self.retrieval_stats['total_queries'], 1)
        }
