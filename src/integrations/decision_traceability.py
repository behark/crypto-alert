"""
Final Traceability Layer - Phase 4 Final Integration
Complete audit trails for every decision with strategy ID, memory insights, confidence, context, and outcome.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    RISK_ADJUSTMENT = "risk_adjustment"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"

class DecisionOutcome(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

@dataclass
class MemoryInfluence:
    """Memory influence on decision"""
    memory_id: str
    memory_type: str
    influence_weight: float
    pattern_match_score: float
    historical_success_rate: float
    key_insight: str
    confidence_adjustment: float

@dataclass
class StrategyContext:
    """Strategy context for decision"""
    strategy_id: str
    strategy_type: str
    generation: int
    fitness_score: float
    mutation_history: List[str]
    performance_metrics: Dict[str, float]
    health_status: str

@dataclass
class EnvironmentalContext:
    """Environmental context for decision"""
    market_regime: str
    volatility_level: float
    risk_factors: List[str]
    sentiment_score: float
    macro_conditions: Dict[str, Any]
    correlation_matrix: Dict[str, float]

@dataclass
class ConfidenceFactors:
    """Confidence factors breakdown"""
    base_confidence: float
    memory_adjustment: float
    strategy_confidence: float
    environmental_adjustment: float
    risk_adjustment: float
    final_confidence: float
    confidence_sources: List[str]

@dataclass
class DecisionTrace:
    """Complete decision trace record"""
    trace_id: str
    timestamp: datetime
    symbol: str
    decision_type: DecisionType
    decision_value: Any
    
    # Core traceability components
    strategy_context: StrategyContext
    memory_influences: List[MemoryInfluence]
    confidence_factors: ConfidenceFactors
    environmental_context: EnvironmentalContext
    
    # Decision process
    decision_logic: str
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    expected_outcome: Dict[str, float]
    
    # Execution tracking
    execution_time: Optional[datetime]
    actual_outcome: Optional[DecisionOutcome]
    performance_metrics: Optional[Dict[str, float]]
    lessons_learned: Optional[str]
    
    # Metadata
    user_override: bool
    emergency_decision: bool
    cross_bot_influence: bool
    audit_flags: List[str]

class DecisionTraceabilityEngine:
    """
    Complete Decision Traceability Engine.
    Tracks every decision with full context, influences, and outcomes.
    """
    
    def __init__(self, data_dir: str = "data/traceability"):
        """Initialize Decision Traceability Engine"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Trace storage
        self.decision_traces: Dict[str, DecisionTrace] = {}
        self.trace_index = {
            'by_symbol': {},
            'by_strategy': {},
            'by_date': {},
            'by_outcome': {}
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'pending_decisions': 0,
            'average_confidence': 0.0,
            'memory_influence_rate': 0.0,
            'strategy_success_rate': {},
            'last_trace': None
        }
        
        # Background processing
        self.running = True
        self.trace_processor_thread = threading.Thread(target=self._trace_processing_loop, daemon=True)
        self.trace_processor_thread.start()
        
        logger.info("Decision Traceability Engine initialized - full audit trail active")
    
    def create_decision_trace(self, symbol: str, decision_type: DecisionType, 
                            decision_value: Any, strategy_context: StrategyContext,
                            memory_influences: List[MemoryInfluence] = None,
                            environmental_context: EnvironmentalContext = None) -> str:
        """Create new decision trace"""
        try:
            trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}_{decision_type.value}"
            
            # Calculate confidence factors
            confidence_factors = self._calculate_confidence_factors(
                strategy_context, memory_influences or [], environmental_context
            )
            
            # Generate decision logic explanation
            decision_logic = self._generate_decision_logic(
                decision_type, decision_value, strategy_context, 
                memory_influences or [], confidence_factors
            )
            
            # Assess risk
            risk_assessment = self._assess_decision_risk(
                symbol, decision_type, decision_value, strategy_context, environmental_context
            )
            
            # Generate expected outcome
            expected_outcome = self._generate_expected_outcome(
                decision_type, decision_value, confidence_factors, risk_assessment
            )
            
            # Create decision trace
            trace = DecisionTrace(
                trace_id=trace_id,
                timestamp=datetime.now(),
                symbol=symbol,
                decision_type=decision_type,
                decision_value=decision_value,
                strategy_context=strategy_context,
                memory_influences=memory_influences or [],
                confidence_factors=confidence_factors,
                environmental_context=environmental_context or self._get_default_environmental_context(),
                decision_logic=decision_logic,
                alternative_options=self._generate_alternative_options(decision_type, decision_value),
                risk_assessment=risk_assessment,
                expected_outcome=expected_outcome,
                execution_time=None,
                actual_outcome=None,
                performance_metrics=None,
                lessons_learned=None,
                user_override=False,
                emergency_decision=False,
                cross_bot_influence=len(memory_influences or []) > 0,
                audit_flags=[]
            )
            
            # Store trace
            self.decision_traces[trace_id] = trace
            self._update_trace_index(trace)
            
            # Update statistics
            self.performance_stats['total_decisions'] += 1
            self.performance_stats['pending_decisions'] += 1
            self.performance_stats['last_trace'] = datetime.now()
            
            # Save to disk
            self._save_trace_to_disk(trace)
            
            logger.info(f"Decision trace created: {trace_id} for {symbol} {decision_type.value}")
            return trace_id
            
        except Exception as e:
            logger.error(f"Error creating decision trace: {e}")
            return ""
    
    def update_trace_outcome(self, trace_id: str, outcome: DecisionOutcome,
                           performance_metrics: Dict[str, float] = None,
                           lessons_learned: str = None) -> bool:
        """Update decision trace with outcome"""
        try:
            if trace_id not in self.decision_traces:
                logger.error(f"Trace not found: {trace_id}")
                return False
            
            trace = self.decision_traces[trace_id]
            
            # Update outcome
            trace.actual_outcome = outcome
            trace.performance_metrics = performance_metrics or {}
            trace.lessons_learned = lessons_learned
            trace.execution_time = datetime.now()
            
            # Update statistics
            self.performance_stats['pending_decisions'] -= 1
            
            if outcome == DecisionOutcome.SUCCESS:
                self.performance_stats['successful_decisions'] += 1
            elif outcome == DecisionOutcome.FAILURE:
                self.performance_stats['failed_decisions'] += 1
            
            # Update strategy success rate
            strategy_id = trace.strategy_context.strategy_id
            if strategy_id not in self.performance_stats['strategy_success_rate']:
                self.performance_stats['strategy_success_rate'][strategy_id] = {'total': 0, 'successful': 0}
            
            self.performance_stats['strategy_success_rate'][strategy_id]['total'] += 1
            if outcome == DecisionOutcome.SUCCESS:
                self.performance_stats['strategy_success_rate'][strategy_id]['successful'] += 1
            
            # Update trace index
            self._update_trace_index(trace)
            
            # Save updated trace
            self._save_trace_to_disk(trace)
            
            logger.info(f"Trace outcome updated: {trace_id} -> {outcome.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating trace outcome: {e}")
            return False
    
    def get_decision_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        """Get decision trace by ID"""
        return self.decision_traces.get(trace_id)
    
    def search_traces(self, symbol: str = None, strategy_id: str = None,
                     decision_type: DecisionType = None, outcome: DecisionOutcome = None,
                     date_range: Tuple[datetime, datetime] = None,
                     limit: int = 50) -> List[DecisionTrace]:
        """Search decision traces with filters"""
        try:
            results = []
            
            for trace in self.decision_traces.values():
                # Apply filters
                if symbol and trace.symbol != symbol:
                    continue
                if strategy_id and trace.strategy_context.strategy_id != strategy_id:
                    continue
                if decision_type and trace.decision_type != decision_type:
                    continue
                if outcome and trace.actual_outcome != outcome:
                    continue
                if date_range:
                    start_date, end_date = date_range
                    if not (start_date <= trace.timestamp <= end_date):
                        continue
                
                results.append(trace)
                
                if len(results) >= limit:
                    break
            
            # Sort by timestamp (most recent first)
            results.sort(key=lambda t: t.timestamp, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching traces: {e}")
            return []
    
    def generate_audit_report(self, symbol: str = None, 
                            date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        try:
            # Get traces for report
            traces = self.search_traces(symbol=symbol, date_range=date_range, limit=1000)
            
            if not traces:
                return {'error': 'No traces found for audit report'}
            
            # Calculate metrics
            total_traces = len(traces)
            successful_traces = len([t for t in traces if t.actual_outcome == DecisionOutcome.SUCCESS])
            failed_traces = len([t for t in traces if t.actual_outcome == DecisionOutcome.FAILURE])
            pending_traces = len([t for t in traces if t.actual_outcome is None or t.actual_outcome == DecisionOutcome.PENDING])
            
            # Memory influence analysis
            memory_influenced = len([t for t in traces if t.memory_influences])
            memory_success_rate = 0.0
            if memory_influenced > 0:
                memory_successful = len([t for t in traces if t.memory_influences and t.actual_outcome == DecisionOutcome.SUCCESS])
                memory_success_rate = memory_successful / memory_influenced
            
            # Strategy performance analysis
            strategy_performance = {}
            for trace in traces:
                strategy_id = trace.strategy_context.strategy_id
                if strategy_id not in strategy_performance:
                    strategy_performance[strategy_id] = {'total': 0, 'successful': 0, 'avg_confidence': 0.0}
                
                strategy_performance[strategy_id]['total'] += 1
                if trace.actual_outcome == DecisionOutcome.SUCCESS:
                    strategy_performance[strategy_id]['successful'] += 1
                strategy_performance[strategy_id]['avg_confidence'] += trace.confidence_factors.final_confidence
            
            # Calculate averages
            for strategy_id in strategy_performance:
                perf = strategy_performance[strategy_id]
                perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0.0
                perf['avg_confidence'] = perf['avg_confidence'] / perf['total'] if perf['total'] > 0 else 0.0
            
            # Decision type analysis
            decision_type_stats = {}
            for decision_type in DecisionType:
                type_traces = [t for t in traces if t.decision_type == decision_type]
                if type_traces:
                    successful = len([t for t in type_traces if t.actual_outcome == DecisionOutcome.SUCCESS])
                    decision_type_stats[decision_type.value] = {
                        'total': len(type_traces),
                        'successful': successful,
                        'success_rate': successful / len(type_traces),
                        'avg_confidence': sum(t.confidence_factors.final_confidence for t in type_traces) / len(type_traces)
                    }
            
            # Risk assessment analysis
            high_risk_decisions = len([t for t in traces if t.risk_assessment.get('overall_risk', 0) > 0.7])
            low_risk_decisions = len([t for t in traces if t.risk_assessment.get('overall_risk', 0) < 0.3])
            
            # Generate report
            report = {
                'report_metadata': {
                    'generated_at': datetime.now(),
                    'symbol_filter': symbol,
                    'date_range': date_range,
                    'total_traces_analyzed': total_traces
                },
                'overall_performance': {
                    'total_decisions': total_traces,
                    'successful_decisions': successful_traces,
                    'failed_decisions': failed_traces,
                    'pending_decisions': pending_traces,
                    'success_rate': successful_traces / total_traces if total_traces > 0 else 0.0,
                    'failure_rate': failed_traces / total_traces if total_traces > 0 else 0.0
                },
                'memory_influence_analysis': {
                    'memory_influenced_decisions': memory_influenced,
                    'memory_influence_rate': memory_influenced / total_traces if total_traces > 0 else 0.0,
                    'memory_enhanced_success_rate': memory_success_rate,
                    'memory_effectiveness': memory_success_rate - (successful_traces / total_traces) if total_traces > 0 else 0.0
                },
                'strategy_performance': strategy_performance,
                'decision_type_analysis': decision_type_stats,
                'risk_analysis': {
                    'high_risk_decisions': high_risk_decisions,
                    'low_risk_decisions': low_risk_decisions,
                    'risk_distribution': {
                        'high_risk_rate': high_risk_decisions / total_traces if total_traces > 0 else 0.0,
                        'low_risk_rate': low_risk_decisions / total_traces if total_traces > 0 else 0.0
                    }
                },
                'confidence_analysis': {
                    'average_confidence': sum(t.confidence_factors.final_confidence for t in traces) / total_traces if total_traces > 0 else 0.0,
                    'high_confidence_decisions': len([t for t in traces if t.confidence_factors.final_confidence > 0.8]),
                    'low_confidence_decisions': len([t for t in traces if t.confidence_factors.final_confidence < 0.4])
                },
                'recent_traces': [
                    {
                        'trace_id': t.trace_id,
                        'timestamp': t.timestamp,
                        'symbol': t.symbol,
                        'decision_type': t.decision_type.value,
                        'outcome': t.actual_outcome.value if t.actual_outcome else 'pending',
                        'confidence': t.confidence_factors.final_confidence,
                        'memory_influenced': len(t.memory_influences) > 0
                    }
                    for t in traces[:10]
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_factors(self, strategy_context: StrategyContext,
                                    memory_influences: List[MemoryInfluence],
                                    environmental_context: Optional[EnvironmentalContext]) -> ConfidenceFactors:
        """Calculate comprehensive confidence factors"""
        try:
            # Base confidence from strategy
            base_confidence = strategy_context.fitness_score
            
            # Memory adjustment
            memory_adjustment = 0.0
            if memory_influences:
                memory_weights = sum(inf.influence_weight for inf in memory_influences)
                memory_confidence = sum(inf.confidence_adjustment * inf.influence_weight for inf in memory_influences)
                memory_adjustment = memory_confidence / memory_weights if memory_weights > 0 else 0.0
            
            # Strategy confidence
            strategy_confidence = strategy_context.fitness_score
            
            # Environmental adjustment
            environmental_adjustment = 0.0
            if environmental_context:
                # Adjust based on volatility and risk factors
                volatility_penalty = -environmental_context.volatility_level * 0.1
                risk_penalty = -len(environmental_context.risk_factors) * 0.05
                environmental_adjustment = volatility_penalty + risk_penalty
            
            # Risk adjustment (conservative approach)
            risk_adjustment = -0.05  # Small penalty for uncertainty
            
            # Calculate final confidence
            final_confidence = max(0.1, min(0.9, 
                base_confidence + memory_adjustment + environmental_adjustment + risk_adjustment
            ))
            
            # Identify confidence sources
            confidence_sources = ['strategy_fitness']
            if memory_influences:
                confidence_sources.append('memory_patterns')
            if environmental_context:
                confidence_sources.append('environmental_context')
            
            return ConfidenceFactors(
                base_confidence=base_confidence,
                memory_adjustment=memory_adjustment,
                strategy_confidence=strategy_confidence,
                environmental_adjustment=environmental_adjustment,
                risk_adjustment=risk_adjustment,
                final_confidence=final_confidence,
                confidence_sources=confidence_sources
            )
            
        except Exception as e:
            logger.error(f"Error calculating confidence factors: {e}")
            return ConfidenceFactors(0.5, 0.0, 0.5, 0.0, 0.0, 0.5, ['error'])
    
    def _generate_decision_logic(self, decision_type: DecisionType, decision_value: Any,
                               strategy_context: StrategyContext, memory_influences: List[MemoryInfluence],
                               confidence_factors: ConfidenceFactors) -> str:
        """Generate human-readable decision logic explanation"""
        try:
            logic_parts = []
            
            # Strategy component
            logic_parts.append(f"Strategy {strategy_context.strategy_id} (Gen {strategy_context.generation}, "
                             f"Fitness: {strategy_context.fitness_score:.3f}) recommended {decision_type.value}")
            
            # Memory component
            if memory_influences:
                memory_count = len(memory_influences)
                avg_success_rate = sum(inf.historical_success_rate for inf in memory_influences) / memory_count
                logic_parts.append(f"{memory_count} memory patterns support this decision "
                                 f"(Avg historical success: {avg_success_rate:.1%})")
                
                # Add key insights
                for inf in memory_influences[:2]:  # Top 2 influences
                    logic_parts.append(f"Memory insight: {inf.key_insight}")
            
            # Confidence component
            logic_parts.append(f"Final confidence: {confidence_factors.final_confidence:.1%} "
                             f"(Base: {confidence_factors.base_confidence:.1%}, "
                             f"Memory adj: {confidence_factors.memory_adjustment:+.1%})")
            
            return " | ".join(logic_parts)
            
        except Exception as e:
            logger.error(f"Error generating decision logic: {e}")
            return f"Decision logic generation failed: {str(e)}"
    
    def _assess_decision_risk(self, symbol: str, decision_type: DecisionType, decision_value: Any,
                            strategy_context: StrategyContext, 
                            environmental_context: Optional[EnvironmentalContext]) -> Dict[str, float]:
        """Assess risk factors for decision"""
        try:
            risk_factors = {
                'strategy_risk': max(0.0, 1.0 - strategy_context.fitness_score),
                'volatility_risk': environmental_context.volatility_level if environmental_context else 0.5,
                'market_risk': len(environmental_context.risk_factors) * 0.1 if environmental_context else 0.3,
                'execution_risk': 0.1,  # Base execution risk
            }
            
            # Calculate overall risk
            risk_factors['overall_risk'] = min(1.0, sum(risk_factors.values()) / len(risk_factors))
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error assessing decision risk: {e}")
            return {'overall_risk': 0.5, 'error': str(e)}
    
    def _generate_expected_outcome(self, decision_type: DecisionType, decision_value: Any,
                                 confidence_factors: ConfidenceFactors, 
                                 risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """Generate expected outcome metrics"""
        try:
            confidence = confidence_factors.final_confidence
            risk = risk_assessment.get('overall_risk', 0.5)
            
            # Expected success probability
            success_probability = confidence * (1.0 - risk * 0.5)
            
            # Expected return (simplified)
            if decision_type in [DecisionType.ENTRY, DecisionType.EXIT]:
                expected_return = success_probability * 0.02 - (1 - success_probability) * 0.01
            else:
                expected_return = 0.0
            
            # Risk-adjusted metrics
            sharpe_estimate = expected_return / max(0.01, risk)
            
            return {
                'success_probability': success_probability,
                'expected_return': expected_return,
                'risk_adjusted_return': sharpe_estimate,
                'confidence_score': confidence,
                'risk_score': risk
            }
            
        except Exception as e:
            logger.error(f"Error generating expected outcome: {e}")
            return {'success_probability': 0.5, 'expected_return': 0.0}
    
    def _generate_alternative_options(self, decision_type: DecisionType, 
                                    decision_value: Any) -> List[Dict[str, Any]]:
        """Generate alternative decision options"""
        try:
            alternatives = []
            
            if decision_type == DecisionType.ENTRY:
                alternatives.append({'option': 'wait', 'rationale': 'Wait for better entry conditions'})
                alternatives.append({'option': 'smaller_size', 'rationale': 'Enter with reduced position size'})
            elif decision_type == DecisionType.EXIT:
                alternatives.append({'option': 'partial_exit', 'rationale': 'Exit partial position, keep remainder'})
                alternatives.append({'option': 'hold', 'rationale': 'Hold position for further development'})
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    def _get_default_environmental_context(self) -> EnvironmentalContext:
        """Get default environmental context"""
        return EnvironmentalContext(
            market_regime='unknown',
            volatility_level=0.5,
            risk_factors=[],
            sentiment_score=0.5,
            macro_conditions={},
            correlation_matrix={}
        )
    
    def _update_trace_index(self, trace: DecisionTrace):
        """Update trace index for fast searching"""
        try:
            # Index by symbol
            if trace.symbol not in self.trace_index['by_symbol']:
                self.trace_index['by_symbol'][trace.symbol] = []
            self.trace_index['by_symbol'][trace.symbol].append(trace.trace_id)
            
            # Index by strategy
            strategy_id = trace.strategy_context.strategy_id
            if strategy_id not in self.trace_index['by_strategy']:
                self.trace_index['by_strategy'][strategy_id] = []
            self.trace_index['by_strategy'][strategy_id].append(trace.trace_id)
            
            # Index by date
            date_key = trace.timestamp.strftime('%Y-%m-%d')
            if date_key not in self.trace_index['by_date']:
                self.trace_index['by_date'][date_key] = []
            self.trace_index['by_date'][date_key].append(trace.trace_id)
            
            # Index by outcome (if available)
            if trace.actual_outcome:
                outcome_key = trace.actual_outcome.value
                if outcome_key not in self.trace_index['by_outcome']:
                    self.trace_index['by_outcome'][outcome_key] = []
                self.trace_index['by_outcome'][outcome_key].append(trace.trace_id)
            
        except Exception as e:
            logger.error(f"Error updating trace index: {e}")
    
    def _save_trace_to_disk(self, trace: DecisionTrace):
        """Save trace to disk for persistence"""
        try:
            trace_file = os.path.join(self.data_dir, f"{trace.trace_id}.json")
            trace_data = asdict(trace)
            
            # Convert datetime objects to ISO strings
            trace_data['timestamp'] = trace.timestamp.isoformat()
            if trace.execution_time:
                trace_data['execution_time'] = trace.execution_time.isoformat()
            
            # Convert enums to strings
            trace_data['decision_type'] = trace.decision_type.value
            if trace.actual_outcome:
                trace_data['actual_outcome'] = trace.actual_outcome.value
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving trace to disk: {e}")
    
    def _trace_processing_loop(self):
        """Background trace processing loop"""
        while self.running:
            try:
                time.sleep(300)  # Process every 5 minutes
                
                # Update performance statistics
                self._update_performance_stats()
                
                # Clean up old traces (keep last 30 days)
                self._cleanup_old_traces()
                
            except Exception as e:
                logger.error(f"Error in trace processing loop: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            if not self.decision_traces:
                return
            
            total_traces = len(self.decision_traces)
            successful = len([t for t in self.decision_traces.values() 
                            if t.actual_outcome == DecisionOutcome.SUCCESS])
            failed = len([t for t in self.decision_traces.values() 
                        if t.actual_outcome == DecisionOutcome.FAILURE])
            pending = len([t for t in self.decision_traces.values() 
                         if t.actual_outcome is None or t.actual_outcome == DecisionOutcome.PENDING])
            
            # Update stats
            self.performance_stats.update({
                'total_decisions': total_traces,
                'successful_decisions': successful,
                'failed_decisions': failed,
                'pending_decisions': pending,
                'average_confidence': sum(t.confidence_factors.final_confidence 
                                        for t in self.decision_traces.values()) / total_traces,
                'memory_influence_rate': len([t for t in self.decision_traces.values() 
                                            if t.memory_influences]) / total_traces
            })
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def _cleanup_old_traces(self):
        """Clean up traces older than 30 days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            old_traces = [trace_id for trace_id, trace in self.decision_traces.items() 
                         if trace.timestamp < cutoff_date]
            
            for trace_id in old_traces:
                # Remove from memory
                del self.decision_traces[trace_id]
                
                # Remove trace file
                trace_file = os.path.join(self.data_dir, f"{trace_id}.json")
                if os.path.exists(trace_file):
                    os.remove(trace_file)
            
            if old_traces:
                logger.info(f"Cleaned up {len(old_traces)} old traces")
            
        except Exception as e:
            logger.error(f"Error cleaning up old traces: {e}")
    
    def get_traceability_stats(self) -> Dict[str, Any]:
        """Get traceability engine statistics"""
        return {
            **self.performance_stats,
            'trace_index_size': {
                'by_symbol': len(self.trace_index['by_symbol']),
                'by_strategy': len(self.trace_index['by_strategy']),
                'by_date': len(self.trace_index['by_date']),
                'by_outcome': len(self.trace_index['by_outcome'])
            },
            'engine_health': 'Excellent' if self.running else 'Stopped'
        }
    
    def stop(self):
        """Stop the traceability engine"""
        self.running = False
        logger.info("Decision Traceability Engine stopped")
