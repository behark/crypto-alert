"""
Strategy Evolution Core - Phase 4 Evolution Layer
Genetic Algorithm Framework for evolving trading strategies through mutation and selection.
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
import random
import numpy as np
from collections import defaultdict, deque
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    HYBRID = "hybrid"

class MutationType(Enum):
    PARAMETER_TWEAK = "parameter_tweak"
    LOGIC_MODIFICATION = "logic_modification"
    SIGNAL_WEIGHT_CHANGE = "signal_weight_change"
    RISK_ADJUSTMENT = "risk_adjustment"
    TIMING_SHIFT = "timing_shift"

@dataclass
class StrategyGene:
    """Individual gene in a trading strategy"""
    name: str
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mutation_rate: float = 0.1
    gene_type: str = "float"  # float, int, bool, choice

@dataclass
class TradingStrategy:
    """Complete trading strategy with genetic components"""
    strategy_id: str
    strategy_type: StrategyType
    generation: int
    parent_ids: List[str]
    genes: Dict[str, StrategyGene]
    performance_metrics: Dict[str, float]
    fitness_score: float
    age: int
    trades_executed: int
    creation_time: datetime
    last_update: datetime
    active: bool = True

@dataclass
class EvolutionSession:
    """Evolution session tracking"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    generation_count: int
    population_size: int
    mutation_rate: float
    crossover_rate: float
    selection_method: str
    fitness_improvements: List[float]
    best_strategies: List[str]

@dataclass
class MutationEvent:
    """Record of a mutation event"""
    mutation_id: str
    strategy_id: str
    mutation_type: MutationType
    genes_affected: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    timestamp: datetime
    success: bool

class StrategyEvolutionCore:
    """
    Strategy Evolution Core - Genetic Algorithm Framework for trading strategies.
    Evolves strategies through mutation, crossover, and natural selection.
    """
    
    def __init__(self, data_dir: str = "data/evolution"):
        """Initialize the Strategy Evolution Core"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Evolution parameters
        self.population_size = 20
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_size = 3
        self.max_generations = 100
        
        # Strategy population
        self.strategies: Dict[str, TradingStrategy] = {}
        self.active_strategies: List[str] = []
        self.elite_strategies: List[str] = []
        
        # Evolution tracking
        self.current_generation = 0
        self.evolution_sessions: Dict[str, EvolutionSession] = {}
        self.mutation_history: List[MutationEvent] = []
        
        # Performance tracking
        self.fitness_history: List[float] = []
        self.generation_stats: Dict[int, Dict[str, float]] = {}
        
        # Evolution statistics
        self.evolution_stats = {
            'total_strategies_created': 0,
            'successful_mutations': 0,
            'failed_mutations': 0,
            'crossover_events': 0,
            'generations_completed': 0,
            'best_fitness_ever': 0.0,
            'last_evolution': None
        }
        
        # Load existing strategies
        self._load_strategies()
        
        # Initialize base strategies if population is empty
        if not self.strategies:
            self._initialize_base_population()
        
        # Start evolution monitoring thread
        self.running = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()
        
        logger.info("Strategy Evolution Core initialized with genetic algorithm framework")
    
    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve the current generation of strategies"""
        try:
            evolution_results = {
                'generation': self.current_generation + 1,
                'strategies_created': 0,
                'mutations_applied': 0,
                'crossovers_performed': 0,
                'fitness_improvement': 0.0,
                'best_strategy_id': None,
                'evolution_time': datetime.now()
            }
            
            # Calculate fitness for all strategies
            self._calculate_population_fitness()
            
            # Select elite strategies
            self._select_elite()
            
            # Create new generation
            new_strategies = []
            
            # Keep elite strategies
            for elite_id in self.elite_strategies:
                if elite_id in self.strategies:
                    elite_copy = copy.deepcopy(self.strategies[elite_id])
                    elite_copy.generation = self.current_generation + 1
                    elite_copy.age += 1
                    new_strategies.append(elite_copy)
            
            # Fill remaining population through crossover and mutation
            while len(new_strategies) < self.population_size:
                if random.random() < self.crossover_rate and len(self.elite_strategies) >= 2:
                    # Crossover
                    parent1_id = random.choice(self.elite_strategies)
                    parent2_id = random.choice(self.elite_strategies)
                    
                    if parent1_id != parent2_id:
                        child = self._crossover_strategies(parent1_id, parent2_id)
                        if child:
                            new_strategies.append(child)
                            evolution_results['crossovers_performed'] += 1
                
                # Mutation
                if len(new_strategies) < self.population_size:
                    parent_id = random.choice(self.active_strategies) if self.active_strategies else random.choice(list(self.strategies.keys()))
                    mutated = self._mutate_strategy(parent_id)
                    if mutated:
                        new_strategies.append(mutated)
                        evolution_results['mutations_applied'] += 1
            
            # Update population
            old_fitness = np.mean([s.fitness_score for s in self.strategies.values()])
            
            # Replace old strategies with new generation
            self.strategies.clear()
            for strategy in new_strategies:
                self.strategies[strategy.strategy_id] = strategy
            
            # Update active strategies list
            self.active_strategies = list(self.strategies.keys())
            
            # Calculate new fitness
            self._calculate_population_fitness()
            new_fitness = np.mean([s.fitness_score for s in self.strategies.values()])
            
            # Update generation
            self.current_generation += 1
            evolution_results['generation'] = self.current_generation
            evolution_results['strategies_created'] = len(new_strategies)
            evolution_results['fitness_improvement'] = new_fitness - old_fitness
            
            # Find best strategy
            best_strategy = max(self.strategies.values(), key=lambda s: s.fitness_score)
            evolution_results['best_strategy_id'] = best_strategy.strategy_id
            
            # Update statistics
            self.evolution_stats['generations_completed'] += 1
            self.evolution_stats['best_fitness_ever'] = max(self.evolution_stats['best_fitness_ever'], best_strategy.fitness_score)
            self.evolution_stats['last_evolution'] = datetime.now()
            
            # Store generation stats
            self.generation_stats[self.current_generation] = {
                'avg_fitness': new_fitness,
                'best_fitness': best_strategy.fitness_score,
                'population_size': len(self.strategies),
                'mutations': evolution_results['mutations_applied'],
                'crossovers': evolution_results['crossovers_performed']
            }
            
            # Save evolved strategies
            self._save_generation()
            
            logger.info(f"Generation {self.current_generation} evolved: {evolution_results}")
            return evolution_results
            
        except Exception as e:
            logger.error(f"Error evolving generation: {e}")
            return {}
    
    def mutate_strategy_now(self, strategy_id: str, mutation_type: Optional[MutationType] = None) -> Optional[TradingStrategy]:
        """Immediately mutate a specific strategy"""
        try:
            if strategy_id not in self.strategies:
                logger.error(f"Strategy {strategy_id} not found for mutation")
                return None
            
            # Select mutation type if not specified
            if mutation_type is None:
                mutation_type = random.choice(list(MutationType))
            
            # Create mutated copy
            original_strategy = self.strategies[strategy_id]
            mutated_strategy = copy.deepcopy(original_strategy)
            
            # Generate new ID for mutated strategy
            mutated_strategy.strategy_id = f"mut_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mutated_strategy.parent_ids = [strategy_id]
            mutated_strategy.generation += 1
            mutated_strategy.age = 0
            mutated_strategy.trades_executed = 0
            mutated_strategy.creation_time = datetime.now()
            mutated_strategy.last_update = datetime.now()
            
            # Apply mutation
            mutation_success = self._apply_mutation(mutated_strategy, mutation_type)
            
            if mutation_success:
                # Add to population
                self.strategies[mutated_strategy.strategy_id] = mutated_strategy
                self.active_strategies.append(mutated_strategy.strategy_id)
                
                # Update statistics
                self.evolution_stats['total_strategies_created'] += 1
                self.evolution_stats['successful_mutations'] += 1
                
                logger.info(f"Strategy {strategy_id} successfully mutated to {mutated_strategy.strategy_id}")
                return mutated_strategy
            else:
                self.evolution_stats['failed_mutations'] += 1
                logger.warning(f"Failed to mutate strategy {strategy_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error in immediate strategy mutation: {e}")
            return None
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy evolution status"""
        try:
            if not self.strategies:
                return {'error': 'No strategies in population'}
            
            # Calculate population statistics
            fitness_scores = [s.fitness_score for s in self.strategies.values()]
            ages = [s.age for s in self.strategies.values()]
            trades = [s.trades_executed for s in self.strategies.values()]
            
            # Strategy type distribution
            type_distribution = defaultdict(int)
            for strategy in self.strategies.values():
                type_distribution[strategy.strategy_type.value] += 1
            
            # Performance metrics
            active_count = len([s for s in self.strategies.values() if s.active])
            
            status = {
                'population_overview': {
                    'total_strategies': len(self.strategies),
                    'active_strategies': active_count,
                    'current_generation': self.current_generation,
                    'elite_strategies': len(self.elite_strategies)
                },
                'fitness_statistics': {
                    'average_fitness': np.mean(fitness_scores),
                    'best_fitness': np.max(fitness_scores),
                    'worst_fitness': np.min(fitness_scores),
                    'fitness_std': np.std(fitness_scores)
                },
                'population_characteristics': {
                    'average_age': np.mean(ages),
                    'total_trades_executed': np.sum(trades),
                    'strategy_types': dict(type_distribution)
                },
                'evolution_progress': {
                    'generations_completed': self.evolution_stats['generations_completed'],
                    'successful_mutations': self.evolution_stats['successful_mutations'],
                    'crossover_events': self.evolution_stats['crossover_events'],
                    'best_fitness_ever': self.evolution_stats['best_fitness_ever']
                },
                'top_performers': []
            }
            
            # Get top performing strategies
            top_strategies = sorted(self.strategies.values(), key=lambda s: s.fitness_score, reverse=True)[:5]
            
            for strategy in top_strategies:
                status['top_performers'].append({
                    'strategy_id': strategy.strategy_id,
                    'strategy_type': strategy.strategy_type.value,
                    'fitness_score': strategy.fitness_score,
                    'generation': strategy.generation,
                    'age': strategy.age,
                    'trades_executed': strategy.trades_executed,
                    'performance_metrics': strategy.performance_metrics
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {'error': str(e)}
    
    def _initialize_base_population(self):
        """Initialize base population of strategies"""
        try:
            base_strategies = [
                # Trend Following Strategy
                {
                    'type': StrategyType.TREND_FOLLOWING,
                    'genes': {
                        'trend_threshold': StrategyGene('trend_threshold', 0.02, 0.01, 0.05, 0.1),
                        'stop_loss': StrategyGene('stop_loss', 0.03, 0.01, 0.1, 0.1),
                        'take_profit': StrategyGene('take_profit', 0.06, 0.02, 0.2, 0.1),
                        'position_size': StrategyGene('position_size', 0.1, 0.05, 0.3, 0.05)
                    }
                },
                # Mean Reversion Strategy
                {
                    'type': StrategyType.MEAN_REVERSION,
                    'genes': {
                        'reversion_threshold': StrategyGene('reversion_threshold', 0.05, 0.02, 0.1, 0.1),
                        'mean_period': StrategyGene('mean_period', 20, 10, 50, 0.1, gene_type='int'),
                        'stop_loss': StrategyGene('stop_loss', 0.02, 0.01, 0.05, 0.1),
                        'position_size': StrategyGene('position_size', 0.08, 0.03, 0.25, 0.05)
                    }
                },
                # Momentum Strategy
                {
                    'type': StrategyType.MOMENTUM,
                    'genes': {
                        'momentum_threshold': StrategyGene('momentum_threshold', 0.03, 0.01, 0.08, 0.1),
                        'lookback_period': StrategyGene('lookback_period', 14, 5, 30, 0.1, gene_type='int'),
                        'stop_loss': StrategyGene('stop_loss', 0.04, 0.02, 0.08, 0.1),
                        'position_size': StrategyGene('position_size', 0.12, 0.05, 0.3, 0.05)
                    }
                },
                # Breakout Strategy
                {
                    'type': StrategyType.BREAKOUT,
                    'genes': {
                        'breakout_threshold': StrategyGene('breakout_threshold', 0.025, 0.01, 0.06, 0.1),
                        'volume_multiplier': StrategyGene('volume_multiplier', 1.5, 1.1, 3.0, 0.1),
                        'stop_loss': StrategyGene('stop_loss', 0.035, 0.015, 0.07, 0.1),
                        'position_size': StrategyGene('position_size', 0.15, 0.08, 0.35, 0.05)
                    }
                }
            ]
            
            for i, strategy_config in enumerate(base_strategies):
                strategy_id = f"base_{strategy_config['type'].value}_{i}_{datetime.now().strftime('%Y%m%d')}"
                
                strategy = TradingStrategy(
                    strategy_id=strategy_id,
                    strategy_type=strategy_config['type'],
                    generation=0,
                    parent_ids=[],
                    genes=strategy_config['genes'],
                    performance_metrics={
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0,
                        'profit_factor': 1.0
                    },
                    fitness_score=0.5,  # Neutral starting fitness
                    age=0,
                    trades_executed=0,
                    creation_time=datetime.now(),
                    last_update=datetime.now()
                )
                
                self.strategies[strategy_id] = strategy
                self.active_strategies.append(strategy_id)
            
            self.evolution_stats['total_strategies_created'] = len(base_strategies)
            logger.info(f"Initialized base population with {len(base_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error initializing base population: {e}")
    
    def _calculate_population_fitness(self):
        """Calculate fitness scores for all strategies"""
        try:
            for strategy in self.strategies.values():
                # Multi-objective fitness function
                metrics = strategy.performance_metrics
                
                # Base fitness components
                return_component = metrics.get('total_return', 0.0) * 0.3
                sharpe_component = min(metrics.get('sharpe_ratio', 0.0) / 3.0, 1.0) * 0.25
                drawdown_component = (1.0 - min(abs(metrics.get('max_drawdown', 0.0)), 1.0)) * 0.2
                win_rate_component = metrics.get('win_rate', 0.0) * 0.15
                profit_factor_component = min(metrics.get('profit_factor', 1.0) / 2.0, 1.0) * 0.1
                
                # Age penalty (older strategies get slight penalty to encourage evolution)
                age_penalty = min(strategy.age * 0.01, 0.1)
                
                # Calculate final fitness
                fitness = (return_component + sharpe_component + drawdown_component + 
                          win_rate_component + profit_factor_component - age_penalty)
                
                # Ensure fitness is between 0 and 1
                strategy.fitness_score = max(0.0, min(1.0, fitness))
            
        except Exception as e:
            logger.error(f"Error calculating population fitness: {e}")
    
    def _select_elite(self):
        """Select elite strategies for next generation"""
        try:
            # Sort strategies by fitness
            sorted_strategies = sorted(self.strategies.values(), key=lambda s: s.fitness_score, reverse=True)
            
            # Select top performers as elite
            self.elite_strategies = [s.strategy_id for s in sorted_strategies[:self.elite_size]]
            
            logger.info(f"Selected {len(self.elite_strategies)} elite strategies")
            
        except Exception as e:
            logger.error(f"Error selecting elite strategies: {e}")
    
    def _evolution_loop(self):
        """Background evolution monitoring loop"""
        while self.running:
            try:
                time.sleep(3600)  # Run every hour
                
                # Auto-evolve if conditions are met
                if len(self.strategies) >= self.population_size and self.current_generation < self.max_generations:
                    # Check if it's time for evolution (e.g., daily at 3 AM)
                    if datetime.now().hour == 3:
                        self.evolve_generation()
                
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
    
    def _save_generation(self):
        """Save current generation to disk"""
        try:
            generation_file = os.path.join(self.data_dir, f"generation_{self.current_generation}.json")
            
            generation_data = {
                'generation': self.current_generation,
                'timestamp': datetime.now().isoformat(),
                'strategies': {sid: asdict(strategy) for sid, strategy in self.strategies.items()},
                'elite_strategies': self.elite_strategies,
                'stats': self.generation_stats.get(self.current_generation, {})
            }
            
            with open(generation_file, 'w') as f:
                json.dump(generation_data, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving generation: {e}")
    
    def _load_strategies(self):
        """Load existing strategies from disk"""
        try:
            if not os.path.exists(self.data_dir):
                return
            
            # Find latest generation file
            generation_files = [f for f in os.listdir(self.data_dir) if f.startswith('generation_')]
            if not generation_files:
                return
            
            latest_file = max(generation_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            file_path = os.path.join(self.data_dir, latest_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct strategies (simplified loading)
            self.current_generation = data.get('generation', 0)
            
            logger.info(f"Loaded strategies from generation {self.current_generation}")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def stop(self):
        """Stop the evolution engine"""
        self.running = False
        logger.info("Strategy Evolution Core stopped")
