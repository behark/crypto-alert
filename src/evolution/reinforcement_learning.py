"""
Reinforcement Learning Module - Phase 4 Evolution Layer
Q-Learning and Policy Gradient methods for optimal trading policy optimization.
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

class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

class StateFeature(Enum):
    PRICE_MOMENTUM = "price_momentum"
    VOLUME_PROFILE = "volume_profile"
    VOLATILITY = "volatility"
    TREND_STRENGTH = "trend_strength"
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_REGIME = "market_regime"
    RISK_LEVEL = "risk_level"

@dataclass
class MarketState:
    """Market state representation for RL agent"""
    timestamp: datetime
    symbol: str
    features: Dict[StateFeature, float]
    normalized_features: np.ndarray
    state_hash: str

@dataclass
class TradingAction:
    """Trading action with parameters"""
    action_type: ActionType
    position_size: float
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: MarketState
    action: TradingAction
    reward: float
    next_state: MarketState
    done: bool
    timestamp: datetime

@dataclass
class Episode:
    """Complete trading episode"""
    episode_id: str
    start_time: datetime
    end_time: datetime
    experiences: List[Experience]
    total_reward: float
    final_pnl: float
    actions_taken: int
    success: bool

class QTable:
    """Q-Table for Q-Learning algorithm"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.q_table: Dict[str, np.ndarray] = {}
        self.visit_counts: Dict[str, np.ndarray] = {}
    
    def get_q_values(self, state_hash: str) -> np.ndarray:
        """Get Q-values for a state"""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.action_size)
            self.visit_counts[state_hash] = np.zeros(self.action_size)
        return self.q_table[state_hash]
    
    def update_q_value(self, state_hash: str, action_idx: int, td_error: float):
        """Update Q-value using TD error"""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.action_size)
            self.visit_counts[state_hash] = np.zeros(self.action_size)
        
        self.visit_counts[state_hash][action_idx] += 1
        
        # Adaptive learning rate based on visit count
        adaptive_lr = self.learning_rate / (1 + 0.01 * self.visit_counts[state_hash][action_idx])
        
        self.q_table[state_hash][action_idx] += adaptive_lr * td_error
    
    def get_best_action(self, state_hash: str, epsilon: float = 0.0) -> int:
        """Get best action using epsilon-greedy policy"""
        q_values = self.get_q_values(state_hash)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(q_values)

class ReinforcementLearningAgent:
    """
    Reinforcement Learning Agent for trading optimization.
    Implements Q-Learning and experience replay for continuous learning.
    """
    
    def __init__(self, data_dir: str = "data/rl_agent"):
        """Initialize the RL Agent"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # RL Parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # State and action spaces
        self.state_features = list(StateFeature)
        self.action_types = list(ActionType)
        self.state_size = len(self.state_features)
        self.action_size = len(self.action_types)
        
        # Q-Learning components
        self.q_table = QTable(self.state_size, self.action_size, self.learning_rate)
        
        # Experience replay
        self.replay_buffer: deque = deque(maxlen=10000)
        self.batch_size = 32
        
        # Episode tracking
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        
        # Performance tracking
        self.training_stats = {
            'episodes_completed': 0,
            'total_experiences': 0,
            'average_reward': 0.0,
            'best_episode_reward': 0.0,
            'learning_progress': [],
            'last_training': None
        }
        
        # State normalization
        self.feature_stats = {
            'means': np.zeros(self.state_size),
            'stds': np.ones(self.state_size),
            'mins': np.full(self.state_size, float('inf')),
            'maxs': np.full(self.state_size, float('-inf'))
        }
        
        # Load existing model
        self._load_model()
        
        # Start training thread
        self.running = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("Reinforcement Learning Agent initialized with Q-Learning")
    
    def start_episode(self, symbol: str) -> str:
        """Start a new trading episode"""
        try:
            episode_id = f"episode_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_episode = Episode(
                episode_id=episode_id,
                start_time=datetime.now(),
                end_time=None,
                experiences=[],
                total_reward=0.0,
                final_pnl=0.0,
                actions_taken=0,
                success=False
            )
            
            logger.info(f"Started new episode: {episode_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Error starting episode: {e}")
            return ""
    
    def get_action(self, market_state: MarketState) -> TradingAction:
        """Get trading action based on current market state"""
        try:
            # Normalize state features
            normalized_state = self._normalize_state(market_state)
            state_hash = self._get_state_hash(normalized_state)
            
            # Get action from Q-table using epsilon-greedy
            action_idx = self.q_table.get_best_action(state_hash, self.epsilon)
            action_type = self.action_types[action_idx]
            
            # Generate action parameters based on Q-values and state
            q_values = self.q_table.get_q_values(state_hash)
            confidence = self._calculate_action_confidence(q_values, action_idx)
            position_size = self._calculate_position_size(market_state, confidence)
            
            action = TradingAction(
                action_type=action_type,
                position_size=position_size,
                confidence=confidence,
                stop_loss=self._calculate_stop_loss(market_state, action_type),
                take_profit=self._calculate_take_profit(market_state, action_type)
            )
            
            logger.info(f"RL Agent action: {action_type.value} with confidence {confidence:.2f}")
            return action
            
        except Exception as e:
            logger.error(f"Error getting RL action: {e}")
            # Return safe default action
            return TradingAction(
                action_type=ActionType.HOLD,
                position_size=0.0,
                confidence=0.5
            )
    
    def add_experience(self, state: MarketState, action: TradingAction, reward: float, 
                      next_state: MarketState, done: bool):
        """Add experience to current episode and replay buffer"""
        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                timestamp=datetime.now()
            )
            
            # Add to current episode
            if self.current_episode:
                self.current_episode.experiences.append(experience)
                self.current_episode.total_reward += reward
                self.current_episode.actions_taken += 1
            
            # Add to replay buffer
            self.replay_buffer.append(experience)
            
            # Update training stats
            self.training_stats['total_experiences'] += 1
            
            # Perform online learning update
            self._update_q_values(experience)
            
            logger.debug(f"Added experience with reward: {reward:.4f}")
            
        except Exception as e:
            logger.error(f"Error adding experience: {e}")
    
    def end_episode(self, final_pnl: float) -> Dict[str, Any]:
        """End current episode and return results"""
        try:
            if not self.current_episode:
                return {'error': 'No active episode'}
            
            # Finalize episode
            self.current_episode.end_time = datetime.now()
            self.current_episode.final_pnl = final_pnl
            self.current_episode.success = final_pnl > 0
            
            # Store episode
            self.episodes[self.current_episode.episode_id] = self.current_episode
            
            # Update training statistics
            self.training_stats['episodes_completed'] += 1
            
            # Calculate average reward
            if self.training_stats['episodes_completed'] > 0:
                total_reward = sum(ep.total_reward for ep in self.episodes.values())
                self.training_stats['average_reward'] = total_reward / self.training_stats['episodes_completed']
            
            # Update best episode reward
            if self.current_episode.total_reward > self.training_stats['best_episode_reward']:
                self.training_stats['best_episode_reward'] = self.current_episode.total_reward
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            episode_results = {
                'episode_id': self.current_episode.episode_id,
                'duration': (self.current_episode.end_time - self.current_episode.start_time).total_seconds(),
                'total_reward': self.current_episode.total_reward,
                'final_pnl': final_pnl,
                'actions_taken': self.current_episode.actions_taken,
                'success': self.current_episode.success,
                'current_epsilon': self.epsilon
            }
            
            # Clear current episode
            self.current_episode = None
            
            logger.info(f"Episode completed: {episode_results}")
            return episode_results
            
        except Exception as e:
            logger.error(f"Error ending episode: {e}")
            return {'error': str(e)}
    
    def train_from_replay(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Train agent using experience replay"""
        try:
            if len(self.replay_buffer) < (batch_size or self.batch_size):
                return {'error': 'Insufficient experiences for training'}
            
            batch_size = batch_size or self.batch_size
            
            # Sample random batch from replay buffer
            batch = random.sample(list(self.replay_buffer), batch_size)
            
            training_loss = 0.0
            updates_made = 0
            
            for experience in batch:
                # Calculate TD target
                current_state_hash = self._get_state_hash(experience.state.normalized_features)
                next_state_hash = self._get_state_hash(experience.next_state.normalized_features)
                
                current_q_values = self.q_table.get_q_values(current_state_hash)
                next_q_values = self.q_table.get_q_values(next_state_hash)
                
                action_idx = self.action_types.index(experience.action.action_type)
                
                if experience.done:
                    td_target = experience.reward
                else:
                    td_target = experience.reward + self.discount_factor * np.max(next_q_values)
                
                td_error = td_target - current_q_values[action_idx]
                
                # Update Q-value
                self.q_table.update_q_value(current_state_hash, action_idx, td_error)
                
                training_loss += abs(td_error)
                updates_made += 1
            
            avg_loss = training_loss / updates_made if updates_made > 0 else 0.0
            
            # Update training progress
            self.training_stats['learning_progress'].append({
                'timestamp': datetime.now(),
                'avg_loss': avg_loss,
                'epsilon': self.epsilon,
                'experiences_used': batch_size
            })
            
            # Keep only recent progress (last 100 entries)
            if len(self.training_stats['learning_progress']) > 100:
                self.training_stats['learning_progress'] = self.training_stats['learning_progress'][-100:]
            
            self.training_stats['last_training'] = datetime.now()
            
            training_results = {
                'batch_size': batch_size,
                'updates_made': updates_made,
                'average_loss': avg_loss,
                'current_epsilon': self.epsilon,
                'replay_buffer_size': len(self.replay_buffer)
            }
            
            logger.info(f"Replay training completed: {training_results}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error in replay training: {e}")
            return {'error': str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        try:
            # Calculate recent performance
            recent_episodes = list(self.episodes.values())[-10:] if self.episodes else []
            recent_rewards = [ep.total_reward for ep in recent_episodes]
            recent_success_rate = np.mean([ep.success for ep in recent_episodes]) if recent_episodes else 0.0
            
            # Q-table statistics
            total_states = len(self.q_table.q_table)
            avg_q_values = []
            for q_vals in self.q_table.q_table.values():
                avg_q_values.extend(q_vals)
            
            stats = {
                'training_overview': {
                    'episodes_completed': self.training_stats['episodes_completed'],
                    'total_experiences': self.training_stats['total_experiences'],
                    'replay_buffer_size': len(self.replay_buffer),
                    'current_epsilon': self.epsilon
                },
                'performance_metrics': {
                    'average_reward': self.training_stats['average_reward'],
                    'best_episode_reward': self.training_stats['best_episode_reward'],
                    'recent_average_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
                    'recent_success_rate': recent_success_rate
                },
                'learning_progress': {
                    'total_states_discovered': total_states,
                    'average_q_value': np.mean(avg_q_values) if avg_q_values else 0.0,
                    'q_value_std': np.std(avg_q_values) if avg_q_values else 0.0,
                    'learning_stability': self._calculate_learning_stability()
                },
                'recent_episodes': []
            }
            
            # Add recent episode details
            for episode in recent_episodes[-5:]:
                stats['recent_episodes'].append({
                    'episode_id': episode.episode_id,
                    'total_reward': episode.total_reward,
                    'final_pnl': episode.final_pnl,
                    'actions_taken': episode.actions_taken,
                    'success': episode.success,
                    'duration': (episode.end_time - episode.start_time).total_seconds() if episode.end_time else 0
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {'error': str(e)}
    
    def _normalize_state(self, market_state: MarketState) -> np.ndarray:
        """Normalize market state features"""
        try:
            features = np.array([market_state.features[feature] for feature in self.state_features])
            
            # Update feature statistics
            self.feature_stats['mins'] = np.minimum(self.feature_stats['mins'], features)
            self.feature_stats['maxs'] = np.maximum(self.feature_stats['maxs'], features)
            
            # Simple min-max normalization
            ranges = self.feature_stats['maxs'] - self.feature_stats['mins']
            ranges[ranges == 0] = 1  # Avoid division by zero
            
            normalized = (features - self.feature_stats['mins']) / ranges
            
            # Store normalized features in market state
            market_state.normalized_features = normalized
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing state: {e}")
            return np.zeros(self.state_size)
    
    def _get_state_hash(self, normalized_features: np.ndarray) -> str:
        """Get hash representation of state for Q-table lookup"""
        try:
            # Discretize continuous features for Q-table
            discretized = np.round(normalized_features * 10).astype(int)
            return '_'.join(map(str, discretized))
            
        except Exception as e:
            logger.error(f"Error getting state hash: {e}")
            return "default_state"
    
    def _update_q_values(self, experience: Experience):
        """Update Q-values based on single experience"""
        try:
            current_state_hash = self._get_state_hash(experience.state.normalized_features)
            next_state_hash = self._get_state_hash(experience.next_state.normalized_features)
            
            current_q_values = self.q_table.get_q_values(current_state_hash)
            next_q_values = self.q_table.get_q_values(next_state_hash)
            
            action_idx = self.action_types.index(experience.action.action_type)
            
            # Calculate TD target
            if experience.done:
                td_target = experience.reward
            else:
                td_target = experience.reward + self.discount_factor * np.max(next_q_values)
            
            td_error = td_target - current_q_values[action_idx]
            
            # Update Q-value
            self.q_table.update_q_value(current_state_hash, action_idx, td_error)
            
        except Exception as e:
            logger.error(f"Error updating Q-values: {e}")
    
    def _calculate_action_confidence(self, q_values: np.ndarray, action_idx: int) -> float:
        """Calculate confidence in selected action"""
        try:
            if len(q_values) == 0:
                return 0.5
            
            # Confidence based on Q-value relative to others
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            
            if max_q == min_q:
                return 0.5
            
            # Normalize selected action's Q-value
            confidence = (q_values[action_idx] - min_q) / (max_q - min_q)
            return max(0.1, min(0.9, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating action confidence: {e}")
            return 0.5
    
    def _calculate_position_size(self, market_state: MarketState, confidence: float) -> float:
        """Calculate position size based on confidence and market conditions"""
        try:
            # Base position size scaled by confidence
            base_size = 0.1  # 10% base allocation
            confidence_multiplier = 0.5 + confidence  # Range: 0.6 to 1.4
            
            # Adjust for volatility
            volatility = market_state.features.get(StateFeature.VOLATILITY, 0.5)
            volatility_adjustment = max(0.5, 1.0 - volatility)
            
            position_size = base_size * confidence_multiplier * volatility_adjustment
            
            return max(0.01, min(0.3, position_size))  # Cap between 1% and 30%
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.05
    
    def _calculate_stop_loss(self, market_state: MarketState, action_type: ActionType) -> Optional[float]:
        """Calculate stop loss based on market conditions"""
        try:
            if action_type == ActionType.HOLD:
                return None
            
            volatility = market_state.features.get(StateFeature.VOLATILITY, 0.02)
            base_stop = 0.02  # 2% base stop loss
            
            # Adjust for volatility
            adjusted_stop = base_stop * (1 + volatility)
            
            return max(0.01, min(0.05, adjusted_stop))
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return 0.02
    
    def _calculate_take_profit(self, market_state: MarketState, action_type: ActionType) -> Optional[float]:
        """Calculate take profit based on market conditions"""
        try:
            if action_type == ActionType.HOLD:
                return None
            
            trend_strength = market_state.features.get(StateFeature.TREND_STRENGTH, 0.5)
            base_tp = 0.04  # 4% base take profit
            
            # Adjust for trend strength
            adjusted_tp = base_tp * (1 + trend_strength)
            
            return max(0.02, min(0.1, adjusted_tp))
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return 0.04
    
    def _calculate_learning_stability(self) -> float:
        """Calculate learning stability metric"""
        try:
            if len(self.training_stats['learning_progress']) < 10:
                return 0.5
            
            recent_losses = [entry['avg_loss'] for entry in self.training_stats['learning_progress'][-10:]]
            loss_std = np.std(recent_losses)
            
            # Lower standard deviation = higher stability
            stability = max(0.0, 1.0 - min(loss_std, 1.0))
            return stability
            
        except Exception as e:
            logger.error(f"Error calculating learning stability: {e}")
            return 0.5
    
    def _training_loop(self):
        """Background training loop"""
        while self.running:
            try:
                time.sleep(300)  # Train every 5 minutes
                
                # Perform replay training if enough experiences
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_from_replay()
                
                # Save model periodically
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    self._save_model()
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
    
    def _save_model(self):
        """Save RL model to disk"""
        try:
            model_data = {
                'q_table': {k: v.tolist() for k, v in self.q_table.q_table.items()},
                'visit_counts': {k: v.tolist() for k, v in self.q_table.visit_counts.items()},
                'feature_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in self.feature_stats.items()},
                'training_stats': self.training_stats,
                'parameters': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.epsilon,
                    'epsilon_decay': self.epsilon_decay,
                    'min_epsilon': self.min_epsilon
                },
                'timestamp': datetime.now().isoformat()
            }
            
            model_file = os.path.join(self.data_dir, 'rl_model.json')
            with open(model_file, 'w') as f:
                json.dump(model_data, f, default=str, indent=2)
                
            logger.info("RL model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def _load_model(self):
        """Load RL model from disk"""
        try:
            model_file = os.path.join(self.data_dir, 'rl_model.json')
            if not os.path.exists(model_file):
                return
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Load Q-table
            for state_hash, q_values in model_data.get('q_table', {}).items():
                self.q_table.q_table[state_hash] = np.array(q_values)
            
            for state_hash, visit_counts in model_data.get('visit_counts', {}).items():
                self.q_table.visit_counts[state_hash] = np.array(visit_counts)
            
            # Load feature statistics
            feature_stats = model_data.get('feature_stats', {})
            for key, value in feature_stats.items():
                if isinstance(value, list):
                    self.feature_stats[key] = np.array(value)
            
            # Load training statistics
            self.training_stats.update(model_data.get('training_stats', {}))
            
            # Load parameters
            params = model_data.get('parameters', {})
            self.epsilon = params.get('epsilon', self.epsilon)
            
            logger.info("RL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
    
    def stop(self):
        """Stop the RL agent"""
        self.running = False
        self._save_model()
        logger.info("Reinforcement Learning Agent stopped")
