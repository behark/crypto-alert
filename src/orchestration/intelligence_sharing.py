"""
Cross-Bot Intelligence Sharing System
====================================
Enable bots to learn from each other's experiences and share collective intelligence
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict

from .command_center import CrossBotCommand, CommandResult

logger = logging.getLogger(__name__)

@dataclass
class SharedIntelligence:
    """Shared intelligence data structure"""
    intelligence_id: str
    source_bot_id: str
    intelligence_type: str  # 'forecast', 'pattern', 'regime', 'correlation'
    symbol: str
    timeframe: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    validation_count: int = 0
    success_rate: float = 0.0

@dataclass
class CrossBotPattern:
    """Pattern recognized across multiple bots"""
    pattern_id: str
    pattern_name: str
    contributing_bots: List[str]
    symbol: str
    timeframe: str
    pattern_data: Dict[str, Any]
    success_rate: float
    occurrence_count: int
    last_seen: datetime

class IntelligenceSharing:
    """
    Cross-Bot Intelligence Sharing System
    
    Enables bots to share forecasts, patterns, and market insights to create
    a collective intelligence that improves the entire platform's performance.
    """
    
    def __init__(self, data_dir: str = "data/intelligence_sharing"):
        """Initialize the intelligence sharing system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Shared intelligence storage
        self.shared_intelligence: Dict[str, SharedIntelligence] = {}
        self.cross_bot_patterns: Dict[str, CrossBotPattern] = {}
        self.bot_correlations: Dict[Tuple[str, str], float] = {}  # (bot1, bot2) -> correlation
        
        # Performance tracking
        self.intelligence_performance: Dict[str, List[float]] = defaultdict(list)
        self.pattern_validation: Dict[str, List[bool]] = defaultdict(list)
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_shared_intelligence()
        
        logger.info("Intelligence Sharing System initialized")
    
    def share_forecast(self, bot_id: str, symbol: str, timeframe: str, 
                      forecast_data: Dict[str, Any], confidence: float) -> str:
        """
        Share a forecast with other bots
        
        Args:
            bot_id: ID of the bot sharing the forecast
            symbol: Trading symbol
            timeframe: Chart timeframe
            forecast_data: Forecast details
            confidence: Forecast confidence (0-100)
            
        Returns:
            str: Intelligence ID for tracking
        """
        try:
            with self._lock:
                intelligence_id = f"forecast_{bot_id}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                shared_intel = SharedIntelligence(
                    intelligence_id=intelligence_id,
                    source_bot_id=bot_id,
                    intelligence_type='forecast',
                    symbol=symbol,
                    timeframe=timeframe,
                    data=forecast_data,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                
                self.shared_intelligence[intelligence_id] = shared_intel
                
                # Check for similar forecasts from other bots
                self._check_forecast_consensus(shared_intel)
                
                # Save to disk
                self._save_intelligence(intelligence_id, shared_intel)
                
                logger.info(f"Shared forecast from {bot_id}: {symbol} {timeframe} (confidence: {confidence}%)")
                return intelligence_id
                
        except Exception as e:
            logger.error(f"Failed to share forecast: {e}")
            return ""
    
    def share_pattern(self, bot_id: str, symbol: str, timeframe: str,
                     pattern_name: str, pattern_data: Dict[str, Any]) -> str:
        """
        Share a recognized pattern with other bots
        
        Args:
            bot_id: ID of the bot sharing the pattern
            symbol: Trading symbol
            timeframe: Chart timeframe
            pattern_name: Name of the pattern
            pattern_data: Pattern details
            
        Returns:
            str: Intelligence ID for tracking
        """
        try:
            with self._lock:
                intelligence_id = f"pattern_{bot_id}_{symbol}_{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                shared_intel = SharedIntelligence(
                    intelligence_id=intelligence_id,
                    source_bot_id=bot_id,
                    intelligence_type='pattern',
                    symbol=symbol,
                    timeframe=timeframe,
                    data={
                        'pattern_name': pattern_name,
                        'pattern_data': pattern_data
                    },
                    confidence=pattern_data.get('confidence', 50.0),
                    timestamp=datetime.now()
                )
                
                self.shared_intelligence[intelligence_id] = shared_intel
                
                # Update cross-bot pattern tracking
                self._update_cross_bot_pattern(pattern_name, bot_id, symbol, timeframe, pattern_data)
                
                # Save to disk
                self._save_intelligence(intelligence_id, shared_intel)
                
                logger.info(f"Shared pattern from {bot_id}: {pattern_name} on {symbol} {timeframe}")
                return intelligence_id
                
        except Exception as e:
            logger.error(f"Failed to share pattern: {e}")
            return ""
    
    def get_shared_intelligence(self, intelligence_type: str = None, 
                              symbol: str = None, max_age_hours: int = 24) -> List[SharedIntelligence]:
        """
        Get shared intelligence from other bots
        
        Args:
            intelligence_type: Filter by type ('forecast', 'pattern', etc.)
            symbol: Filter by symbol
            max_age_hours: Maximum age in hours
            
        Returns:
            List of shared intelligence objects
        """
        try:
            with self._lock:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                results = []
                for intel in self.shared_intelligence.values():
                    # Apply filters
                    if intel.timestamp < cutoff_time:
                        continue
                    if intelligence_type and intel.intelligence_type != intelligence_type:
                        continue
                    if symbol and intel.symbol != symbol:
                        continue
                    
                    results.append(intel)
                
                # Sort by confidence and timestamp
                results.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
                return results
                
        except Exception as e:
            logger.error(f"Failed to get shared intelligence: {e}")
            return []
    
    def get_forecast_consensus(self, symbol: str, timeframe: str, 
                             max_age_hours: int = 4) -> Optional[Dict[str, Any]]:
        """
        Get consensus forecast from multiple bots
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            max_age_hours: Maximum age for forecasts
            
        Returns:
            Dict containing consensus forecast or None
        """
        try:
            forecasts = self.get_shared_intelligence(
                intelligence_type='forecast',
                symbol=symbol,
                max_age_hours=max_age_hours
            )
            
            if len(forecasts) < 2:
                return None
            
            # Calculate consensus
            directions = []
            confidences = []
            targets = []
            
            for forecast in forecasts:
                data = forecast.data
                if 'direction' in data:
                    directions.append(data['direction'])
                if 'confidence' in data:
                    confidences.append(data['confidence'])
                if 'target' in data:
                    targets.append(data['target'])
            
            if not directions:
                return None
            
            # Determine consensus direction
            bullish_count = directions.count('bullish')
            bearish_count = directions.count('bearish')
            
            if bullish_count > bearish_count:
                consensus_direction = 'bullish'
                consensus_strength = bullish_count / len(directions)
            elif bearish_count > bullish_count:
                consensus_direction = 'bearish'
                consensus_strength = bearish_count / len(directions)
            else:
                consensus_direction = 'neutral'
                consensus_strength = 0.5
            
            consensus = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': consensus_direction,
                'strength': consensus_strength,
                'contributing_bots': len(forecasts),
                'average_confidence': sum(confidences) / len(confidences) if confidences else 50.0,
                'timestamp': datetime.now()
            }
            
            if targets:
                consensus['average_target'] = sum(targets) / len(targets)
            
            logger.info(f"Generated consensus forecast for {symbol} {timeframe}: {consensus_direction} ({consensus_strength:.2f})")
            return consensus
            
        except Exception as e:
            logger.error(f"Failed to get forecast consensus: {e}")
            return None
    
    def get_cross_bot_patterns(self, symbol: str = None, 
                              min_occurrence_count: int = 2) -> List[CrossBotPattern]:
        """
        Get patterns recognized by multiple bots
        
        Args:
            symbol: Filter by symbol
            min_occurrence_count: Minimum number of occurrences
            
        Returns:
            List of cross-bot patterns
        """
        try:
            with self._lock:
                results = []
                for pattern in self.cross_bot_patterns.values():
                    if symbol and pattern.symbol != symbol:
                        continue
                    if pattern.occurrence_count < min_occurrence_count:
                        continue
                    
                    results.append(pattern)
                
                # Sort by success rate and occurrence count
                results.sort(key=lambda x: (x.success_rate, x.occurrence_count), reverse=True)
                return results
                
        except Exception as e:
            logger.error(f"Failed to get cross-bot patterns: {e}")
            return []
    
    def validate_intelligence(self, intelligence_id: str, success: bool, 
                            outcome_data: Dict[str, Any] = None) -> bool:
        """
        Validate shared intelligence with actual market outcome
        
        Args:
            intelligence_id: ID of intelligence to validate
            success: Whether the intelligence was successful
            outcome_data: Additional outcome data
            
        Returns:
            bool: Success status
        """
        try:
            with self._lock:
                if intelligence_id not in self.shared_intelligence:
                    return False
                
                intel = self.shared_intelligence[intelligence_id]
                intel.validation_count += 1
                
                # Update success rate
                current_successes = intel.success_rate * (intel.validation_count - 1) / 100
                if success:
                    current_successes += 1
                
                intel.success_rate = (current_successes / intel.validation_count) * 100
                
                # Track performance for the source bot
                self.intelligence_performance[intel.source_bot_id].append(intel.success_rate)
                
                # Update pattern validation if applicable
                if intel.intelligence_type == 'pattern':
                    pattern_name = intel.data.get('pattern_name', '')
                    if pattern_name:
                        self.pattern_validation[pattern_name].append(success)
                        
                        # Update cross-bot pattern success rate
                        pattern_key = f"{pattern_name}_{intel.symbol}_{intel.timeframe}"
                        if pattern_key in self.cross_bot_patterns:
                            pattern = self.cross_bot_patterns[pattern_key]
                            validations = self.pattern_validation[pattern_name]
                            pattern.success_rate = (sum(validations) / len(validations)) * 100
                
                # Save updated intelligence
                self._save_intelligence(intelligence_id, intel)
                
                logger.info(f"Validated intelligence {intelligence_id}: {'success' if success else 'failure'}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to validate intelligence: {e}")
            return False
    
    def share_command_results(self, command: CrossBotCommand, 
                            results: List[CommandResult]) -> None:
        """
        Share results from cross-bot command execution
        
        Args:
            command: The executed command
            results: Results from all bots
        """
        try:
            if command.command in ['forecast', 'analyze']:
                # Extract and share intelligence from command results
                for result in results:
                    if result.success and result.result:
                        # Parse result and share as intelligence
                        if command.command == 'forecast':
                            self._extract_and_share_forecast(result, command)
                        elif command.command == 'analyze':
                            self._extract_and_share_analysis(result, command)
            
        except Exception as e:
            logger.error(f"Failed to share command results: {e}")
    
    def get_bot_correlation(self, bot1_id: str, bot2_id: str) -> float:
        """
        Get correlation between two bots' performance
        
        Args:
            bot1_id: First bot ID
            bot2_id: Second bot ID
            
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        try:
            key = tuple(sorted([bot1_id, bot2_id]))
            return self.bot_correlations.get(key, 0.0)
            
        except Exception as e:
            logger.error(f"Failed to get bot correlation: {e}")
            return 0.0
    
    def calculate_bot_correlations(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate correlations between all bot pairs
        
        Returns:
            Dict mapping bot pairs to correlation coefficients
        """
        try:
            from scipy.stats import pearsonr
            import numpy as np
            
            # Get all bots with performance data
            bots_with_data = [
                bot_id for bot_id, performance in self.intelligence_performance.items()
                if len(performance) >= 5  # Minimum data points
            ]
            
            correlations = {}
            
            for i, bot1 in enumerate(bots_with_data):
                for bot2 in bots_with_data[i+1:]:
                    perf1 = self.intelligence_performance[bot1]
                    perf2 = self.intelligence_performance[bot2]
                    
                    # Align data points (take minimum length)
                    min_len = min(len(perf1), len(perf2))
                    if min_len >= 5:
                        correlation, _ = pearsonr(perf1[-min_len:], perf2[-min_len:])
                        if not np.isnan(correlation):
                            key = tuple(sorted([bot1, bot2]))
                            correlations[key] = correlation
                            self.bot_correlations[key] = correlation
            
            logger.info(f"Calculated correlations for {len(correlations)} bot pairs")
            return correlations
            
        except ImportError:
            logger.warning("scipy not available for correlation calculation")
            return {}
        except Exception as e:
            logger.error(f"Failed to calculate bot correlations: {e}")
            return {}
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """
        Get summary of shared intelligence
        
        Returns:
            Dict containing intelligence summary
        """
        try:
            with self._lock:
                total_intelligence = len(self.shared_intelligence)
                
                # Count by type
                type_counts = defaultdict(int)
                for intel in self.shared_intelligence.values():
                    type_counts[intel.intelligence_type] += 1
                
                # Calculate average success rates
                validated_intelligence = [
                    intel for intel in self.shared_intelligence.values()
                    if intel.validation_count > 0
                ]
                
                avg_success_rate = 0.0
                if validated_intelligence:
                    avg_success_rate = sum(intel.success_rate for intel in validated_intelligence) / len(validated_intelligence)
                
                return {
                    'timestamp': datetime.now(),
                    'total_intelligence': total_intelligence,
                    'validated_intelligence': len(validated_intelligence),
                    'average_success_rate': round(avg_success_rate, 2),
                    'intelligence_by_type': dict(type_counts),
                    'cross_bot_patterns': len(self.cross_bot_patterns),
                    'bot_correlations': len(self.bot_correlations),
                    'top_performing_bots': self._get_top_performing_bots()
                }
                
        except Exception as e:
            logger.error(f"Failed to get intelligence summary: {e}")
            return {'error': str(e)}
    
    def _check_forecast_consensus(self, new_forecast: SharedIntelligence):
        """Check if new forecast creates consensus with existing forecasts"""
        try:
            similar_forecasts = [
                intel for intel in self.shared_intelligence.values()
                if (intel.intelligence_type == 'forecast' and
                    intel.symbol == new_forecast.symbol and
                    intel.timeframe == new_forecast.timeframe and
                    intel.source_bot_id != new_forecast.source_bot_id and
                    (datetime.now() - intel.timestamp).total_seconds() < 3600)  # 1 hour
            ]
            
            if len(similar_forecasts) >= 1:
                logger.info(f"Found {len(similar_forecasts)} similar forecasts for consensus analysis")
                
        except Exception as e:
            logger.error(f"Failed to check forecast consensus: {e}")
    
    def _update_cross_bot_pattern(self, pattern_name: str, bot_id: str, 
                                symbol: str, timeframe: str, pattern_data: Dict[str, Any]):
        """Update cross-bot pattern tracking"""
        try:
            pattern_key = f"{pattern_name}_{symbol}_{timeframe}"
            
            if pattern_key in self.cross_bot_patterns:
                pattern = self.cross_bot_patterns[pattern_key]
                if bot_id not in pattern.contributing_bots:
                    pattern.contributing_bots.append(bot_id)
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now()
            else:
                pattern = CrossBotPattern(
                    pattern_id=pattern_key,
                    pattern_name=pattern_name,
                    contributing_bots=[bot_id],
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_data=pattern_data,
                    success_rate=0.0,
                    occurrence_count=1,
                    last_seen=datetime.now()
                )
                self.cross_bot_patterns[pattern_key] = pattern
            
        except Exception as e:
            logger.error(f"Failed to update cross-bot pattern: {e}")
    
    def _extract_and_share_forecast(self, result: CommandResult, command: CrossBotCommand):
        """Extract forecast data from command result and share it"""
        # Implementation would parse the forecast result and share it
        pass
    
    def _extract_and_share_analysis(self, result: CommandResult, command: CrossBotCommand):
        """Extract analysis data from command result and share it"""
        # Implementation would parse the analysis result and share it
        pass
    
    def _get_top_performing_bots(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing bots by intelligence success rate"""
        try:
            bot_performance = []
            for bot_id, performance_data in self.intelligence_performance.items():
                if len(performance_data) >= 3:  # Minimum data points
                    avg_performance = sum(performance_data) / len(performance_data)
                    bot_performance.append({
                        'bot_id': bot_id,
                        'average_success_rate': round(avg_performance, 2),
                        'intelligence_count': len(performance_data)
                    })
            
            # Sort by success rate
            bot_performance.sort(key=lambda x: x['average_success_rate'], reverse=True)
            return bot_performance[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top performing bots: {e}")
            return []
    
    def _save_intelligence(self, intelligence_id: str, intelligence: SharedIntelligence):
        """Save intelligence to disk"""
        try:
            file_path = self.data_dir / f"{intelligence_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(intelligence)
                data['timestamp'] = intelligence.timestamp.isoformat()
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save intelligence {intelligence_id}: {e}")
    
    def _load_shared_intelligence(self):
        """Load shared intelligence from disk"""
        try:
            if not self.data_dir.exists():
                return
            
            loaded_count = 0
            for file_path in self.data_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Convert timestamp back to datetime
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    intelligence = SharedIntelligence(**data)
                    self.shared_intelligence[intelligence.intelligence_id] = intelligence
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load intelligence from {file_path}: {e}")
            
            logger.info(f"Loaded {loaded_count} shared intelligence entries")
            
        except Exception as e:
            logger.error(f"Failed to load shared intelligence: {e}")


# Global instance
_intelligence_sharing = None

def get_intelligence_sharing() -> IntelligenceSharing:
    """Get global intelligence sharing instance"""
    global _intelligence_sharing
    if _intelligence_sharing is None:
        _intelligence_sharing = IntelligenceSharing()
    return _intelligence_sharing
