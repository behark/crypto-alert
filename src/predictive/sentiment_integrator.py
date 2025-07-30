"""
Market Sentiment & External Data Integration
==========================================
On-chain data, social sentiment, and funding rate analysis for predictive intelligence
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SentimentType(Enum):
    """Sentiment type enumeration"""
    SOCIAL = "social"
    ON_CHAIN = "on_chain"
    FUNDING = "funding"
    NEWS = "news"
    WHALE = "whale"
    TECHNICAL = "technical"

class SentimentPolarity(Enum):
    """Sentiment polarity enumeration"""
    EXTREMELY_BEARISH = "extremely_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    EXTREMELY_BULLISH = "extremely_bullish"

@dataclass
class SentimentData:
    """Individual sentiment data point"""
    data_id: str
    symbol: str
    sentiment_type: SentimentType
    polarity: SentimentPolarity
    score: float  # -1 to 1
    confidence: float
    source: str
    raw_data: Dict[str, Any]
    timestamp: datetime
    weight: float

@dataclass
class CompositeSentiment:
    """Composite sentiment analysis"""
    symbol: str
    overall_polarity: SentimentPolarity
    overall_score: float
    confidence: float
    sentiment_breakdown: Dict[str, float]
    key_drivers: List[str]
    sentiment_shift: float  # Change from previous
    prediction_bias: str  # bullish/bearish/neutral
    timestamp: datetime

@dataclass
class OnChainMetrics:
    """On-chain analysis metrics"""
    symbol: str
    active_addresses: int
    transaction_volume: float
    whale_movements: List[Dict[str, Any]]
    exchange_flows: Dict[str, float]  # inflow/outflow
    holder_distribution: Dict[str, float]
    network_growth: float
    timestamp: datetime

@dataclass
class FundingAnalysis:
    """Funding rate analysis"""
    symbol: str
    current_funding_rate: float
    funding_trend: str  # increasing/decreasing/stable
    predicted_funding_shift: float
    long_short_ratio: float
    open_interest: float
    liquidation_levels: Dict[str, List[float]]
    bias_signal: str  # bullish/bearish/neutral
    timestamp: datetime

class SentimentIntegrator:
    """
    Market Sentiment & External Data Integration
    
    Integrates multiple sentiment sources including social media, on-chain data,
    funding rates, and news to generate predictive sentiment intelligence.
    """
    
    def __init__(self, data_dir: str = "data/sentiment"):
        """Initialize the sentiment integrator"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentiment data storage
        self.sentiment_data: Dict[str, List[SentimentData]] = {}
        self.composite_sentiment: Dict[str, CompositeSentiment] = {}
        self.on_chain_metrics: Dict[str, OnChainMetrics] = {}
        self.funding_analysis: Dict[str, FundingAnalysis] = {}
        
        # Sentiment weights by type
        self.sentiment_weights = {
            SentimentType.ON_CHAIN: 0.25,
            SentimentType.FUNDING: 0.20,
            SentimentType.SOCIAL: 0.15,
            SentimentType.WHALE: 0.15,
            SentimentType.NEWS: 0.15,
            SentimentType.TECHNICAL: 0.10
        }
        
        # Configuration
        self.sentiment_history_limit = 1000
        self.sentiment_decay_hours = 24  # How long sentiment data remains relevant
        
        # External data sources (mock for now)
        self.data_sources = {
            'twitter_api': {'enabled': False, 'weight': 0.3},
            'reddit_api': {'enabled': False, 'weight': 0.2},
            'blockchain_api': {'enabled': False, 'weight': 0.4},
            'funding_api': {'enabled': False, 'weight': 0.5},
            'news_api': {'enabled': False, 'weight': 0.3}
        }
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Sentiment Integrator initialized")
    
    def add_sentiment_data(self, symbol: str, sentiment_type: SentimentType,
                          score: float, source: str, raw_data: Dict[str, Any] = None,
                          confidence: float = 0.8) -> str:
        """
        Add sentiment data point
        
        Args:
            symbol: Trading symbol
            sentiment_type: Type of sentiment data
            score: Sentiment score (-1 to 1)
            source: Data source identifier
            raw_data: Raw data from source
            confidence: Confidence in the data
            
        Returns:
            str: Data ID
        """
        try:
            data_id = f"sentiment_{symbol}_{sentiment_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine polarity from score
            if score <= -0.6:
                polarity = SentimentPolarity.EXTREMELY_BEARISH
            elif score <= -0.2:
                polarity = SentimentPolarity.BEARISH
            elif score >= 0.6:
                polarity = SentimentPolarity.EXTREMELY_BULLISH
            elif score >= 0.2:
                polarity = SentimentPolarity.BULLISH
            else:
                polarity = SentimentPolarity.NEUTRAL
            
            # Calculate weight based on type and confidence
            base_weight = self.sentiment_weights.get(sentiment_type, 0.1)
            weight = base_weight * confidence
            
            # Create sentiment data
            sentiment_data = SentimentData(
                data_id=data_id,
                symbol=symbol,
                sentiment_type=sentiment_type,
                polarity=polarity,
                score=np.clip(score, -1, 1),
                confidence=confidence,
                source=source,
                raw_data=raw_data or {},
                timestamp=datetime.now(),
                weight=weight
            )
            
            # Add to storage
            if symbol not in self.sentiment_data:
                self.sentiment_data[symbol] = []
            
            self.sentiment_data[symbol].append(sentiment_data)
            
            # Keep only recent data
            cutoff_time = datetime.now() - timedelta(hours=self.sentiment_decay_hours)
            self.sentiment_data[symbol] = [
                data for data in self.sentiment_data[symbol]
                if data.timestamp > cutoff_time
            ]
            
            # Update composite sentiment
            self._update_composite_sentiment(symbol)
            
            logger.info(f"Added sentiment data: {data_id} ({sentiment_type.value}: {score:.2f})")
            return data_id
            
        except Exception as e:
            logger.error(f"Failed to add sentiment data: {e}")
            return ""
    
    def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """
        Get current sentiment score for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing sentiment analysis
        """
        try:
            if symbol not in self.composite_sentiment:
                return {
                    'overall_score': 0.0,
                    'polarity': 'neutral',
                    'confidence': 0.0,
                    'prediction_bias': 'neutral',
                    'reasoning': 'No sentiment data available'
                }
            
            composite = self.composite_sentiment[symbol]
            
            return {
                'overall_score': composite.overall_score,
                'polarity': composite.overall_polarity.value,
                'confidence': composite.confidence,
                'sentiment_breakdown': composite.sentiment_breakdown,
                'key_drivers': composite.key_drivers,
                'sentiment_shift': composite.sentiment_shift,
                'prediction_bias': composite.prediction_bias,
                'data_freshness': (datetime.now() - composite.timestamp).total_seconds() / 3600,  # Hours
                'timestamp': composite.timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to get sentiment score: {e}")
            return {'error': str(e)}
    
    def analyze_on_chain_data(self, symbol: str, blockchain_data: Dict[str, Any]) -> OnChainMetrics:
        """
        Analyze on-chain data for sentiment
        
        Args:
            symbol: Trading symbol
            blockchain_data: Raw blockchain data
            
        Returns:
            OnChainMetrics: On-chain analysis
        """
        try:
            # Extract metrics from blockchain data (simplified)
            active_addresses = blockchain_data.get('active_addresses', 0)
            transaction_volume = blockchain_data.get('transaction_volume', 0.0)
            
            # Analyze whale movements
            whale_movements = []
            large_transactions = blockchain_data.get('large_transactions', [])
            for tx in large_transactions:
                if tx.get('amount', 0) > 1000000:  # $1M+ transactions
                    whale_movements.append({
                        'amount': tx['amount'],
                        'direction': tx.get('direction', 'unknown'),
                        'timestamp': tx.get('timestamp', datetime.now())
                    })
            
            # Exchange flows
            exchange_flows = {
                'inflow': blockchain_data.get('exchange_inflow', 0.0),
                'outflow': blockchain_data.get('exchange_outflow', 0.0),
                'net_flow': blockchain_data.get('exchange_inflow', 0.0) - blockchain_data.get('exchange_outflow', 0.0)
            }
            
            # Holder distribution
            holder_distribution = blockchain_data.get('holder_distribution', {
                'whales': 0.1,
                'large_holders': 0.2,
                'retail': 0.7
            })
            
            # Network growth
            current_addresses = active_addresses
            previous_addresses = blockchain_data.get('previous_active_addresses', current_addresses)
            network_growth = (current_addresses - previous_addresses) / previous_addresses if previous_addresses > 0 else 0.0
            
            # Create metrics
            metrics = OnChainMetrics(
                symbol=symbol,
                active_addresses=active_addresses,
                transaction_volume=transaction_volume,
                whale_movements=whale_movements,
                exchange_flows=exchange_flows,
                holder_distribution=holder_distribution,
                network_growth=network_growth,
                timestamp=datetime.now()
            )
            
            # Store metrics
            self.on_chain_metrics[symbol] = metrics
            
            # Generate sentiment from on-chain data
            on_chain_sentiment = self._calculate_on_chain_sentiment(metrics)
            self.add_sentiment_data(
                symbol=symbol,
                sentiment_type=SentimentType.ON_CHAIN,
                score=on_chain_sentiment,
                source='blockchain_analysis',
                raw_data=blockchain_data,
                confidence=0.9
            )
            
            logger.info(f"Analyzed on-chain data for {symbol}: sentiment {on_chain_sentiment:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze on-chain data: {e}")
            return None
    
    def analyze_funding_rates(self, symbol: str, funding_data: Dict[str, Any]) -> FundingAnalysis:
        """
        Analyze funding rates for sentiment
        
        Args:
            symbol: Trading symbol
            funding_data: Funding rate data
            
        Returns:
            FundingAnalysis: Funding analysis
        """
        try:
            current_funding_rate = funding_data.get('current_funding_rate', 0.0)
            historical_rates = funding_data.get('historical_rates', [current_funding_rate])
            
            # Determine funding trend
            if len(historical_rates) >= 3:
                recent_trend = np.polyfit(range(len(historical_rates[-3:])), historical_rates[-3:], 1)[0]
                if recent_trend > 0.0001:
                    funding_trend = 'increasing'
                elif recent_trend < -0.0001:
                    funding_trend = 'decreasing'
                else:
                    funding_trend = 'stable'
            else:
                funding_trend = 'stable'
            
            # Predict funding shift
            predicted_funding_shift = 0.0
            if funding_trend == 'increasing' and current_funding_rate > 0.01:  # 1% funding
                predicted_funding_shift = -0.005  # Expect decrease
            elif funding_trend == 'decreasing' and current_funding_rate < -0.01:
                predicted_funding_shift = 0.005  # Expect increase
            
            # Long/short ratio and open interest
            long_short_ratio = funding_data.get('long_short_ratio', 1.0)
            open_interest = funding_data.get('open_interest', 0.0)
            
            # Liquidation levels
            liquidation_levels = {
                'long_liquidations': funding_data.get('long_liquidation_levels', []),
                'short_liquidations': funding_data.get('short_liquidation_levels', [])
            }
            
            # Generate bias signal
            bias_signal = self._calculate_funding_bias(current_funding_rate, funding_trend, long_short_ratio)
            
            # Create analysis
            analysis = FundingAnalysis(
                symbol=symbol,
                current_funding_rate=current_funding_rate,
                funding_trend=funding_trend,
                predicted_funding_shift=predicted_funding_shift,
                long_short_ratio=long_short_ratio,
                open_interest=open_interest,
                liquidation_levels=liquidation_levels,
                bias_signal=bias_signal,
                timestamp=datetime.now()
            )
            
            # Store analysis
            self.funding_analysis[symbol] = analysis
            
            # Generate sentiment from funding data
            funding_sentiment = self._calculate_funding_sentiment(analysis)
            self.add_sentiment_data(
                symbol=symbol,
                sentiment_type=SentimentType.FUNDING,
                score=funding_sentiment,
                source='funding_analysis',
                raw_data=funding_data,
                confidence=0.85
            )
            
            logger.info(f"Analyzed funding rates for {symbol}: bias {bias_signal}, sentiment {funding_sentiment:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze funding rates: {e}")
            return None
    
    def get_external_sentiment_index(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get external sentiment index for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dict containing sentiment index
        """
        try:
            if not symbols:
                return {'index_score': 0.0, 'index_polarity': 'neutral', 'symbol_count': 0}
            
            # Collect sentiment scores
            sentiment_scores = []
            symbol_sentiments = {}
            
            for symbol in symbols:
                sentiment = self.get_sentiment_score(symbol)
                if 'overall_score' in sentiment:
                    sentiment_scores.append(sentiment['overall_score'])
                    symbol_sentiments[symbol] = sentiment
            
            if not sentiment_scores:
                return {'index_score': 0.0, 'index_polarity': 'neutral', 'symbol_count': 0}
            
            # Calculate index metrics
            index_score = np.mean(sentiment_scores)
            index_std = np.std(sentiment_scores)
            
            # Determine index polarity
            if index_score >= 0.4:
                index_polarity = 'bullish'
            elif index_score <= -0.4:
                index_polarity = 'bearish'
            else:
                index_polarity = 'neutral'
            
            # Identify sentiment leaders and laggards
            sorted_sentiments = sorted(symbol_sentiments.items(), key=lambda x: x[1]['overall_score'], reverse=True)
            sentiment_leaders = [item[0] for item in sorted_sentiments[:3]]
            sentiment_laggards = [item[0] for item in sorted_sentiments[-3:]]
            
            return {
                'index_score': index_score,
                'index_polarity': index_polarity,
                'index_volatility': index_std,
                'symbol_count': len(sentiment_scores),
                'sentiment_leaders': sentiment_leaders,
                'sentiment_laggards': sentiment_laggards,
                'symbol_breakdown': symbol_sentiments,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get external sentiment index: {e}")
            return {'error': str(e)}
    
    def get_sentiment_status(self) -> Dict[str, Any]:
        """Get sentiment integrator status"""
        try:
            total_sentiment_points = sum(len(data) for data in self.sentiment_data.values())
            active_symbols = len(self.sentiment_data)
            
            # Recent activity
            recent_data = 0
            for symbol_data in self.sentiment_data.values():
                recent_data += len([d for d in symbol_data 
                                 if d.timestamp > datetime.now() - timedelta(hours=1)])
            
            # Data source status
            enabled_sources = sum(1 for source in self.data_sources.values() if source['enabled'])
            
            return {
                'active_symbols': active_symbols,
                'total_sentiment_points': total_sentiment_points,
                'recent_data_1h': recent_data,
                'composite_sentiments': len(self.composite_sentiment),
                'on_chain_metrics': len(self.on_chain_metrics),
                'funding_analyses': len(self.funding_analysis),
                'enabled_data_sources': enabled_sources,
                'total_data_sources': len(self.data_sources),
                'monitoring_active': not self._should_stop,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get sentiment status: {e}")
            return {'error': str(e)}
    
    def _update_composite_sentiment(self, symbol: str):
        """Update composite sentiment for symbol"""
        try:
            if symbol not in self.sentiment_data or not self.sentiment_data[symbol]:
                return
            
            sentiment_data = self.sentiment_data[symbol]
            
            # Calculate weighted sentiment scores by type
            sentiment_breakdown = {}
            total_weighted_score = 0.0
            total_weight = 0.0
            key_drivers = []
            
            for sentiment_type in SentimentType:
                type_data = [d for d in sentiment_data if d.sentiment_type == sentiment_type]
                if type_data:
                    # Use most recent data points (last 10)
                    recent_data = sorted(type_data, key=lambda x: x.timestamp, reverse=True)[:10]
                    
                    # Calculate weighted average for this type
                    type_scores = []
                    type_weights = []
                    
                    for data in recent_data:
                        # Apply time decay
                        hours_old = (datetime.now() - data.timestamp).total_seconds() / 3600
                        time_decay = max(0.1, 1.0 - (hours_old / self.sentiment_decay_hours))
                        
                        adjusted_weight = data.weight * time_decay
                        type_scores.append(data.score)
                        type_weights.append(adjusted_weight)
                    
                    if type_weights:
                        type_avg_score = np.average(type_scores, weights=type_weights)
                        type_total_weight = sum(type_weights)
                        
                        sentiment_breakdown[sentiment_type.value] = type_avg_score
                        total_weighted_score += type_avg_score * type_total_weight
                        total_weight += type_total_weight
                        
                        # Add to key drivers if significant
                        if abs(type_avg_score) > 0.3:
                            key_drivers.append(f"{sentiment_type.value}: {type_avg_score:.2f}")
            
            # Calculate overall sentiment
            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
            else:
                overall_score = 0.0
            
            # Determine overall polarity
            if overall_score <= -0.6:
                overall_polarity = SentimentPolarity.EXTREMELY_BEARISH
            elif overall_score <= -0.2:
                overall_polarity = SentimentPolarity.BEARISH
            elif overall_score >= 0.6:
                overall_polarity = SentimentPolarity.EXTREMELY_BULLISH
            elif overall_score >= 0.2:
                overall_polarity = SentimentPolarity.BULLISH
            else:
                overall_polarity = SentimentPolarity.NEUTRAL
            
            # Calculate confidence based on data quantity and agreement
            data_quantity_factor = min(len(sentiment_data) / 20, 1.0)  # More data = higher confidence
            
            # Agreement factor (lower standard deviation = higher confidence)
            if len(sentiment_breakdown) > 1:
                score_std = np.std(list(sentiment_breakdown.values()))
                agreement_factor = max(0.3, 1.0 - score_std)
            else:
                agreement_factor = 0.5
            
            confidence = (data_quantity_factor * 0.6 + agreement_factor * 0.4)
            
            # Calculate sentiment shift
            previous_composite = self.composite_sentiment.get(symbol)
            if previous_composite:
                sentiment_shift = overall_score - previous_composite.overall_score
            else:
                sentiment_shift = 0.0
            
            # Determine prediction bias
            if overall_score > 0.3:
                prediction_bias = 'bullish'
            elif overall_score < -0.3:
                prediction_bias = 'bearish'
            else:
                prediction_bias = 'neutral'
            
            # Create composite sentiment
            composite = CompositeSentiment(
                symbol=symbol,
                overall_polarity=overall_polarity,
                overall_score=overall_score,
                confidence=confidence,
                sentiment_breakdown=sentiment_breakdown,
                key_drivers=key_drivers,
                sentiment_shift=sentiment_shift,
                prediction_bias=prediction_bias,
                timestamp=datetime.now()
            )
            
            self.composite_sentiment[symbol] = composite
            
        except Exception as e:
            logger.error(f"Failed to update composite sentiment: {e}")
    
    def _calculate_on_chain_sentiment(self, metrics: OnChainMetrics) -> float:
        """Calculate sentiment from on-chain metrics"""
        try:
            sentiment_factors = []
            
            # Network growth factor
            if metrics.network_growth > 0.05:  # 5% growth
                sentiment_factors.append(0.3)
            elif metrics.network_growth < -0.05:
                sentiment_factors.append(-0.3)
            else:
                sentiment_factors.append(0.0)
            
            # Exchange flow factor
            net_flow = metrics.exchange_flows.get('net_flow', 0)
            if net_flow < -1000000:  # Large outflow (bullish)
                sentiment_factors.append(0.4)
            elif net_flow > 1000000:  # Large inflow (bearish)
                sentiment_factors.append(-0.4)
            else:
                sentiment_factors.append(0.0)
            
            # Whale movement factor
            whale_sentiment = 0.0
            for movement in metrics.whale_movements:
                if movement.get('direction') == 'accumulation':
                    whale_sentiment += 0.1
                elif movement.get('direction') == 'distribution':
                    whale_sentiment -= 0.1
            
            sentiment_factors.append(np.clip(whale_sentiment, -0.3, 0.3))
            
            # Calculate overall sentiment
            overall_sentiment = np.mean(sentiment_factors)
            return float(np.clip(overall_sentiment, -1, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate on-chain sentiment: {e}")
            return 0.0
    
    def _calculate_funding_bias(self, funding_rate: float, trend: str, long_short_ratio: float) -> str:
        """Calculate bias from funding data"""
        try:
            # Extreme funding rates suggest contrarian opportunities
            if funding_rate > 0.02:  # 2% funding (very high)
                return 'bearish'  # Longs paying too much
            elif funding_rate < -0.02:  # -2% funding (very low)
                return 'bullish'  # Shorts paying too much
            
            # Long/short ratio analysis
            if long_short_ratio > 3.0:  # Too many longs
                return 'bearish'
            elif long_short_ratio < 0.33:  # Too many shorts
                return 'bullish'
            
            # Trend analysis
            if trend == 'increasing' and funding_rate > 0.005:
                return 'bearish'  # Increasing positive funding
            elif trend == 'decreasing' and funding_rate < -0.005:
                return 'bullish'  # Decreasing negative funding
            
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Failed to calculate funding bias: {e}")
            return 'neutral'
    
    def _calculate_funding_sentiment(self, analysis: FundingAnalysis) -> float:
        """Calculate sentiment from funding analysis"""
        try:
            sentiment_score = 0.0
            
            # Funding rate component
            if analysis.current_funding_rate > 0.01:
                sentiment_score -= 0.4  # High positive funding = bearish
            elif analysis.current_funding_rate < -0.01:
                sentiment_score += 0.4  # High negative funding = bullish
            
            # Long/short ratio component
            if analysis.long_short_ratio > 2.0:
                sentiment_score -= 0.3  # Too many longs = bearish
            elif analysis.long_short_ratio < 0.5:
                sentiment_score += 0.3  # Too many shorts = bullish
            
            # Trend component
            if analysis.funding_trend == 'increasing' and analysis.current_funding_rate > 0:
                sentiment_score -= 0.2
            elif analysis.funding_trend == 'decreasing' and analysis.current_funding_rate < 0:
                sentiment_score += 0.2
            
            return float(np.clip(sentiment_score, -1, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate funding sentiment: {e}")
            return 0.0
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="SentimentMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started sentiment monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Update composite sentiments for all symbols
                for symbol in list(self.sentiment_data.keys()):
                    self._update_composite_sentiment(symbol)
                
                # Clean up old data
                cutoff_time = datetime.now() - timedelta(hours=self.sentiment_decay_hours * 2)
                for symbol in list(self.sentiment_data.keys()):
                    self.sentiment_data[symbol] = [
                        data for data in self.sentiment_data[symbol]
                        if data.timestamp > cutoff_time
                    ]
                    
                    # Remove empty symbol entries
                    if not self.sentiment_data[symbol]:
                        del self.sentiment_data[symbol]
                
                # Sleep
                threading.Event().wait(300.0)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in sentiment monitoring: {e}")
    
    def stop(self):
        """Stop the sentiment integrator"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Sentiment Integrator stopped")


# Global instance
_sentiment_integrator = None

def get_sentiment_integrator() -> SentimentIntegrator:
    """Get global sentiment integrator instance"""
    global _sentiment_integrator
    if _sentiment_integrator is None:
        _sentiment_integrator = SentimentIntegrator()
    return _sentiment_integrator
