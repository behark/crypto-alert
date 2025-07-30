"""
Environmental Risk Intelligence Engine
=====================================
Macro-level and systemic risk intelligence for predictive trading decisions
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import json
import aiohttp
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Environmental risk level enumeration"""
    GREEN = "green"      # Low risk, normal operations
    YELLOW = "yellow"    # Moderate risk, increased caution
    ORANGE = "orange"    # High risk, reduced exposure
    RED = "red"          # Critical risk, defensive mode

class RiskCategory(Enum):
    """Risk category enumeration"""
    MACRO_ECONOMIC = "macro_economic"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    ON_CHAIN = "on_chain"
    SYSTEMIC = "systemic"

@dataclass
class MacroIndicator:
    """Macro economic indicator"""
    indicator_id: str
    name: str
    current_value: float
    historical_avg: float
    percentile: float  # 0-100 percentile vs history
    trend_direction: str  # "up", "down", "sideways"
    risk_impact: float  # -1 to 1, negative = bearish
    last_updated: datetime

@dataclass
class RiskEvent:
    """Risk event detection"""
    event_id: str
    category: RiskCategory
    severity: RiskLevel
    title: str
    description: str
    impact_score: float  # 0-1
    confidence: float    # 0-1
    time_horizon: timedelta
    affected_assets: List[str]
    mitigation_actions: List[str]
    detected_at: datetime
    expires_at: datetime

@dataclass
class OnChainMetrics:
    """On-chain risk metrics"""
    symbol: str
    exchange_netflow: float  # Positive = inflow to exchanges
    whale_activity_score: float  # 0-1
    network_congestion: float   # 0-1
    funding_rate_extremes: float  # Absolute funding rate
    liquidation_cascade_risk: float  # 0-1
    stablecoin_dominance: float  # USDT/USDC dominance
    timestamp: datetime

@dataclass
class NewsSignal:
    """News sentiment signal"""
    signal_id: str
    source: str
    headline: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0-1
    risk_keywords: List[str]
    impact_estimate: float  # 0-1
    published_at: datetime

@dataclass
class EnvironmentalRiskAssessment:
    """Complete environmental risk assessment"""
    overall_risk_level: RiskLevel
    risk_score: float  # 0-1
    confidence: float  # 0-1
    active_events: List[RiskEvent]
    macro_indicators: Dict[str, MacroIndicator]
    on_chain_metrics: Dict[str, OnChainMetrics]
    news_signals: List[NewsSignal]
    recommended_actions: List[str]
    confidence_multiplier: float  # Trading confidence adjustment
    position_size_multiplier: float  # Position sizing adjustment
    timestamp: datetime

class EnvironmentalRiskEngine:
    """
    Environmental Risk Intelligence Engine
    
    Monitors macro-economic indicators, on-chain metrics, news sentiment,
    and geopolitical events to assess systemic risk and adjust trading behavior.
    """
    
    def __init__(self, data_dir: str = "data/environmental_risk"):
        """Initialize the environmental risk engine"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk state
        self.current_assessment: Optional[EnvironmentalRiskAssessment] = None
        self.risk_events: List[RiskEvent] = []
        self.macro_indicators: Dict[str, MacroIndicator] = {}
        self.on_chain_metrics: Dict[str, OnChainMetrics] = {}
        self.news_signals: List[NewsSignal] = []
        
        # Risk thresholds
        self.risk_thresholds = {
            'green_max': 0.3,
            'yellow_max': 0.5,
            'orange_max': 0.8,
            'red_min': 0.8
        }
        
        # Macro indicator configurations
        self.macro_config = {
            'VIX': {'weight': 0.25, 'invert': True},  # High VIX = high risk
            'DXY': {'weight': 0.15, 'threshold': 105},  # Strong dollar stress
            'US10Y': {'weight': 0.2, 'threshold': 4.5},  # High yields = risk
            'GOLD': {'weight': 0.1, 'safe_haven': True},
            'OIL': {'weight': 0.1, 'volatility_factor': True},
            'SPY': {'weight': 0.2, 'correlation_factor': True}
        }
        
        # News risk keywords
        self.risk_keywords = {
            'critical': ['hack', 'exploit', 'crash', 'collapse', 'emergency', 'halt', 'suspend'],
            'high': ['regulation', 'ban', 'investigation', 'lawsuit', 'seizure', 'raid'],
            'medium': ['concern', 'warning', 'volatility', 'uncertainty', 'pressure'],
            'low': ['caution', 'watch', 'monitor', 'review']
        }
        
        # Monitoring thread
        self._monitor_thread = None
        self._should_stop = False
        
        self._start_monitoring()
        
        logger.info("Environmental Risk Engine initialized")
    
    async def assess_environmental_risk(self, symbols: List[str] = None) -> EnvironmentalRiskAssessment:
        """
        Perform comprehensive environmental risk assessment
        
        Args:
            symbols: List of symbols to assess (default: major cryptos)
            
        Returns:
            EnvironmentalRiskAssessment: Complete risk analysis
        """
        try:
            if symbols is None:
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            
            # 1. Update macro indicators
            await self._update_macro_indicators()
            
            # 2. Update on-chain metrics
            await self._update_on_chain_metrics(symbols)
            
            # 3. Scan news and events
            await self._scan_news_events()
            
            # 4. Calculate composite risk score
            risk_score = self._calculate_composite_risk_score()
            
            # 5. Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # 6. Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_level, risk_score)
            
            # 7. Calculate trading adjustments
            confidence_multiplier = self._calculate_confidence_multiplier(risk_level, risk_score)
            position_multiplier = self._calculate_position_multiplier(risk_level, risk_score)
            
            # 8. Filter active events
            active_events = [e for e in self.risk_events if e.expires_at > datetime.now()]
            
            # Create assessment
            assessment = EnvironmentalRiskAssessment(
                overall_risk_level=risk_level,
                risk_score=risk_score,
                confidence=self._calculate_assessment_confidence(),
                active_events=active_events,
                macro_indicators=self.macro_indicators.copy(),
                on_chain_metrics=self.on_chain_metrics.copy(),
                news_signals=self.news_signals[-50:],  # Recent signals
                recommended_actions=recommendations,
                confidence_multiplier=confidence_multiplier,
                position_size_multiplier=position_multiplier,
                timestamp=datetime.now()
            )
            
            self.current_assessment = assessment
            
            logger.info(f"Environmental risk assessment: {risk_level.value} (score: {risk_score:.2f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess environmental risk: {e}")
            # Return safe default assessment
            return EnvironmentalRiskAssessment(
                overall_risk_level=RiskLevel.YELLOW,
                risk_score=0.5,
                confidence=0.3,
                active_events=[],
                macro_indicators={},
                on_chain_metrics={},
                news_signals=[],
                recommended_actions=["Monitor market conditions"],
                confidence_multiplier=0.8,
                position_size_multiplier=0.8,
                timestamp=datetime.now()
            )
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk engine status"""
        try:
            if not self.current_assessment:
                return {'error': 'No risk assessment available'}
            
            assessment = self.current_assessment
            
            # Risk breakdown by category
            risk_breakdown = defaultdict(list)
            for event in assessment.active_events:
                risk_breakdown[event.category.value].append({
                    'severity': event.severity.value,
                    'impact': event.impact_score,
                    'title': event.title
                })
            
            # Macro indicator summary
            macro_summary = {}
            for indicator_id, indicator in assessment.macro_indicators.items():
                macro_summary[indicator_id] = {
                    'value': indicator.current_value,
                    'percentile': indicator.percentile,
                    'trend': indicator.trend_direction,
                    'risk_impact': indicator.risk_impact
                }
            
            return {
                'overall_risk_level': assessment.overall_risk_level.value,
                'risk_score': assessment.risk_score,
                'confidence': assessment.confidence,
                'active_events_count': len(assessment.active_events),
                'confidence_multiplier': assessment.confidence_multiplier,
                'position_multiplier': assessment.position_size_multiplier,
                'risk_breakdown': dict(risk_breakdown),
                'macro_summary': macro_summary,
                'news_signals_24h': len([s for s in assessment.news_signals 
                                       if s.published_at > datetime.now() - timedelta(hours=24)]),
                'last_updated': assessment.timestamp,
                'monitoring_active': not self._should_stop
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk status: {e}")
            return {'error': str(e)}
    
    def get_trading_adjustments(self) -> Dict[str, float]:
        """Get current trading adjustments based on environmental risk"""
        try:
            if not self.current_assessment:
                return {
                    'confidence_multiplier': 1.0,
                    'position_multiplier': 1.0,
                    'max_drawdown_adjustment': 1.0
                }
            
            assessment = self.current_assessment
            
            # Additional adjustments based on specific risks
            max_drawdown_adjustment = 1.0
            if assessment.overall_risk_level == RiskLevel.RED:
                max_drawdown_adjustment = 0.5  # Tighter stops
            elif assessment.overall_risk_level == RiskLevel.ORANGE:
                max_drawdown_adjustment = 0.7
            elif assessment.overall_risk_level == RiskLevel.YELLOW:
                max_drawdown_adjustment = 0.85
            
            return {
                'confidence_multiplier': assessment.confidence_multiplier,
                'position_multiplier': assessment.position_size_multiplier,
                'max_drawdown_adjustment': max_drawdown_adjustment,
                'risk_level': assessment.overall_risk_level.value,
                'risk_score': assessment.risk_score
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading adjustments: {e}")
            return {
                'confidence_multiplier': 0.8,
                'position_multiplier': 0.8,
                'max_drawdown_adjustment': 0.8
            }
    
    async def _update_macro_indicators(self):
        """Update macro economic indicators"""
        try:
            # Simulated macro data (in production, connect to real APIs)
            mock_data = {
                'VIX': {'value': 18.5, 'avg': 20.0},
                'DXY': {'value': 103.2, 'avg': 100.0},
                'US10Y': {'value': 4.2, 'avg': 3.5},
                'GOLD': {'value': 2020.0, 'avg': 1950.0},
                'OIL': {'value': 82.5, 'avg': 80.0},
                'SPY': {'value': 4450.0, 'avg': 4400.0}
            }
            
            for indicator_id, data in mock_data.items():
                current_value = data['value']
                historical_avg = data['avg']
                
                # Calculate percentile (simplified)
                deviation = (current_value - historical_avg) / historical_avg
                percentile = 50 + (deviation * 30)  # Rough percentile
                percentile = max(0, min(100, percentile))
                
                # Determine trend (simplified)
                trend_direction = "up" if deviation > 0.02 else "down" if deviation < -0.02 else "sideways"
                
                # Calculate risk impact
                config = self.macro_config.get(indicator_id, {})
                if config.get('invert'):
                    risk_impact = deviation  # High VIX = high risk
                else:
                    risk_impact = -deviation if config.get('safe_haven') else deviation
                
                # Normalize risk impact
                risk_impact = max(-1, min(1, risk_impact))
                
                indicator = MacroIndicator(
                    indicator_id=indicator_id,
                    name=indicator_id,
                    current_value=current_value,
                    historical_avg=historical_avg,
                    percentile=percentile,
                    trend_direction=trend_direction,
                    risk_impact=risk_impact,
                    last_updated=datetime.now()
                )
                
                self.macro_indicators[indicator_id] = indicator
            
            logger.debug("Updated macro indicators")
            
        except Exception as e:
            logger.error(f"Failed to update macro indicators: {e}")
    
    async def _update_on_chain_metrics(self, symbols: List[str]):
        """Update on-chain metrics"""
        try:
            # Simulated on-chain data (in production, connect to real APIs)
            for symbol in symbols:
                # Mock on-chain metrics
                metrics = OnChainMetrics(
                    symbol=symbol,
                    exchange_netflow=np.random.normal(0, 1000),  # Random netflow
                    whale_activity_score=np.random.uniform(0, 1),
                    network_congestion=np.random.uniform(0, 0.8),
                    funding_rate_extremes=abs(np.random.normal(0, 0.01)),
                    liquidation_cascade_risk=np.random.uniform(0, 0.6),
                    stablecoin_dominance=np.random.uniform(0.6, 0.8),
                    timestamp=datetime.now()
                )
                
                self.on_chain_metrics[symbol] = metrics
            
            logger.debug(f"Updated on-chain metrics for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to update on-chain metrics: {e}")
    
    async def _scan_news_events(self):
        """Scan news and events for risk signals"""
        try:
            # Simulated news scanning (in production, connect to real news APIs)
            mock_headlines = [
                "Bitcoin ETF sees record inflows amid institutional adoption",
                "Federal Reserve hints at potential rate cuts in Q4",
                "Major exchange reports security upgrade completion",
                "Regulatory clarity improves for digital assets",
                "Ethereum network upgrade scheduled for next month"
            ]
            
            current_time = datetime.now()
            
            for i, headline in enumerate(mock_headlines):
                # Analyze sentiment (simplified)
                sentiment_score = self._analyze_news_sentiment(headline)
                relevance_score = self._calculate_news_relevance(headline)
                risk_keywords = self._extract_risk_keywords(headline)
                impact_estimate = len(risk_keywords) * 0.2
                
                signal = NewsSignal(
                    signal_id=f"news_{current_time.strftime('%H%M%S')}_{i}",
                    source="mock_source",
                    headline=headline,
                    sentiment_score=sentiment_score,
                    relevance_score=relevance_score,
                    risk_keywords=risk_keywords,
                    impact_estimate=impact_estimate,
                    published_at=current_time - timedelta(minutes=i*30)
                )
                
                self.news_signals.append(signal)
            
            # Keep only recent signals
            cutoff_time = current_time - timedelta(hours=48)
            self.news_signals = [s for s in self.news_signals if s.published_at > cutoff_time]
            
            logger.debug(f"Scanned news, found {len(self.news_signals)} recent signals")
            
        except Exception as e:
            logger.error(f"Failed to scan news events: {e}")
    
    def _analyze_news_sentiment(self, headline: str) -> float:
        """Analyze news sentiment (simplified NLP)"""
        try:
            positive_words = ['upgrade', 'adoption', 'growth', 'bullish', 'positive', 'surge', 'rally']
            negative_words = ['crash', 'hack', 'ban', 'regulation', 'concern', 'drop', 'fall']
            
            headline_lower = headline.lower()
            
            positive_count = sum(1 for word in positive_words if word in headline_lower)
            negative_count = sum(1 for word in negative_words if word in headline_lower)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            return (positive_count - negative_count) / (positive_count + negative_count)
            
        except Exception as e:
            logger.error(f"Failed to analyze news sentiment: {e}")
            return 0.0
    
    def _calculate_news_relevance(self, headline: str) -> float:
        """Calculate news relevance to crypto markets"""
        try:
            crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft', 'btc', 'eth']
            financial_keywords = ['fed', 'rate', 'inflation', 'market', 'trading', 'investment']
            
            headline_lower = headline.lower()
            
            crypto_score = sum(1 for word in crypto_keywords if word in headline_lower)
            financial_score = sum(0.5 for word in financial_keywords if word in headline_lower)
            
            relevance = (crypto_score + financial_score) / 3  # Normalize
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Failed to calculate news relevance: {e}")
            return 0.5
    
    def _extract_risk_keywords(self, headline: str) -> List[str]:
        """Extract risk keywords from headline"""
        try:
            found_keywords = []
            headline_lower = headline.lower()
            
            for severity, keywords in self.risk_keywords.items():
                for keyword in keywords:
                    if keyword in headline_lower:
                        found_keywords.append(f"{severity}:{keyword}")
            
            return found_keywords
            
        except Exception as e:
            logger.error(f"Failed to extract risk keywords: {e}")
            return []
    
    def _calculate_composite_risk_score(self) -> float:
        """Calculate composite environmental risk score"""
        try:
            risk_components = []
            
            # 1. Macro risk component
            macro_risk = 0.0
            if self.macro_indicators:
                macro_impacts = [ind.risk_impact for ind in self.macro_indicators.values()]
                macro_risk = np.mean([abs(impact) for impact in macro_impacts])
                risk_components.append(('macro', macro_risk, 0.3))
            
            # 2. On-chain risk component
            onchain_risk = 0.0
            if self.on_chain_metrics:
                onchain_scores = []
                for metrics in self.on_chain_metrics.values():
                    # Combine various on-chain risk factors
                    score = (
                        abs(metrics.exchange_netflow) / 10000 * 0.2 +  # Normalize netflow
                        metrics.whale_activity_score * 0.3 +
                        metrics.network_congestion * 0.2 +
                        metrics.funding_rate_extremes * 100 * 0.2 +  # Funding rate to 0-1
                        metrics.liquidation_cascade_risk * 0.1
                    )
                    onchain_scores.append(min(1.0, score))
                
                onchain_risk = np.mean(onchain_scores) if onchain_scores else 0.0
                risk_components.append(('onchain', onchain_risk, 0.25))
            
            # 3. News sentiment risk component
            news_risk = 0.0
            if self.news_signals:
                recent_signals = [s for s in self.news_signals 
                                if s.published_at > datetime.now() - timedelta(hours=24)]
                if recent_signals:
                    # Weight by impact and recency
                    weighted_sentiment = 0.0
                    total_weight = 0.0
                    
                    for signal in recent_signals:
                        hours_old = (datetime.now() - signal.published_at).total_seconds() / 3600
                        recency_weight = max(0.1, 1.0 - hours_old / 24)  # Decay over 24h
                        weight = signal.relevance_score * recency_weight
                        
                        # Convert sentiment to risk (negative sentiment = higher risk)
                        sentiment_risk = max(0, -signal.sentiment_score) + signal.impact_estimate
                        
                        weighted_sentiment += sentiment_risk * weight
                        total_weight += weight
                    
                    news_risk = weighted_sentiment / total_weight if total_weight > 0 else 0.0
                    risk_components.append(('news', news_risk, 0.2))
            
            # 4. Active events risk component
            events_risk = 0.0
            if self.risk_events:
                active_events = [e for e in self.risk_events if e.expires_at > datetime.now()]
                if active_events:
                    # Weight by severity and impact
                    severity_weights = {
                        RiskLevel.GREEN: 0.1,
                        RiskLevel.YELLOW: 0.3,
                        RiskLevel.ORANGE: 0.7,
                        RiskLevel.RED: 1.0
                    }
                    
                    event_scores = []
                    for event in active_events:
                        severity_weight = severity_weights.get(event.severity, 0.5)
                        event_score = event.impact_score * severity_weight
                        event_scores.append(event_score)
                    
                    events_risk = np.mean(event_scores)
                    risk_components.append(('events', events_risk, 0.25))
            
            # Calculate weighted composite score
            if risk_components:
                total_score = sum(score * weight for _, score, weight in risk_components)
                total_weight = sum(weight for _, _, weight in risk_components)
                composite_score = total_score / total_weight if total_weight > 0 else 0.0
            else:
                composite_score = 0.3  # Default moderate risk
            
            return float(np.clip(composite_score, 0, 1))
            
        except Exception as e:
            logger.error(f"Failed to calculate composite risk score: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from composite score"""
        if risk_score <= self.risk_thresholds['green_max']:
            return RiskLevel.GREEN
        elif risk_score <= self.risk_thresholds['yellow_max']:
            return RiskLevel.YELLOW
        elif risk_score <= self.risk_thresholds['orange_max']:
            return RiskLevel.ORANGE
        else:
            return RiskLevel.RED
    
    def _generate_risk_recommendations(self, risk_level: RiskLevel, risk_score: float) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.GREEN:
            recommendations.extend([
                "Normal trading operations",
                "Monitor for emerging risks",
                "Consider increasing position sizes"
            ])
        elif risk_level == RiskLevel.YELLOW:
            recommendations.extend([
                "Maintain cautious approach",
                "Reduce leverage slightly",
                "Monitor news and macro indicators closely"
            ])
        elif risk_level == RiskLevel.ORANGE:
            recommendations.extend([
                "Reduce position sizes",
                "Tighten stop losses",
                "Avoid high-risk trades",
                "Increase cash reserves"
            ])
        else:  # RED
            recommendations.extend([
                "Enter defensive mode",
                "Close risky positions",
                "Halt new position opening",
                "Maintain high cash reserves",
                "Wait for risk environment to improve"
            ])
        
        return recommendations
    
    def _calculate_confidence_multiplier(self, risk_level: RiskLevel, risk_score: float) -> float:
        """Calculate trading confidence multiplier"""
        base_multipliers = {
            RiskLevel.GREEN: 1.0,
            RiskLevel.YELLOW: 0.85,
            RiskLevel.ORANGE: 0.6,
            RiskLevel.RED: 0.3
        }
        
        base_multiplier = base_multipliers.get(risk_level, 0.8)
        
        # Fine-tune based on exact risk score
        score_adjustment = 1.0 - (risk_score * 0.3)  # Max 30% reduction
        
        return float(np.clip(base_multiplier * score_adjustment, 0.1, 1.0))
    
    def _calculate_position_multiplier(self, risk_level: RiskLevel, risk_score: float) -> float:
        """Calculate position size multiplier"""
        base_multipliers = {
            RiskLevel.GREEN: 1.0,
            RiskLevel.YELLOW: 0.8,
            RiskLevel.ORANGE: 0.5,
            RiskLevel.RED: 0.2
        }
        
        base_multiplier = base_multipliers.get(risk_level, 0.7)
        
        # Fine-tune based on exact risk score
        score_adjustment = 1.0 - (risk_score * 0.4)  # Max 40% reduction
        
        return float(np.clip(base_multiplier * score_adjustment, 0.1, 1.0))
    
    def _calculate_assessment_confidence(self) -> float:
        """Calculate confidence in the risk assessment"""
        try:
            confidence_factors = []
            
            # Data freshness factor
            if self.macro_indicators:
                avg_age = np.mean([(datetime.now() - ind.last_updated).total_seconds() / 3600 
                                 for ind in self.macro_indicators.values()])
                freshness_factor = max(0.3, 1.0 - avg_age / 24)  # Decay over 24h
                confidence_factors.append(freshness_factor)
            
            # Data completeness factor
            expected_indicators = len(self.macro_config)
            actual_indicators = len(self.macro_indicators)
            completeness_factor = actual_indicators / expected_indicators if expected_indicators > 0 else 0.5
            confidence_factors.append(completeness_factor)
            
            # Signal consistency factor (simplified)
            consistency_factor = 0.8  # Default
            confidence_factors.append(consistency_factor)
            
            return float(np.mean(confidence_factors)) if confidence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Failed to calculate assessment confidence: {e}")
            return 0.5
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        self._should_stop = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="EnvironmentalRiskMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started environmental risk monitoring thread")
    
    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        while not self._should_stop:
            try:
                # Clean up expired events
                current_time = datetime.now()
                self.risk_events = [e for e in self.risk_events if e.expires_at > current_time]
                
                # Clean up old news signals
                cutoff_time = current_time - timedelta(hours=48)
                self.news_signals = [s for s in self.news_signals if s.published_at > cutoff_time]
                
                # Periodic risk assessment (every 15 minutes)
                if hasattr(self, '_last_assessment'):
                    if current_time - self._last_assessment > timedelta(minutes=15):
                        asyncio.create_task(self.assess_environmental_risk())
                        self._last_assessment = current_time
                else:
                    self._last_assessment = current_time
                
                # Sleep
                threading.Event().wait(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in environmental risk monitoring: {e}")
    
    def stop(self):
        """Stop the environmental risk engine"""
        self._should_stop = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Environmental Risk Engine stopped")


# Global instance
_environmental_risk_engine = None

def get_environmental_risk_engine() -> EnvironmentalRiskEngine:
    """Get global environmental risk engine instance"""
    global _environmental_risk_engine
    if _environmental_risk_engine is None:
        _environmental_risk_engine = EnvironmentalRiskEngine()
    return _environmental_risk_engine
