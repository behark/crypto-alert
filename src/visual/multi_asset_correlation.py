"""
Multi-Asset Visual Correlation System
====================================
Side-by-side forecasting and correlation analysis of related tokens
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import asyncio
from scipy.stats import pearsonr
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class AssetCorrelation:
    """Correlation data between two assets"""
    asset1: str
    asset2: str
    correlation: float
    timeframe: str
    lookback_periods: int
    p_value: float
    strength: str  # 'strong', 'moderate', 'weak'
    timestamp: datetime

@dataclass
class MultiAssetForecast:
    """Combined forecast for multiple assets"""
    assets: List[str]
    timeframe: str
    correlations: Dict[Tuple[str, str], float]
    individual_forecasts: Dict[str, Dict[str, Any]]
    consensus_direction: str
    confidence_score: float
    risk_assessment: str
    timestamp: datetime

class MultiAssetCorrelator:
    """
    Multi-Asset Visual Correlation System
    
    Provides side-by-side forecasting and correlation analysis of related tokens
    to identify cross-asset opportunities and risks.
    """
    
    def __init__(self, data_dir: str = "data/multi_asset"):
        """Initialize the multi-asset correlator"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Correlation cache
        self.correlation_cache: Dict[str, AssetCorrelation] = {}
        self.correlation_history: Dict[str, List[float]] = {}
        
        # Asset groupings for analysis
        self.asset_groups = {
            'major_crypto': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT'],
            'defi_tokens': ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'SUSHIUSDT', 'CRVUSDT'],
            'layer1_chains': ['ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'SOLUSDT', 'AVAXUSDT'],
            'meme_coins': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT'],
            'ai_tokens': ['FETUSDT', 'AGIXUSDT', 'OCEANUSDT', 'RNDRУСDT']
        }
        
        logger.info("Multi-Asset Correlator initialized")
    
    async def analyze_correlation_matrix(self, assets: List[str], timeframe: str = '1h',
                                       lookback_periods: int = 100) -> Dict[str, Any]:
        """
        Analyze correlation matrix for multiple assets
        
        Args:
            assets: List of asset symbols
            timeframe: Chart timeframe
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dict containing correlation matrix and analysis
        """
        try:
            logger.info(f"Analyzing correlation matrix for {len(assets)} assets")
            
            # Get price data for all assets (mock implementation)
            price_data = {}
            for asset in assets:
                # In real implementation, this would fetch actual price data
                price_data[asset] = self._generate_mock_price_data(asset, lookback_periods)
            
            # Calculate correlation matrix
            correlation_matrix = {}
            correlations = []
            
            for i, asset1 in enumerate(assets):
                correlation_matrix[asset1] = {}
                for j, asset2 in enumerate(assets):
                    if i == j:
                        correlation = 1.0
                        p_value = 0.0
                    else:
                        correlation, p_value = pearsonr(
                            price_data[asset1], 
                            price_data[asset2]
                        )
                    
                    correlation_matrix[asset1][asset2] = correlation
                    
                    if i < j:  # Avoid duplicates
                        correlations.append(AssetCorrelation(
                            asset1=asset1,
                            asset2=asset2,
                            correlation=correlation,
                            timeframe=timeframe,
                            lookback_periods=lookback_periods,
                            p_value=p_value,
                            strength=self._classify_correlation_strength(correlation),
                            timestamp=datetime.now()
                        ))
            
            # Identify strongest correlations
            strong_correlations = [
                corr for corr in correlations 
                if abs(corr.correlation) >= 0.7
            ]
            
            # Generate correlation heatmap
            heatmap_path = await self._generate_correlation_heatmap(
                correlation_matrix, assets, timeframe
            )
            
            return {
                'correlation_matrix': correlation_matrix,
                'correlations': correlations,
                'strong_correlations': strong_correlations,
                'average_correlation': np.mean([abs(c.correlation) for c in correlations]),
                'heatmap_path': heatmap_path,
                'analysis_timestamp': datetime.now(),
                'timeframe': timeframe,
                'lookback_periods': lookback_periods
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze correlation matrix: {e}")
            return {'error': str(e)}
    
    async def generate_multi_asset_forecast(self, assets: List[str], timeframe: str = '1h') -> MultiAssetForecast:
        """
        Generate coordinated forecast for multiple assets
        
        Args:
            assets: List of asset symbols
            timeframe: Chart timeframe
            
        Returns:
            MultiAssetForecast object
        """
        try:
            logger.info(f"Generating multi-asset forecast for {assets}")
            
            # Get individual forecasts (mock implementation)
            individual_forecasts = {}
            for asset in assets:
                individual_forecasts[asset] = self._generate_mock_forecast(asset, timeframe)
            
            # Calculate current correlations
            correlation_analysis = await self.analyze_correlation_matrix(assets, timeframe)
            correlations = {}
            
            for corr in correlation_analysis.get('correlations', []):
                key = (corr.asset1, corr.asset2)
                correlations[key] = corr.correlation
            
            # Determine consensus direction
            directions = [forecast['direction'] for forecast in individual_forecasts.values()]
            bullish_count = directions.count('bullish')
            bearish_count = directions.count('bearish')
            
            if bullish_count > bearish_count:
                consensus_direction = 'bullish'
                confidence_multiplier = bullish_count / len(directions)
            elif bearish_count > bullish_count:
                consensus_direction = 'bearish'
                confidence_multiplier = bearish_count / len(directions)
            else:
                consensus_direction = 'neutral'
                confidence_multiplier = 0.5
            
            # Calculate confidence score
            individual_confidences = [forecast['confidence'] for forecast in individual_forecasts.values()]
            base_confidence = np.mean(individual_confidences)
            confidence_score = base_confidence * confidence_multiplier
            
            # Assess risk based on correlations
            avg_correlation = correlation_analysis.get('average_correlation', 0.5)
            if avg_correlation >= 0.8:
                risk_assessment = 'high_correlation_risk'
            elif avg_correlation >= 0.6:
                risk_assessment = 'moderate_correlation_risk'
            else:
                risk_assessment = 'diversified_risk'
            
            multi_forecast = MultiAssetForecast(
                assets=assets,
                timeframe=timeframe,
                correlations=correlations,
                individual_forecasts=individual_forecasts,
                consensus_direction=consensus_direction,
                confidence_score=confidence_score,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
            return multi_forecast
            
        except Exception as e:
            logger.error(f"Failed to generate multi-asset forecast: {e}")
            return None
    
    async def generate_correlation_chart(self, assets: List[str], timeframe: str = '1h') -> str:
        """
        Generate side-by-side correlation chart
        
        Args:
            assets: List of asset symbols
            timeframe: Chart timeframe
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Generating correlation chart for {assets}")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, len(assets), height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
            
            # Set dark theme
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e1e1e')
            
            # Generate mock price data
            price_data = {}
            for asset in assets:
                price_data[asset] = self._generate_mock_price_data(asset, 100)
            
            # Plot individual price charts
            for i, asset in enumerate(assets):
                ax = fig.add_subplot(gs[0, i])
                
                # Price line
                ax.plot(price_data[asset], color='#00ff88', linewidth=2, alpha=0.8)
                ax.set_title(f'{asset} Price', color='white', fontsize=12, fontweight='bold')
                ax.set_facecolor('#2d2d2d')
                ax.grid(True, alpha=0.3)
                
                # Style
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#444444')
            
            # Plot correlation matrix heatmap
            correlation_analysis = await self.analyze_correlation_matrix(assets, timeframe)
            correlation_matrix = correlation_analysis.get('correlation_matrix', {})
            
            if correlation_matrix:
                ax_heatmap = fig.add_subplot(gs[1, :])
                
                # Convert to DataFrame for seaborn
                corr_df = pd.DataFrame(correlation_matrix)
                
                # Create heatmap
                sns.heatmap(
                    corr_df, 
                    annot=True, 
                    cmap='RdYlBu_r', 
                    center=0,
                    square=True,
                    ax=ax_heatmap,
                    cbar_kws={'label': 'Correlation Coefficient'}
                )
                ax_heatmap.set_title('Asset Correlation Matrix', color='white', fontsize=14, fontweight='bold')
            
            # Plot correlation timeline
            ax_timeline = fig.add_subplot(gs[2, :])
            
            # Mock correlation timeline data
            timeline_data = {}
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    pair = f"{asset1}-{asset2}"
                    timeline_data[pair] = np.random.normal(0.5, 0.2, 50)  # Mock data
            
            for pair, data in timeline_data.items():
                ax_timeline.plot(data, label=pair, alpha=0.7, linewidth=1.5)
            
            ax_timeline.set_title('Correlation Timeline', color='white', fontsize=12, fontweight='bold')
            ax_timeline.set_ylabel('Correlation', color='white')
            ax_timeline.set_xlabel('Time Periods', color='white')
            ax_timeline.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_timeline.grid(True, alpha=0.3)
            ax_timeline.set_facecolor('#2d2d2d')
            
            # Add watermark
            fig.text(0.99, 0.01, '🧠 Living Trading Intelligence Platform', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.data_dir / f"correlation_chart_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', 
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close()
            
            logger.info(f"Generated correlation chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate correlation chart: {e}")
            return ""
    
    def get_asset_group_correlations(self, group_name: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Get correlations for a predefined asset group
        
        Args:
            group_name: Name of asset group
            timeframe: Chart timeframe
            
        Returns:
            Dict containing group correlation analysis
        """
        try:
            if group_name not in self.asset_groups:
                return {'error': f'Asset group {group_name} not found'}
            
            assets = self.asset_groups[group_name]
            
            # This would be async in real implementation
            # For now, return mock analysis
            return {
                'group_name': group_name,
                'assets': assets,
                'average_correlation': 0.65,
                'strongest_pair': (assets[0], assets[1]),
                'weakest_pair': (assets[-2], assets[-1]),
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get asset group correlations: {e}")
            return {'error': str(e)}
    
    def find_correlation_opportunities(self, base_asset: str, 
                                     correlation_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find trading opportunities based on correlation analysis
        
        Args:
            base_asset: Base asset to find correlations for
            correlation_threshold: Minimum correlation threshold
            
        Returns:
            List of correlation opportunities
        """
        try:
            opportunities = []
            
            # Check all asset groups for correlations with base asset
            for group_name, assets in self.asset_groups.items():
                if base_asset in assets:
                    for asset in assets:
                        if asset != base_asset:
                            # Mock correlation calculation
                            correlation = np.random.uniform(0.3, 0.9)
                            
                            if abs(correlation) >= correlation_threshold:
                                opportunity = {
                                    'correlated_asset': asset,
                                    'correlation': correlation,
                                    'group': group_name,
                                    'opportunity_type': 'positive_correlation' if correlation > 0 else 'negative_correlation',
                                    'confidence': abs(correlation) * 100,
                                    'recommendation': self._generate_correlation_recommendation(correlation)
                                }
                                opportunities.append(opportunity)
            
            # Sort by correlation strength
            opportunities.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to find correlation opportunities: {e}")
            return []
    
    async def _generate_correlation_heatmap(self, correlation_matrix: Dict[str, Dict[str, float]], 
                                          assets: List[str], timeframe: str) -> str:
        """Generate correlation heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e1e1e')
            
            # Convert to DataFrame
            corr_df = pd.DataFrame(correlation_matrix)
            
            # Create heatmap
            sns.heatmap(
                corr_df,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            
            ax.set_title(f'Asset Correlation Matrix - {timeframe}', 
                        color='white', fontsize=14, fontweight='bold')
            
            # Save heatmap
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            heatmap_path = self.data_dir / f"correlation_heatmap_{timestamp}.png"
            
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight',
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close()
            
            return str(heatmap_path)
            
        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap: {e}")
            return ""
    
    def _generate_mock_price_data(self, asset: str, periods: int) -> List[float]:
        """Generate mock price data for testing"""
        np.random.seed(hash(asset) % 2**32)  # Consistent seed per asset
        
        # Generate realistic price movement
        returns = np.random.normal(0.001, 0.02, periods)  # Small daily returns with volatility
        prices = [100.0]  # Starting price
        
        for return_rate in returns:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        return prices[1:]  # Remove starting price
    
    def _generate_mock_forecast(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """Generate mock forecast for testing"""
        np.random.seed(hash(f"{asset}_{timeframe}") % 2**32)
        
        directions = ['bullish', 'bearish', 'neutral']
        direction = np.random.choice(directions, p=[0.4, 0.4, 0.2])
        confidence = np.random.uniform(50, 90)
        
        return {
            'asset': asset,
            'timeframe': timeframe,
            'direction': direction,
            'confidence': confidence,
            'target': np.random.uniform(0.02, 0.08),  # 2-8% target
            'stop_loss': np.random.uniform(0.01, 0.03),  # 1-3% stop loss
            'timestamp': datetime.now()
        }
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_correlation_recommendation(self, correlation: float) -> str:
        """Generate trading recommendation based on correlation"""
        if correlation >= 0.8:
            return 'strong_positive_correlation'
        elif correlation >= 0.5:
            return 'moderate_positive_correlation'
        elif correlation <= -0.8:
            return 'strong_negative_correlation'
        elif correlation <= -0.5:
            return 'moderate_negative_correlation'
        else:
            return 'weak_correlation'


# Global instance
_multi_asset_correlator = None

def get_multi_asset_correlator() -> MultiAssetCorrelator:
    """Get global multi-asset correlator instance"""
    global _multi_asset_correlator
    if _multi_asset_correlator is None:
        _multi_asset_correlator = MultiAssetCorrelator()
    return _multi_asset_correlator
