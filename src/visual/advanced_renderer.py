"""
Advanced Chart Renderer
=======================
Enhanced chart generation with multi-asset, interactive, and streaming capabilities
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import seaborn as sns

logger = logging.getLogger(__name__)

class AdvancedChartRenderer:
    """
    Advanced Chart Renderer
    
    Enhanced chart generation system supporting multi-timeframe overlays,
    cross-asset correlation charts, interactive annotations, and real-time streaming.
    """
    
    def __init__(self, output_dir: str = "data/advanced_charts"):
        """Initialize the advanced chart renderer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart themes
        self.themes = {
            'dark': {
                'background': '#1e1e1e',
                'chart_background': '#2d2d2d',
                'text_color': 'white',
                'grid_color': '#444444',
                'grid_alpha': 0.3
            },
            'light': {
                'background': '#ffffff',
                'chart_background': '#f8f9fa',
                'text_color': 'black',
                'grid_color': '#cccccc',
                'grid_alpha': 0.5
            }
        }
        
        # Color schemes
        self.color_schemes = {
            'crypto': {
                'bullish': '#00ff88',
                'bearish': '#ff4444',
                'neutral': '#ffaa00',
                'volume': '#6666ff',
                'support': '#44ff44',
                'resistance': '#ff4444'
            },
            'professional': {
                'bullish': '#26a69a',
                'bearish': '#ef5350',
                'neutral': '#ffa726',
                'volume': '#42a5f5',
                'support': '#66bb6a',
                'resistance': '#ef5350'
            }
        }
        
        logger.info("Advanced Chart Renderer initialized")
    
    def render_multi_timeframe_chart(self, symbol: str, timeframes: List[str],
                                   chart_data: Dict[str, Any], theme: str = 'dark') -> str:
        """
        Render multi-timeframe overlay chart
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to display
            chart_data: Chart data for each timeframe
            theme: Chart theme ('dark' or 'light')
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Rendering multi-timeframe chart for {symbol}")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(len(timeframes), 1, height_ratios=[1] * len(timeframes), hspace=0.3)
            
            # Apply theme
            theme_config = self.themes.get(theme, self.themes['dark'])
            plt.style.use('dark_background' if theme == 'dark' else 'default')
            fig.patch.set_facecolor(theme_config['background'])
            
            # Render each timeframe
            for i, timeframe in enumerate(timeframes):
                ax = fig.add_subplot(gs[i])
                
                # Generate mock data for each timeframe
                tf_data = self._generate_timeframe_data(symbol, timeframe, 100)
                
                # Plot price line
                ax.plot(tf_data['timestamps'], tf_data['prices'], 
                       color=self.color_schemes['crypto']['bullish'], 
                       linewidth=2, alpha=0.8, label=f'{timeframe} Price')
                
                # Add volume bars
                ax2 = ax.twinx()
                ax2.bar(tf_data['timestamps'], tf_data['volumes'], 
                       color=self.color_schemes['crypto']['volume'], 
                       alpha=0.3, width=0.8, label='Volume')
                
                # Styling
                ax.set_title(f'{symbol} - {timeframe}', 
                           color=theme_config['text_color'], fontsize=14, fontweight='bold')
                ax.set_facecolor(theme_config['chart_background'])
                ax.grid(True, alpha=theme_config['grid_alpha'], color=theme_config['grid_color'])
                ax.tick_params(colors=theme_config['text_color'])
                ax2.tick_params(colors=theme_config['text_color'])
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color(theme_config['grid_color'])
                for spine in ax2.spines.values():
                    spine.set_color(theme_config['grid_color'])
                
                # Add legends
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
            
            # Add overall title
            fig.suptitle(f'{symbol} Multi-Timeframe Analysis', 
                        color=theme_config['text_color'], fontsize=18, fontweight='bold')
            
            # Add watermark
            fig.text(0.99, 0.01, '🧠 Advanced Trading Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f"multi_timeframe_{symbol}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor=theme_config['background'], edgecolor='none')
            plt.close()
            
            logger.info(f"Generated multi-timeframe chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to render multi-timeframe chart: {e}")
            return ""
    
    def render_correlation_matrix_chart(self, assets: List[str], 
                                      correlation_data: Dict[str, Dict[str, float]],
                                      theme: str = 'dark') -> str:
        """
        Render correlation matrix heatmap
        
        Args:
            assets: List of asset symbols
            correlation_data: Correlation matrix data
            theme: Chart theme
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Rendering correlation matrix for {len(assets)} assets")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            theme_config = self.themes.get(theme, self.themes['dark'])
            plt.style.use('dark_background' if theme == 'dark' else 'default')
            fig.patch.set_facecolor(theme_config['background'])
            
            # Convert correlation data to matrix
            import pandas as pd
            corr_df = pd.DataFrame(correlation_data)
            
            # Create heatmap
            sns.heatmap(
                corr_df,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'},
                fmt='.2f'
            )
            
            # Styling
            ax.set_title('Asset Correlation Matrix', 
                        color=theme_config['text_color'], fontsize=16, fontweight='bold')
            ax.tick_params(colors=theme_config['text_color'])
            
            # Add watermark
            fig.text(0.99, 0.01, '🔗 Correlation Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f"correlation_matrix_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor=theme_config['background'], edgecolor='none')
            plt.close()
            
            logger.info(f"Generated correlation matrix chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to render correlation matrix: {e}")
            return ""
    
    def render_interactive_annotated_chart(self, symbol: str, timeframe: str,
                                         chart_data: Dict[str, Any],
                                         annotations: List[Dict[str, Any]],
                                         theme: str = 'dark') -> str:
        """
        Render chart with interactive annotations
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            chart_data: Base chart data
            annotations: List of annotations to add
            theme: Chart theme
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Rendering interactive annotated chart for {symbol}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            theme_config = self.themes.get(theme, self.themes['dark'])
            plt.style.use('dark_background' if theme == 'dark' else 'default')
            fig.patch.set_facecolor(theme_config['background'])
            ax.set_facecolor(theme_config['chart_background'])
            
            # Generate base chart data
            base_data = self._generate_timeframe_data(symbol, timeframe, 100)
            
            # Plot base price chart
            ax.plot(base_data['timestamps'], base_data['prices'],
                   color=self.color_schemes['crypto']['bullish'],
                   linewidth=2, alpha=0.8, label='Price')
            
            # Add annotations
            for annotation in annotations:
                self._add_chart_annotation(ax, annotation, base_data)
            
            # Styling
            ax.set_title(f'{symbol} {timeframe} - Interactive Chart',
                        color=theme_config['text_color'], fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', color=theme_config['text_color'])
            ax.set_ylabel('Price', color=theme_config['text_color'])
            ax.grid(True, alpha=theme_config['grid_alpha'], color=theme_config['grid_color'])
            ax.tick_params(colors=theme_config['text_color'])
            ax.legend()
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color(theme_config['grid_color'])
            
            # Add watermark
            fig.text(0.99, 0.01, '🎨 Interactive Chart Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f"interactive_{symbol}_{timeframe}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor=theme_config['background'], edgecolor='none')
            plt.close()
            
            logger.info(f"Generated interactive annotated chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to render interactive chart: {e}")
            return ""
    
    def render_streaming_chart_snapshot(self, symbol: str, timeframe: str,
                                      streaming_data: List[Dict[str, Any]],
                                      forecast_overlay: Optional[Dict[str, Any]] = None,
                                      theme: str = 'dark') -> str:
        """
        Render snapshot of streaming chart
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            streaming_data: Real-time streaming data
            forecast_overlay: Optional forecast overlay
            theme: Chart theme
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Rendering streaming chart snapshot for {symbol}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            theme_config = self.themes.get(theme, self.themes['dark'])
            plt.style.use('dark_background' if theme == 'dark' else 'default')
            fig.patch.set_facecolor(theme_config['background'])
            ax.set_facecolor(theme_config['chart_background'])
            
            # Extract data from streaming points
            if streaming_data:
                timestamps = [datetime.fromisoformat(point['timestamp']) if isinstance(point['timestamp'], str) 
                            else point['timestamp'] for point in streaming_data]
                prices = [point['price'] for point in streaming_data]
                
                # Plot streaming price line
                ax.plot(timestamps, prices,
                       color=self.color_schemes['crypto']['bullish'],
                       linewidth=2, alpha=0.8, label='Live Price')
                
                # Add forecast overlay if provided
                if forecast_overlay:
                    self._add_forecast_overlay(ax, forecast_overlay, timestamps, prices)
            
            # Styling
            latest_price = prices[-1] if prices else 0
            ax.set_title(f'{symbol} {timeframe} - Live Stream: ${latest_price:.4f}',
                        color=theme_config['text_color'], fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', color=theme_config['text_color'])
            ax.set_ylabel('Price', color=theme_config['text_color'])
            ax.grid(True, alpha=theme_config['grid_alpha'], color=theme_config['grid_color'])
            ax.tick_params(colors=theme_config['text_color'])
            ax.legend()
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color(theme_config['grid_color'])
            
            # Add live indicator
            ax.text(0.02, 0.98, '🔴 LIVE', transform=ax.transAxes,
                   color='red', fontsize=12, fontweight='bold', va='top')
            
            # Add watermark
            fig.text(0.99, 0.01, '📡 Live Streaming Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f"streaming_{symbol}_{timeframe}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor=theme_config['background'], edgecolor='none')
            plt.close()
            
            logger.info(f"Generated streaming chart snapshot: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to render streaming chart: {e}")
            return ""
    
    def render_pattern_recognition_chart(self, symbol: str, timeframe: str,
                                       chart_data: Dict[str, Any],
                                       recognized_patterns: List[Dict[str, Any]],
                                       theme: str = 'dark') -> str:
        """
        Render chart with pattern recognition highlights
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            chart_data: Base chart data
            recognized_patterns: List of recognized patterns
            theme: Chart theme
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Rendering pattern recognition chart for {symbol}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            theme_config = self.themes.get(theme, self.themes['dark'])
            plt.style.use('dark_background' if theme == 'dark' else 'default')
            fig.patch.set_facecolor(theme_config['background'])
            ax.set_facecolor(theme_config['chart_background'])
            
            # Generate base chart data
            base_data = self._generate_timeframe_data(symbol, timeframe, 100)
            
            # Plot base price chart
            ax.plot(base_data['timestamps'], base_data['prices'],
                   color=self.color_schemes['crypto']['bullish'],
                   linewidth=2, alpha=0.8, label='Price')
            
            # Highlight recognized patterns
            pattern_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            
            for i, pattern in enumerate(recognized_patterns):
                color = pattern_colors[i % len(pattern_colors)]
                confidence = pattern.get('confidence', 50)
                pattern_name = pattern.get('pattern_name', f'Pattern {i+1}')
                
                # Highlight pattern area (mock implementation)
                start_idx = max(0, len(base_data['timestamps']) - 20)
                end_idx = len(base_data['timestamps']) - 5
                
                ax.fill_between(
                    base_data['timestamps'][start_idx:end_idx],
                    base_data['prices'][start_idx:end_idx],
                    alpha=confidence/200,  # Transparency based on confidence
                    color=color,
                    label=f'{pattern_name} ({confidence:.1f}%)'
                )
                
                # Add pattern label
                mid_idx = (start_idx + end_idx) // 2
                ax.annotate(
                    f'{pattern_name}\n{confidence:.1f}%',
                    xy=(base_data['timestamps'][mid_idx], base_data['prices'][mid_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    color='white', fontsize=8, fontweight='bold'
                )
            
            # Styling
            ax.set_title(f'{symbol} {timeframe} - Pattern Recognition',
                        color=theme_config['text_color'], fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', color=theme_config['text_color'])
            ax.set_ylabel('Price', color=theme_config['text_color'])
            ax.grid(True, alpha=theme_config['grid_alpha'], color=theme_config['grid_color'])
            ax.tick_params(colors=theme_config['text_color'])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color(theme_config['grid_color'])
            
            # Add watermark
            fig.text(0.99, 0.01, '🔍 Pattern Recognition Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.output_dir / f"patterns_{symbol}_{timeframe}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor=theme_config['background'], edgecolor='none')
            plt.close()
            
            logger.info(f"Generated pattern recognition chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to render pattern recognition chart: {e}")
            return ""
    
    def _generate_timeframe_data(self, symbol: str, timeframe: str, periods: int) -> Dict[str, List]:
        """Generate mock timeframe data"""
        np.random.seed(hash(f"{symbol}_{timeframe}") % 2**32)
        
        # Generate timestamps based on timeframe
        timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        minutes = timeframe_minutes.get(timeframe, 60)
        
        timestamps = [
            datetime.now() - timedelta(minutes=minutes * i) 
            for i in range(periods-1, -1, -1)
        ]
        
        # Generate price data
        returns = np.random.normal(0.001, 0.02, periods)
        prices = [100.0]
        
        for return_rate in returns:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(new_price, 0.01))
        
        prices = prices[1:]
        
        # Generate volume data
        volumes = np.random.uniform(1000, 10000, periods)
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'volumes': volumes
        }
    
    def _add_chart_annotation(self, ax, annotation: Dict[str, Any], base_data: Dict[str, List]):
        """Add annotation to chart"""
        try:
            annotation_type = annotation.get('type', 'text')
            
            if annotation_type == 'support_resistance':
                price = annotation.get('price', 0)
                level_type = annotation.get('level_type', 'support')
                color = self.color_schemes['crypto']['support'] if level_type == 'support' else self.color_schemes['crypto']['resistance']
                
                ax.axhline(y=price, color=color, alpha=0.6, linewidth=2, linestyle='--',
                          label=f'{level_type.title()} {price}')
            
            elif annotation_type == 'text':
                x = annotation.get('x', base_data['timestamps'][-1])
                y = annotation.get('y', base_data['prices'][-1])
                text = annotation.get('text', 'Annotation')
                
                ax.text(x, y, text, color='yellow', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
        except Exception as e:
            logger.warning(f"Failed to add annotation: {e}")
    
    def _add_forecast_overlay(self, ax, forecast_overlay: Dict[str, Any], 
                            timestamps: List[datetime], prices: List[float]):
        """Add forecast overlay to chart"""
        try:
            # Entry zone
            entry_zone = forecast_overlay.get('entry_zone', (0, 0))
            if entry_zone[0] > 0 and entry_zone[1] > 0:
                ax.axhspan(entry_zone[0], entry_zone[1], alpha=0.2, color='yellow', label='Entry Zone')
            
            # Target zones
            target_zones = forecast_overlay.get('target_zones', [])
            for i, (target_min, target_max) in enumerate(target_zones):
                if target_min > 0 and target_max > 0:
                    ax.axhspan(target_min, target_max, alpha=0.15, color='cyan',
                              label=f'Target {i+1}' if i == 0 else "")
            
            # Stop loss
            stop_loss = forecast_overlay.get('stop_loss', 0)
            if stop_loss > 0:
                ax.axhline(y=stop_loss, color='red', alpha=0.6, linewidth=1, 
                          linestyle=':', label='Stop Loss')
            
            # Confidence indicator
            confidence = forecast_overlay.get('confidence', 50)
            direction = forecast_overlay.get('direction', 'neutral')
            
            ax.text(0.02, 0.02, f'{direction.upper()}\nConfidence: {confidence:.1f}%',
                   transform=ax.transAxes, color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
            
        except Exception as e:
            logger.warning(f"Failed to add forecast overlay: {e}")


# Global instance
_advanced_renderer = None

def get_advanced_renderer() -> AdvancedChartRenderer:
    """Get global advanced renderer instance"""
    global _advanced_renderer
    if _advanced_renderer is None:
        _advanced_renderer = AdvancedChartRenderer()
    return _advanced_renderer
