"""
Visual Chart Generator for Trading Forecasts
Advanced chart generation with regime overlays, confidence zones, and professional styling
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Polygon
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import logging
import os

logger = logging.getLogger(__name__)

class ForecastChartGenerator:
    """
    Professional chart generator for trading forecasts with visual intelligence
    """
    
    def __init__(self, theme: str = "dark", watermark: Optional[str] = None):
        """
        Initialize chart generator
        
        Args:
            theme: Chart theme ('dark' or 'light')
            watermark: Optional watermark text
        """
        self.theme = theme
        self.watermark = watermark
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib styling based on theme"""
        if self.theme == "dark":
            plt.style.use('dark_background')
            self.colors = {
                'background': '#0d1117',
                'text': '#f0f6fc',
                'grid': '#21262d',
                'bullish': '#00d4aa',
                'bearish': '#f85149',
                'neutral': '#7c3aed',
                'confidence_high': '#00d4aa',
                'confidence_medium': '#fbbf24',
                'confidence_low': '#f87171',
                'regime_bull': 'rgba(0, 212, 170, 0.2)',
                'regime_bear': 'rgba(248, 81, 73, 0.2)',
                'regime_sideways': 'rgba(124, 58, 237, 0.2)',
                'forecast_line': '#60a5fa',
                'support': '#10b981',
                'resistance': '#ef4444'
            }
        else:
            plt.style.use('default')
            self.colors = {
                'background': '#ffffff',
                'text': '#1f2937',
                'grid': '#e5e7eb',
                'bullish': '#059669',
                'bearish': '#dc2626',
                'neutral': '#7c3aed',
                'confidence_high': '#059669',
                'confidence_medium': '#d97706',
                'confidence_low': '#dc2626',
                'regime_bull': 'rgba(5, 150, 105, 0.2)',
                'regime_bear': 'rgba(220, 38, 38, 0.2)',
                'regime_sideways': 'rgba(124, 58, 237, 0.2)',
                'forecast_line': '#2563eb',
                'support': '#059669',
                'resistance': '#dc2626'
            }
    
    def generate_forecast_chart(self, 
                              data: Dict,
                              forecast_data: Dict,
                              signal: Optional[Dict] = None,
                              regime_zones: Optional[List[Dict]] = None,
                              confidence_score: float = 85.0,
                              timeframe: str = "1h") -> bytes:
        """
        Generate comprehensive forecast chart
        
        Args:
            data: Historical price data
            forecast_data: Forecast projections
            signal: Current trading signal
            regime_zones: Market regime zones
            confidence_score: Forecast confidence percentage
            timeframe: Chart timeframe
            
        Returns:
            bytes: PNG image data
        """
        try:
            # Create figure with high DPI for crisp images
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                         facecolor=self.colors['background'],
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Main price chart
            self._plot_price_data(ax1, data, forecast_data)
            self._plot_regime_zones(ax1, regime_zones)
            self._plot_forecast_projection(ax1, forecast_data, confidence_score)
            self._plot_entry_exit_points(ax1, signal)
            self._add_confidence_annotations(ax1, confidence_score)
            
            # Volume/indicator subplot
            self._plot_volume_indicators(ax2, data)
            
            # Styling and annotations
            self._style_chart(ax1, ax2, timeframe)
            self._add_title_and_labels(fig, ax1, signal, confidence_score, timeframe)
            
            # Add watermark if specified
            if self.watermark:
                self._add_watermark(fig)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight',
                       facecolor=self.colors['background'], edgecolor='none')
            img_buffer.seek(0)
            
            plt.close(fig)  # Clean up memory
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate forecast chart: {e}")
            plt.close('all')  # Clean up any open figures
            raise
    
    def _plot_price_data(self, ax, data: Dict, forecast_data: Dict):
        """Plot historical price data with candlesticks"""
        if 'timestamps' not in data or 'close' not in data:
            return
            
        timestamps = pd.to_datetime(data['timestamps'])
        
        # Plot candlesticks (simplified as OHLC lines for now)
        if all(k in data for k in ['open', 'high', 'low', 'close']):
            for i in range(len(timestamps)):
                color = self.colors['bullish'] if data['close'][i] >= data['open'][i] else self.colors['bearish']
                
                # High-low line
                ax.plot([timestamps[i], timestamps[i]], 
                       [data['low'][i], data['high'][i]], 
                       color=color, linewidth=1, alpha=0.8)
                
                # Open-close body
                body_height = abs(data['close'][i] - data['open'][i])
                body_bottom = min(data['open'][i], data['close'][i])
                
                rect = Rectangle((mdates.date2num(timestamps[i]) - 0.0003, body_bottom),
                               0.0006, body_height, 
                               facecolor=color, alpha=0.8, edgecolor=color)
                ax.add_patch(rect)
        else:
            # Fallback to line chart
            ax.plot(timestamps, data['close'], color=self.colors['bullish'], 
                   linewidth=2, label='Price')
    
    def _plot_regime_zones(self, ax, regime_zones: Optional[List[Dict]]):
        """Plot market regime zones as background overlays"""
        if not regime_zones:
            return
            
        for zone in regime_zones:
            start_time = pd.to_datetime(zone.get('start_time'))
            end_time = pd.to_datetime(zone.get('end_time'))
            regime_type = zone.get('type', 'sideways')
            
            color_map = {
                'bullish': self.colors['regime_bull'],
                'bearish': self.colors['regime_bear'],
                'sideways': self.colors['regime_sideways']
            }
            
            color = color_map.get(regime_type, self.colors['regime_sideways'])
            
            ax.axvspan(start_time, end_time, alpha=0.3, color=color, 
                      label=f'{regime_type.title()} Regime' if regime_zones.index(zone) == 0 else "")
    
    def _plot_forecast_projection(self, ax, forecast_data: Dict, confidence_score: float):
        """Plot forecast projection with confidence bands"""
        if 'timestamps' not in forecast_data or 'predicted_prices' not in forecast_data:
            return
            
        forecast_times = pd.to_datetime(forecast_data['timestamps'])
        predicted_prices = forecast_data['predicted_prices']
        
        # Main forecast line
        ax.plot(forecast_times, predicted_prices, 
               color=self.colors['forecast_line'], linewidth=3, 
               linestyle='--', alpha=0.9, label='Forecast')
        
        # Confidence bands
        if 'confidence_upper' in forecast_data and 'confidence_lower' in forecast_data:
            ax.fill_between(forecast_times, 
                           forecast_data['confidence_lower'],
                           forecast_data['confidence_upper'],
                           alpha=0.2, color=self.colors['forecast_line'],
                           label=f'{confidence_score:.0f}% Confidence')
    
    def _plot_entry_exit_points(self, ax, signal: Optional[Dict]):
        """Plot entry and exit points from trading signal"""
        if not signal:
            return
            
        entry_time = pd.to_datetime(signal.get('timestamp', datetime.now()))
        entry_price = signal.get('price', 0)
        profit_target = signal.get('profit_target')
        stop_loss = signal.get('stop_loss')
        direction = signal.get('direction', 'LONG')
        
        # Entry point
        color = self.colors['bullish'] if direction == 'LONG' else self.colors['bearish']
        marker = '^' if direction == 'LONG' else 'v'
        
        ax.scatter(entry_time, entry_price, color=color, s=200, 
                  marker=marker, zorder=10, edgecolor='white', linewidth=2,
                  label=f'{direction} Entry')
        
        # Profit target line
        if profit_target:
            ax.axhline(y=profit_target, color=self.colors['support'], 
                      linestyle=':', alpha=0.8, linewidth=2, label='Profit Target')
        
        # Stop loss line
        if stop_loss:
            ax.axhline(y=stop_loss, color=self.colors['resistance'], 
                      linestyle=':', alpha=0.8, linewidth=2, label='Stop Loss')
    
    def _add_confidence_annotations(self, ax, confidence_score: float):
        """Add confidence score annotations"""
        # Determine confidence color
        if confidence_score >= 80:
            conf_color = self.colors['confidence_high']
            conf_text = "HIGH"
        elif confidence_score >= 60:
            conf_color = self.colors['confidence_medium']
            conf_text = "MEDIUM"
        else:
            conf_color = self.colors['confidence_low']
            conf_text = "LOW"
        
        # Add confidence badge
        ax.text(0.02, 0.98, f'CONFIDENCE: {confidence_score:.0f}% ({conf_text})',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=conf_color, alpha=0.8),
               color='white', verticalalignment='top')
    
    def _plot_volume_indicators(self, ax, data: Dict):
        """Plot volume and technical indicators in subplot"""
        if 'timestamps' not in data:
            return
            
        timestamps = pd.to_datetime(data['timestamps'])
        
        # Volume bars
        if 'volume' in data:
            colors = [self.colors['bullish'] if data['close'][i] >= data['open'][i] 
                     else self.colors['bearish'] for i in range(len(data['close']))]
            ax.bar(timestamps, data['volume'], color=colors, alpha=0.6, width=0.0008)
            ax.set_ylabel('Volume', color=self.colors['text'])
        
        # RSI or other indicators could be added here
        if 'rsi' in data:
            ax2 = ax.twinx()
            ax2.plot(timestamps, data['rsi'], color=self.colors['neutral'], 
                    linewidth=2, alpha=0.8, label='RSI')
            ax2.axhline(y=70, color=self.colors['resistance'], linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color=self.colors['support'], linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI', color=self.colors['text'])
    
    def _style_chart(self, ax1, ax2, timeframe: str):
        """Apply professional styling to charts"""
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.tick_params(colors=self.colors['text'])
            
            # Format x-axis for time
            if timeframe in ['1m', '5m', '15m', '30m']:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            elif timeframe in ['1h', '4h']:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        # Add legend to main chart
        ax1.legend(loc='upper left', facecolor=self.colors['background'], 
                  edgecolor=self.colors['grid'], labelcolor=self.colors['text'])
    
    def _add_title_and_labels(self, fig, ax1, signal: Optional[Dict], 
                            confidence_score: float, timeframe: str):
        """Add title and axis labels"""
        symbol = signal.get('symbol', 'CRYPTO') if signal else 'MARKET'
        strategy = signal.get('strategy_name', 'Analysis') if signal else 'Forecast'
        
        title = f"🧠 {strategy} Forecast: {symbol} ({timeframe})"
        fig.suptitle(title, fontsize=18, fontweight='bold', 
                    color=self.colors['text'], y=0.95)
        
        ax1.set_ylabel('Price', color=self.colors['text'], fontsize=12)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                ha='right', va='bottom', fontsize=8, 
                color=self.colors['text'], alpha=0.7)
    
    def _add_watermark(self, fig):
        """Add watermark to chart"""
        fig.text(0.5, 0.5, self.watermark, fontsize=40, alpha=0.1,
                ha='center', va='center', rotation=45, 
                color=self.colors['text'])
    
    def generate_simple_chart(self, symbol: str, price: float, 
                            direction: str, confidence: float) -> bytes:
        """
        Generate a simple chart for quick forecasts
        
        Args:
            symbol: Trading symbol
            price: Current price
            direction: Trade direction
            confidence: Confidence score
            
        Returns:
            bytes: PNG image data
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=self.colors['background'])
            
            # Create simple price visualization
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                end=datetime.now() + timedelta(hours=12), freq='1h')
            
            # Generate sample price movement
            base_price = price
            price_data = []
            for i, time in enumerate(times):
                if i < 24:  # Historical data
                    noise = np.random.normal(0, base_price * 0.01)
                    price_data.append(base_price + noise)
                else:  # Forecast data
                    trend = 0.02 if direction == 'LONG' else -0.02
                    forecast_price = base_price * (1 + trend * (i - 24) / 12)
                    price_data.append(forecast_price)
            
            # Plot historical vs forecast
            historical_times = times[:24]
            forecast_times = times[24:]
            historical_prices = price_data[:24]
            forecast_prices = price_data[24:]
            
            ax.plot(historical_times, historical_prices, 
                   color=self.colors['bullish'], linewidth=2, label='Historical')
            ax.plot(forecast_times, forecast_prices, 
                   color=self.colors['forecast_line'], linewidth=3, 
                   linestyle='--', label='Forecast')
            
            # Add current price marker
            ax.scatter(times[23], price, color=self.colors['neutral'], 
                      s=200, zorder=10, edgecolor='white', linewidth=2)
            
            # Styling
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.tick_params(colors=self.colors['text'])
            ax.legend(facecolor=self.colors['background'], 
                     edgecolor=self.colors['grid'], labelcolor=self.colors['text'])
            
            # Title and labels
            direction_emoji = "🟢" if direction == 'LONG' else "🔴"
            title = f"{direction_emoji} {symbol} Forecast - {confidence:.0f}% Confidence"
            ax.set_title(title, fontsize=16, fontweight='bold', 
                        color=self.colors['text'], pad=20)
            
            ax.set_ylabel('Price', color=self.colors['text'])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight',
                       facecolor=self.colors['background'], edgecolor='none')
            img_buffer.seek(0)
            
            plt.close(fig)
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate simple chart: {e}")
            plt.close('all')
            raise


# Global instance
chart_generator = ForecastChartGenerator(theme="dark", watermark="TRADING INTELLIGENCE")
