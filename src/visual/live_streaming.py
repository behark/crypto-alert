"""
Real-Time Chart Streaming System
===============================
Live updating charts with real-time forecast overlays and streaming ML confidence updates
"""

import logging
import asyncio
import websockets
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
from collections import deque
import io
import base64

logger = logging.getLogger(__name__)

@dataclass
class StreamingData:
    """Real-time streaming data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    forecast_overlay: Optional[Dict[str, Any]] = None
    ml_confidence: Optional[float] = None

@dataclass
class LiveForecastOverlay:
    """Live forecast overlay data"""
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    regime: str
    entry_zone: Tuple[float, float]
    target_zones: List[Tuple[float, float]]
    stop_loss: float
    timestamp: datetime

class LiveChartStreamer:
    """
    Real-Time Chart Streaming System
    
    Provides live updating charts with real-time forecast overlays,
    streaming ML confidence updates, and WebSocket-based price feeds.
    """
    
    def __init__(self, data_dir: str = "data/live_streaming"):
        """Initialize the live chart streamer"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Streaming data storage
        self.streaming_data: Dict[str, deque] = {}  # symbol -> deque of StreamingData
        self.active_streams: Dict[str, bool] = {}  # symbol -> active status
        self.forecast_overlays: Dict[str, LiveForecastOverlay] = {}  # symbol -> overlay
        
        # WebSocket connections
        self.websocket_connections: Dict[str, Any] = {}
        self.price_feeds: Dict[str, queue.Queue] = {}
        
        # Chart animation
        self.animated_charts: Dict[str, Any] = {}
        self.chart_update_callbacks: Dict[str, List[Callable]] = {}
        
        # Configuration
        self.max_data_points = 200  # Maximum data points to keep in memory
        self.update_interval = 1.0  # Update interval in seconds
        
        logger.info("Live Chart Streamer initialized")
    
    async def start_streaming(self, symbol: str, timeframe: str, 
                            websocket_url: str = None) -> bool:
        """
        Start live streaming for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            websocket_url: Optional WebSocket URL for price feed
            
        Returns:
            bool: Success status
        """
        try:
            stream_key = f"{symbol}_{timeframe}"
            
            if stream_key in self.active_streams and self.active_streams[stream_key]:
                logger.warning(f"Stream already active for {stream_key}")
                return True
            
            # Initialize data storage
            self.streaming_data[stream_key] = deque(maxlen=self.max_data_points)
            self.price_feeds[stream_key] = queue.Queue()
            self.active_streams[stream_key] = True
            
            # Start WebSocket connection if URL provided
            if websocket_url:
                asyncio.create_task(self._websocket_price_feed(symbol, websocket_url, stream_key))
            else:
                # Start mock price feed for testing
                threading.Thread(
                    target=self._mock_price_feed,
                    args=(symbol, stream_key),
                    daemon=True
                ).start()
            
            # Start data processing
            threading.Thread(
                target=self._process_streaming_data,
                args=(stream_key,),
                daemon=True
            ).start()
            
            logger.info(f"Started live streaming for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self, symbol: str, timeframe: str) -> bool:
        """
        Stop live streaming for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            
        Returns:
            bool: Success status
        """
        try:
            stream_key = f"{symbol}_{timeframe}"
            
            if stream_key in self.active_streams:
                self.active_streams[stream_key] = False
                
                # Close WebSocket connection if exists
                if stream_key in self.websocket_connections:
                    # WebSocket cleanup would go here
                    pass
                
                logger.info(f"Stopped live streaming for {symbol} {timeframe}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop streaming: {e}")
            return False
    
    def update_forecast_overlay(self, symbol: str, timeframe: str, 
                              forecast_data: Dict[str, Any]) -> bool:
        """
        Update forecast overlay for live chart
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            forecast_data: Forecast data to overlay
            
        Returns:
            bool: Success status
        """
        try:
            overlay = LiveForecastOverlay(
                symbol=symbol,
                timeframe=timeframe,
                direction=forecast_data.get('direction', 'neutral'),
                confidence=forecast_data.get('confidence', 50.0),
                regime=forecast_data.get('regime', 'sideways'),
                entry_zone=forecast_data.get('entry_zone', (0.0, 0.0)),
                target_zones=forecast_data.get('target_zones', []),
                stop_loss=forecast_data.get('stop_loss', 0.0),
                timestamp=datetime.now()
            )
            
            stream_key = f"{symbol}_{timeframe}"
            self.forecast_overlays[stream_key] = overlay
            
            # Trigger chart update
            self._trigger_chart_update(stream_key)
            
            logger.info(f"Updated forecast overlay for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update forecast overlay: {e}")
            return False
    
    def create_live_chart(self, symbol: str, timeframe: str, 
                         chart_type: str = 'line') -> str:
        """
        Create live updating chart
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            chart_type: Type of chart ('line', 'candlestick')
            
        Returns:
            str: Chart identifier
        """
        try:
            stream_key = f"{symbol}_{timeframe}"
            chart_id = f"live_chart_{stream_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(16, 10))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2d2d2d')
            
            # Initialize empty plot
            line, = ax.plot([], [], color='#00ff88', linewidth=2, alpha=0.8, label='Price')
            
            # Setup chart styling
            ax.set_title(f'{symbol} {timeframe} - Live Chart', 
                        color='white', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', color='white', fontsize=12)
            ax.set_ylabel('Price', color='white', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Style axes
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            
            # Store chart components
            self.animated_charts[chart_id] = {
                'fig': fig,
                'ax': ax,
                'line': line,
                'symbol': symbol,
                'timeframe': timeframe,
                'stream_key': stream_key,
                'chart_type': chart_type
            }
            
            # Setup animation
            ani = animation.FuncAnimation(
                fig, 
                self._update_chart_animation,
                fargs=(chart_id,),
                interval=int(self.update_interval * 1000),
                blit=False
            )
            
            self.animated_charts[chart_id]['animation'] = ani
            
            logger.info(f"Created live chart: {chart_id}")
            return chart_id
            
        except Exception as e:
            logger.error(f"Failed to create live chart: {e}")
            return ""
    
    def stream_to_telegram(self, chat_id: str, symbol: str, timeframe: str) -> bool:
        """
        Stream live chart updates to Telegram
        
        Args:
            chat_id: Telegram chat ID
            symbol: Trading symbol
            timeframe: Chart timeframe
            
        Returns:
            bool: Success status
        """
        try:
            # This would integrate with the Telegram bot to send live updates
            # For now, just log the request
            logger.info(f"Starting Telegram stream for {symbol} {timeframe} to chat {chat_id}")
            
            # Create callback for Telegram updates
            def telegram_update_callback(chart_data):
                # This would send updated chart to Telegram
                logger.info(f"Sending chart update to Telegram chat {chat_id}")
            
            # Register callback
            stream_key = f"{symbol}_{timeframe}"
            if stream_key not in self.chart_update_callbacks:
                self.chart_update_callbacks[stream_key] = []
            
            self.chart_update_callbacks[stream_key].append(telegram_update_callback)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Telegram streaming: {e}")
            return False
    
    def get_streaming_status(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get status of active streams
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dict containing streaming status
        """
        try:
            status = {
                'timestamp': datetime.now(),
                'total_active_streams': sum(self.active_streams.values()),
                'streams': []
            }
            
            for stream_key, is_active in self.active_streams.items():
                if not is_active:
                    continue
                
                stream_symbol = stream_key.split('_')[0]
                if symbol and stream_symbol != symbol:
                    continue
                
                stream_info = {
                    'stream_key': stream_key,
                    'symbol': stream_symbol,
                    'active': is_active,
                    'data_points': len(self.streaming_data.get(stream_key, [])),
                    'has_forecast_overlay': stream_key in self.forecast_overlays,
                    'has_websocket': stream_key in self.websocket_connections
                }
                
                status['streams'].append(stream_info)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get streaming status: {e}")
            return {'error': str(e)}
    
    def get_live_chart_snapshot(self, chart_id: str) -> str:
        """
        Get snapshot of live chart as base64 image
        
        Args:
            chart_id: Chart identifier
            
        Returns:
            str: Base64 encoded image
        """
        try:
            if chart_id not in self.animated_charts:
                return ""
            
            chart = self.animated_charts[chart_id]
            fig = chart['fig']
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1e1e1e', edgecolor='none')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to get chart snapshot: {e}")
            return ""
    
    async def _websocket_price_feed(self, symbol: str, websocket_url: str, stream_key: str):
        """WebSocket price feed handler"""
        try:
            async with websockets.connect(websocket_url) as websocket:
                self.websocket_connections[stream_key] = websocket
                
                # Subscribe to symbol
                subscribe_message = {
                    "method": "SUBSCRIBE",
                    "params": [f"{symbol.lower()}@ticker"],
                    "id": 1
                }
                await websocket.send(json.dumps(subscribe_message))
                
                while self.active_streams.get(stream_key, False):
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Parse price data
                        if 'c' in data:  # Current price
                            price = float(data['c'])
                            volume = float(data.get('v', 0))
                            
                            # Add to queue
                            streaming_data = StreamingData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                price=price,
                                volume=volume
                            )
                            
                            self.price_feeds[stream_key].put(streaming_data)
                    
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning(f"WebSocket error: {e}")
                        break
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    
    def _mock_price_feed(self, symbol: str, stream_key: str):
        """Mock price feed for testing"""
        try:
            base_price = 100.0
            current_price = base_price
            
            while self.active_streams.get(stream_key, False):
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.5)  # Small random changes
                current_price = max(current_price + price_change, 0.01)
                
                volume = np.random.uniform(1000, 10000)
                
                streaming_data = StreamingData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    volume=volume
                )
                
                self.price_feeds[stream_key].put(streaming_data)
                
                # Sleep for update interval
                threading.Event().wait(self.update_interval)
                
        except Exception as e:
            logger.error(f"Mock price feed error: {e}")
    
    def _process_streaming_data(self, stream_key: str):
        """Process streaming data from queue"""
        try:
            while self.active_streams.get(stream_key, False):
                try:
                    # Get data from queue with timeout
                    data = self.price_feeds[stream_key].get(timeout=1.0)
                    
                    # Add to streaming data storage
                    self.streaming_data[stream_key].append(data)
                    
                    # Trigger chart updates
                    self._trigger_chart_update(stream_key)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.warning(f"Data processing error: {e}")
                    
        except Exception as e:
            logger.error(f"Streaming data processing failed: {e}")
    
    def _trigger_chart_update(self, stream_key: str):
        """Trigger chart update callbacks"""
        try:
            if stream_key in self.chart_update_callbacks:
                chart_data = {
                    'stream_key': stream_key,
                    'latest_data': list(self.streaming_data.get(stream_key, []))[-10:],  # Last 10 points
                    'forecast_overlay': self.forecast_overlays.get(stream_key),
                    'timestamp': datetime.now()
                }
                
                for callback in self.chart_update_callbacks[stream_key]:
                    try:
                        callback(chart_data)
                    except Exception as e:
                        logger.warning(f"Chart update callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to trigger chart update: {e}")
    
    def _update_chart_animation(self, frame, chart_id: str):
        """Update animated chart"""
        try:
            if chart_id not in self.animated_charts:
                return
            
            chart = self.animated_charts[chart_id]
            stream_key = chart['stream_key']
            
            if stream_key not in self.streaming_data:
                return
            
            # Get recent data
            data_points = list(self.streaming_data[stream_key])
            if not data_points:
                return
            
            # Extract timestamps and prices
            timestamps = [point.timestamp for point in data_points]
            prices = [point.price for point in data_points]
            
            # Update line plot
            line = chart['line']
            line.set_data(timestamps, prices)
            
            # Update axes
            ax = chart['ax']
            if timestamps and prices:
                ax.set_xlim(timestamps[0], timestamps[-1])
                ax.set_ylim(min(prices) * 0.99, max(prices) * 1.01)
            
            # Add forecast overlay if available
            if stream_key in self.forecast_overlays:
                overlay = self.forecast_overlays[stream_key]
                
                # Clear previous overlays
                for artist in ax.patches + ax.lines[1:]:  # Keep main price line
                    artist.remove()
                
                # Add entry zone
                if overlay.entry_zone[0] > 0 and overlay.entry_zone[1] > 0:
                    ax.axhspan(overlay.entry_zone[0], overlay.entry_zone[1], 
                              alpha=0.2, color='yellow', label='Entry Zone')
                
                # Add target zones
                for i, (target_min, target_max) in enumerate(overlay.target_zones):
                    if target_min > 0 and target_max > 0:
                        ax.axhspan(target_min, target_max, alpha=0.15, color='cyan',
                                  label=f'Target {i+1}' if i == 0 else "")
                
                # Add stop loss
                if overlay.stop_loss > 0:
                    ax.axhline(y=overlay.stop_loss, color='red', alpha=0.6,
                              linewidth=1, linestyle=':', label='Stop Loss')
                
                # Update legend
                ax.legend()
            
            # Update title with latest price
            if prices:
                latest_price = prices[-1]
                chart['ax'].set_title(
                    f"{chart['symbol']} {chart['timeframe']} - Live: ${latest_price:.4f}",
                    color='white', fontsize=16, fontweight='bold'
                )
            
        except Exception as e:
            logger.error(f"Chart animation update error: {e}")


# Global instance
_live_chart_streamer = None

def get_live_chart_streamer() -> LiveChartStreamer:
    """Get global live chart streamer instance"""
    global _live_chart_streamer
    if _live_chart_streamer is None:
        _live_chart_streamer = LiveChartStreamer()
    return _live_chart_streamer
