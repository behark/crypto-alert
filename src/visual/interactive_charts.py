"""
Interactive Chart Commands System
================================
Allow users to draw support/resistance zones and track setups visually
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class SupportResistanceLevel:
    """Support or resistance level annotation"""
    level_id: str
    user_id: str
    symbol: str
    timeframe: str
    level_type: str  # 'support', 'resistance'
    price: float
    strength: int  # 1-5 strength rating
    notes: str
    created_at: datetime
    last_tested: Optional[datetime] = None
    test_count: int = 0
    broken: bool = False

@dataclass
class SetupTracker:
    """Visual setup tracking"""
    setup_id: str
    user_id: str
    symbol: str
    timeframe: str
    setup_type: str  # 'breakout', 'pullback', 'reversal', etc.
    entry_zone: Tuple[float, float]  # (min_price, max_price)
    target_zones: List[Tuple[float, float]]
    stop_loss: float
    status: str  # 'pending', 'triggered', 'completed', 'stopped'
    notes: str
    created_at: datetime
    triggered_at: Optional[datetime] = None

@dataclass
class ChartAnnotation:
    """General chart annotation"""
    annotation_id: str
    user_id: str
    symbol: str
    timeframe: str
    annotation_type: str  # 'text', 'arrow', 'rectangle', 'trendline'
    coordinates: Dict[str, Any]
    text: str
    style: Dict[str, Any]
    created_at: datetime

class InteractiveChartHandler:
    """
    Interactive Chart Commands System
    
    Enables users to draw support/resistance levels, track setups visually,
    and add interactive annotations to charts.
    """
    
    def __init__(self, data_dir: str = "data/interactive_charts"):
        """Initialize the interactive chart handler"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for interactive elements
        self.sr_levels: Dict[str, SupportResistanceLevel] = {}
        self.setup_trackers: Dict[str, SetupTracker] = {}
        self.chart_annotations: Dict[str, ChartAnnotation] = {}
        
        # Load existing data
        self._load_interactive_data()
        
        logger.info("Interactive Chart Handler initialized")
    
    def draw_support_resistance(self, user_id: str, symbol: str, timeframe: str,
                               level_type: str, price: float, strength: int = 3,
                               notes: str = "") -> str:
        """
        Draw support or resistance level on chart
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            level_type: 'support' or 'resistance'
            price: Price level
            strength: Strength rating (1-5)
            notes: Optional notes
            
        Returns:
            str: Level ID
        """
        try:
            level_id = f"sr_{user_id}_{symbol}_{level_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            sr_level = SupportResistanceLevel(
                level_id=level_id,
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                level_type=level_type,
                price=price,
                strength=strength,
                notes=notes,
                created_at=datetime.now()
            )
            
            self.sr_levels[level_id] = sr_level
            self._save_sr_level(sr_level)
            
            logger.info(f"Drew {level_type} level: {price} for {symbol} {timeframe}")
            return level_id
            
        except Exception as e:
            logger.error(f"Failed to draw support/resistance: {e}")
            return ""
    
    def track_setup(self, user_id: str, symbol: str, timeframe: str,
                   setup_type: str, entry_zone: Tuple[float, float],
                   target_zones: List[Tuple[float, float]], stop_loss: float,
                   notes: str = "") -> str:
        """
        Track a visual setup
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            setup_type: Type of setup
            entry_zone: Entry price range (min, max)
            target_zones: List of target price ranges
            stop_loss: Stop loss price
            notes: Optional notes
            
        Returns:
            str: Setup ID
        """
        try:
            setup_id = f"setup_{user_id}_{symbol}_{setup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            setup = SetupTracker(
                setup_id=setup_id,
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                setup_type=setup_type,
                entry_zone=entry_zone,
                target_zones=target_zones,
                stop_loss=stop_loss,
                status='pending',
                notes=notes,
                created_at=datetime.now()
            )
            
            self.setup_trackers[setup_id] = setup
            self._save_setup_tracker(setup)
            
            logger.info(f"Tracking setup: {setup_type} for {symbol} {timeframe}")
            return setup_id
            
        except Exception as e:
            logger.error(f"Failed to track setup: {e}")
            return ""
    
    def add_chart_annotation(self, user_id: str, symbol: str, timeframe: str,
                           annotation_type: str, coordinates: Dict[str, Any],
                           text: str = "", style: Dict[str, Any] = None) -> str:
        """
        Add annotation to chart
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            annotation_type: Type of annotation
            coordinates: Position coordinates
            text: Annotation text
            style: Style properties
            
        Returns:
            str: Annotation ID
        """
        try:
            annotation_id = f"ann_{user_id}_{symbol}_{annotation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if style is None:
                style = {'color': '#ffff00', 'fontsize': 10}
            
            annotation = ChartAnnotation(
                annotation_id=annotation_id,
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                annotation_type=annotation_type,
                coordinates=coordinates,
                text=text,
                style=style,
                created_at=datetime.now()
            )
            
            self.chart_annotations[annotation_id] = annotation
            self._save_chart_annotation(annotation)
            
            logger.info(f"Added {annotation_type} annotation for {symbol} {timeframe}")
            return annotation_id
            
        except Exception as e:
            logger.error(f"Failed to add chart annotation: {e}")
            return ""
    
    def generate_interactive_chart(self, symbol: str, timeframe: str, 
                                 user_id: str = None) -> str:
        """
        Generate chart with interactive elements
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            user_id: Optional user filter
            
        Returns:
            str: Path to generated chart
        """
        try:
            logger.info(f"Generating interactive chart for {symbol} {timeframe}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2d2d2d')
            
            # Generate mock price data
            price_data = self._generate_mock_price_data(symbol, 100)
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(price_data)-1, -1, -1)]
            
            # Plot price chart
            ax.plot(timestamps, price_data, color='#00ff88', linewidth=2, alpha=0.8, label='Price')
            
            # Add support/resistance levels
            sr_levels = [
                level for level in self.sr_levels.values()
                if (level.symbol == symbol and level.timeframe == timeframe and
                    (user_id is None or level.user_id == user_id))
            ]
            
            for level in sr_levels:
                color = '#ff4444' if level.level_type == 'resistance' else '#44ff44'
                alpha = min(0.3 + (level.strength * 0.1), 0.8)
                
                ax.axhline(y=level.price, color=color, alpha=alpha, 
                          linewidth=2, linestyle='--', 
                          label=f'{level.level_type.title()} {level.price}')
                
                # Add strength indicator
                ax.text(timestamps[-1], level.price, f' S{level.strength}', 
                       color=color, fontsize=8, va='center')
            
            # Add setup trackers
            setups = [
                setup for setup in self.setup_trackers.values()
                if (setup.symbol == symbol and setup.timeframe == timeframe and
                    (user_id is None or setup.user_id == user_id))
            ]
            
            for setup in setups:
                # Entry zone
                entry_min, entry_max = setup.entry_zone
                ax.axhspan(entry_min, entry_max, alpha=0.2, color='#ffff00', 
                          label=f'{setup.setup_type} Entry Zone')
                
                # Target zones
                for i, (target_min, target_max) in enumerate(setup.target_zones):
                    ax.axhspan(target_min, target_max, alpha=0.15, color='#00ffff',
                              label=f'Target {i+1}' if i == 0 else "")
                
                # Stop loss
                ax.axhline(y=setup.stop_loss, color='#ff0000', alpha=0.6,
                          linewidth=1, linestyle=':', label='Stop Loss' if setup == setups[0] else "")
            
            # Add chart annotations
            annotations = [
                ann for ann in self.chart_annotations.values()
                if (ann.symbol == symbol and ann.timeframe == timeframe and
                    (user_id is None or ann.user_id == user_id))
            ]
            
            for ann in annotations:
                if ann.annotation_type == 'text':
                    x = ann.coordinates.get('x', timestamps[-1])
                    y = ann.coordinates.get('y', price_data[-1])
                    ax.text(x, y, ann.text, color=ann.style.get('color', '#ffff00'),
                           fontsize=ann.style.get('fontsize', 10))
                
                elif ann.annotation_type == 'arrow':
                    x1 = ann.coordinates.get('x1', timestamps[-10])
                    y1 = ann.coordinates.get('y1', price_data[-10])
                    x2 = ann.coordinates.get('x2', timestamps[-5])
                    y2 = ann.coordinates.get('y2', price_data[-5])
                    
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', 
                                             color=ann.style.get('color', '#ffff00'),
                                             lw=2))
            
            # Styling
            ax.set_title(f'{symbol} {timeframe} - Interactive Chart', 
                        color='white', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Time', color='white', fontsize=12)
            ax.set_ylabel('Price', color='white', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Style axes
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            
            # Add watermark
            fig.text(0.99, 0.01, '🎨 Interactive Trading Intelligence', 
                    fontsize=10, color='#666666', ha='right', va='bottom')
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = self.data_dir / f"interactive_chart_{symbol}_{timeframe}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close()
            
            logger.info(f"Generated interactive chart: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate interactive chart: {e}")
            return ""
    
    def update_setup_status(self, setup_id: str, status: str, 
                           triggered_at: datetime = None) -> bool:
        """
        Update setup status
        
        Args:
            setup_id: Setup identifier
            status: New status
            triggered_at: Optional trigger timestamp
            
        Returns:
            bool: Success status
        """
        try:
            if setup_id not in self.setup_trackers:
                return False
            
            setup = self.setup_trackers[setup_id]
            setup.status = status
            
            if triggered_at:
                setup.triggered_at = triggered_at
            
            self._save_setup_tracker(setup)
            
            logger.info(f"Updated setup {setup_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update setup status: {e}")
            return False
    
    def get_user_interactive_elements(self, user_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Get all interactive elements for a user
        
        Args:
            user_id: User identifier
            symbol: Optional symbol filter
            
        Returns:
            Dict containing user's interactive elements
        """
        try:
            # Filter elements by user and symbol
            user_sr_levels = [
                level for level in self.sr_levels.values()
                if (level.user_id == user_id and 
                    (symbol is None or level.symbol == symbol))
            ]
            
            user_setups = [
                setup for setup in self.setup_trackers.values()
                if (setup.user_id == user_id and
                    (symbol is None or setup.symbol == symbol))
            ]
            
            user_annotations = [
                ann for ann in self.chart_annotations.values()
                if (ann.user_id == user_id and
                    (symbol is None or ann.symbol == symbol))
            ]
            
            return {
                'user_id': user_id,
                'symbol_filter': symbol,
                'sr_levels': len(user_sr_levels),
                'active_setups': len([s for s in user_setups if s.status == 'pending']),
                'completed_setups': len([s for s in user_setups if s.status == 'completed']),
                'annotations': len(user_annotations),
                'total_elements': len(user_sr_levels) + len(user_setups) + len(user_annotations),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user interactive elements: {e}")
            return {'error': str(e)}
    
    def analyze_price_zones(self, symbol: str, timeframe: str, 
                           price_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Analyze a specific price zone
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_range: Price range to analyze (min, max)
            
        Returns:
            Dict containing zone analysis
        """
        try:
            min_price, max_price = price_range
            
            # Find relevant S/R levels in the zone
            relevant_levels = [
                level for level in self.sr_levels.values()
                if (level.symbol == symbol and level.timeframe == timeframe and
                    min_price <= level.price <= max_price)
            ]
            
            # Find setups in the zone
            relevant_setups = [
                setup for setup in self.setup_trackers.values()
                if (setup.symbol == symbol and setup.timeframe == timeframe and
                    (min_price <= setup.entry_zone[0] <= max_price or
                     min_price <= setup.entry_zone[1] <= max_price))
            ]
            
            # Calculate zone characteristics
            zone_width = max_price - min_price
            zone_center = (min_price + max_price) / 2
            
            # Determine zone significance
            total_strength = sum(level.strength for level in relevant_levels)
            significance = 'high' if total_strength >= 15 else 'medium' if total_strength >= 8 else 'low'
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'price_range': price_range,
                'zone_width': zone_width,
                'zone_center': zone_center,
                'sr_levels_count': len(relevant_levels),
                'setups_count': len(relevant_setups),
                'total_strength': total_strength,
                'significance': significance,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze price zone: {e}")
            return {'error': str(e)}
    
    def _generate_mock_price_data(self, symbol: str, periods: int) -> List[float]:
        """Generate mock price data for testing"""
        np.random.seed(hash(symbol) % 2**32)
        
        returns = np.random.normal(0.001, 0.02, periods)
        prices = [100.0]
        
        for return_rate in returns:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(new_price, 0.01))
        
        return prices[1:]
    
    def _save_sr_level(self, sr_level: SupportResistanceLevel):
        """Save S/R level to disk"""
        try:
            file_path = self.data_dir / f"sr_{sr_level.level_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(sr_level)
                data['created_at'] = sr_level.created_at.isoformat()
                if sr_level.last_tested:
                    data['last_tested'] = sr_level.last_tested.isoformat()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save S/R level: {e}")
    
    def _save_setup_tracker(self, setup: SetupTracker):
        """Save setup tracker to disk"""
        try:
            file_path = self.data_dir / f"setup_{setup.setup_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(setup)
                data['created_at'] = setup.created_at.isoformat()
                if setup.triggered_at:
                    data['triggered_at'] = setup.triggered_at.isoformat()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save setup tracker: {e}")
    
    def _save_chart_annotation(self, annotation: ChartAnnotation):
        """Save chart annotation to disk"""
        try:
            file_path = self.data_dir / f"ann_{annotation.annotation_id}.json"
            with open(file_path, 'w') as f:
                data = asdict(annotation)
                data['created_at'] = annotation.created_at.isoformat()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chart annotation: {e}")
    
    def _load_interactive_data(self):
        """Load interactive data from disk"""
        try:
            if not self.data_dir.exists():
                return
            
            # Load S/R levels
            for file_path in self.data_dir.glob("sr_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    if data.get('last_tested'):
                        data['last_tested'] = datetime.fromisoformat(data['last_tested'])
                    
                    sr_level = SupportResistanceLevel(**data)
                    self.sr_levels[sr_level.level_id] = sr_level
                except Exception as e:
                    logger.warning(f"Failed to load S/R level from {file_path}: {e}")
            
            # Load setup trackers
            for file_path in self.data_dir.glob("setup_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    if data.get('triggered_at'):
                        data['triggered_at'] = datetime.fromisoformat(data['triggered_at'])
                    
                    setup = SetupTracker(**data)
                    self.setup_trackers[setup.setup_id] = setup
                except Exception as e:
                    logger.warning(f"Failed to load setup tracker from {file_path}: {e}")
            
            # Load annotations
            for file_path in self.data_dir.glob("ann_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    
                    annotation = ChartAnnotation(**data)
                    self.chart_annotations[annotation.annotation_id] = annotation
                except Exception as e:
                    logger.warning(f"Failed to load annotation from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.sr_levels)} S/R levels, {len(self.setup_trackers)} setups, {len(self.chart_annotations)} annotations")
            
        except Exception as e:
            logger.error(f"Failed to load interactive data: {e}")


# Global instance
_interactive_chart_handler = None

def get_interactive_chart_handler() -> InteractiveChartHandler:
    """Get global interactive chart handler instance"""
    global _interactive_chart_handler
    if _interactive_chart_handler is None:
        _interactive_chart_handler = InteractiveChartHandler()
    return _interactive_chart_handler
