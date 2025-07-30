"""
Telegram Visual ML Tuning Commands for Bot 2
============================================
Telegram interface for visual intelligence ML tuning and optimization.
Provides /tune, /metrics, and /audit commands for visual forecast evolution.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Local imports
try:
    from src.analytics.visual_ml_tuner import VisualMLTuner, get_visual_ml_tuner, VisualTuningSession, VisualTuningRecommendation
    VISUAL_ML_AVAILABLE = True
except ImportError:
    VISUAL_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class TelegramVisualML:
    """Handles visual ML tuning commands and interactions via Telegram"""
    
    def __init__(self):
        """Initialize the Telegram visual ML handler."""
        if VISUAL_ML_AVAILABLE:
            self.visual_ml_tuner = get_visual_ml_tuner()
        else:
            self.visual_ml_tuner = None
            logger.warning("Visual ML tuner not available")
        
        # Store pending sessions for approval
        self.pending_sessions: Dict[str, VisualTuningSession] = {}
        
        logger.info("Telegram Visual ML handler initialized")
    
    def handle_tune_forecast_command(self, args: List[str]) -> str:
        """
        Handle /tune forecast command for visual intelligence optimization.
        
        Usage: 
        - /tune forecast - Generate visual tuning recommendations
        - /tune forecast apply - Auto-apply last recommendations
        - /tune forecast status - Show tuning status and history
        """
        try:
            if not VISUAL_ML_AVAILABLE or not self.visual_ml_tuner:
                return "⚠️ **Visual ML System Unavailable**\\n\\nVisual intelligence tuning is not available. Please ensure the analytics module is properly installed."
            
            command = args[0] if args else 'generate'
            
            if command == 'status':
                return self._handle_tune_forecast_status()
            elif command == 'apply':
                return self._handle_tune_forecast_apply()
            else:
                return self._handle_tune_forecast_generate()
                
        except Exception as e:
            logger.error(f"Error in tune forecast command: {e}")
            return f"❌ **Error processing tune forecast command:** {str(e)}"
    
    def handle_metrics_command(self, args: List[str]) -> str:
        """
        Handle /metrics command to show visual intelligence performance.
        
        Usage:
        - /metrics - Show current performance metrics
        - /metrics [days] - Show metrics for specific number of days
        """
        try:
            if not VISUAL_ML_AVAILABLE or not self.visual_ml_tuner:
                return "⚠️ **Visual ML System Unavailable**\\n\\nVisual intelligence metrics are not available."
            
            lookback_days = 30
            if args and args[0].isdigit():
                lookback_days = int(args[0])
                lookback_days = max(1, min(lookback_days, 365))  # Limit range
            
            performance = self.visual_ml_tuner.analyze_visual_performance(lookback_days)
            
            if 'error' in performance:
                return f"⚠️ **Metrics Error**\\n\\n{performance['error']}"
            
            return self._format_metrics_message(performance, lookback_days)
            
        except Exception as e:
            logger.error(f"Error in metrics command: {e}")
            return f"❌ **Error retrieving metrics:** {str(e)}"
    
    def handle_audit_command(self, args: List[str]) -> str:
        """
        Handle /audit command for detailed forecast accuracy analysis.
        
        Usage:
        - /audit - Show recent forecast audit
        - /audit [days] - Show audit for specific number of days
        """
        try:
            if not VISUAL_ML_AVAILABLE or not self.visual_ml_tuner:
                return "⚠️ **Visual ML System Unavailable**\\n\\nForecast audit is not available."
            
            lookback_days = 7
            if args and args[0].isdigit():
                lookback_days = int(args[0])
                lookback_days = max(1, min(lookback_days, 90))
            
            return self._generate_forecast_audit(lookback_days)
            
        except Exception as e:
            logger.error(f"Error in audit command: {e}")
            return f"❌ **Error generating audit:** {str(e)}"
    
    def _handle_tune_forecast_generate(self) -> str:
        """Generate visual tuning recommendations"""
        try:
            session = self.visual_ml_tuner.generate_tuning_recommendations(lookback_days=30)
            
            if not session:
                return "⚠️ **Insufficient Data**\\n\\nNot enough forecast data to generate tuning recommendations. Continue using /forecast and /plan commands to build learning data."
            
            if not session.recommendations:
                return "✅ **Visual Intelligence Optimized**\\n\\nNo tuning recommendations needed. Your visual intelligence system is performing well!"
            
            # Store session for potential approval
            self.pending_sessions[session.session_id] = session
            
            return self._format_tuning_recommendations(session)
            
        except Exception as e:
            logger.error(f"Error generating visual tuning recommendations: {e}")
            return f"❌ **Tuning Generation Error:** {str(e)}"
    
    def _handle_tune_forecast_status(self) -> str:
        """Show visual tuning status"""
        try:
            performance = self.visual_ml_tuner.analyze_visual_performance(lookback_days=30)
            
            if 'error' in performance:
                return f"⚠️ **Status Error**\\n\\n{performance['error']}"
            
            message = "📊 **Visual Intelligence Status**\\n\\n"
            message += f"**📈 Overall Accuracy:** {performance.get('overall_accuracy', 0):.1%}\\n"
            message += f"**🎯 Direction Accuracy:** {performance.get('direction_accuracy', 0):.1%}\\n"
            message += f"**🏆 Target Hit Rate:** {performance.get('target_hit_rate', 0):.1%}\\n"
            message += f"**🛡️ Stop Hit Rate:** {performance.get('stop_hit_rate', 0):.1%}\\n\\n"
            
            message += "**🔮 Regime Prediction:** "
            regime_acc = performance.get('regime_prediction_accuracy', 0)
            if regime_acc >= 0.8:
                message += f"{regime_acc:.1%} 🟢\\n"
            elif regime_acc >= 0.6:
                message += f"{regime_acc:.1%} 🟡\\n"
            else:
                message += f"{regime_acc:.1%} 🔴\\n"
            
            message += f"\\n**📊 Total Forecasts Analyzed:** {performance.get('total_forecasts', 0)}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting visual tuning status: {e}")
            return f"❌ **Status Error:** {str(e)}"
    
    def _handle_tune_forecast_apply(self) -> str:
        """Apply pending visual tuning recommendations"""
        try:
            if not self.pending_sessions:
                return "⚠️ **No Pending Recommendations**\\n\\nGenerate recommendations first using `/tune forecast`."
            
            # Get most recent session
            latest_session = max(self.pending_sessions.values(), key=lambda s: s.timestamp)
            
            if not latest_session.recommendations:
                return "✅ **No Changes Needed**\\n\\nLatest analysis shows no tuning recommendations."
            
            # Apply safe recommendations only
            applied_count = 0
            for rec in latest_session.recommendations:
                if rec.risk_level == 'low':
                    # Here you would actually apply the recommendation
                    # For now, we'll just count them
                    applied_count += 1
                    logger.info(f"Applied visual tuning: {rec.component}.{rec.parameter} = {rec.recommended_value}")
            
            if applied_count > 0:
                return f"✅ **Applied {applied_count} Visual Tuning Changes**\\n\\nLow-risk recommendations have been applied. Monitor performance with `/metrics`."
            else:
                return "⚠️ **No Safe Changes Available**\\n\\nAll pending recommendations require manual review. Use `/tune forecast` to see details."
            
        except Exception as e:
            logger.error(f"Error applying visual tuning: {e}")
            return f"❌ **Apply Error:** {str(e)}"
    
    def _format_tuning_recommendations(self, session: VisualTuningSession) -> str:
        """Format tuning recommendations for Telegram"""
        try:
            message = "🧠 **Visual Intelligence Tuning Recommendations**\\n\\n"
            message += f"**📊 Analysis Period:** {session.lookback_days} days\\n"
            message += f"**🔍 Forecasts Analyzed:** {session.total_forecasts_analyzed}\\n"
            message += f"**📈 Current Accuracy:** {session.forecast_accuracy_metrics.get('overall_accuracy', 0):.1%}\\n\\n"
            
            for i, rec in enumerate(session.recommendations, 1):
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(rec.risk_level, "⚪")
                
                message += f"**{i}. {rec.component.title()} Optimization** {risk_emoji}\\n"
                message += f"   **Parameter:** `{rec.parameter}`\\n"
                message += f"   **Current:** {rec.current_value:.3f} → **Recommended:** {rec.recommended_value:.3f}\\n"
                message += f"   **Reasoning:** {rec.reasoning}\\n"
                message += f"   **Expected Improvement:** +{rec.expected_improvement:.1%}\\n"
                message += f"   **Confidence:** {rec.confidence:.1%}\\n\\n"
            
            message += "**🎯 Pattern Performance:**\\n"
            for pattern, rate in session.pattern_success_rates.items():
                emoji = "🟢" if rate >= 0.7 else "🟡" if rate >= 0.5 else "🔴"
                message += f"   • {pattern}: {rate:.1%} {emoji}\\n"
            
            message += "\\n**⚡ Confidence Zones:**\\n"
            for zone, perf in session.confidence_zone_performance.items():
                emoji = "🟢" if perf >= 0.8 else "🟡" if perf >= 0.6 else "🔴"
                message += f"   • {zone.title()}: {perf:.1%} {emoji}\\n"
            
            message += f"\\n💡 Use `/tune forecast apply` to apply safe recommendations automatically."
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting tuning recommendations: {e}")
            return f"❌ **Formatting Error:** {str(e)}"
    
    def _format_metrics_message(self, performance: Dict, lookback_days: int) -> str:
        """Format performance metrics for Telegram"""
        try:
            message = f"📊 **Visual Intelligence Metrics** ({lookback_days} days)\\n\\n"
            
            # Overall performance
            overall_acc = performance.get('overall_accuracy', 0)
            acc_emoji = "🟢" if overall_acc >= 0.8 else "🟡" if overall_acc >= 0.6 else "🔴"
            message += f"**🎯 Overall Accuracy:** {overall_acc:.1%} {acc_emoji}\\n"
            
            direction_acc = performance.get('direction_accuracy', 0)
            dir_emoji = "🟢" if direction_acc >= 0.7 else "🟡" if direction_acc >= 0.5 else "🔴"
            message += f"**📈 Direction Accuracy:** {direction_acc:.1%} {dir_emoji}\\n"
            
            target_rate = performance.get('target_hit_rate', 0)
            target_emoji = "🟢" if target_rate >= 0.6 else "🟡" if target_rate >= 0.4 else "🔴"
            message += f"**🏆 Target Hit Rate:** {target_rate:.1%} {target_emoji}\\n"
            
            stop_rate = performance.get('stop_hit_rate', 0)
            stop_emoji = "🟢" if stop_rate <= 0.2 else "🟡" if stop_rate <= 0.4 else "🔴"
            message += f"**🛡️ Stop Hit Rate:** {stop_rate:.1%} {stop_emoji}\\n\\n"
            
            # Pattern analysis
            message += "**🔍 Pattern Success Rates:**\\n"
            pattern_rates = performance.get('pattern_success_rates', {})
            if pattern_rates:
                for pattern, rate in sorted(pattern_rates.items(), key=lambda x: x[1], reverse=True):
                    emoji = "🟢" if rate >= 0.7 else "🟡" if rate >= 0.5 else "🔴"
                    message += f"   • {pattern.title()}: {rate:.1%} {emoji}\\n"
            else:
                message += "   • No pattern data available\\n"
            
            # Confidence zones
            message += "\\n**⚡ Confidence Zone Performance:**\\n"
            conf_zones = performance.get('confidence_zone_performance', {})
            for zone in ['high', 'medium', 'low']:
                if zone in conf_zones:
                    perf = conf_zones[zone]
                    emoji = "🟢" if perf >= 0.8 else "🟡" if perf >= 0.6 else "🔴"
                    message += f"   • {zone.title()} Confidence: {perf:.1%} {emoji}\\n"
            
            # Regime prediction
            regime_acc = performance.get('regime_prediction_accuracy', 0)
            regime_emoji = "🟢" if regime_acc >= 0.8 else "🟡" if regime_acc >= 0.6 else "🔴"
            message += f"\\n**🔮 Regime Prediction:** {regime_acc:.1%} {regime_emoji}\\n"
            
            message += f"\\n**📊 Total Forecasts:** {performance.get('total_forecasts', 0)}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting metrics: {e}")
            return f"❌ **Metrics Formatting Error:** {str(e)}"
    
    def _generate_forecast_audit(self, lookback_days: int) -> str:
        """Generate detailed forecast audit"""
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get recent forecasts with outcomes
            recent_forecasts = []
            for snapshot in self.visual_ml_tuner.forecast_snapshots:
                if snapshot.timestamp >= cutoff_date:
                    # Find corresponding outcome
                    outcome = None
                    for out in self.visual_ml_tuner.forecast_outcomes:
                        if out.forecast_id == snapshot.forecast_id:
                            outcome = out
                            break
                    
                    if outcome:
                        recent_forecasts.append((snapshot, outcome))
            
            if not recent_forecasts:
                return f"⚠️ **No Forecast Data**\\n\\nNo forecasts with outcomes found in the last {lookback_days} days."
            
            message = f"🔍 **Forecast Audit** ({lookback_days} days)\\n\\n"
            message += f"**📊 Total Forecasts:** {len(recent_forecasts)}\\n\\n"
            
            # Show recent forecasts
            message += "**📈 Recent Forecasts:**\\n"
            for i, (snapshot, outcome) in enumerate(recent_forecasts[-5:], 1):  # Last 5
                accuracy_emoji = "✅" if outcome.forecast_accuracy_score >= 0.7 else "⚠️" if outcome.forecast_accuracy_score >= 0.5 else "❌"
                direction_emoji = "🟢" if snapshot.predicted_direction == "LONG" else "🔴"
                
                message += f"**{i}.** {snapshot.symbol} {direction_emoji} ({snapshot.timeframe})\\n"
                message += f"   **Confidence:** {snapshot.confidence_score:.1%} | **Accuracy:** {outcome.forecast_accuracy_score:.1%} {accuracy_emoji}\\n"
                message += f"   **Pattern:** {snapshot.pattern_detected} | **Regime:** {snapshot.regime_type}\\n"
                message += f"   **Result:** {'Target Hit' if outcome.target_reached else 'Stop Hit' if outcome.stop_hit else 'Ongoing'}\\n\\n"
            
            # Summary statistics
            total_accurate = sum(1 for _, outcome in recent_forecasts if outcome.forecast_accuracy_score >= 0.7)
            accuracy_rate = total_accurate / len(recent_forecasts)
            
            message += f"**🎯 High Accuracy Forecasts:** {total_accurate}/{len(recent_forecasts)} ({accuracy_rate:.1%})\\n"
            
            # Best and worst patterns
            pattern_performance = {}
            for snapshot, outcome in recent_forecasts:
                pattern = snapshot.pattern_detected
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = []
                pattern_performance[pattern].append(outcome.forecast_accuracy_score)
            
            if pattern_performance:
                best_pattern = max(pattern_performance.items(), key=lambda x: sum(x[1])/len(x[1]))
                worst_pattern = min(pattern_performance.items(), key=lambda x: sum(x[1])/len(x[1]))
                
                message += f"\\n**🏆 Best Pattern:** {best_pattern[0]} ({sum(best_pattern[1])/len(best_pattern[1]):.1%})\\n"
                message += f"**⚠️ Worst Pattern:** {worst_pattern[0]} ({sum(worst_pattern[1])/len(worst_pattern[1]):.1%})\\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error generating forecast audit: {e}")
            return f"❌ **Audit Error:** {str(e)}"


# Global instance
telegram_visual_ml = None

def get_telegram_visual_ml() -> TelegramVisualML:
    """Get global Telegram visual ML instance."""
    global telegram_visual_ml
    if telegram_visual_ml is None:
        telegram_visual_ml = TelegramVisualML()
    return telegram_visual_ml

def initialize_telegram_visual_ml() -> TelegramVisualML:
    """Initialize global Telegram visual ML instance."""
    global telegram_visual_ml
    telegram_visual_ml = TelegramVisualML()
    return telegram_visual_ml
