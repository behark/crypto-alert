"""
Unified Telegram Interface
=========================
Single Telegram interface for all orchestration and visual intelligence commands
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

# Import orchestration components
from ..orchestration.command_center import get_command_center, BotCapability
from ..orchestration.bot_registry import get_bot_registry
from ..orchestration.intelligence_sharing import get_intelligence_sharing
from ..orchestration.risk_coordinator import get_risk_coordinator

# Import visual intelligence components
from ..visual.pattern_learning import get_pattern_learner
from ..visual.multi_asset_correlation import get_multi_asset_correlator
from ..visual.interactive_charts import get_interactive_chart_handler
from ..visual.live_streaming import get_live_chart_streamer
from ..visual.advanced_renderer import get_advanced_renderer

logger = logging.getLogger(__name__)

class UnifiedTelegramInterface:
    """
    Unified Telegram Interface
    
    Single interface for all orchestration and visual intelligence commands,
    providing a centralized command center for the Living Trading Intelligence Platform.
    """
    
    def __init__(self, telegram_notifier=None):
        """Initialize the unified Telegram interface"""
        self.telegram_notifier = telegram_notifier
        
        # Get component instances
        self.command_center = get_command_center()
        self.bot_registry = get_bot_registry()
        self.intelligence_sharing = get_intelligence_sharing()
        self.risk_coordinator = get_risk_coordinator()
        
        self.pattern_learner = get_pattern_learner()
        self.multi_asset_correlator = get_multi_asset_correlator()
        self.interactive_chart_handler = get_interactive_chart_handler()
        self.live_chart_streamer = get_live_chart_streamer()
        self.advanced_renderer = get_advanced_renderer()
        
        logger.info("Unified Telegram Interface initialized")
    
    async def handle_dashboard_command(self, update: Update, context: CallbackContext):
        """Handle /dashboard command - Unified bot status and performance"""
        try:
            logger.info("Processing /dashboard command")
            
            # Get comprehensive dashboard data
            dashboard_data = self.command_center.get_unified_dashboard_data()
            
            if 'error' in dashboard_data:
                await update.message.reply_text(f"❌ **Dashboard Error**\n\n{dashboard_data['error']}")
                return
            
            # Format dashboard message
            message = "🧠 **LIVING TRADING INTELLIGENCE PLATFORM**\n"
            message += "=" * 45 + "\n\n"
            
            # Overall statistics
            message += f"📊 **Platform Overview**\n"
            message += f"• Total Bots: {dashboard_data['total_bots']}\n"
            message += f"• Online Bots: {dashboard_data['online_bots']}\n"
            message += f"• Recent Commands: {dashboard_data['recent_commands']}\n\n"
            
            # Bot capabilities summary
            capabilities = dashboard_data.get('capabilities_summary', {})
            message += f"🤖 **Bot Capabilities**\n"
            for capability, count in capabilities.items():
                message += f"• {capability.title()}: {count} bots\n"
            message += "\n"
            
            # Individual bot status
            message += f"🔧 **Bot Status**\n"
            for bot in dashboard_data.get('bots', [])[:5]:  # Show first 5 bots
                status_emoji = "🟢" if bot['status'] == 'online' else "🔴"
                message += f"{status_emoji} **{bot['name']}**\n"
                message += f"   Status: {bot['status']}\n"
                message += f"   Capabilities: {', '.join(bot['capabilities'])}\n\n"
            
            # Risk status
            risk_status = dashboard_data.get('risk_status', {})
            if risk_status and 'error' not in risk_status:
                message += f"⚠️ **Risk Status**\n"
                message += f"• Risk Utilization: {risk_status.get('risk_utilization', 0):.1%}\n"
                message += f"• Overall Risk: {risk_status.get('overall_risk_level', 'unknown').title()}\n\n"
            
            # Intelligence sharing summary
            intel_summary = dashboard_data.get('cross_bot_intelligence', {})
            if intel_summary and 'error' not in intel_summary:
                message += f"🧠 **Shared Intelligence**\n"
                message += f"• Total Intelligence: {intel_summary.get('total_intelligence', 0)}\n"
                message += f"• Success Rate: {intel_summary.get('average_success_rate', 0):.1f}%\n\n"
            
            message += f"⏰ **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Create inline keyboard for quick actions
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="dashboard_refresh")],
                [InlineKeyboardButton("🤖 Bot Details", callback_data="bot_details"),
                 InlineKeyboardButton("⚠️ Risk Details", callback_data="risk_details")],
                [InlineKeyboardButton("🧠 Intelligence", callback_data="intelligence_details")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle dashboard command: {e}")
            await update.message.reply_text(f"❌ **Dashboard Error:** {str(e)}")
    
    async def handle_bots_command(self, update: Update, context: CallbackContext):
        """Handle /bots command - List all registered bots and capabilities"""
        try:
            logger.info("Processing /bots command")
            
            bot_status = self.command_center.get_bot_status()
            
            if 'error' in bot_status:
                await update.message.reply_text(f"❌ **Bots Error**\n\n{bot_status['error']}")
                return
            
            message = "🤖 **REGISTERED TRADING BOTS**\n"
            message += "=" * 35 + "\n\n"
            
            message += f"📊 **Summary**\n"
            message += f"• Total Bots: {bot_status['total_bots']}\n\n"
            
            # List individual bots
            for bot_id, bot_info in bot_status.get('bots', {}).items():
                status_emoji = "🟢" if bot_info['status'] == 'online' else "🔴"
                message += f"{status_emoji} **{bot_info['name']}** (`{bot_id}`)\n"
                message += f"   Status: {bot_info['status']}\n"
                message += f"   Capabilities: {', '.join(bot_info['capabilities'])}\n\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle bots command: {e}")
            await update.message.reply_text(f"❌ **Bots Error:** {str(e)}")
    
    async def handle_cross_forecast_command(self, update: Update, context: CallbackContext):
        """Handle /cross forecast command - Run forecast across all capable bots"""
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ **Usage:** `/cross forecast SYMBOL TIMEFRAME`\n\n"
                    "**Example:** `/cross forecast BTCUSDT 1h`"
                )
                return
            
            symbol = args[0].upper()
            timeframe = args[1]
            
            logger.info(f"Processing cross-bot forecast: {symbol} {timeframe}")
            
            # Execute cross-bot command
            command_id = self.command_center.execute_cross_bot_command(
                command='forecast',
                args=[symbol, timeframe],
                requester=f"telegram_user_{update.effective_user.id}"
            )
            
            if command_id.startswith("❌"):
                await update.message.reply_text(command_id)
                return
            
            # Send initial response
            await update.message.reply_text(
                f"🚀 **Cross-Bot Forecast Initiated**\n\n"
                f"📊 **Symbol:** {symbol}\n"
                f"⏰ **Timeframe:** {timeframe}\n"
                f"🆔 **Command ID:** `{command_id}`\n\n"
                f"⏳ Processing across all capable bots..."
            )
            
            # Wait for results (in real implementation, this would be async)
            await asyncio.sleep(2)  # Mock processing time
            
            # Get command results
            results = self.command_center.get_command_results(command_id)
            
            if not results:
                await update.message.reply_text("⏳ **Results not ready yet.** Please check back in a moment.")
                return
            
            # Format results message
            message = f"📈 **CROSS-BOT FORECAST RESULTS**\n"
            message += f"Symbol: {symbol} | Timeframe: {timeframe}\n"
            message += "=" * 40 + "\n\n"
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            message += f"✅ **Successful:** {len(successful_results)} bots\n"
            message += f"❌ **Failed:** {len(failed_results)} bots\n\n"
            
            # Show individual results
            for result in successful_results[:3]:  # Show first 3 successful results
                message += f"🤖 **Bot:** {result.bot_id}\n"
                message += f"⚡ **Execution Time:** {result.execution_time:.2f}s\n"
                message += f"📊 **Result:** {str(result.result)[:100]}...\n\n"
            
            # Get consensus forecast if available
            consensus = self.intelligence_sharing.get_forecast_consensus(symbol, timeframe)
            if consensus:
                message += f"🧠 **CONSENSUS FORECAST**\n"
                message += f"Direction: {consensus['direction'].upper()}\n"
                message += f"Strength: {consensus['strength']:.1%}\n"
                message += f"Confidence: {consensus['average_confidence']:.1f}%\n"
                message += f"Contributing Bots: {consensus['contributing_bots']}\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle cross forecast command: {e}")
            await update.message.reply_text(f"❌ **Cross Forecast Error:** {str(e)}")
    
    async def handle_risk_status_command(self, update: Update, context: CallbackContext):
        """Handle /risk status command - Portfolio-level risk analysis"""
        try:
            logger.info("Processing /risk status command")
            
            risk_status = self.risk_coordinator.get_portfolio_risk_status()
            
            if 'error' in risk_status:
                await update.message.reply_text(f"❌ **Risk Status Error**\n\n{risk_status['error']}")
                return
            
            message = "⚠️ **PORTFOLIO RISK STATUS**\n"
            message += "=" * 30 + "\n\n"
            
            # Overall risk metrics
            risk_level = risk_status.get('overall_risk_level', 'unknown')
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(risk_level, "⚪")
            
            message += f"{risk_emoji} **Overall Risk Level:** {risk_level.upper()}\n\n"
            
            message += f"💰 **Exposure**\n"
            message += f"• Total: ${risk_status.get('total_exposure', 0):,.2f}\n"
            message += f"• Maximum: ${risk_status.get('max_exposure', 0):,.2f}\n"
            message += f"• Utilization: {risk_status.get('risk_utilization', 0):.1%}\n\n"
            
            message += f"📊 **Risk Metrics**\n"
            message += f"• Correlation Risk: {risk_status.get('correlation_risk', 0):.1%}\n"
            message += f"• Concentration Risk: {risk_status.get('concentration_risk', 0):.1%}\n"
            message += f"• Volatility Risk: {risk_status.get('volatility_risk', 0):.1%}\n\n"
            
            # Individual bot risks
            bot_risks = risk_status.get('bot_risks', [])
            if bot_risks:
                message += f"🤖 **Bot Risk Breakdown**\n"
                for bot_risk in bot_risks[:5]:  # Show first 5 bots
                    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(bot_risk['risk_level'], "⚪")
                    message += f"{risk_emoji} {bot_risk['bot_id']}: {bot_risk['utilization']:.1f}%\n"
                message += "\n"
            
            message += f"🚨 **Emergency Threshold:** {risk_status.get('emergency_threshold', 0):.1%}\n"
            message += f"⏰ **Last Updated:** {risk_status.get('timestamp', datetime.now()).strftime('%H:%M:%S')}"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle risk status command: {e}")
            await update.message.reply_text(f"❌ **Risk Status Error:** {str(e)}")
    
    async def handle_pattern_train_command(self, update: Update, context: CallbackContext):
        """Handle /pattern train command - Start custom pattern training session"""
        try:
            args = context.args
            user_id = str(update.effective_user.id)
            
            if not args:
                # Show pattern training help
                message = "🎯 **CUSTOM PATTERN TRAINING**\n"
                message += "=" * 30 + "\n\n"
                message += "**Available Commands:**\n"
                message += "• `/pattern train SYMBOL TIMEFRAME PATTERN_NAME` - Annotate pattern\n"
                message += "• `/pattern status` - View training status\n"
                message += "• `/pattern recognize SYMBOL TIMEFRAME` - Recognize patterns\n\n"
                message += "**Example:**\n"
                message += "`/pattern train BTCUSDT 1h double_bottom`"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                return
            
            if len(args) < 3:
                await update.message.reply_text(
                    "❌ **Usage:** `/pattern train SYMBOL TIMEFRAME PATTERN_NAME`\n\n"
                    "**Example:** `/pattern train BTCUSDT 1h double_bottom`"
                )
                return
            
            symbol = args[0].upper()
            timeframe = args[1]
            pattern_name = args[2]
            
            logger.info(f"Processing pattern training: {pattern_name} for {symbol} {timeframe}")
            
            # Generate mock chart data for annotation
            chart_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'ohlcv': [[i, 100+i*0.1, 101+i*0.1, 99+i*0.1, 100.5+i*0.1, 1000] for i in range(50)]
            }
            
            # Annotate pattern
            annotation_id = self.pattern_learner.annotate_pattern(
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                pattern_name=pattern_name,
                chart_data=chart_data
            )
            
            if not annotation_id:
                await update.message.reply_text("❌ **Failed to annotate pattern**")
                return
            
            message = f"✅ **PATTERN ANNOTATED**\n\n"
            message += f"🎯 **Pattern:** {pattern_name}\n"
            message += f"📊 **Symbol:** {symbol}\n"
            message += f"⏰ **Timeframe:** {timeframe}\n"
            message += f"🆔 **Annotation ID:** `{annotation_id}`\n\n"
            message += f"📝 **Next Steps:**\n"
            message += f"• Wait for market outcome\n"
            message += f"• Update with success/failure\n"
            message += f"• Train ML model with `/pattern status`"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle pattern train command: {e}")
            await update.message.reply_text(f"❌ **Pattern Training Error:** {str(e)}")
    
    async def handle_correlate_command(self, update: Update, context: CallbackContext):
        """Handle /correlate command - Multi-asset correlation analysis"""
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ **Usage:** `/correlate ASSET1 ASSET2 [ASSET3...]`\n\n"
                    "**Example:** `/correlate BTCUSDT ETHUSDT ADAUSDT`"
                )
                return
            
            assets = [arg.upper() for arg in args]
            
            logger.info(f"Processing correlation analysis for: {assets}")
            
            # Analyze correlation matrix
            correlation_analysis = await self.multi_asset_correlator.analyze_correlation_matrix(
                assets=assets,
                timeframe='1h'
            )
            
            if 'error' in correlation_analysis:
                await update.message.reply_text(f"❌ **Correlation Error**\n\n{correlation_analysis['error']}")
                return
            
            # Generate correlation chart
            chart_path = await self.multi_asset_correlator.generate_correlation_chart(
                assets=assets,
                timeframe='1h'
            )
            
            # Format correlation message
            message = f"🔗 **MULTI-ASSET CORRELATION**\n"
            message += f"Assets: {', '.join(assets)}\n"
            message += "=" * 35 + "\n\n"
            
            correlations = correlation_analysis.get('correlations', [])
            strong_correlations = correlation_analysis.get('strong_correlations', [])
            
            message += f"📊 **Summary**\n"
            message += f"• Total Pairs: {len(correlations)}\n"
            message += f"• Strong Correlations: {len(strong_correlations)}\n"
            message += f"• Average Correlation: {correlation_analysis.get('average_correlation', 0):.3f}\n\n"
            
            # Show strongest correlations
            if strong_correlations:
                message += f"💪 **Strongest Correlations**\n"
                for corr in strong_correlations[:3]:
                    strength_emoji = "🔥" if abs(corr.correlation) >= 0.8 else "⚡"
                    message += f"{strength_emoji} {corr.asset1}-{corr.asset2}: {corr.correlation:.3f}\n"
                message += "\n"
            
            message += f"📈 **Chart Generated:** Correlation matrix heatmap\n"
            message += f"⏰ **Analysis Time:** {correlation_analysis.get('analysis_timestamp', datetime.now()).strftime('%H:%M:%S')}"
            
            # Send chart if generated
            if chart_path and self.telegram_notifier:
                try:
                    await self.telegram_notifier.send_photo(
                        chat_id=update.effective_chat.id,
                        photo_path=chart_path,
                        caption=message
                    )
                except Exception as e:
                    logger.warning(f"Failed to send correlation chart: {e}")
                    await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle correlate command: {e}")
            await update.message.reply_text(f"❌ **Correlation Error:** {str(e)}")
    
    async def handle_draw_sr_command(self, update: Update, context: CallbackContext):
        """Handle /draw sr command - Interactive chart drawing"""
        try:
            args = context.args
            if len(args) < 4:
                await update.message.reply_text(
                    "❌ **Usage:** `/draw sr SYMBOL TIMEFRAME LEVEL_TYPE PRICE`\n\n"
                    "**Level Types:** support, resistance\n"
                    "**Example:** `/draw sr BTCUSDT 1h support 45000`"
                )
                return
            
            symbol = args[0].upper()
            timeframe = args[1]
            level_type = args[2].lower()
            price = float(args[3])
            user_id = str(update.effective_user.id)
            
            if level_type not in ['support', 'resistance']:
                await update.message.reply_text("❌ **Level type must be 'support' or 'resistance'**")
                return
            
            logger.info(f"Drawing {level_type} level: {price} for {symbol} {timeframe}")
            
            # Draw support/resistance level
            level_id = self.interactive_chart_handler.draw_support_resistance(
                user_id=user_id,
                symbol=symbol,
                timeframe=timeframe,
                level_type=level_type,
                price=price,
                strength=3,
                notes=f"Drawn via Telegram at {datetime.now().strftime('%H:%M:%S')}"
            )
            
            if not level_id:
                await update.message.reply_text("❌ **Failed to draw level**")
                return
            
            # Generate interactive chart
            chart_path = self.interactive_chart_handler.generate_interactive_chart(
                symbol=symbol,
                timeframe=timeframe,
                user_id=user_id
            )
            
            message = f"🎨 **LEVEL DRAWN SUCCESSFULLY**\n\n"
            message += f"📊 **Symbol:** {symbol}\n"
            message += f"⏰ **Timeframe:** {timeframe}\n"
            message += f"📏 **Level:** {level_type.title()} @ ${price:,.2f}\n"
            message += f"🆔 **Level ID:** `{level_id}`\n\n"
            message += f"📈 **Interactive chart generated with your level**"
            
            # Send chart if generated
            if chart_path and self.telegram_notifier:
                try:
                    await self.telegram_notifier.send_photo(
                        chat_id=update.effective_chat.id,
                        photo_path=chart_path,
                        caption=message
                    )
                except Exception as e:
                    logger.warning(f"Failed to send interactive chart: {e}")
                    await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text(message, parse_mode='Markdown')
            
        except ValueError:
            await update.message.reply_text("❌ **Invalid price value**")
        except Exception as e:
            logger.error(f"Failed to handle draw sr command: {e}")
            await update.message.reply_text(f"❌ **Draw Error:** {str(e)}")
    
    async def handle_stream_command(self, update: Update, context: CallbackContext):
        """Handle /stream command - Start live chart streaming"""
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ **Usage:** `/stream SYMBOL TIMEFRAME`\n\n"
                    "**Example:** `/stream BTCUSDT 1h`"
                )
                return
            
            symbol = args[0].upper()
            timeframe = args[1]
            chat_id = str(update.effective_chat.id)
            
            logger.info(f"Starting live stream: {symbol} {timeframe}")
            
            # Start streaming
            success = await self.live_chart_streamer.start_streaming(
                symbol=symbol,
                timeframe=timeframe
            )
            
            if not success:
                await update.message.reply_text("❌ **Failed to start streaming**")
                return
            
            # Setup Telegram streaming
            telegram_success = self.live_chart_streamer.stream_to_telegram(
                chat_id=chat_id,
                symbol=symbol,
                timeframe=timeframe
            )
            
            message = f"📡 **LIVE STREAMING STARTED**\n\n"
            message += f"📊 **Symbol:** {symbol}\n"
            message += f"⏰ **Timeframe:** {timeframe}\n"
            message += f"💬 **Chat ID:** {chat_id}\n\n"
            
            if telegram_success:
                message += f"✅ **Telegram streaming enabled**\n"
                message += f"📈 You'll receive live chart updates\n\n"
            else:
                message += f"⚠️ **Telegram streaming setup failed**\n\n"
            
            message += f"🔄 **Commands:**\n"
            message += f"• `/stream stop {symbol} {timeframe}` - Stop streaming\n"
            message += f"• `/stream status` - View active streams"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to handle stream command: {e}")
            await update.message.reply_text(f"❌ **Streaming Error:** {str(e)}")
    
    def get_command_handlers(self) -> Dict[str, callable]:
        """Get all command handlers for registration"""
        return {
            'dashboard': self.handle_dashboard_command,
            'bots': self.handle_bots_command,
            'cross': self.handle_cross_forecast_command,
            'risk': self.handle_risk_status_command,
            'pattern': self.handle_pattern_train_command,
            'correlate': self.handle_correlate_command,
            'draw': self.handle_draw_sr_command,
            'stream': self.handle_stream_command
        }


# Global instance
_unified_telegram = None

def get_unified_telegram(telegram_notifier=None) -> UnifiedTelegramInterface:
    """Get global unified Telegram interface instance"""
    global _unified_telegram
    if _unified_telegram is None:
        _unified_telegram = UnifiedTelegramInterface(telegram_notifier)
    return _unified_telegram
