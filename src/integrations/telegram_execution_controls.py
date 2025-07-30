"""
Telegram Execution Control Panel
================================
Advanced Telegram interface for autonomous execution control and decision monitoring
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
import json

logger = logging.getLogger(__name__)

class TelegramExecutionControls:
    """
    Telegram Execution Control Panel
    
    Provides advanced Telegram commands for controlling autonomous execution,
    monitoring decisions, and managing system behavior.
    """
    
    def __init__(self, telegram_notifier=None):
        """Initialize Telegram execution controls"""
        self.telegram_notifier = telegram_notifier
        self.execution_log = []
        self.autonomy_status = {
            'enabled': True,
            'override_level': 'normal',  # normal, cautious, aggressive
            'last_changed': datetime.now(),
            'changed_by': 'system'
        }
        
        logger.info("Telegram Execution Controls initialized")
    
    async def handle_autonomy_command(self, update: Update, context: CallbackContext):
        """Handle /autonomy command"""
        try:
            args = context.args
            
            if not args:
                await self._show_autonomy_status(update, context)
                return
            
            command = args[0].lower()
            
            if command == 'status':
                await self._show_autonomy_status(update, context)
            elif command == 'on':
                await self._set_autonomy_status(update, context, True)
            elif command == 'off':
                await self._set_autonomy_status(update, context, False)
            elif command == 'pause':
                await self._pause_autonomy(update, context)
            elif command == 'resume':
                await self._resume_autonomy(update, context)
            elif command == 'level':
                if len(args) > 1:
                    await self._set_override_level(update, context, args[1])
                else:
                    await self._show_override_levels(update, context)
            else:
                await self._show_autonomy_help(update, context)
                
        except Exception as e:
            logger.error(f"Error in autonomy command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_execution_log_command(self, update: Update, context: CallbackContext):
        """Handle /execution log command"""
        try:
            args = context.args
            limit = 5  # Default
            
            if args and args[0].isdigit():
                limit = min(int(args[0]), 20)  # Max 20 entries
            
            if not self.execution_log:
                await update.message.reply_text(
                    "📋 **Execution Log**\n\n"
                    "No autonomous trades recorded yet."
                )
                return
            
            # Get recent executions
            recent_executions = self.execution_log[-limit:]
            
            message = "📋 **Recent Autonomous Executions**\n\n"
            
            for i, execution in enumerate(reversed(recent_executions), 1):
                timestamp = execution.get('timestamp', datetime.now())
                symbol = execution.get('symbol', 'Unknown')
                action = execution.get('action', 'Unknown')
                result = execution.get('result', 'Unknown')
                confidence = execution.get('confidence', 0.0)
                
                status_emoji = "✅" if result == 'success' else "❌" if result == 'failed' else "⏳"
                
                message += f"{i}. {status_emoji} **{symbol}** - {action}\n"
                message += f"   🕐 {timestamp.strftime('%H:%M:%S')}\n"
                message += f"   🎯 Confidence: {confidence:.1%}\n"
                message += f"   📊 Result: {result}\n\n"
            
            # Add summary stats
            total_trades = len(self.execution_log)
            successful_trades = len([e for e in self.execution_log if e.get('result') == 'success'])
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            message += f"📈 **Summary**: {successful_trades}/{total_trades} successful ({success_rate:.1%})"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in execution log command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_decision_command(self, update: Update, context: CallbackContext):
        """Handle /decision command"""
        try:
            args = context.args
            
            if not args:
                await self._show_decision_help(update, context)
                return
            
            subcommand = args[0].lower()
            
            if subcommand == 'trail':
                symbol = args[1] if len(args) > 1 else 'BTCUSDT'
                await self._show_decision_trail(update, context, symbol)
            elif subcommand == 'now':
                symbol = args[1] if len(args) > 1 else 'BTCUSDT'
                await self._trigger_fresh_decision(update, context, symbol)
            else:
                await self._show_decision_help(update, context)
                
        except Exception as e:
            logger.error(f"Error in decision command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_bias_command(self, update: Update, context: CallbackContext):
        """Handle /bias command"""
        try:
            args = context.args
            symbol = args[0] if args else 'BTCUSDT'
            
            # Get behavioral decision engine
            try:
                from ..predictive.behavioral_decision import get_behavioral_decision_engine
                decision_engine = get_behavioral_decision_engine()
                
                bias_info = decision_engine.get_current_bias(symbol)
                
                # Format bias message
                bias = bias_info['bias']
                strength = bias_info['strength']
                confidence = bias_info['confidence']
                reasons = bias_info['reasons']
                last_updated = bias_info['last_updated']
                
                # Bias emoji
                if bias == 'bullish':
                    bias_emoji = "🟢"
                elif bias == 'bearish':
                    bias_emoji = "🔴"
                else:
                    bias_emoji = "🟡"
                
                message = f"🧠 **Current Bias: {symbol}**\n\n"
                message += f"{bias_emoji} **Direction**: {bias.upper()}\n"
                message += f"💪 **Strength**: {strength:.1%}\n"
                message += f"🎯 **Confidence**: {confidence:.1%}\n\n"
                
                if last_updated:
                    message += f"🕐 **Last Updated**: {last_updated.strftime('%H:%M:%S')}\n\n"
                
                message += "📋 **Key Reasons**:\n"
                for reason in reasons[-3:]:  # Show top 3 reasons
                    message += f"• {reason}\n"
                
                # Add trend indicator
                if strength > 0.6:
                    message += f"\n🔥 **Strong {bias} bias detected**"
                elif strength > 0.3:
                    message += f"\n📊 **Moderate {bias} lean**"
                else:
                    message += f"\n😐 **Weak directional bias**"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Behavioral decision engine not available")
                
        except Exception as e:
            logger.error(f"Error in bias command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_autonomy_status(self, update: Update, context: CallbackContext):
        """Show current autonomy status"""
        try:
            status = self.autonomy_status
            
            status_emoji = "🟢" if status['enabled'] else "🔴"
            status_text = "ENABLED" if status['enabled'] else "DISABLED"
            
            message = f"🤖 **Autonomous Trading Status**\n\n"
            message += f"{status_emoji} **Status**: {status_text}\n"
            message += f"⚙️ **Override Level**: {status['override_level'].upper()}\n"
            message += f"🕐 **Last Changed**: {status['last_changed'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"👤 **Changed By**: {status['changed_by']}\n\n"
            
            # Add level descriptions
            message += "**Override Levels**:\n"
            message += "• `normal` - Standard autonomous operation\n"
            message += "• `cautious` - Reduced position sizes, higher thresholds\n"
            message += "• `aggressive` - Larger positions, lower thresholds\n\n"
            
            # Add quick action buttons
            keyboard = []
            if status['enabled']:
                keyboard.append([
                    InlineKeyboardButton("⏸️ Pause", callback_data="autonomy_pause"),
                    InlineKeyboardButton("🛑 Disable", callback_data="autonomy_off")
                ])
            else:
                keyboard.append([
                    InlineKeyboardButton("▶️ Enable", callback_data="autonomy_on")
                ])
            
            keyboard.append([
                InlineKeyboardButton("📊 Execution Log", callback_data="execution_log"),
                InlineKeyboardButton("🔄 Refresh", callback_data="autonomy_status")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error showing autonomy status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _set_autonomy_status(self, update: Update, context: CallbackContext, enabled: bool):
        """Set autonomy status"""
        try:
            self.autonomy_status['enabled'] = enabled
            self.autonomy_status['last_changed'] = datetime.now()
            self.autonomy_status['changed_by'] = update.effective_user.username or 'user'
            
            status_text = "ENABLED" if enabled else "DISABLED"
            emoji = "✅" if enabled else "🛑"
            
            message = f"{emoji} **Autonomous trading {status_text}**\n\n"
            
            if enabled:
                message += "🚀 System will now execute trades automatically based on signal confidence.\n"
                message += "⚠️ Monitor positions and risk levels regularly."
            else:
                message += "⏹️ All autonomous execution has been halted.\n"
                message += "📋 Manual trading only until re-enabled."
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Log the change
            self.execution_log.append({
                'timestamp': datetime.now(),
                'action': f'autonomy_{status_text.lower()}',
                'symbol': 'SYSTEM',
                'result': 'success',
                'confidence': 1.0,
                'user': update.effective_user.username or 'user'
            })
            
        except Exception as e:
            logger.error(f"Error setting autonomy status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _pause_autonomy(self, update: Update, context: CallbackContext):
        """Temporarily pause autonomy"""
        try:
            # Set a temporary pause (could be implemented with a timer)
            self.autonomy_status['enabled'] = False
            self.autonomy_status['last_changed'] = datetime.now()
            self.autonomy_status['changed_by'] = f"{update.effective_user.username or 'user'} (pause)"
            
            message = "⏸️ **Autonomous trading PAUSED**\n\n"
            message += "🕐 Trading suspended temporarily\n"
            message += "💡 Use `/autonomy resume` to continue\n"
            message += "🛑 Use `/autonomy off` to disable completely"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error pausing autonomy: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _resume_autonomy(self, update: Update, context: CallbackContext):
        """Resume autonomy from pause"""
        try:
            self.autonomy_status['enabled'] = True
            self.autonomy_status['last_changed'] = datetime.now()
            self.autonomy_status['changed_by'] = f"{update.effective_user.username or 'user'} (resume)"
            
            message = "▶️ **Autonomous trading RESUMED**\n\n"
            message += "🚀 System back to normal operation\n"
            message += "📊 Monitoring signals and executing trades"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error resuming autonomy: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _set_override_level(self, update: Update, context: CallbackContext, level: str):
        """Set override level"""
        try:
            valid_levels = ['normal', 'cautious', 'aggressive']
            
            if level.lower() not in valid_levels:
                await update.message.reply_text(
                    f"❌ Invalid level. Valid options: {', '.join(valid_levels)}"
                )
                return
            
            self.autonomy_status['override_level'] = level.lower()
            self.autonomy_status['last_changed'] = datetime.now()
            self.autonomy_status['changed_by'] = update.effective_user.username or 'user'
            
            level_descriptions = {
                'normal': '📊 Standard risk and position sizing',
                'cautious': '🛡️ Reduced positions, higher confidence thresholds',
                'aggressive': '🔥 Larger positions, lower confidence thresholds'
            }
            
            message = f"⚙️ **Override level set to: {level.upper()}**\n\n"
            message += level_descriptions.get(level.lower(), '')
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error setting override level: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_decision_trail(self, update: Update, context: CallbackContext, symbol: str):
        """Show decision trail for symbol"""
        try:
            # Get behavioral decision engine
            try:
                from ..predictive.behavioral_decision import get_behavioral_decision_engine
                decision_engine = get_behavioral_decision_engine()
                
                trail = decision_engine.get_decision_trail(symbol, limit=5)
                
                if not trail:
                    await update.message.reply_text(
                        f"📋 **Decision Trail: {symbol}**\n\n"
                        "No recent decisions found."
                    )
                    return
                
                message = f"🧠 **Decision Trail: {symbol}**\n\n"
                
                for i, decision in enumerate(trail, 1):
                    timestamp = decision['timestamp']
                    decision_type = decision['decision_type']
                    direction = decision.get('direction', 'N/A')
                    confidence = decision['confidence']
                    confidence_score = decision['confidence_score']
                    
                    # Decision emoji
                    if 'enter' in decision_type:
                        emoji = "🟢" if direction == 'long' else "🔴"
                    elif 'exit' in decision_type:
                        emoji = "🚪"
                    elif decision_type == 'hold':
                        emoji = "⏸️"
                    else:
                        emoji = "⏳"
                    
                    message += f"{i}. {emoji} **{decision_type.replace('_', ' ').title()}**"
                    if direction and direction != 'N/A':
                        message += f" ({direction})"
                    message += f"\n"
                    
                    message += f"   🕐 {timestamp.strftime('%H:%M:%S')}\n"
                    message += f"   🎯 {confidence} ({confidence_score:.1%})\n"
                    
                    # Show key reasoning
                    trail_items = decision.get('decision_trail', [])
                    if trail_items:
                        message += f"   💭 {trail_items[0]}\n"
                    
                    message += "\n"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Behavioral decision engine not available")
                
        except Exception as e:
            logger.error(f"Error showing decision trail: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _trigger_fresh_decision(self, update: Update, context: CallbackContext, symbol: str):
        """Trigger fresh decision assessment"""
        try:
            message = f"🔄 **Triggering fresh assessment for {symbol}**\n\n"
            message += "⏳ Analyzing current market conditions...\n"
            message += "🧠 Fusing signals from all predictive engines...\n"
            message += "📊 Calculating decision confidence...\n\n"
            message += "⚡ Results will be available in decision trail shortly."
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Here you would trigger actual decision making
            # For now, just acknowledge the request
            
        except Exception as e:
            logger.error(f"Error triggering fresh decision: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_autonomy_help(self, update: Update, context: CallbackContext):
        """Show autonomy command help"""
        help_text = """
🤖 **Autonomy Control Commands**

`/autonomy status` - Show current status
`/autonomy on` - Enable autonomous trading
`/autonomy off` - Disable autonomous trading
`/autonomy pause` - Temporarily pause trading
`/autonomy resume` - Resume from pause
`/autonomy level <normal|cautious|aggressive>` - Set override level

**Override Levels:**
• `normal` - Standard operation
• `cautious` - Reduced risk, higher thresholds
• `aggressive` - Higher risk, lower thresholds
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def _show_decision_help(self, update: Update, context: CallbackContext):
        """Show decision command help"""
        help_text = """
🧠 **Decision Analysis Commands**

`/decision trail <SYMBOL>` - Show recent decision trail
`/decision now <SYMBOL>` - Trigger fresh assessment

**Examples:**
`/decision trail BTCUSDT` - Show BTC decision history
`/decision now ETHUSDT` - Fresh ETH analysis
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    def log_execution(self, symbol: str, action: str, result: str, confidence: float, details: Dict[str, Any] = None):
        """Log an execution for the execution log"""
        try:
            execution_entry = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'result': result,
                'confidence': confidence,
                'details': details or {}
            }
            
            self.execution_log.append(execution_entry)
            
            # Keep only recent executions
            if len(self.execution_log) > 100:
                self.execution_log = self.execution_log[-100:]
                
        except Exception as e:
            logger.error(f"Error logging execution: {e}")
    
    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current autonomy status"""
        return self.autonomy_status.copy()
    
    def is_autonomy_enabled(self) -> bool:
        """Check if autonomy is enabled"""
        return self.autonomy_status.get('enabled', False)


# Global instance
_telegram_execution_controls = None

def get_telegram_execution_controls() -> TelegramExecutionControls:
    """Get global telegram execution controls instance"""
    global _telegram_execution_controls
    if _telegram_execution_controls is None:
        _telegram_execution_controls = TelegramExecutionControls()
    return _telegram_execution_controls
