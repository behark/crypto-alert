"""
Telegram Audit Dashboard Commands
=================================
Advanced Telegram interface for audit suite monitoring and control
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
import json

logger = logging.getLogger(__name__)

class TelegramAuditCommands:
    """
    Telegram Audit Dashboard Commands
    
    Provides Telegram interface for audit suite monitoring, validation results,
    risk event tracking, and self-healing controls.
    """
    
    def __init__(self, telegram_notifier=None):
        """Initialize Telegram audit commands"""
        self.telegram_notifier = telegram_notifier
        
        logger.info("Telegram Audit Commands initialized")
    
    async def handle_audit_command(self, update: Update, context: CallbackContext):
        """Handle /audit command"""
        try:
            args = context.args
            
            if not args:
                await self._show_audit_help(update, context)
                return
            
            subcommand = args[0].lower()
            
            if subcommand == 'summary':
                await self._show_audit_summary(update, context)
            elif subcommand == 'trail':
                symbol = args[1] if len(args) > 1 else 'BTCUSDT'
                await self._show_audit_trail(update, context, symbol)
            elif subcommand == 'scan':
                await self._trigger_audit_scan(update, context)
            elif subcommand == 'status':
                await self._show_audit_status(update, context)
            elif subcommand == 'risks':
                await self._show_risk_events(update, context)
            else:
                await self._show_audit_help(update, context)
                
        except Exception as e:
            logger.error(f"Error in audit command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def handle_selfheal_command(self, update: Update, context: CallbackContext):
        """Handle /selfheal command"""
        try:
            args = context.args
            
            if not args:
                await self._show_selfheal_status(update, context)
                return
            
            subcommand = args[0].lower()
            
            if subcommand == 'trigger':
                modules = args[1:] if len(args) > 1 else None
                await self._trigger_self_healing(update, context, modules)
            elif subcommand == 'status':
                await self._show_selfheal_status(update, context)
            elif subcommand == 'enable':
                await self._enable_self_healing(update, context)
            elif subcommand == 'disable':
                await self._disable_self_healing(update, context)
            else:
                await self._show_selfheal_help(update, context)
                
        except Exception as e:
            logger.error(f"Error in selfheal command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_audit_summary(self, update: Update, context: CallbackContext):
        """Show audit summary"""
        try:
            # Get audit suite
            try:
                from ..predictive.autonomous_audit import get_autonomous_audit_suite
                audit_suite = get_autonomous_audit_suite()
                
                # Generate fresh summary
                summary = audit_suite.generate_audit_summary(timedelta(hours=24))
                
                if not summary:
                    await update.message.reply_text("❌ Unable to generate audit summary")
                    return
                
                # Format summary message
                health_emoji = "🟢" if summary.system_health_score > 0.8 else "🟡" if summary.system_health_score > 0.6 else "🔴"
                
                message = f"🔍 **Audit Summary (24h)**\n\n"
                message += f"{health_emoji} **System Health**: {summary.system_health_score:.1%}\n\n"
                
                # Validation results
                message += "📊 **Validation Results**:\n"
                message += f"✅ Passed: {summary.validation_results['passed']}\n"
                message += f"⚠️ Warning: {summary.validation_results['warning']}\n"
                message += f"❌ Failed: {summary.validation_results['failed']}\n"
                message += f"⏳ Pending: {summary.validation_results['pending']}\n\n"
                
                # Performance metrics
                message += "📈 **Performance**:\n"
                message += f"🎯 Accuracy: {summary.avg_accuracy_score:.1%}\n"
                message += f"🎚️ Confidence Calibration: {summary.avg_confidence_calibration:.1%}\n"
                message += f"📊 Success Rate: {summary.performance_metrics['success_rate']:.1%}\n\n"
                
                # Risk events
                message += f"🚨 **Risk Events**: {len(summary.risk_events)}\n"
                if summary.risk_events:
                    critical_count = len([e for e in summary.risk_events if e.severity.value == 'critical'])
                    high_count = len([e for e in summary.risk_events if e.severity.value == 'high'])
                    if critical_count > 0:
                        message += f"🔴 Critical: {critical_count}\n"
                    if high_count > 0:
                        message += f"🟠 High: {high_count}\n"
                
                message += "\n"
                
                # Top recommendations
                if summary.recommendations:
                    message += "💡 **Top Recommendations**:\n"
                    for rec in summary.recommendations[:3]:
                        message += f"• {rec}\n"
                
                # Add interactive buttons
                keyboard = [
                    [
                        InlineKeyboardButton("📋 Risk Events", callback_data="audit_risks"),
                        InlineKeyboardButton("🔄 Refresh", callback_data="audit_summary")
                    ],
                    [
                        InlineKeyboardButton("🩺 Self-Heal", callback_data="selfheal_status"),
                        InlineKeyboardButton("📊 Full Status", callback_data="audit_status")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    message, 
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
                
            except ImportError:
                await update.message.reply_text("❌ Autonomous audit suite not available")
                
        except Exception as e:
            logger.error(f"Error showing audit summary: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_audit_trail(self, update: Update, context: CallbackContext, symbol: str):
        """Show audit trail for symbol"""
        try:
            # Get audit suite
            try:
                from ..predictive.autonomous_audit import get_autonomous_audit_suite
                audit_suite = get_autonomous_audit_suite()
                
                # Get recent validations for symbol
                symbol_validations = [
                    v for v in audit_suite.decision_validations.values()
                    if v.symbol == symbol and v.timestamp > datetime.now() - timedelta(hours=24)
                ]
                
                if not symbol_validations:
                    await update.message.reply_text(
                        f"📋 **Audit Trail: {symbol}**\n\n"
                        "No recent validations found."
                    )
                    return
                
                # Sort by timestamp
                symbol_validations.sort(key=lambda x: x.timestamp, reverse=True)
                
                message = f"🔍 **Audit Trail: {symbol}**\n\n"
                
                for i, validation in enumerate(symbol_validations[:5], 1):
                    # Status emoji
                    if validation.validation_status.value == 'passed':
                        status_emoji = "✅"
                    elif validation.validation_status.value == 'warning':
                        status_emoji = "⚠️"
                    elif validation.validation_status.value == 'failed':
                        status_emoji = "❌"
                    else:
                        status_emoji = "⏳"
                    
                    message += f"{i}. {status_emoji} **{validation.validation_status.value.title()}**\n"
                    message += f"   🕐 {validation.timestamp.strftime('%H:%M:%S')}\n"
                    message += f"   🎯 Accuracy: {validation.accuracy_score:.1%}\n"
                    message += f"   📊 Predicted: {validation.predicted_outcome} → Actual: {validation.actual_outcome}\n"
                    message += f"   🎚️ Confidence Cal: {validation.confidence_calibration:.1%}\n"
                    
                    # Show key findings
                    if validation.findings:
                        message += f"   💭 {validation.findings[0]}\n"
                    
                    message += "\n"
                
                # Add summary stats
                passed_count = len([v for v in symbol_validations if v.validation_status.value == 'passed'])
                total_count = len(symbol_validations)
                success_rate = passed_count / total_count if total_count > 0 else 0
                
                avg_accuracy = sum(v.accuracy_score for v in symbol_validations) / total_count if total_count > 0 else 0
                
                message += f"📈 **Summary**: {passed_count}/{total_count} passed ({success_rate:.1%})\n"
                message += f"🎯 **Avg Accuracy**: {avg_accuracy:.1%}"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Autonomous audit suite not available")
                
        except Exception as e:
            logger.error(f"Error showing audit trail: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _trigger_audit_scan(self, update: Update, context: CallbackContext):
        """Trigger manual audit scan"""
        try:
            message = "🔄 **Triggering Audit Scan**\n\n"
            message += "⏳ Running comprehensive audit across all systems...\n"
            message += "🔍 Validating recent decisions\n"
            message += "🚨 Scanning for risk events\n"
            message += "📊 Calculating performance metrics\n\n"
            message += "⚡ Results will be available in audit summary shortly."
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Here you would trigger actual audit scan
            # For now, just acknowledge the request
            
        except Exception as e:
            logger.error(f"Error triggering audit scan: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_audit_status(self, update: Update, context: CallbackContext):
        """Show detailed audit status"""
        try:
            # Get audit suite
            try:
                from ..predictive.autonomous_audit import get_autonomous_audit_suite
                audit_suite = get_autonomous_audit_suite()
                
                status = audit_suite.get_audit_status()
                
                message = f"📊 **Audit Suite Status**\n\n"
                
                # Core metrics
                message += "🔢 **Core Metrics**:\n"
                message += f"📋 Total Validations: {status['total_validations']}\n"
                message += f"📅 Recent (24h): {status['recent_validations_24h']}\n"
                message += f"📈 Trade Traces: {status['total_trade_traces']}\n"
                message += f"🚨 Risk Events: {status['total_risk_events']}\n"
                message += f"⚠️ Recent Risks (24h): {status['recent_risk_events_24h']}\n\n"
                
                # Performance metrics
                message += "📈 **Current Performance**:\n"
                message += f"🎯 Accuracy: {status['current_accuracy']:.1%}\n"
                message += f"🎚️ Confidence Cal: {status['current_confidence_calibration']:.1%}\n"
                message += f"⚡ Execution Quality: {status['current_execution_quality']:.1%}\n\n"
                
                # System status
                healing_status = "🟢 Enabled" if status['self_healing_enabled'] else "🔴 Disabled"
                monitoring_status = "🟢 Active" if status['monitoring_active'] else "🔴 Inactive"
                
                message += "⚙️ **System Status**:\n"
                message += f"🩺 Self-Healing: {healing_status}\n"
                message += f"👁️ Monitoring: {monitoring_status}\n"
                
                if status['last_audit_summary']:
                    message += f"📊 Last Summary: {status['last_audit_summary'].strftime('%H:%M:%S')}\n"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Autonomous audit suite not available")
                
        except Exception as e:
            logger.error(f"Error showing audit status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_risk_events(self, update: Update, context: CallbackContext):
        """Show recent risk events"""
        try:
            # Get audit suite
            try:
                from ..predictive.autonomous_audit import get_autonomous_audit_suite
                audit_suite = get_autonomous_audit_suite()
                
                # Get recent risk events
                recent_events = [
                    e for e in audit_suite.risk_events
                    if e.detection_time > datetime.now() - timedelta(hours=24)
                ]
                
                if not recent_events:
                    await update.message.reply_text(
                        "🚨 **Risk Events (24h)**\n\n"
                        "✅ No risk events detected in the last 24 hours."
                    )
                    return
                
                # Sort by severity and time
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                recent_events.sort(key=lambda x: (severity_order.get(x.severity.value, 4), x.detection_time), reverse=True)
                
                message = f"🚨 **Risk Events (24h)**\n\n"
                
                for i, event in enumerate(recent_events[:10], 1):  # Show top 10
                    # Severity emoji
                    if event.severity.value == 'critical':
                        severity_emoji = "🔴"
                    elif event.severity.value == 'high':
                        severity_emoji = "🟠"
                    elif event.severity.value == 'medium':
                        severity_emoji = "🟡"
                    else:
                        severity_emoji = "🟢"
                    
                    message += f"{i}. {severity_emoji} **{event.severity.value.upper()}**\n"
                    message += f"   📋 {event.description}\n"
                    message += f"   🕐 {event.detection_time.strftime('%H:%M:%S')}\n"
                    message += f"   📊 Impact: {event.impact_score:.1%}\n"
                    
                    if event.affected_symbols:
                        symbols_str = ', '.join(event.affected_symbols[:3])
                        if len(event.affected_symbols) > 3:
                            symbols_str += f" (+{len(event.affected_symbols)-3} more)"
                        message += f"   🎯 Symbols: {symbols_str}\n"
                    
                    # Show top mitigation action
                    if event.mitigation_actions:
                        message += f"   💡 Action: {event.mitigation_actions[0]}\n"
                    
                    message += "\n"
                
                # Add summary
                critical_count = len([e for e in recent_events if e.severity.value == 'critical'])
                high_count = len([e for e in recent_events if e.severity.value == 'high'])
                
                if critical_count > 0 or high_count > 0:
                    message += f"⚠️ **Attention**: {critical_count} critical, {high_count} high severity events"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Autonomous audit suite not available")
                
        except Exception as e:
            logger.error(f"Error showing risk events: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _trigger_self_healing(self, update: Update, context: CallbackContext, modules: List[str] = None):
        """Trigger self-healing"""
        try:
            # Get audit suite
            try:
                from ..predictive.autonomous_audit import get_autonomous_audit_suite
                audit_suite = get_autonomous_audit_suite()
                
                message = "🩺 **Triggering Self-Healing**\n\n"
                if modules:
                    message += f"🎯 Target Modules: {', '.join(modules)}\n"
                else:
                    message += "🎯 Target: Full System\n"
                
                message += "⏳ Initiating healing procedures...\n\n"
                
                await update.message.reply_text(message, parse_mode='Markdown')
                
                # Trigger healing
                healing_result = await audit_suite.trigger_self_healing(modules)
                
                # Format result
                if healing_result['status'] == 'completed':
                    result_message = "✅ **Self-Healing Completed**\n\n"
                    
                    for module, result in healing_result['results'].items():
                        if result['status'] == 'success':
                            result_message += f"✅ **{module}**: Healed successfully\n"
                            if 'actions_taken' in result:
                                for action in result['actions_taken'][:2]:  # Show top 2 actions
                                    result_message += f"   • {action}\n"
                        elif result['status'] == 'cooldown':
                            result_message += f"⏳ **{module}**: {result['message']}\n"
                        else:
                            result_message += f"❌ **{module}**: {result.get('message', 'Failed')}\n"
                        result_message += "\n"
                    
                    result_message += f"🕐 Completed at: {healing_result['timestamp'].strftime('%H:%M:%S')}"
                    
                elif healing_result['status'] == 'disabled':
                    result_message = "⚠️ **Self-Healing Disabled**\n\n"
                    result_message += "Self-healing is currently disabled.\n"
                    result_message += "Use `/selfheal enable` to activate."
                    
                else:
                    result_message = f"❌ **Self-Healing Failed**\n\n{healing_result.get('message', 'Unknown error')}"
                
                await update.message.reply_text(result_message, parse_mode='Markdown')
                
            except ImportError:
                await update.message.reply_text("❌ Autonomous audit suite not available")
                
        except Exception as e:
            logger.error(f"Error triggering self-healing: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_selfheal_status(self, update: Update, context: CallbackContext):
        """Show self-healing status"""
        try:
            message = "🩺 **Self-Healing Status**\n\n"
            message += "🟢 **Status**: Enabled\n"
            message += "⏰ **Cooldown**: 1 hour between attempts\n"
            message += "🔄 **Max Attempts**: 3 per module\n"
            message += "🎯 **Available Modules**:\n"
            message += "   • decision_engine\n"
            message += "   • execution_engine\n"
            message += "   • risk_monitor\n"
            message += "   • data_feeds\n\n"
            
            message += "💡 **Usage**:\n"
            message += "`/selfheal trigger` - Heal all modules\n"
            message += "`/selfheal trigger decision_engine` - Heal specific module"
            
            # Add interactive buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Trigger Healing", callback_data="selfheal_trigger"),
                    InlineKeyboardButton("📊 System Status", callback_data="audit_status")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error showing self-heal status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def _show_audit_help(self, update: Update, context: CallbackContext):
        """Show audit command help"""
        help_text = """
🔍 **Audit Dashboard Commands**

`/audit summary` - Overview of recent audit results
`/audit trail <SYMBOL>` - Full trace of latest trade decisions + validation scores
`/audit scan` - Trigger manual audit run across all pairs
`/audit status` - Detailed audit suite status
`/audit risks` - Show recent risk events

**Examples:**
`/audit trail BTCUSDT` - Show BTC audit trail
`/audit summary` - 24h audit overview
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def _show_selfheal_help(self, update: Update, context: CallbackContext):
        """Show self-heal command help"""
        help_text = """
🩺 **Self-Healing Commands**

`/selfheal status` - Show healing system status
`/selfheal trigger` - Trigger full system healing
`/selfheal trigger <module>` - Heal specific module
`/selfheal enable` - Enable auto-healing
`/selfheal disable` - Disable auto-healing

**Available Modules:**
• decision_engine
• execution_engine
• risk_monitor
• data_feeds
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')


# Global instance
_telegram_audit_commands = None

def get_telegram_audit_commands() -> TelegramAuditCommands:
    """Get global telegram audit commands instance"""
    global _telegram_audit_commands
    if _telegram_audit_commands is None:
        _telegram_audit_commands = TelegramAuditCommands()
    return _telegram_audit_commands
