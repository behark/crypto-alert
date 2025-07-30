"""
Telegram Memory Commands - Phase 4 Evolution Layer
Telegram interface for Strategic Memory System with /memory recall and /memory consolidate commands.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

from ..memory.strategic_memory import StrategicMemoryEngine, MarketCondition
from ..memory.memory_retrieval import MemoryRetrievalSystem, MemoryQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramMemoryCommands:
    """
    Telegram Memory Commands Interface - Strategic memory control and insights.
    Provides comprehensive memory recall, consolidation, and analysis commands.
    """
    
    def __init__(self, memory_engine: StrategicMemoryEngine, retrieval_system: MemoryRetrievalSystem):
        """Initialize Telegram Memory Commands"""
        self.memory_engine = memory_engine
        self.retrieval_system = retrieval_system
        
        logger.info("Telegram Memory Commands initialized with strategic memory interface")
    
    async def handle_memory_command(self, update: Update, context: CallbackContext):
        """Handle /memory command with subcommands"""
        try:
            message_text = update.message.text.strip()
            parts = message_text.split()
            
            if len(parts) < 2:
                await self._show_memory_help(update, context)
                return
            
            subcommand = parts[1].lower()
            
            if subcommand == 'recall':
                await self._handle_memory_recall(update, context, parts[2:])
            elif subcommand == 'consolidate':
                await self._handle_memory_consolidate(update, context)
            elif subcommand == 'insights':
                await self._handle_memory_insights(update, context, parts[2:])
            elif subcommand == 'patterns':
                await self._handle_memory_patterns(update, context, parts[2:])
            elif subcommand == 'stats':
                await self._handle_memory_stats(update, context)
            elif subcommand == 'clear':
                await self._handle_memory_clear(update, context)
            else:
                await self._show_memory_help(update, context)
                
        except Exception as e:
            logger.error(f"Error handling memory command: {e}")
            await update.message.reply_text(f"❌ Error processing memory command: {str(e)}")
    
    async def _handle_memory_recall(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /memory recall [symbol|condition] command"""
        try:
            if not args:
                await update.message.reply_text("❌ Please specify symbol or condition for recall")
                return
            
            query_param = args[0].upper()
            
            # Determine if it's a symbol or condition
            if query_param in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']:
                # Symbol recall
                memories = self.memory_engine.recall_by_symbol(query_param, limit=10)
                await self._send_symbol_memories(update, query_param, memories)
            else:
                # Try to match market condition
                condition_map = {
                    'BULL': MarketCondition.BULL_TREND,
                    'BEAR': MarketCondition.BEAR_TREND,
                    'SIDEWAYS': MarketCondition.SIDEWAYS,
                    'VOLATILE': MarketCondition.HIGH_VOLATILITY,
                    'CALM': MarketCondition.LOW_VOLATILITY,
                    'BREAKOUT': MarketCondition.BREAKOUT,
                    'REVERSAL': MarketCondition.REVERSAL
                }
                
                condition = condition_map.get(query_param)
                if condition:
                    memories = self.memory_engine.recall_by_condition(condition, limit=10)
                    await self._send_condition_memories(update, condition, memories)
                else:
                    await update.message.reply_text(f"❌ Unknown symbol or condition: {query_param}")
            
        except Exception as e:
            logger.error(f"Error in memory recall: {e}")
            await update.message.reply_text(f"❌ Memory recall failed: {str(e)}")
    
    async def _handle_memory_consolidate(self, update: Update, context: CallbackContext):
        """Handle /memory consolidate command"""
        try:
            # Send initial message
            processing_msg = await update.message.reply_text("🧠 **MEMORY CONSOLIDATION INITIATED**\n\n⏳ Processing memories and discovering patterns...")
            
            # Perform consolidation
            results = self.memory_engine.consolidate_memories()
            
            if results:
                # Format consolidation results
                message = f"""🧠 **MEMORY CONSOLIDATION COMPLETE**

📊 **Results Summary:**
• **Patterns Discovered:** {results['patterns_discovered']}
• **Insights Generated:** {results['insights_generated']}
• **Memories Pruned:** {results['memories_pruned']}
• **Consolidation Time:** {results['consolidation_time'].strftime('%Y-%m-%d %H:%M:%S')}

💡 **Key Insights:**"""
                
                for insight in results.get('knowledge_distilled', [])[:5]:
                    message += f"\n• {insight}"
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("📈 View Patterns", callback_data="memory_patterns")],
                    [InlineKeyboardButton("📊 Memory Stats", callback_data="memory_stats")],
                    [InlineKeyboardButton("🔄 Refresh", callback_data="memory_consolidate")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text("❌ Memory consolidation failed - no results generated")
                
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
            await update.message.reply_text(f"❌ Memory consolidation failed: {str(e)}")
    
    async def _handle_memory_insights(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /memory insights [symbol] command"""
        try:
            symbol = args[0].upper() if args else None
            
            insights = self.memory_engine.get_memory_insights(symbol)
            
            if insights:
                symbol_text = f" for {symbol}" if symbol else ""
                message = f"""🧠 **MEMORY INSIGHTS{symbol_text.upper()}**

📊 **Performance Overview:**
• **Total Memories:** {insights['total_memories']}
• **Success Rate:** {insights['success_rate']:.1%}
• **Average Profit:** {insights['average_profit']:.4f}
• **Memory Health:** {insights['memory_health']}

🎯 **Top Performing Conditions:**"""
                
                for condition, success_rate, count in insights['top_performing_conditions'][:3]:
                    message += f"\n• **{condition.replace('_', ' ').title()}:** {success_rate:.1%} ({count} trades)"
                
                if insights['pattern_effectiveness']:
                    message += f"\n\n🔍 **Most Effective Patterns:**"
                    for pattern in insights['pattern_effectiveness'][:3]:
                        message += f"\n• **{pattern['type']}:** {pattern['success_rate']:.1%} success, {pattern['frequency']} uses"
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("📈 Success Patterns", callback_data=f"memory_success_{symbol or 'all'}")],
                    [InlineKeyboardButton("⚠️ Failure Analysis", callback_data=f"memory_failures_{symbol or 'all'}")],
                    [InlineKeyboardButton("🔄 Refresh", callback_data=f"memory_insights_{symbol or 'all'}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ No memory insights available{' for ' + symbol if symbol else ''}")
                
        except Exception as e:
            logger.error(f"Error generating memory insights: {e}")
            await update.message.reply_text(f"❌ Memory insights failed: {str(e)}")
    
    async def _handle_memory_patterns(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /memory patterns [symbol] command"""
        try:
            symbol = args[0].upper() if args else None
            
            # Get success patterns
            success_patterns = self.retrieval_system.extract_success_patterns(symbol, lookback_days=90)
            
            if success_patterns:
                symbol_text = f" for {symbol}" if symbol else ""
                message = f"""🎯 **SUCCESS PATTERNS{symbol_text.upper()}**

📈 **Discovered Patterns (Last 90 Days):**"""
                
                for i, pattern in enumerate(success_patterns[:5], 1):
                    conditions_text = ", ".join([c.value.replace('_', ' ').title() for c in pattern.conditions])
                    message += f"""

**{i}. {pattern.pattern_type.replace('_', ' ').title()}**
• **Success Rate:** {pattern.success_rate:.1%}
• **Frequency:** {pattern.frequency} occurrences
• **Avg Profit:** {pattern.average_profit:.4f}
• **Conditions:** {conditions_text}
• **Recommendation:** {pattern.recommendation}"""
                
                # Add failure avoidance insights
                failure_insights = self.retrieval_system.get_failure_avoidance_insights(symbol, lookback_days=90)
                
                if failure_insights:
                    message += f"\n\n⚠️ **PATTERNS TO AVOID:**"
                    for insight in failure_insights[:3]:
                        message += f"\n• **{insight['pattern'].replace('_', ' ').title()}:** {insight['failure_count']} failures, avg loss {insight['average_loss']:.4f}"
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("🔍 Deep Analysis", callback_data=f"memory_deep_patterns_{symbol or 'all'}")],
                    [InlineKeyboardButton("📊 Pattern Stats", callback_data=f"memory_pattern_stats_{symbol or 'all'}")],
                    [InlineKeyboardButton("🔄 Refresh", callback_data=f"memory_patterns_{symbol or 'all'}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ No success patterns found{' for ' + symbol if symbol else ''}")
                
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {e}")
            await update.message.reply_text(f"❌ Memory pattern analysis failed: {str(e)}")
    
    async def _handle_memory_stats(self, update: Update, context: CallbackContext):
        """Handle /memory stats command"""
        try:
            # Get memory engine stats
            memory_stats = self.memory_engine.memory_stats
            
            # Get retrieval system stats
            retrieval_stats = self.retrieval_system.get_retrieval_stats()
            
            message = f"""📊 **STRATEGIC MEMORY STATISTICS**

🧠 **Memory Engine:**
• **Total Memories:** {memory_stats['total_memories']}
• **Successful Trades:** {memory_stats['successful_trades']}
• **Failed Trades:** {memory_stats['failed_trades']}
• **Regime Changes:** {memory_stats['regime_changes']}
• **Pattern Discoveries:** {memory_stats['pattern_discoveries']}
• **Last Consolidation:** {memory_stats['last_consolidation'].strftime('%Y-%m-%d %H:%M') if memory_stats['last_consolidation'] else 'Never'}

🔍 **Retrieval System:**
• **Total Queries:** {retrieval_stats['total_queries']}
• **Successful Retrievals:** {retrieval_stats['successful_retrievals']}
• **Pattern Matches:** {retrieval_stats['pattern_matches']}
• **Context Matches:** {retrieval_stats['context_matches']}
• **Cache Hit Rate:** {retrieval_stats['cache_hit_rate']:.1%}
• **Cache Size:** {retrieval_stats['cache_size']} entries

⚡ **System Health:** {'🟢 Excellent' if memory_stats['total_memories'] > 100 else '🟡 Growing' if memory_stats['total_memories'] > 10 else '🔴 Limited'}"""
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("🧠 Consolidate", callback_data="memory_consolidate")],
                [InlineKeyboardButton("🗑️ Clear Cache", callback_data="memory_clear_cache")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="memory_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            await update.message.reply_text(f"❌ Memory stats failed: {str(e)}")
    
    async def _handle_memory_clear(self, update: Update, context: CallbackContext):
        """Handle /memory clear command"""
        try:
            # Clear retrieval cache
            self.retrieval_system.clear_cache()
            
            message = """🗑️ **MEMORY CACHE CLEARED**

✅ **Actions Completed:**
• Retrieval query cache cleared
• Memory indices refreshed
• System ready for fresh queries

💡 **Note:** Core memories remain intact - only cache cleared for performance optimization."""
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("📊 View Stats", callback_data="memory_stats")],
                [InlineKeyboardButton("🧠 Consolidate", callback_data="memory_consolidate")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error clearing memory cache: {e}")
            await update.message.reply_text(f"❌ Memory cache clear failed: {str(e)}")
    
    async def _send_symbol_memories(self, update: Update, symbol: str, memories: List):
        """Send formatted symbol memories"""
        try:
            if not memories:
                await update.message.reply_text(f"❌ No memories found for symbol: {symbol}")
                return
            
            message = f"""🧠 **MEMORY RECALL: {symbol}**

📈 **Recent Trading Memories ({len(memories)} found):**"""
            
            for i, memory in enumerate(memories[:5], 1):
                success_emoji = "✅" if memory.success else "❌"
                profit_emoji = "📈" if memory.profit_loss > 0 else "📉"
                
                message += f"""

**{i}. {success_emoji} {memory.entry_time.strftime('%Y-%m-%d %H:%M')}**
• **Entry:** {memory.entry_price:.6f} → **Exit:** {memory.exit_price:.6f}
• **P&L:** {profit_emoji} {memory.profit_loss:.6f}
• **Confidence:** {memory.confidence_score:.2f}
• **Condition:** {memory.market_condition.value.replace('_', ' ').title()}
• **Lesson:** {memory.lessons_learned[:100]}..."""
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("📊 Symbol Insights", callback_data=f"memory_insights_{symbol}")],
                [InlineKeyboardButton("🎯 Success Patterns", callback_data=f"memory_patterns_{symbol}")],
                [InlineKeyboardButton("🔄 Refresh", callback_data=f"memory_recall_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error sending symbol memories: {e}")
            await update.message.reply_text(f"❌ Error displaying symbol memories: {str(e)}")
    
    async def _send_condition_memories(self, update: Update, condition: MarketCondition, memories: List):
        """Send formatted condition memories"""
        try:
            if not memories:
                await update.message.reply_text(f"❌ No memories found for condition: {condition.value}")
                return
            
            condition_name = condition.value.replace('_', ' ').title()
            message = f"""🧠 **MEMORY RECALL: {condition_name}**

📊 **Trading Memories in {condition_name} Conditions ({len(memories)} found):**"""
            
            success_count = sum(1 for m in memories if m.success)
            success_rate = success_count / len(memories)
            avg_profit = sum(m.profit_loss for m in memories) / len(memories)
            
            message += f"""

📈 **Condition Performance:**
• **Success Rate:** {success_rate:.1%} ({success_count}/{len(memories)})
• **Average P&L:** {avg_profit:.6f}

🎯 **Recent Examples:**"""
            
            for i, memory in enumerate(memories[:3], 1):
                success_emoji = "✅" if memory.success else "❌"
                message += f"""

**{i}. {success_emoji} {memory.symbol} - {memory.entry_time.strftime('%m-%d %H:%M')}**
• **P&L:** {memory.profit_loss:.6f} | **Confidence:** {memory.confidence_score:.2f}"""
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("📈 Condition Analysis", callback_data=f"memory_condition_analysis_{condition.value}")],
                [InlineKeyboardButton("🎯 Best Practices", callback_data=f"memory_condition_patterns_{condition.value}")],
                [InlineKeyboardButton("🔄 Refresh", callback_data=f"memory_recall_{condition.value}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error sending condition memories: {e}")
            await update.message.reply_text(f"❌ Error displaying condition memories: {str(e)}")
    
    async def _show_memory_help(self, update: Update, context: CallbackContext):
        """Show memory command help"""
        try:
            message = """🧠 **STRATEGIC MEMORY COMMANDS**

🔍 **Available Commands:**

**📋 Basic Commands:**
• `/memory recall BTCUSDT` - Recall symbol trading history
• `/memory recall BULL` - Recall bull market memories
• `/memory consolidate` - Trigger memory consolidation
• `/memory stats` - View memory system statistics

**📊 Analysis Commands:**
• `/memory insights [SYMBOL]` - Get memory insights
• `/memory patterns [SYMBOL]` - Analyze success patterns
• `/memory clear` - Clear memory cache

**🎯 Market Conditions:**
• `BULL` - Bull trend memories
• `BEAR` - Bear trend memories  
• `SIDEWAYS` - Sideways market memories
• `VOLATILE` - High volatility memories
• `CALM` - Low volatility memories
• `BREAKOUT` - Breakout pattern memories
• `REVERSAL` - Reversal pattern memories

**💡 Examples:**
• `/memory recall ETHUSDT` - Get ETHUSDT trading history
• `/memory recall VOLATILE` - Get high volatility memories
• `/memory insights BTCUSDT` - Get BTCUSDT insights
• `/memory patterns` - Get all success patterns"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing memory help: {e}")
            await update.message.reply_text("❌ Error displaying memory help")
    
    async def handle_callback_query(self, update: Update, context: CallbackContext):
        """Handle callback queries from inline buttons"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data == "memory_consolidate":
                await self._handle_memory_consolidate(query, context)
            elif data == "memory_stats":
                await self._handle_memory_stats(query, context)
            elif data == "memory_clear_cache":
                await self._handle_memory_clear(query, context)
            elif data.startswith("memory_"):
                # Handle other memory-related callbacks
                await query.edit_message_text("🔄 Processing memory request...")
                
        except Exception as e:
            logger.error(f"Error handling memory callback: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
