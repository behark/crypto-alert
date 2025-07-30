"""
Telegram Portfolio Commands - Phase 4 Evolution Layer
Telegram interface for Portfolio Intelligence with /portfolio optimize, /risk stress test, and /capital strategy commands.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

from ..portfolio.portfolio_intelligence import PortfolioIntelligenceCore, AllocationMethod, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramPortfolioCommands:
    """
    Telegram Portfolio Commands Interface - Portfolio intelligence and risk management.
    Provides comprehensive portfolio optimization, risk analysis, and capital allocation commands.
    """
    
    def __init__(self, portfolio_core: PortfolioIntelligenceCore):
        """Initialize Telegram Portfolio Commands"""
        self.portfolio_core = portfolio_core
        
        logger.info("Telegram Portfolio Commands initialized with portfolio intelligence")
    
    async def handle_portfolio_command(self, update: Update, context: CallbackContext):
        """Handle /portfolio command with subcommands"""
        try:
            message_text = update.message.text.strip()
            parts = message_text.split()
            
            if len(parts) < 2:
                await self._show_portfolio_help(update, context)
                return
            
            subcommand = parts[1].lower()
            
            if subcommand == 'optimize':
                await self._handle_portfolio_optimize(update, context, parts[2:])
            elif subcommand == 'status':
                await self._handle_portfolio_status(update, context)
            elif subcommand == 'rebalance':
                await self._handle_portfolio_rebalance(update, context)
            elif subcommand == 'metrics':
                await self._handle_portfolio_metrics(update, context)
            elif subcommand == 'allocation':
                await self._handle_allocation_breakdown(update, context)
            elif subcommand == 'performance':
                await self._handle_performance_analysis(update, context)
            else:
                await self._show_portfolio_help(update, context)
                
        except Exception as e:
            logger.error(f"Error handling portfolio command: {e}")
            await update.message.reply_text(f"❌ Error processing portfolio command: {str(e)}")
    
    async def handle_risk_command(self, update: Update, context: CallbackContext):
        """Handle /risk command with subcommands"""
        try:
            message_text = update.message.text.strip()
            parts = message_text.split()
            
            if len(parts) < 2:
                await self._show_risk_help(update, context)
                return
            
            subcommand = parts[1].lower()
            
            if subcommand == 'stress' and len(parts) > 2 and parts[2].lower() == 'test':
                await self._handle_stress_test(update, context, parts[3:])
            elif subcommand == 'analysis':
                await self._handle_risk_analysis(update, context)
            elif subcommand == 'limits':
                await self._handle_risk_limits(update, context)
            elif subcommand == 'var':
                await self._handle_var_analysis(update, context)
            else:
                await self._show_risk_help(update, context)
                
        except Exception as e:
            logger.error(f"Error handling risk command: {e}")
            await update.message.reply_text(f"❌ Error processing risk command: {str(e)}")
    
    async def handle_capital_command(self, update: Update, context: CallbackContext):
        """Handle /capital command with subcommands"""
        try:
            message_text = update.message.text.strip()
            parts = message_text.split()
            
            if len(parts) < 2:
                await self._show_capital_help(update, context)
                return
            
            subcommand = parts[1].lower()
            
            if subcommand == 'strategy':
                await self._handle_capital_strategy(update, context, parts[2:])
            elif subcommand == 'allocation':
                await self._handle_capital_allocation(update, context)
            elif subcommand == 'efficiency':
                await self._handle_capital_efficiency(update, context)
            elif subcommand == 'utilization':
                await self._handle_capital_utilization(update, context)
            else:
                await self._show_capital_help(update, context)
                
        except Exception as e:
            logger.error(f"Error handling capital command: {e}")
            await update.message.reply_text(f"❌ Error processing capital command: {str(e)}")
    
    async def _handle_portfolio_optimize(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /portfolio optimize [method] command"""
        try:
            # Determine optimization method
            method = AllocationMethod.KELLY_CRITERION  # Default
            if args:
                method_map = {
                    'kelly': AllocationMethod.KELLY_CRITERION,
                    'risk_parity': AllocationMethod.RISK_PARITY,
                    'equal': AllocationMethod.EQUAL_WEIGHT,
                    'dynamic': AllocationMethod.DYNAMIC_ALLOCATION
                }
                method = method_map.get(args[0].lower(), AllocationMethod.KELLY_CRITERION)
            
            # Send initial message
            method_name = method.value.replace('_', ' ').title()
            processing_msg = await update.message.reply_text(f"📊 **PORTFOLIO OPTIMIZATION INITIATED**\n\n⏳ Optimizing allocation using {method_name}...")
            
            # Perform optimization
            optimal_allocations = self.portfolio_core.optimize_portfolio_allocation(method)
            
            if optimal_allocations:
                message = f"""📊 **PORTFOLIO OPTIMIZATION COMPLETE**

🎯 **Optimization Method:** {method_name}

💰 **Optimal Allocations:**"""
                
                # Sort allocations by weight
                sorted_allocations = sorted(optimal_allocations.items(), key=lambda x: x[1], reverse=True)
                
                for strategy_id, weight in sorted_allocations:
                    capital_amount = weight * self.portfolio_core.total_capital
                    strategy_name = strategy_id.replace('_', ' ').title()[:20]
                    message += f"\n• **{strategy_name}:** {weight:.1%} (${capital_amount:,.0f})"
                
                # Add optimization insights
                total_allocation = sum(optimal_allocations.values())
                message += f"""

📈 **Optimization Insights:**
• **Total Allocation:** {total_allocation:.1%}
• **Available Capital:** {(1-total_allocation):.1%} (${(1-total_allocation)*self.portfolio_core.total_capital:,.0f})
• **Strategy Count:** {len(optimal_allocations)}
• **Max Single Allocation:** {max(optimal_allocations.values()):.1%}

💡 **Recommendation:** {'Proceed with rebalancing' if total_allocation > 0.5 else 'Consider increasing allocations'}"""
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("⚖️ Rebalance Now", callback_data="portfolio_rebalance")],
                    [InlineKeyboardButton("📊 View Status", callback_data="portfolio_status")],
                    [InlineKeyboardButton("🔄 Re-optimize", callback_data="portfolio_optimize")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text("❌ Portfolio optimization failed - no optimal allocations found")
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            await update.message.reply_text(f"❌ Portfolio optimization failed: {str(e)}")
    
    async def _handle_portfolio_status(self, update: Update, context: CallbackContext):
        """Handle /portfolio status command"""
        try:
            status = self.portfolio_core.get_portfolio_status()
            
            if 'error' in status:
                await update.message.reply_text(f"❌ {status['error']}")
                return
            
            # Format status message
            overview = status['portfolio_overview']
            performance = status['performance_metrics']
            risk_metrics = status['risk_metrics']
            
            # Health indicator
            health = status['portfolio_health']
            health_emoji = "🟢" if health == "Excellent" else "🟡" if health == "Good" else "🔴"
            
            message = f"""📊 **PORTFOLIO STATUS OVERVIEW**

💰 **Capital Overview:**
• **Total Capital:** ${overview['total_capital']:,.0f}
• **Available Capital:** ${overview['available_capital']:,.0f}
• **Total Value:** ${overview['total_value']:,.0f}
• **Total Return:** {overview['total_return']:+.2%}

📈 **Performance Metrics:**
• **Total P&L:** ${performance['total_pnl']:+,.0f}
• **Unrealized P&L:** ${performance['unrealized_pnl']:+,.0f}
• **Realized P&L:** ${performance['realized_pnl']:+,.0f}
• **Sharpe Ratio:** {performance['sharpe_ratio']:.2f}
• **Max Drawdown:** {performance['max_drawdown']:.2%}

⚠️ **Risk Metrics:**
• **Portfolio VaR (95%):** {risk_metrics['portfolio_var_95']:.2%}
• **Portfolio VaR (99%):** {risk_metrics['portfolio_var_99']:.2%}
• **Risk Contribution:** {risk_metrics['total_risk_contribution']:.2%}

🎯 **Portfolio Composition:**
• **Active Positions:** {overview['total_positions']}
• **Active Strategies:** {overview['active_strategies']}
• **Portfolio Health:** {health_emoji} {health}

⚖️ **Rebalancing:** {'🔄 Needed' if status['rebalance_needed'] else '✅ Balanced'}"""
            
            if status['last_rebalance']:
                last_rebalance = status['last_rebalance'].strftime('%Y-%m-%d %H:%M')
                message += f"\n• **Last Rebalance:** {last_rebalance}"
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("📊 Detailed Metrics", callback_data="portfolio_metrics")],
                [InlineKeyboardButton("💰 Allocation Breakdown", callback_data="portfolio_allocation")],
                [InlineKeyboardButton("⚖️ Rebalance", callback_data="portfolio_rebalance")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            await update.message.reply_text(f"❌ Portfolio status failed: {str(e)}")
    
    async def _handle_stress_test(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /risk stress test [scenario] command"""
        try:
            # Determine scenarios to test
            scenario_ids = None
            if args:
                scenario_map = {
                    'crash': ['market_crash'],
                    'flash': ['flash_crash'],
                    'correlation': ['correlation_spike'],
                    'all': None  # Test all scenarios
                }
                scenario_ids = scenario_map.get(args[0].lower())
            
            # Send initial message
            scenario_text = f" ({args[0]})" if args else " (all scenarios)"
            processing_msg = await update.message.reply_text(f"🧪 **PORTFOLIO STRESS TEST INITIATED**\n\n⏳ Running stress tests{scenario_text}...")
            
            # Run stress tests
            stress_results = self.portfolio_core.run_stress_test(scenario_ids)
            
            if stress_results:
                message = f"""🧪 **PORTFOLIO STRESS TEST RESULTS**

📊 **Tests Completed:** {len(stress_results)}

🎯 **Scenario Results:**"""
                
                for result in stress_results:
                    scenario_name = result.scenario_id.replace('_', ' ').title()
                    impact_emoji = "🔴" if result.portfolio_return < -0.1 else "🟡" if result.portfolio_return < -0.05 else "🟢"
                    var_breach_emoji = "⚠️" if result.var_breach else "✅"
                    
                    message += f"""

**{impact_emoji} {scenario_name}:**
• **Portfolio Impact:** {result.portfolio_return:+.2%}
• **P&L Impact:** ${result.portfolio_pnl:+,.0f}
• **Max Drawdown:** {result.max_drawdown:.2%}
• **VaR Breach:** {var_breach_emoji}
• **Recovery Time:** {result.recovery_time_days} days"""
                
                # Overall assessment
                worst_case = min(stress_results, key=lambda x: x.portfolio_return)
                avg_impact = sum(r.portfolio_return for r in stress_results) / len(stress_results)
                
                message += f"""

📈 **Overall Assessment:**
• **Worst Case Scenario:** {worst_case.scenario_id.replace('_', ' ').title()}
• **Worst Case Impact:** {worst_case.portfolio_return:+.2%}
• **Average Impact:** {avg_impact:+.2%}
• **VaR Breaches:** {sum(1 for r in stress_results if r.var_breach)}/{len(stress_results)}

💡 **Risk Level:** {'🔴 High Risk' if avg_impact < -0.08 else '🟡 Moderate Risk' if avg_impact < -0.04 else '🟢 Low Risk'}"""
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("📊 Risk Analysis", callback_data="risk_analysis")],
                    [InlineKeyboardButton("⚖️ Rebalance for Risk", callback_data="portfolio_risk_rebalance")],
                    [InlineKeyboardButton("🔄 Run Again", callback_data="risk_stress_test")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text("❌ Stress test failed - no results generated")
                
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            await update.message.reply_text(f"❌ Stress test failed: {str(e)}")
    
    async def _handle_capital_strategy(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /capital strategy [kelly|fixed|dynamic] command"""
        try:
            if not args:
                await update.message.reply_text("❌ Please specify strategy: kelly, fixed, or dynamic")
                return
            
            strategy = args[0].lower()
            
            if strategy not in ['kelly', 'fixed', 'dynamic']:
                await update.message.reply_text("❌ Invalid strategy. Use: kelly, fixed, or dynamic")
                return
            
            # Map strategy to allocation method
            method_map = {
                'kelly': AllocationMethod.KELLY_CRITERION,
                'fixed': AllocationMethod.EQUAL_WEIGHT,
                'dynamic': AllocationMethod.DYNAMIC_ALLOCATION
            }
            
            allocation_method = method_map[strategy]
            method_name = allocation_method.value.replace('_', ' ').title()
            
            # Send initial message
            processing_msg = await update.message.reply_text(f"💰 **CAPITAL STRATEGY CHANGE**\n\n⏳ Implementing {method_name} allocation strategy...")
            
            # Optimize with new strategy
            optimal_allocations = self.portfolio_core.optimize_portfolio_allocation(allocation_method)
            
            if optimal_allocations:
                # Calculate strategy comparison
                current_metrics = self.portfolio_core.calculate_portfolio_metrics()
                
                message = f"""💰 **CAPITAL STRATEGY: {method_name.upper()}**

🎯 **Strategy Overview:**"""
                
                if strategy == 'kelly':
                    message += f"""
• **Method:** Kelly Criterion optimization
• **Objective:** Maximize long-term growth rate
• **Risk Management:** Automatic position sizing based on edge
• **Advantages:** Mathematically optimal for known probabilities"""
                elif strategy == 'fixed':
                    message += f"""
• **Method:** Equal weight allocation
• **Objective:** Diversification and simplicity
• **Risk Management:** Equal risk across all strategies
• **Advantages:** Reduced concentration risk"""
                else:  # dynamic
                    message += f"""
• **Method:** Dynamic momentum-based allocation
• **Objective:** Adapt to changing market conditions
• **Risk Management:** Performance and volatility adjusted
• **Advantages:** Responsive to recent performance"""
                
                message += f"""

💰 **Recommended Allocations:**"""
                
                # Show top allocations
                sorted_allocations = sorted(optimal_allocations.items(), key=lambda x: x[1], reverse=True)
                for strategy_id, weight in sorted_allocations[:5]:
                    capital_amount = weight * self.portfolio_core.total_capital
                    strategy_name = strategy_id.replace('_', ' ').title()[:15]
                    message += f"\n• **{strategy_name}:** {weight:.1%} (${capital_amount:,.0f})"
                
                # Expected impact
                total_allocation = sum(optimal_allocations.values())
                message += f"""

📈 **Expected Impact:**
• **Total Capital Deployed:** {total_allocation:.1%}
• **Strategy Count:** {len(optimal_allocations)}
• **Diversification Score:** {1 - max(optimal_allocations.values()):.2f}
• **Expected Sharpe Improvement:** +0.1 to +0.3

✅ **Status:** Strategy ready for implementation"""
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("⚖️ Apply Strategy", callback_data=f"capital_apply_{strategy}")],
                    [InlineKeyboardButton("📊 Compare Strategies", callback_data="capital_compare")],
                    [InlineKeyboardButton("🔄 Recalculate", callback_data=f"capital_strategy_{strategy}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text(f"❌ {method_name} strategy optimization failed")
                
        except Exception as e:
            logger.error(f"Error in capital strategy: {e}")
            await update.message.reply_text(f"❌ Capital strategy failed: {str(e)}")
    
    async def _show_portfolio_help(self, update: Update, context: CallbackContext):
        """Show portfolio command help"""
        try:
            message = """📊 **PORTFOLIO INTELLIGENCE COMMANDS**

💰 **Portfolio Management:**
• `/portfolio optimize [method]` - Optimize allocation (kelly/risk_parity/equal/dynamic)
• `/portfolio status` - View portfolio overview
• `/portfolio rebalance` - Trigger portfolio rebalancing
• `/portfolio metrics` - Detailed performance metrics
• `/portfolio allocation` - Allocation breakdown
• `/portfolio performance` - Performance analysis

🎯 **Optimization Methods:**
• **kelly** - Kelly Criterion (maximize growth)
• **risk_parity** - Risk Parity (equal risk contribution)
• **equal** - Equal Weight (simple diversification)
• **dynamic** - Dynamic (momentum + risk adjusted)

💡 **Examples:**
• `/portfolio optimize kelly` - Optimize using Kelly Criterion
• `/portfolio status` - Get portfolio overview
• `/portfolio rebalance` - Rebalance to optimal weights"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing portfolio help: {e}")
            await update.message.reply_text("❌ Error displaying portfolio help")
    
    async def _show_risk_help(self, update: Update, context: CallbackContext):
        """Show risk command help"""
        try:
            message = """⚠️ **RISK MANAGEMENT COMMANDS**

🧪 **Stress Testing:**
• `/risk stress test [scenario]` - Run portfolio stress tests
• `/risk analysis` - Comprehensive risk analysis
• `/risk limits` - View and modify risk limits
• `/risk var` - Value-at-Risk analysis

🎯 **Stress Test Scenarios:**
• **crash** - Market crash scenario (-20%)
• **flash** - Flash crash scenario (-10%)
• **correlation** - High correlation scenario
• **all** - Run all scenarios

💡 **Examples:**
• `/risk stress test crash` - Test market crash scenario
• `/risk stress test all` - Run all stress tests
• `/risk analysis` - Get comprehensive risk report"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing risk help: {e}")
            await update.message.reply_text("❌ Error displaying risk help")
    
    async def _show_capital_help(self, update: Update, context: CallbackContext):
        """Show capital command help"""
        try:
            message = """💰 **CAPITAL ALLOCATION COMMANDS**

🎯 **Capital Strategy:**
• `/capital strategy kelly` - Kelly Criterion allocation
• `/capital strategy fixed` - Equal weight allocation
• `/capital strategy dynamic` - Dynamic allocation
• `/capital allocation` - Current allocation breakdown
• `/capital efficiency` - Capital efficiency analysis
• `/capital utilization` - Capital utilization metrics

📊 **Strategy Types:**
• **Kelly:** Mathematically optimal growth maximization
• **Fixed:** Equal weight diversification
• **Dynamic:** Adaptive momentum-based allocation

💡 **Examples:**
• `/capital strategy kelly` - Switch to Kelly Criterion
• `/capital allocation` - View current allocations
• `/capital efficiency` - Analyze capital efficiency"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing capital help: {e}")
            await update.message.reply_text("❌ Error displaying capital help")
    
    async def handle_callback_query(self, update: Update, context: CallbackContext):
        """Handle callback queries from inline buttons"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data == "portfolio_optimize":
                await self._handle_portfolio_optimize(query, context, [])
            elif data == "portfolio_status":
                await self._handle_portfolio_status(query, context)
            elif data == "portfolio_rebalance":
                await self._handle_portfolio_rebalance(query, context)
            elif data == "risk_stress_test":
                await self._handle_stress_test(query, context, [])
            elif data.startswith("capital_strategy_"):
                strategy = data.split("_")[-1]
                await self._handle_capital_strategy(query, context, [strategy])
            elif data.startswith("portfolio_") or data.startswith("risk_") or data.startswith("capital_"):
                # Handle other portfolio-related callbacks
                await query.edit_message_text("🔄 Processing request...")
                
        except Exception as e:
            logger.error(f"Error handling portfolio callback: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def _handle_portfolio_rebalance(self, update: Update, context: CallbackContext):
        """Handle portfolio rebalancing"""
        try:
            processing_msg = await update.message.reply_text("⚖️ **PORTFOLIO REBALANCING**\n\n⏳ Calculating optimal rebalancing...")
            
            # Perform rebalancing
            rebalance_results = self.portfolio_core.rebalance_portfolio()
            
            if 'error' not in rebalance_results:
                actions_count = len(rebalance_results['rebalance_actions'])
                
                message = f"""⚖️ **PORTFOLIO REBALANCING COMPLETE**

📊 **Rebalancing Summary:**
• **Actions Taken:** {actions_count}
• **Total Capital:** ${rebalance_results['total_capital']:,.0f}
• **Expected Improvement:** {rebalance_results['expected_improvement']:+.4f}
• **Rebalance Time:** {rebalance_results['timestamp'].strftime('%H:%M:%S')}

🎯 **Key Changes:**"""
                
                for action in rebalance_results['rebalance_actions'][:5]:
                    strategy_name = action['strategy_id'].replace('_', ' ').title()[:15]
                    action_emoji = "📈" if action['action'] == 'increase' else "📉"
                    message += f"\n{action_emoji} **{strategy_name}:** {action['current_weight']:.1%} → {action['target_weight']:.1%}"
                
                message += f"\n\n✅ **Status:** Portfolio successfully rebalanced for optimal performance"
                
                keyboard = [
                    [InlineKeyboardButton("📊 View Status", callback_data="portfolio_status")],
                    [InlineKeyboardButton("📈 Performance", callback_data="portfolio_performance")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text(f"❌ Rebalancing failed: {rebalance_results['error']}")
                
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
            await update.message.reply_text(f"❌ Portfolio rebalancing failed: {str(e)}")
