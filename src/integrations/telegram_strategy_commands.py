"""
Telegram Strategy Commands - Phase 4 Evolution Layer
Telegram interface for Strategy Adaptation Engine with /strategy evolve, status, and mutate commands.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

from ..evolution.strategy_evolution import StrategyEvolutionCore, MutationType
from ..evolution.reinforcement_learning import ReinforcementLearningAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramStrategyCommands:
    """
    Telegram Strategy Commands Interface - Strategy evolution and RL control.
    Provides comprehensive strategy evolution, mutation, and learning commands.
    """
    
    def __init__(self, evolution_core: StrategyEvolutionCore, rl_agent: ReinforcementLearningAgent):
        """Initialize Telegram Strategy Commands"""
        self.evolution_core = evolution_core
        self.rl_agent = rl_agent
        
        logger.info("Telegram Strategy Commands initialized with evolution and RL systems")
    
    async def handle_strategy_command(self, update: Update, context: CallbackContext):
        """Handle /strategy command with subcommands"""
        try:
            message_text = update.message.text.strip()
            parts = message_text.split()
            
            if len(parts) < 2:
                await self._show_strategy_help(update, context)
                return
            
            subcommand = parts[1].lower()
            
            if subcommand == 'evolve':
                await self._handle_strategy_evolve(update, context)
            elif subcommand == 'status':
                await self._handle_strategy_status(update, context)
            elif subcommand == 'mutate':
                await self._handle_strategy_mutate(update, context, parts[2:])
            elif subcommand == 'rl':
                await self._handle_rl_status(update, context)
            elif subcommand == 'population':
                await self._handle_population_overview(update, context)
            elif subcommand == 'history':
                await self._handle_evolution_history(update, context)
            else:
                await self._show_strategy_help(update, context)
                
        except Exception as e:
            logger.error(f"Error handling strategy command: {e}")
            await update.message.reply_text(f"❌ Error processing strategy command: {str(e)}")
    
    async def _handle_strategy_evolve(self, update: Update, context: CallbackContext):
        """Handle /strategy evolve command"""
        try:
            # Send initial message
            processing_msg = await update.message.reply_text("🧬 **STRATEGY EVOLUTION INITIATED**\n\n⏳ Evolving next generation of trading strategies...")
            
            # Perform evolution
            results = self.evolution_core.evolve_generation()
            
            if results:
                # Format evolution results
                fitness_change = "📈" if results['fitness_improvement'] > 0 else "📉" if results['fitness_improvement'] < 0 else "➡️"
                
                message = f"""🧬 **STRATEGY EVOLUTION COMPLETE**

📊 **Generation {results['generation']} Results:**
• **Strategies Created:** {results['strategies_created']}
• **Mutations Applied:** {results['mutations_applied']}
• **Crossovers Performed:** {results['crossovers_performed']}
• **Fitness Change:** {fitness_change} {results['fitness_improvement']:+.4f}
• **Best Strategy:** `{results['best_strategy_id']}`
• **Evolution Time:** {results['evolution_time'].strftime('%H:%M:%S')}

🎯 **Next Generation Status:**
Population evolved with genetic diversity and performance optimization."""
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("📈 View Status", callback_data="strategy_status")],
                    [InlineKeyboardButton("🧬 Evolve Again", callback_data="strategy_evolve")],
                    [InlineKeyboardButton("📊 Population", callback_data="strategy_population")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text("❌ Strategy evolution failed - no results generated")
                
        except Exception as e:
            logger.error(f"Error in strategy evolution: {e}")
            await update.message.reply_text(f"❌ Strategy evolution failed: {str(e)}")
    
    async def _handle_strategy_status(self, update: Update, context: CallbackContext):
        """Handle /strategy status command"""
        try:
            status = self.evolution_core.get_strategy_status()
            
            if 'error' in status:
                await update.message.reply_text(f"❌ {status['error']}")
                return
            
            # Format status message
            pop_overview = status['population_overview']
            fitness_stats = status['fitness_statistics']
            pop_chars = status['population_characteristics']
            evolution_progress = status['evolution_progress']
            
            message = f"""📊 **STRATEGY EVOLUTION STATUS**

🧬 **Population Overview:**
• **Total Strategies:** {pop_overview['total_strategies']}
• **Active Strategies:** {pop_overview['active_strategies']}
• **Current Generation:** {pop_overview['current_generation']}
• **Elite Strategies:** {pop_overview['elite_strategies']}

📈 **Fitness Statistics:**
• **Average Fitness:** {fitness_stats['average_fitness']:.4f}
• **Best Fitness:** {fitness_stats['best_fitness']:.4f}
• **Worst Fitness:** {fitness_stats['worst_fitness']:.4f}
• **Fitness Std Dev:** {fitness_stats['fitness_std']:.4f}

🎯 **Population Characteristics:**
• **Average Age:** {pop_chars['average_age']:.1f} generations
• **Total Trades:** {pop_chars['total_trades_executed']}
• **Strategy Types:** {len(pop_chars['strategy_types'])} different types

⚡ **Evolution Progress:**
• **Generations Completed:** {evolution_progress['generations_completed']}
• **Successful Mutations:** {evolution_progress['successful_mutations']}
• **Crossover Events:** {evolution_progress['crossover_events']}
• **Best Fitness Ever:** {evolution_progress['best_fitness_ever']:.4f}"""
            
            # Add top performers
            if status['top_performers']:
                message += f"\n\n🏆 **Top Performing Strategies:**"
                for i, performer in enumerate(status['top_performers'][:3], 1):
                    strategy_type = performer['strategy_type'].replace('_', ' ').title()
                    message += f"\n**{i}. {strategy_type}** (Gen {performer['generation']})"
                    message += f"\n   • Fitness: {performer['fitness_score']:.4f} | Age: {performer['age']} | Trades: {performer['trades_executed']}"
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("🧬 Evolve Now", callback_data="strategy_evolve")],
                [InlineKeyboardButton("🎲 Mutate Strategy", callback_data="strategy_mutate_random")],
                [InlineKeyboardButton("📊 Population Details", callback_data="strategy_population")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            await update.message.reply_text(f"❌ Strategy status failed: {str(e)}")
    
    async def _handle_strategy_mutate(self, update: Update, context: CallbackContext, args: List[str]):
        """Handle /strategy mutate [now] command"""
        try:
            if not args or args[0].lower() != 'now':
                await update.message.reply_text("❌ Use `/strategy mutate now` to trigger immediate mutation")
                return
            
            # Get a random strategy to mutate
            if not self.evolution_core.active_strategies:
                await update.message.reply_text("❌ No active strategies available for mutation")
                return
            
            import random
            strategy_id = random.choice(self.evolution_core.active_strategies)
            
            # Send initial message
            processing_msg = await update.message.reply_text(f"🎲 **STRATEGY MUTATION INITIATED**\n\n⏳ Mutating strategy `{strategy_id[:12]}...`")
            
            # Perform mutation
            mutated_strategy = self.evolution_core.mutate_strategy_now(strategy_id)
            
            if mutated_strategy:
                mutation_type = "Random Genetic Mutation"
                
                message = f"""🎲 **STRATEGY MUTATION COMPLETE**

🧬 **Mutation Results:**
• **Original Strategy:** `{strategy_id[:12]}...`
• **New Strategy:** `{mutated_strategy.strategy_id[:12]}...`
• **Mutation Type:** {mutation_type}
• **Strategy Type:** {mutated_strategy.strategy_type.value.replace('_', ' ').title()}
• **Generation:** {mutated_strategy.generation}
• **Parent Generation:** {mutated_strategy.generation - 1}

🎯 **Genetic Profile:**
• **Genes Modified:** Multiple parameters adjusted
• **Fitness Score:** {mutated_strategy.fitness_score:.4f} (initial)
• **Creation Time:** {mutated_strategy.creation_time.strftime('%H:%M:%S')}

✅ **Status:** New strategy added to active population and ready for trading."""
                
                # Add interactive buttons
                keyboard = [
                    [InlineKeyboardButton("📊 View Status", callback_data="strategy_status")],
                    [InlineKeyboardButton("🎲 Mutate Again", callback_data="strategy_mutate_random")],
                    [InlineKeyboardButton("🧬 Full Evolution", callback_data="strategy_evolve")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await processing_msg.edit_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await processing_msg.edit_text("❌ Strategy mutation failed - unable to create viable mutant")
                
        except Exception as e:
            logger.error(f"Error in strategy mutation: {e}")
            await update.message.reply_text(f"❌ Strategy mutation failed: {str(e)}")
    
    async def _handle_rl_status(self, update: Update, context: CallbackContext):
        """Handle /strategy rl command"""
        try:
            stats = self.rl_agent.get_learning_stats()
            
            if 'error' in stats:
                await update.message.reply_text(f"❌ RL Status Error: {stats['error']}")
                return
            
            training_overview = stats['training_overview']
            performance_metrics = stats['performance_metrics']
            learning_progress = stats['learning_progress']
            
            message = f"""🤖 **REINFORCEMENT LEARNING STATUS**

📚 **Training Overview:**
• **Episodes Completed:** {training_overview['episodes_completed']}
• **Total Experiences:** {training_overview['total_experiences']}
• **Replay Buffer Size:** {training_overview['replay_buffer_size']}
• **Current Exploration:** {training_overview['current_epsilon']:.3f}

📈 **Performance Metrics:**
• **Average Reward:** {performance_metrics['average_reward']:.4f}
• **Best Episode Reward:** {performance_metrics['best_episode_reward']:.4f}
• **Recent Avg Reward:** {performance_metrics['recent_average_reward']:.4f}
• **Recent Success Rate:** {performance_metrics['recent_success_rate']:.1%}

🧠 **Learning Progress:**
• **States Discovered:** {learning_progress['total_states_discovered']}
• **Average Q-Value:** {learning_progress['average_q_value']:.4f}
• **Q-Value Std Dev:** {learning_progress['q_value_std']:.4f}
• **Learning Stability:** {learning_progress['learning_stability']:.1%}"""
            
            # Add recent episodes if available
            if stats['recent_episodes']:
                message += f"\n\n🎯 **Recent Episodes:**"
                for episode in stats['recent_episodes'][-3:]:
                    success_emoji = "✅" if episode['success'] else "❌"
                    message += f"\n{success_emoji} **{episode['episode_id'][:12]}...** | Reward: {episode['total_reward']:.3f} | P&L: {episode['final_pnl']:.4f}"
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("🎓 Train Now", callback_data="rl_train")],
                [InlineKeyboardButton("📊 Detailed Stats", callback_data="rl_detailed_stats")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="strategy_rl")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting RL status: {e}")
            await update.message.reply_text(f"❌ RL status failed: {str(e)}")
    
    async def _handle_population_overview(self, update: Update, context: CallbackContext):
        """Handle /strategy population command"""
        try:
            status = self.evolution_core.get_strategy_status()
            
            if 'error' in status:
                await update.message.reply_text(f"❌ {status['error']}")
                return
            
            pop_chars = status['population_characteristics']
            strategy_types = pop_chars['strategy_types']
            
            message = f"""👥 **STRATEGY POPULATION OVERVIEW**

🧬 **Population Composition:**"""
            
            # Strategy type breakdown
            total_strategies = sum(strategy_types.values())
            for strategy_type, count in strategy_types.items():
                percentage = (count / total_strategies) * 100 if total_strategies > 0 else 0
                type_name = strategy_type.replace('_', ' ').title()
                message += f"\n• **{type_name}:** {count} strategies ({percentage:.1f}%)"
            
            message += f"""

📊 **Population Statistics:**
• **Total Population:** {total_strategies}
• **Average Age:** {pop_chars['average_age']:.1f} generations
• **Total Trading Experience:** {pop_chars['total_trades_executed']} trades
• **Genetic Diversity:** {len(strategy_types)} different strategy types"""
            
            # Add top performers with more details
            if status['top_performers']:
                message += f"\n\n🏆 **Elite Performers:**"
                for i, performer in enumerate(status['top_performers'][:5], 1):
                    strategy_type = performer['strategy_type'].replace('_', ' ').title()
                    age_indicator = "🆕" if performer['age'] < 3 else "🧓" if performer['age'] > 10 else "👤"
                    
                    message += f"\n**{i}. {age_indicator} {strategy_type}**"
                    message += f"\n   • **Fitness:** {performer['fitness_score']:.4f}"
                    message += f"\n   • **Generation:** {performer['generation']} (Age: {performer['age']})"
                    message += f"\n   • **Trades:** {performer['trades_executed']}"
                    
                    # Add performance metrics if available
                    metrics = performer.get('performance_metrics', {})
                    if metrics.get('win_rate'):
                        message += f"\n   • **Win Rate:** {metrics['win_rate']:.1%}"
                    if metrics.get('sharpe_ratio'):
                        message += f"\n   • **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}"
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("🧬 Evolve Population", callback_data="strategy_evolve")],
                [InlineKeyboardButton("📈 Fitness Analysis", callback_data="strategy_fitness_analysis")],
                [InlineKeyboardButton("📊 Evolution History", callback_data="strategy_history")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting population overview: {e}")
            await update.message.reply_text(f"❌ Population overview failed: {str(e)}")
    
    async def _handle_evolution_history(self, update: Update, context: CallbackContext):
        """Handle /strategy history command"""
        try:
            history = self.evolution_core.get_evolution_history(generations=10)
            
            if not history or not history.get('generation_stats'):
                await update.message.reply_text("❌ No evolution history available")
                return
            
            generation_stats = history['generation_stats']
            fitness_trend = history.get('fitness_trend', [])
            trend_direction = history.get('trend_direction', 'stable')
            evolution_stats = history.get('evolution_stats', {})
            
            # Trend indicators
            trend_emoji = "📈" if trend_direction == 'improving' else "📉" if trend_direction == 'declining' else "➡️"
            
            message = f"""📚 **STRATEGY EVOLUTION HISTORY**

📊 **Evolution Trend:** {trend_emoji} {trend_direction.title()}

🧬 **Recent Generations:**"""
            
            # Show recent generations
            sorted_generations = sorted(generation_stats.items(), reverse=True)
            for gen, stats in sorted_generations[:5]:
                message += f"\n**Generation {gen}:**"
                message += f"\n   • Avg Fitness: {stats['avg_fitness']:.4f}"
                message += f"\n   • Best Fitness: {stats['best_fitness']:.4f}"
                message += f"\n   • Population: {stats['population_size']}"
                message += f"\n   • Mutations: {stats['mutations']} | Crossovers: {stats['crossovers']}"
            
            message += f"""

📈 **Overall Progress:**
• **Total Generations:** {evolution_stats.get('generations_completed', 0)}
• **Successful Mutations:** {evolution_stats.get('successful_mutations', 0)}
• **Crossover Events:** {evolution_stats.get('crossover_events', 0)}
• **Best Fitness Ever:** {evolution_stats.get('best_fitness_ever', 0.0):.4f}"""
            
            # Add fitness trend analysis
            if fitness_trend:
                avg_improvement = sum(fitness_trend) / len(fitness_trend)
                message += f"\n• **Avg Fitness Change:** {avg_improvement:+.4f} per generation"
            
            # Add recent mutations if available
            mutation_history = history.get('mutation_history', [])
            if mutation_history:
                message += f"\n\n🎲 **Recent Mutations:**"
                for mutation in mutation_history[-3:]:
                    success_emoji = "✅" if mutation['success'] else "❌"
                    mutation_type = mutation['mutation_type'].replace('_', ' ').title()
                    message += f"\n{success_emoji} **{mutation_type}** - {len(mutation['genes_affected'])} genes affected"
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("🧬 Evolve Next Gen", callback_data="strategy_evolve")],
                [InlineKeyboardButton("📊 Current Status", callback_data="strategy_status")],
                [InlineKeyboardButton("🔄 Refresh History", callback_data="strategy_history")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting evolution history: {e}")
            await update.message.reply_text(f"❌ Evolution history failed: {str(e)}")
    
    async def _show_strategy_help(self, update: Update, context: CallbackContext):
        """Show strategy command help"""
        try:
            message = """🧬 **STRATEGY EVOLUTION COMMANDS**

🔬 **Available Commands:**

**🧬 Evolution Commands:**
• `/strategy evolve` - Evolve next generation of strategies
• `/strategy status` - View current population status
• `/strategy mutate now` - Trigger immediate strategy mutation
• `/strategy population` - View population composition
• `/strategy history` - View evolution history

**🤖 Reinforcement Learning:**
• `/strategy rl` - View RL agent status and performance

**📊 Analysis Commands:**
• Population fitness statistics
• Strategy type distribution
• Evolution trend analysis
• Top performer tracking

**🎯 Evolution Process:**
1. **Genetic Algorithm:** Strategies evolve through mutation and crossover
2. **Natural Selection:** Best performers survive to next generation
3. **Diversity Maintenance:** Multiple strategy types maintained
4. **Continuous Learning:** RL agent learns from trading experiences

**💡 Examples:**
• `/strategy evolve` - Start evolution process
• `/strategy status` - Check population health
• `/strategy mutate now` - Create new variant
• `/strategy rl` - View learning progress"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing strategy help: {e}")
            await update.message.reply_text("❌ Error displaying strategy help")
    
    async def handle_callback_query(self, update: Update, context: CallbackContext):
        """Handle callback queries from inline buttons"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data == "strategy_evolve":
                await self._handle_strategy_evolve(query, context)
            elif data == "strategy_status":
                await self._handle_strategy_status(query, context)
            elif data == "strategy_population":
                await self._handle_population_overview(query, context)
            elif data == "strategy_history":
                await self._handle_evolution_history(query, context)
            elif data == "strategy_rl":
                await self._handle_rl_status(query, context)
            elif data == "strategy_mutate_random":
                await self._handle_strategy_mutate(query, context, ['now'])
            elif data.startswith("strategy_"):
                # Handle other strategy-related callbacks
                await query.edit_message_text("🔄 Processing strategy request...")
                
        except Exception as e:
            logger.error(f"Error handling strategy callback: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
