#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telegram Commands Handler for the Trading Bot
Implements debug commands and health check capabilities
"""

import os
import logging
import time
from threading import Thread
import json
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Try to import playbook manager
try:
    from src.utils.playbook_manager import playbook_manager
    PLAYBOOK_MANAGER_AVAILABLE = True
except ImportError:
    PLAYBOOK_MANAGER_AVAILABLE = False
    logger.warning("Playbook manager not available - playbook commands will be limited")

# Try to import chart generator
try:
    from src.utils.chart_generator import chart_generator
    CHART_GENERATOR_AVAILABLE = True
except ImportError:
    CHART_GENERATOR_AVAILABLE = False
    logger.warning("Chart generator not available - visual forecasts disabled")

# Try to import market data
try:
    from src.market_data import MarketData
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False
    logger.warning("Market data not available - forecast data limited")

# Try to import visual ML tuning
try:
    from src.integrations.telegram_visual_ml import get_telegram_visual_ml
    VISUAL_ML_AVAILABLE = True
except ImportError:
    VISUAL_ML_AVAILABLE = False
    logger.warning("Visual ML tuning not available - ML features disabled")

class TelegramCommandHandler:
    """Handler for Telegram bot commands"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelegramCommandHandler, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the command handler"""
        self.commands = {
            '/help': self._cmd_help,
            '/status': self._cmd_status,
            '/health': self._cmd_health,
            '/trades': self._cmd_trades,
            '/reset': self._cmd_reset,
            '/set': self._cmd_set,
            '/get': self._cmd_get,
            '/balance': self._cmd_balance,
            '/playbook': self._cmd_playbook,
            '/forecast': self._cmd_forecast,
            '/plan': self._cmd_plan,
            '/tune': self._cmd_tune,
            '/metrics': self._cmd_metrics,
            '/audit': self._cmd_audit
        }
        
        # Map of available parameters that can be set
        self.settable_params = {
            'confidence': {
                'env_var': 'CONFIDENCE_THRESHOLD',
                'type': float,
                'min': 1.0,
                'max': 100.0,
                'default': 95.0,
                'description': 'Signal confidence threshold (%)'
            },
            'max_signals': {
                'env_var': 'MAX_SIGNALS_PER_DAY',
                'type': int,
                'min': 1,
                'max': 100,
                'default': 10,
                'description': 'Maximum signals per day'
            },
            'max_trades': {
                'env_var': 'MAX_TRADES_PER_DAY',
                'type': int,
                'min': 1,
                'max': 50,
                'default': 5,
                'description': 'Maximum trades per day'
            },
            'position_size': {
                'env_var': 'POSITION_SIZE_PERCENT',
                'type': float,
                'min': 1.0,
                'max': 100.0,
                'default': 25.0,
                'description': 'Position size as percentage of available balance'
            }
        }
        
        self.bot_instance = None
        self.telegram_notifier = None
        logger.info("Telegram command handler initialized")
    
    def set_bot_instance(self, bot_instance):
        """Set the bot instance reference for command execution"""
        self.bot_instance = bot_instance
        logger.info("Bot instance reference set for command handler")
    
    def set_telegram_notifier(self, telegram_notifier):
        """Set the telegram notifier reference for sending photos"""
        self.telegram_notifier = telegram_notifier
        logger.info("Telegram notifier reference set for command handler")
    
    def process_command(self, message: str) -> Optional[str]:
        """
        Process a potential command message from Telegram
        
        Args:
            message: The message text from Telegram
            
        Returns:
            Optional response message to send back, or None if not a command
        """
        # Check if it's a command (starts with /)
        if not message.startswith('/'):
            return None
        
        # Extract command and arguments
        parts = message.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        logger.info(f"Processing command: {command} with args: {args}")
        
        # Execute command if it exists
        if command in self.commands:
            try:
                return self.commands[command](args)
            except Exception as e:
                error_msg = f"Error executing command {command}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return f"⚠️ *Command Error*\n\n{error_msg}\n\n```\n{traceback.format_exc()[:500]}...\n```"
        else:
            return f"❓ Unknown command: {command}\n\nType /help for available commands."
    
    def _cmd_help(self, args: List[str] = None) -> str:
        """Show help message with available commands"""
        help_text = "🤖 *Regime Intelligence Bot Commands:*\n\n"
        help_text += "/status - Show bot status\n"
        help_text += "/health - Run health check\n"
        help_text += "/trades - Show recent trades\n"
        help_text += "/reset - Reset daily counters\n"
        help_text += "/set <param> <value> - Set parameter\n"
        help_text += "/get <param> - Get parameter value\n"
        help_text += "/balance - Show account balance\n"
        help_text += "/playbook <active|list|update> - Manage playbooks\n"
        help_text += "/forecast [symbol] [timeframe] - Generate regime forecast\n"
        
        help_text += "\n*Examples:*\n"
        help_text += "/health api - Run API connectivity health check\n"
        help_text += "/reset BTCUSDT - Reset position tracking for BTC\n"
        help_text += "/set confidence 90 - Set confidence threshold to 90%\n"
        help_text += "/playbook active - Show active playbook configuration\n"
        help_text += "/forecast BTCUSDT 4h - Get forecast for BTC on 4h timeframe\n"
        
        return help_text
    
    def _cmd_status(self, args: List[str]) -> str:
        """Show bot status summary"""
        if not self.bot_instance:
            return "⚠️ Cannot access bot instance"
        
        try:
            # Bot version and runtime info
            uptime = time.time() - getattr(self.bot_instance, 'start_time', time.time())
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Get basic status information
            status = {
                'mode': 'TEST' if getattr(self.bot_instance, 'test_mode', True) else 'LIVE',
                'uptime': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
                'signals_today': len(getattr(self.bot_instance, 'daily_signals', [])),
                'trades_today': getattr(self.bot_instance, 'daily_trades_count', 0),
                'active_trades': len(getattr(self.bot_instance, 'active_trades', [])),
                'confidence_threshold': getattr(self.bot_instance, 'confidence_threshold', 'N/A'),
                'max_signals': getattr(self.bot_instance, 'max_signals_per_day', 'N/A'),
                'max_trades': getattr(self.bot_instance, 'max_trades_per_day', 'N/A')
            }
            
            # Format the response
            response = f"🤖 *Trading Bot Status*\n\n"
            response += f"*Mode:* {status['mode']}\n"
            response += f"*Uptime:* {status['uptime']}\n"
            response += f"*Signals Today:* {status['signals_today']}\n"
            response += f"*Trades Today:* {status['trades_today']}\n"
            response += f"*Active Trades:* {status['active_trades']}\n"
            response += f"*Confidence Threshold:* {status['confidence_threshold']}%\n"
            
            # Add active trades list if any
            if status['active_trades'] > 0:
                response += "\n*Active Positions:*\n"
                for symbol in getattr(self.bot_instance, 'active_trades', []):
                    response += f"• {symbol}\n"
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Status Error*\n\n{error_msg}"
    
    def _cmd_health(self, args: List[str]) -> str:
        """Run health check(s)"""
        try:
            from src.utils.health_check import health_check
            
            # Determine which checks to run
            check_types = None
            if args:
                if args[0].lower() == 'all':
                    check_types = None  # Will run all checks
                else:
                    # Map argument to check type
                    check_map = {
                        'api': 'api_connection',
                        'trading': 'trading_permissions',
                        'signals': 'signal_generation',
                        'trades': 'active_trades',
                        'balance': 'balance',
                        'system': 'system'
                    }
                    
                    check_types = []
                    for arg in args:
                        if arg.lower() in check_map:
                            check_types.append(check_map[arg.lower()])
            
            # Run health check and get results
            results = health_check.run_health_check(check_types=check_types, notify=False)
            
            # Format response
            status_emoji = {
                'ok': '✅',
                'warning': '⚠️',
                'error': '❌'
            }
            
            overall_status = results['overall_status']
            emoji = status_emoji.get(overall_status, '❓')
            
            response = [
                f"{emoji} *Health Check Results*",
                f"Status: {overall_status.upper()}",
                f"Execution Time: {results['duration_ms']}ms",
                ""
            ]
            
            # Add individual check results
            for check_name, check_result in results['checks'].items():
                check_emoji = status_emoji.get(check_result['status'], '❓')
                response.append(f"{check_emoji} {check_name}: {check_result['message']}")
            
            # Include limited details for important checks
            if 'details' in results:
                for check_name, details in results['details'].items():
                    if check_name == 'active_trades' and details:
                        response.append("\n*Active Positions:*")
                        positions = details.get('actual_positions', [])
                        if positions:
                            for pos in positions:
                                response.append(f"• {pos.get('symbol', 'Unknown')}: {pos.get('size', 0)} ({pos.get('side', 'unknown')})")
                        else:
                            response.append("No active positions")
                    
                    if check_name == 'balance' and details:
                        response.append(f"\n*Balance:* {details.get('available', 0)} USDT available, {details.get('frozen', 0)} USDT frozen")
                        
            return "\n".join(response)
            
        except ImportError:
            return "⚠️ Health check module not available"
        except Exception as e:
            error_msg = f"Error running health check: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Health Check Error*\n\n{error_msg}"
    
    def _cmd_trades(self, args: List[str]) -> str:
        """Show active trades and positions"""
        if not self.bot_instance:
            return "⚠️ Cannot access bot instance"
        
        try:
            # First get locally tracked trades
            active_trades = getattr(self.bot_instance, 'active_trades', [])
            
            response = f"🔄 *Active Trades*\n\n"
            
            # First show locally tracked trades
            response += "*Tracked Positions:*\n"
            if active_trades:
                for symbol in active_trades:
                    response += f"• {symbol}\n"
            else:
                response += "No tracked positions\n"
            
            # Try to get actual API positions
            try:
                from src.integrations.bidget import TradingAPI
                api = TradingAPI()
                
                if api.is_configured:
                    # Get positions from API
                    positions_endpoint = "/api/mix/v1/position/allPosition?productType=umcbl&marginCoin=USDT"
                    positions_response = api._make_request("GET", positions_endpoint, signed=True)
                    
                    if 'error' not in positions_response and 'data' in positions_response:
                        positions_data = positions_response.get('data', [])
                        
                        response += "\n*Actual Positions:*\n"
                        
                        active_positions = []
                        for position in positions_data:
                            if isinstance(position, dict):
                                total = float(position.get('total', 0))
                                if total > 0:
                                    symbol = position.get('symbol', '').replace('_UMCBL', '')
                                    side = position.get('holdSide', 'unknown')
                                    leverage = position.get('leverage', 'N/A')
                                    
                                    active_positions.append({
                                        'symbol': symbol,
                                        'size': total,
                                        'side': side,
                                        'leverage': leverage
                                    })
                        
                        if active_positions:
                            for pos in active_positions:
                                response += f"• {pos['symbol']}: {pos['size']} ({pos['side']}, {pos['leverage']}x)\n"
                        else:
                            response += "No active positions found\n"
                    else:
                        response += "\n⚠️ Failed to retrieve actual positions from API\n"
                else:
                    response += "\n⚠️ API not configured, showing only tracked trades\n"
            except Exception as e:
                response += f"\n⚠️ Error retrieving API positions: {str(e)[:100]}...\n"
                
            # Check for discrepancies
            tracked_set = {s.replace('/', '') for s in active_trades}
            
            try:
                api_set = {p['symbol'] for p in active_positions}
                missing = [s for s in api_set if s not in tracked_set]
                extra = [s for s in tracked_set if s not in api_set]
                
                if missing or extra:
                    response += "\n⚠️ *Tracking Discrepancies:*\n"
                    if missing:
                        response += f"Positions not tracked: {', '.join(missing)}\n"
                    if extra:
                        response += f"Tracked but no position: {', '.join(extra)}\n"
            except:
                pass
                
            return response
            
        except Exception as e:
            error_msg = f"Error getting trades: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Trades Error*\n\n{error_msg}"
    
    def _cmd_reset(self, args: List[str]) -> str:
        """Reset tracking for a symbol, all symbols, or notification cache"""
        if not args:
            return "⚠️ Missing arguments. Use: /reset all, /reset notifications, or /reset SYMBOL"
        
        try:
            # Reset all active trades tracking
            if args[0].lower() == 'all':
                if self.bot_instance:
                    symbols = list(getattr(self.bot_instance, 'active_trades', []))
                    getattr(self.bot_instance, 'active_trades', set()).clear()
                    return f"✅ Reset tracking for all symbols: {', '.join(symbols) if symbols else 'none'}"
                else:
                    return "⚠️ Cannot access bot instance"
            
            # Reset notification cache
            elif args[0].lower() == 'notifications':
                try:
                    from src.utils.notification_cache import notification_cache
                    notification_cache.reset()
                    return "✅ Notification cache has been reset"
                except ImportError:
                    return "⚠️ Notification cache module not found"
            
            # Reset specific symbol
            else:
                symbol = args[0].upper()
                
                # Format symbol consistently
                if '/' not in symbol and len(symbol) > 3:
                    # Convert BTCUSDT to BTC/USDT format if needed
                    base = symbol[:-4] if symbol.endswith('USDT') else symbol
                    symbol = f"{base}/USDT"
                
                if self.bot_instance:
                    # Use the bot's force reset method
                    if hasattr(self.bot_instance, 'check_and_clean_active_trades'):
                        self.bot_instance.check_and_clean_active_trades(force_reset=True, symbol_to_reset=symbol)
                        return f"✅ Reset tracking for {symbol}"
                    # Fallback to direct manipulation
                    elif symbol in getattr(self.bot_instance, 'active_trades', set()):
                        getattr(self.bot_instance, 'active_trades', set()).remove(symbol)
                        return f"✅ Reset tracking for {symbol}"
                    else:
                        return f"⚠️ {symbol} is not in active trades list"
                else:
                    return "⚠️ Cannot access bot instance"
        except Exception as e:
            error_msg = f"Error during reset: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Reset Error*\n\n{error_msg}"
    
    def _cmd_set(self, args: List[str]) -> str:
        """Set bot parameters"""
        if len(args) < 2:
            params_list = ", ".join(self.settable_params.keys())
            return f"⚠️ Missing arguments. Use: /set parameter value\n\nAvailable parameters: {params_list}"
        
        param_name = args[0].lower()
        value_str = args[1]
        
        if param_name not in self.settable_params:
            params_list = ", ".join(self.settable_params.keys())
            return f"⚠️ Invalid parameter '{param_name}'. Available parameters: {params_list}"
        
        param_info = self.settable_params[param_name]
        
        try:
            # Convert value to correct type
            value = param_info['type'](value_str)
            
            # Validate range
            if value < param_info['min'] or value > param_info['max']:
                return f"⚠️ Value out of range. {param_name} must be between {param_info['min']} and {param_info['max']}"
            
            # Apply setting to bot instance
            if self.bot_instance:
                # Try direct attribute setting
                param_attr = None
                if param_name == 'confidence':
                    param_attr = 'confidence_threshold'
                elif param_name == 'max_signals':
                    param_attr = 'max_signals_per_day'
                elif param_name == 'max_trades':
                    param_attr = 'max_trades_per_day'
                elif param_name == 'position_size':
                    param_attr = 'position_size_percent'
                
                if param_attr and hasattr(self.bot_instance, param_attr):
                    setattr(self.bot_instance, param_attr, value)
                    
                # For debugging
                logger.info(f"Set parameter {param_name} to {value} via attribute {param_attr}")
                
                # Also set the environment variable so it persists on restart
                env_var = param_info['env_var']
                os.environ[env_var] = str(value)
                
                # For debugging
                logger.info(f"Set environment variable {env_var} to {value}")
                
                return f"✅ Set {param_name} to {value}"
            else:
                # If we can't access bot instance, just set the environment variable
                env_var = param_info['env_var']
                os.environ[env_var] = str(value)
                return f"✅ Set {param_name} to {value} (environment only)"
                
        except ValueError:
            return f"⚠️ Invalid value format. {param_name} must be a {param_info['type'].__name__}"
        except Exception as e:
            error_msg = f"Error setting parameter: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Parameter Error*\n\n{error_msg}"
    
    def _cmd_get(self, args: List[str]) -> str:
        """Get current parameter values"""
        response = "🔧 *Current Bot Parameters*\n\n"
        
        for param_name, param_info in self.settable_params.items():
            # Get current value - first try bot instance
            current_value = None
            
            if self.bot_instance:
                param_attr = None
                if param_name == 'confidence':
                    param_attr = 'confidence_threshold'
                elif param_name == 'max_signals':
                    param_attr = 'max_signals_per_day'
                elif param_name == 'max_trades':
                    param_attr = 'max_trades_per_day'
                elif param_name == 'position_size':
                    param_attr = 'position_size_percent'
                
                if param_attr and hasattr(self.bot_instance, param_attr):
                    current_value = getattr(self.bot_instance, param_attr)
            
            # Fall back to environment variable
            if current_value is None:
                env_var = param_info['env_var']
                env_value = os.getenv(env_var)
                if env_value:
                    try:
                        current_value = param_info['type'](env_value)
                    except ValueError:
                        current_value = f"Invalid format: {env_value}"
                else:
                    current_value = param_info['default']
            
            # Add to response
            response += f"*{param_name}*: {current_value} ({param_info['description']})\n"
        
        return response
    
    def _cmd_balance(self, args: List[str]) -> str:
        """Show account balance"""
        try:
            from src.integrations.bidget import TradingAPI
            api = TradingAPI()
            
            if not api.is_configured:
                return "⚠️ API not configured"
            
            # Get account balance
            balance_endpoint = "/api/mix/v1/account/accounts?productType=umcbl"
            balance_response = api._make_request("GET", balance_endpoint, signed=True)
            
            if 'error' in balance_response:
                return f"⚠️ Balance API Error: {balance_response.get('error')}"
            
            balance_data = balance_response.get('data', [])
            if not balance_data:
                return "⚠️ No balance data available"
            
            response = "💰 *Account Balance*\n\n"
            
            for currency_data in balance_data:
                if not isinstance(currency_data, dict):
                    continue
                
                currency = currency_data.get('marginCoin', 'Unknown')
                available = float(currency_data.get('available', 0))
                frozen = float(currency_data.get('locked', 0))
                equity = float(currency_data.get('equity', 0))
                
                response += f"*{currency}*\n"
                response += f"Available: {available:.2f}\n"
                response += f"Frozen: {frozen:.2f}\n"
                response += f"Equity: {equity:.2f}\n"
                response += "\n"
            
            return response
            
        except ImportError:
            return "⚠️ API module not available"
        except Exception as e:
            error_msg = f"Error retrieving balance: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ *Balance Error*\n\n{error_msg}"

# Singleton instance
    def _cmd_playbook(self, args: List[str] = None) -> str:
        """Manage playbooks and view active playbook configuration"""
        if not PLAYBOOK_MANAGER_AVAILABLE:
            return "⚠️ Playbook system not available"
            
        if not args or len(args) == 0:
            return "📒 *Playbook Commands*:\n\n" + \
                   "/playbook active - Show active playbook\n" + \
                   "/playbook list - List all available playbooks\n" + \
                   "/playbook update <regime> <param> <value> - Update playbook parameter\n" + \
                   "/playbook reset - Reset all playbooks to defaults"
        
        # Handle subcommands
        subcommand = args[0].lower()
        
        if subcommand == "active":
            return self._playbook_show_active()
        elif subcommand == "list":
            return self._playbook_list_all()
        elif subcommand == "update" and len(args) >= 4:
            return self._playbook_update(args[1], args[2], args[3])
        elif subcommand == "reset":
            return self._playbook_reset()
        else:
            return "❌ Invalid playbook command. Use /playbook without arguments to see available options."
    
    def _playbook_show_active(self) -> str:
        """Show active playbook configuration"""
        active = playbook_manager.get_active_playbook()
        
        if not active["active_regime"] or not active["playbook"]:
            return "ℹ️ No active playbook currently"
            
        # Format activation timestamp
        activated_at = active.get("activated_at")
        time_str = "Unknown"
        if activated_at:
            try:
                dt = datetime.fromisoformat(activated_at)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass
        
        # Build response message
        playbook = active["playbook"]
        response = f"📊 *Active Playbook for Regime: {active['active_regime']}*\n"
        response += f"Activated: {time_str}\n\n"
        
        # Add playbook details
        response += f"Strategy: {playbook.get('strategy', 'N/A')}\n"
        response += f"Leverage: {playbook.get('leverage', 'N/A')}\n"
        response += f"Entry Type: {playbook.get('entry_type', 'N/A')}\n"
        response += f"Stop Loss: {playbook.get('stop_loss', 'N/A')}\n"
        
        # Handle take profit targets (may be a list)
        take_profit = playbook.get('take_profit', [])
        if isinstance(take_profit, list) and take_profit:
            response += f"Take Profit: {', '.join(take_profit)}\n"
        elif take_profit:
            response += f"Take Profit: {take_profit}\n"
        else:
            response += "Take Profit: N/A\n"
            
        response += f"Risk Level: {playbook.get('risk_level', 'N/A')}\n"
        
        return response
    
    def _playbook_list_all(self) -> str:
        """List all available playbooks"""
        playbooks = playbook_manager.playbooks
        
        if not playbooks:
            return "ℹ️ No playbooks available"
        
        response = "📒 *Available Playbooks:*\n\n"
        
        # Sort playbooks by name for consistent listing
        for regime in sorted(playbooks.keys()):
            playbook = playbooks[regime]
            strategy = playbook.get('strategy', 'N/A')
            risk_level = playbook.get('risk_level', 'N/A')
            
            response += f"*{regime}*: {strategy} (Risk: {risk_level})\n"
        
        response += "\nUse /playbook active to see the currently active playbook"
        return response
    
    def _playbook_update(self, regime: str, param: str, value: str) -> str:
        """Update playbook parameter for a specific regime"""
        # Validate regime exists
        if regime not in playbook_manager.playbooks:
            return f"❌ Regime '{regime}' not found in playbooks"
        
        # Handle different parameter types
        if param == "leverage":
            try:
                value = int(value)
            except ValueError:
                return f"❌ Leverage must be a number"
                
        elif param == "take_profit":
            # Handle take_profit as a comma-separated list
            value = [item.strip() for item in value.split(',')]
        
        # Update the playbook
        success = playbook_manager.update_playbook_config(regime, {param: value})
        
        if success:
            return f"✅ Updated {param} to {value} for regime '{regime}'"
        else:
            return f"❌ Failed to update playbook parameter"
    
    def _playbook_reset(self) -> str:
        """Reset all playbooks to defaults"""
        success = playbook_manager.reset_to_defaults()
        
        if success:
            return "✅ Reset all playbooks to default values"
        else:
            return "❌ Failed to reset playbooks"
    
    def _cmd_forecast(self, args: List[str]) -> str:
        """
        Generate visual forecast with chart and analysis.
        Usage: /forecast [symbol] [timeframe]
        """
        try:
            # Parse arguments with validation
            symbol = args[0].upper() if len(args) > 0 else "BTCUSDT"
            timeframe = args[1] if len(args) > 1 else "1h"
            
            # Validate timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
            if timeframe not in valid_timeframes:
                return f"⚠️ **Invalid Timeframe**\n\nSupported timeframes: {', '.join(valid_timeframes)}\n\n💡 **Usage:** `/forecast [symbol] [timeframe]`"
            
            # Generate forecast analysis and chart
            return self._generate_visual_forecast(symbol, timeframe, forecast_type="forecast")
            
        except Exception as e:
            error_msg = f"Error generating forecast: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ **Forecast Error**\n\n{error_msg}\n\n💡 **Usage:** `/forecast [symbol] [timeframe]`\n**Example:** `/forecast BTCUSDT 1h`"
    
    def _cmd_plan(self, args: List[str]) -> str:
        """
        Generate trading plan with visual chart and strategy context.
        Usage: /plan [symbol] [timeframe]
        """
        try:
            # Parse arguments with validation
            symbol = args[0].upper() if len(args) > 0 else "BTCUSDT"
            timeframe = args[1] if len(args) > 1 else "1h"
            
            # Validate timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
            if timeframe not in valid_timeframes:
                return f"⚠️ **Invalid Timeframe**\n\nSupported timeframes: {', '.join(valid_timeframes)}\n\n💡 **Usage:** `/plan [symbol] [timeframe]`"
            
            # Generate trading plan with chart
            return self._generate_visual_forecast(symbol, timeframe, forecast_type="plan")
            
        except Exception as e:
            error_msg = f"Error generating plan: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"⚠️ **Plan Error**\n\n{error_msg}\n\n💡 **Usage:** `/plan [symbol] [timeframe]`\n**Example:** `/plan BTCUSDT 1h`"
    
    def _generate_visual_forecast(self, symbol: str, timeframe: str, forecast_type: str = "forecast") -> str:
        """
        Generate visual forecast with chart and comprehensive analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            forecast_type: Type of forecast ("forecast" or "plan")
            
        Returns:
            str: Formatted message or error
        """
        try:
            # Get current market data and generate signal
            market_data = None
            current_signal = None
            confidence_score = 75.0
            
            if MARKET_DATA_AVAILABLE and self.bot_instance:
                try:
                    # Get market data from bot instance
                    market_data_instance = getattr(self.bot_instance, 'market_data', None)
                    if market_data_instance:
                        # Get recent price data
                        price_data = market_data_instance.get_price_data(symbol, timeframe, limit=100)
                        if price_data:
                            market_data = {
                                'timestamps': price_data.get('timestamps', []),
                                'open': price_data.get('open', []),
                                'high': price_data.get('high', []),
                                'low': price_data.get('low', []),
                                'close': price_data.get('close', []),
                                'volume': price_data.get('volume', [])
                            }
                            
                            # Generate a mock signal for demonstration
                            current_price = price_data['close'][-1] if price_data['close'] else 50000
                            direction = "LONG" if len(price_data['close']) > 1 and price_data['close'][-1] > price_data['close'][-2] else "SHORT"
                            
                            current_signal = {
                                'symbol': symbol,
                                'direction': direction,
                                'price': current_price,
                                'profit_target': current_price * 1.02 if direction == "LONG" else current_price * 0.98,
                                'stop_loss': current_price * 0.98 if direction == "LONG" else current_price * 1.02,
                                'confidence': confidence_score,
                                'strategy_name': 'Supertrend ADX',
                                'timestamp': datetime.now()
                            }
                            
                            confidence_score = min(95.0, max(60.0, confidence_score + (len(price_data['close']) / 10)))
                except Exception as e:
                    logger.warning(f"Failed to get market data: {e}")
            
            # Generate forecast data (mock for now)
            forecast_data = self._generate_mock_forecast_data(symbol, current_signal)
            
            # Generate regime zones (mock for now)
            regime_zones = self._generate_mock_regime_zones()
            
            # Log forecast snapshot for ML learning (if available)
            forecast_id = None
            if VISUAL_ML_AVAILABLE and current_signal:
                try:
                    visual_ml = get_telegram_visual_ml()
                    if visual_ml and visual_ml.visual_ml_tuner:
                        # Prepare forecast data for logging
                        forecast_log_data = {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'regime_type': regime_zones[0]['type'] if regime_zones else 'sideways',
                            'pattern_detected': 'breakout' if confidence_score > 80 else 'continuation',
                            'confidence_score': confidence_score / 100.0,
                            'predicted_direction': current_signal['direction'],
                            'predicted_price_target': current_signal['profit_target'],
                            'predicted_timeframe_hours': 24,
                            'volatility_level': 'high' if confidence_score > 85 else 'medium',
                            'trend_strength': min(1.0, confidence_score / 100.0),
                            'support_resistance_clarity': 0.7,
                            'volume_confirmation': True,
                            'strategy_name': current_signal['strategy_name'],
                            'entry_price': current_signal['price'],
                            'stop_loss': current_signal['stop_loss'],
                            'profit_target': current_signal['profit_target']
                        }
                        
                        forecast_id = visual_ml.visual_ml_tuner.log_forecast_snapshot(forecast_log_data)
                        logger.info(f"Logged forecast snapshot for ML learning: {forecast_id}")
                except Exception as e:
                    logger.warning(f"Failed to log forecast snapshot: {e}")
            
            # Create chart if chart generator is available
            chart_sent = False
            if CHART_GENERATOR_AVAILABLE and hasattr(self, 'telegram_notifier'):
                try:
                    if market_data:
                        # Generate comprehensive chart
                        chart_data = chart_generator.generate_forecast_chart(
                            data=market_data,
                            forecast_data=forecast_data,
                            signal=current_signal,
                            regime_zones=regime_zones,
                            confidence_score=confidence_score,
                            timeframe=timeframe
                        )
                    else:
                        # Generate simple chart
                        chart_data = chart_generator.generate_simple_chart(
                            symbol=symbol,
                            price=current_signal['price'] if current_signal else 50000,
                            direction=current_signal['direction'] if current_signal else "LONG",
                            confidence=confidence_score
                        )
                    
                    # Format message based on forecast type
                    if forecast_type == "plan":
                        message_text = self._format_plan_message(symbol, timeframe, current_signal, confidence_score)
                    else:
                        message_text = self._format_forecast_message(symbol, timeframe, current_signal, confidence_score)
                    
                    # Send chart with caption
                    if hasattr(self, 'telegram_notifier') and self.telegram_notifier:
                        result = self.telegram_notifier.send_photo(chart_data, message_text)
                        if 'error' not in result:
                            chart_sent = True
                            return "📊 Visual forecast sent successfully!"
                        else:
                            logger.error(f"Failed to send chart: {result.get('error')}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate/send chart: {e}")
            
            # Fallback to text-only forecast if chart failed
            if not chart_sent:
                if forecast_type == "plan":
                    return self._format_plan_message(symbol, timeframe, current_signal, confidence_score)
                else:
                    return self._format_forecast_message(symbol, timeframe, current_signal, confidence_score)
                    
        except Exception as e:
            logger.error(f"Error in visual forecast generation: {e}", exc_info=True)
            return f"⚠️ **{forecast_type.title()} Error**\n\nFailed to generate visual {forecast_type}: {str(e)}"
    
    def _generate_mock_forecast_data(self, symbol: str, signal: Optional[Dict] = None) -> Dict:
        """Generate mock forecast data for demonstration"""
        from datetime import timedelta
        import numpy as np
        
        base_time = datetime.now()
        forecast_times = [base_time + timedelta(hours=i) for i in range(1, 13)]
        
        base_price = signal['price'] if signal else 50000
        trend_direction = 1 if (signal and signal['direction'] == 'LONG') else -1
        
        # Generate realistic price projection
        predicted_prices = []
        for i in range(12):
            trend_factor = trend_direction * 0.001 * i
            noise = np.random.normal(0, 0.005)
            price = base_price * (1 + trend_factor + noise)
            predicted_prices.append(price)
        
        # Generate confidence bands
        confidence_upper = [p * 1.01 for p in predicted_prices]
        confidence_lower = [p * 0.99 for p in predicted_prices]
        
        return {
            'timestamps': forecast_times,
            'predicted_prices': predicted_prices,
            'confidence_upper': confidence_upper,
            'confidence_lower': confidence_lower
        }
    
    def _generate_mock_regime_zones(self) -> List[Dict]:
        """Generate mock regime zones for demonstration"""
        from datetime import timedelta
        
        base_time = datetime.now() - timedelta(hours=24)
        
        return [
            {
                'start_time': base_time,
                'end_time': base_time + timedelta(hours=8),
                'type': 'bearish'
            },
            {
                'start_time': base_time + timedelta(hours=8),
                'end_time': base_time + timedelta(hours=16),
                'type': 'sideways'
            },
            {
                'start_time': base_time + timedelta(hours=16),
                'end_time': base_time + timedelta(hours=24),
                'type': 'bullish'
            }
        ]
    
    def _format_forecast_message(self, symbol: str, timeframe: str, signal: Optional[Dict], confidence: float) -> str:
        """Format forecast message in YAML style"""
        direction_emoji = "🟢" if (signal and signal['direction'] == 'LONG') else "🔴"
        strategy = signal['strategy_name'] if signal else "Technical Analysis"
        
        message = f"🧠 **Forecast: {symbol} Analysis**\n"
        message += f"📉 **Confidence:** {confidence:.0f}%\n"
        message += f"📈 **Strategy:** {strategy} | **Timeframe:** {timeframe}\n"
        message += f"📊 **Chart attached below:**\n\n"
        
        if signal:
            message += f"{direction_emoji} **Signal:** {signal['direction']} @ {signal['price']:.6f}\n"
            message += f"🎯 **Target:** {signal['profit_target']:.6f}\n"
            message += f"🛡️ **Stop:** {signal['stop_loss']:.6f}\n\n"
        
        message += f"⏰ **Generated:** {datetime.now().strftime('%H:%M UTC')}"
        return message
    
    def _format_plan_message(self, symbol: str, timeframe: str, signal: Optional[Dict], confidence: float) -> str:
        """Format trading plan message in YAML style"""
        direction_emoji = "🟢" if (signal and signal['direction'] == 'LONG') else "🔴"
        strategy = signal['strategy_name'] if signal else "Technical Analysis"
        
        message = f"📋 **Trading Plan: {symbol}**\n"
        message += f"📉 **Confidence:** {confidence:.0f}%\n"
        message += f"📈 **Strategy:** {strategy} | **Leverage:** 3x\n"
        message += f"📊 **Chart attached below:**\n\n"
        
        if signal:
            message += f"{direction_emoji} **Entry:** {signal['direction']} @ {signal['price']:.6f}\n"
            message += f"🎯 **Take Profit:** {signal['profit_target']:.6f}\n"
            message += f"🛡️ **Stop Loss:** {signal['stop_loss']:.6f}\n"
            
            # Calculate risk-reward
            risk = abs(signal['price'] - signal['stop_loss'])
            reward = abs(signal['profit_target'] - signal['price'])
            if risk > 0:
                rr_ratio = reward / risk
                message += f"⚖️ **Risk:Reward:** 1:{rr_ratio:.2f}\n"
        
        message += f"\n⏰ **Generated:** {datetime.now().strftime('%H:%M UTC')}"
        return message
    
    def _cmd_tune(self, args: List[str]) -> str:
        """
        Handle ML tuning commands for visual intelligence optimization.
        Usage: /tune forecast [generate|apply|status]
        """
        try:
            if not VISUAL_ML_AVAILABLE:
                return "⚠️ **Visual ML System Unavailable**\n\nML-based tuning is not available. Please ensure the analytics module is properly installed."
            
            if not args:
                return "💡 **Tune Command Usage:**\n\n• `/tune forecast` - Generate visual intelligence recommendations\n• `/tune forecast apply` - Apply safe recommendations\n• `/tune forecast status` - Show tuning status\n\nExample: `/tune forecast`"
            
            subcommand = args[0].lower()
            if subcommand == 'forecast':
                visual_ml = get_telegram_visual_ml()
                return visual_ml.handle_tune_forecast_command(args[1:] if len(args) > 1 else [])
            else:
                return f"⚠️ **Unknown Tune Command**\n\nSupported: `forecast`\n\n💡 **Usage:** `/tune forecast`"
                
        except Exception as e:
            logger.error(f"Error in tune command: {e}")
            return f"❌ **Tune Error:** {str(e)}"
    
    def _cmd_metrics(self, args: List[str]) -> str:
        """
        Handle metrics command to show visual intelligence performance.
        Usage: /metrics [days]
        """
        try:
            if not VISUAL_ML_AVAILABLE:
                return "⚠️ **Visual ML System Unavailable**\n\nPerformance metrics are not available."
            
            visual_ml = get_telegram_visual_ml()
            return visual_ml.handle_metrics_command(args)
            
        except Exception as e:
            logger.error(f"Error in metrics command: {e}")
            return f"❌ **Metrics Error:** {str(e)}"
    
    def _cmd_audit(self, args: List[str]) -> str:
        """
        Handle audit command for detailed forecast accuracy analysis.
        Usage: /audit [days]
        """
        try:
            if not VISUAL_ML_AVAILABLE:
                return "⚠️ **Visual ML System Unavailable**\n\nForecast audit is not available."
            
            visual_ml = get_telegram_visual_ml()
            return visual_ml.handle_audit_command(args)
            
        except Exception as e:
            logger.error(f"Error in audit command: {e}")
            return f"❌ **Audit Error:** {str(e)}"


# Singleton instance
telegram_commands = TelegramCommandHandler()
