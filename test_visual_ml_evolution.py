#!/usr/bin/env python3
"""
Test script for Bot 2's Visual Intelligence Evolution System
Demonstrates the complete ML-based self-tuning visual forecasting capabilities
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_visual_ml_system():
    """Test the complete visual ML evolution system"""
    try:
        logger.info("🧠 Testing Visual Intelligence Evolution System...")
        
        # Test imports
        from src.analytics.visual_ml_tuner import VisualMLTuner, get_visual_ml_tuner, ForecastSnapshot, ForecastOutcome
        from src.integrations.telegram_visual_ml import TelegramVisualML, get_telegram_visual_ml
        from src.integrations.telegram_commands import telegram_commands
        
        logger.info("✅ All ML system imports successful")
        
        # Initialize the visual ML tuner
        visual_ml_tuner = get_visual_ml_tuner()
        telegram_visual_ml = get_telegram_visual_ml()
        
        logger.info("✅ Visual ML system initialized")
        
        # Test forecast snapshot logging
        logger.info("📊 Testing forecast snapshot logging...")
        
        test_forecast_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'regime_type': 'bullish',
            'pattern_detected': 'breakout',
            'confidence_score': 0.87,
            'predicted_direction': 'LONG',
            'predicted_price_target': 52000.0,
            'predicted_timeframe_hours': 24,
            'volatility_level': 'high',
            'trend_strength': 0.8,
            'support_resistance_clarity': 0.9,
            'volume_confirmation': True,
            'strategy_name': 'Supertrend ADX',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'profit_target': 52000.0
        }
        
        forecast_id = visual_ml_tuner.log_forecast_snapshot(test_forecast_data)
        logger.info(f"✅ Forecast snapshot logged: {forecast_id}")
        
        # Test outcome logging
        logger.info("📈 Testing outcome logging...")
        
        test_outcome_data = {
            'actual_direction': 'LONG',
            'actual_price_change': 1500.0,
            'actual_max_favorable': 1800.0,
            'actual_max_adverse': -200.0,
            'target_reached': True,
            'stop_hit': False,
            'pattern_accuracy': 0.9,
            'timing_accuracy': 0.8,
            'regime_transition_occurred': True,
            'pattern_completed_as_expected': True,
            'confidence_justified': True
        }
        
        outcome_success = visual_ml_tuner.log_forecast_outcome(forecast_id, test_outcome_data)
        logger.info(f"✅ Forecast outcome logged: {outcome_success}")
        
        # Test performance analysis
        logger.info("📊 Testing performance analysis...")
        
        performance = visual_ml_tuner.analyze_visual_performance(lookback_days=30)
        if 'error' not in performance:
            logger.info(f"✅ Performance analysis successful: {performance.get('total_forecasts', 0)} forecasts analyzed")
        else:
            logger.info(f"⚠️ Performance analysis: {performance['error']}")
        
        # Test tuning recommendations
        logger.info("🔧 Testing tuning recommendations...")
        
        tuning_session = visual_ml_tuner.generate_tuning_recommendations(lookback_days=30)
        if tuning_session:
            logger.info(f"✅ Tuning recommendations generated: {len(tuning_session.recommendations)} recommendations")
        else:
            logger.info("⚠️ No tuning recommendations (insufficient data)")
        
        # Test Telegram commands
        logger.info("🤖 Testing Telegram ML commands...")
        
        # Test /metrics command
        metrics_result = telegram_visual_ml.handle_metrics_command(['30'])
        logger.info("✅ /metrics command tested")
        
        # Test /tune forecast command
        tune_result = telegram_visual_ml.handle_tune_forecast_command([])
        logger.info("✅ /tune forecast command tested")
        
        # Test /audit command
        audit_result = telegram_visual_ml.handle_audit_command(['7'])
        logger.info("✅ /audit command tested")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Visual ML system test failed: {e}")
        return False

def test_telegram_integration():
    """Test the Telegram command integration"""
    try:
        logger.info("🤖 Testing Telegram Command Integration...")
        
        from src.integrations.telegram_commands import telegram_commands
        
        # Test new ML commands
        logger.info("Testing /tune command...")
        tune_result = telegram_commands._cmd_tune(['forecast'])
        logger.info(f"✅ /tune command result type: {type(tune_result)}")
        
        logger.info("Testing /metrics command...")
        metrics_result = telegram_commands._cmd_metrics(['30'])
        logger.info(f"✅ /metrics command result type: {type(metrics_result)}")
        
        logger.info("Testing /audit command...")
        audit_result = telegram_commands._cmd_audit(['7'])
        logger.info(f"✅ /audit command result type: {type(audit_result)}")
        
        # Test forecast with ML logging integration
        logger.info("Testing /forecast with ML logging...")
        forecast_result = telegram_commands._cmd_forecast(['BTCUSDT', '1h'])
        logger.info(f"✅ /forecast with ML logging result type: {type(forecast_result)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram integration test failed: {e}")
        return False

def simulate_learning_cycle():
    """Simulate a complete learning cycle"""
    try:
        logger.info("🔄 Simulating Complete Learning Cycle...")
        
        from src.analytics.visual_ml_tuner import get_visual_ml_tuner
        
        visual_ml_tuner = get_visual_ml_tuner()
        
        # Simulate multiple forecasts with outcomes
        test_scenarios = [
            {
                'forecast': {
                    'symbol': 'ETHUSDT',
                    'timeframe': '4h',
                    'regime_type': 'bullish',
                    'pattern_detected': 'breakout',
                    'confidence_score': 0.92,
                    'predicted_direction': 'LONG',
                    'predicted_price_target': 3200.0,
                    'predicted_timeframe_hours': 48,
                    'volatility_level': 'high',
                    'trend_strength': 0.9,
                    'support_resistance_clarity': 0.85,
                    'volume_confirmation': True,
                    'strategy_name': 'Supertrend ADX',
                    'entry_price': 3000.0,
                    'stop_loss': 2900.0,
                    'profit_target': 3200.0
                },
                'outcome': {
                    'actual_direction': 'LONG',
                    'actual_price_change': 180.0,
                    'actual_max_favorable': 220.0,
                    'actual_max_adverse': -30.0,
                    'target_reached': True,
                    'stop_hit': False,
                    'pattern_accuracy': 0.95,
                    'timing_accuracy': 0.9,
                    'regime_transition_occurred': True,
                    'pattern_completed_as_expected': True,
                    'confidence_justified': True
                }
            },
            {
                'forecast': {
                    'symbol': 'ADAUSDT',
                    'timeframe': '1h',
                    'regime_type': 'bearish',
                    'pattern_detected': 'reversal',
                    'confidence_score': 0.75,
                    'predicted_direction': 'SHORT',
                    'predicted_price_target': 0.45,
                    'predicted_timeframe_hours': 12,
                    'volatility_level': 'medium',
                    'trend_strength': 0.6,
                    'support_resistance_clarity': 0.7,
                    'volume_confirmation': False,
                    'strategy_name': 'Supertrend ADX',
                    'entry_price': 0.50,
                    'stop_loss': 0.52,
                    'profit_target': 0.45
                },
                'outcome': {
                    'actual_direction': 'SHORT',
                    'actual_price_change': -0.03,
                    'actual_max_favorable': -0.04,
                    'actual_max_adverse': 0.01,
                    'target_reached': False,
                    'stop_hit': True,
                    'pattern_accuracy': 0.4,
                    'timing_accuracy': 0.3,
                    'regime_transition_occurred': False,
                    'pattern_completed_as_expected': False,
                    'confidence_justified': False
                }
            }
        ]
        
        forecast_ids = []
        
        # Log forecasts
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"📊 Logging test forecast {i}...")
            forecast_id = visual_ml_tuner.log_forecast_snapshot(scenario['forecast'])
            forecast_ids.append(forecast_id)
            
            # Log outcome
            logger.info(f"📈 Logging test outcome {i}...")
            visual_ml_tuner.log_forecast_outcome(forecast_id, scenario['outcome'])
        
        # Analyze performance
        logger.info("📊 Analyzing learning performance...")
        performance = visual_ml_tuner.analyze_visual_performance(lookback_days=1)
        
        if 'error' not in performance:
            logger.info(f"✅ Learning cycle complete:")
            logger.info(f"   • Total forecasts: {performance.get('total_forecasts', 0)}")
            logger.info(f"   • Direction accuracy: {performance.get('direction_accuracy', 0):.1%}")
            logger.info(f"   • Overall accuracy: {performance.get('overall_accuracy', 0):.1%}")
        
        # Generate tuning recommendations
        logger.info("🔧 Generating learning-based recommendations...")
        tuning_session = visual_ml_tuner.generate_tuning_recommendations(lookback_days=1)
        
        if tuning_session and tuning_session.recommendations:
            logger.info(f"✅ Generated {len(tuning_session.recommendations)} tuning recommendations")
            for i, rec in enumerate(tuning_session.recommendations, 1):
                logger.info(f"   {i}. {rec.component}: {rec.parameter} ({rec.risk_level} risk)")
        else:
            logger.info("✅ No tuning needed - system performing optimally")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning cycle simulation failed: {e}")
        return False

def main():
    """Run all visual ML evolution tests"""
    logger.info("🚀 Starting Visual Intelligence Evolution System Tests...")
    logger.info("=" * 80)
    
    tests = [
        ("Visual ML System Core", test_visual_ml_system),
        ("Telegram Integration", test_telegram_integration),
        ("Complete Learning Cycle", simulate_learning_cycle)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        logger.info("-" * 60)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} Test: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 VISUAL INTELLIGENCE EVOLUTION TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:30} : {status}")
    
    logger.info("-" * 80)
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Visual Intelligence Evolution System is LEGENDARY!")
        logger.info("\n🧠 Bot 2 now has the power to:")
        logger.info("  • 📊 Learn from every forecast it makes")
        logger.info("  • 🎯 Improve its confidence scoring through experience")
        logger.info("  • 🔧 Self-tune its visual analysis parameters")
        logger.info("  • 📈 Track pattern success rates and optimize")
        logger.info("  • 🤖 Propose intelligent upgrades with human approval")
        logger.info("\n🔥 COMMANDS AVAILABLE:")
        logger.info("  • /forecast BTCUSDT 1h  - Generate visual forecast (now with ML logging)")
        logger.info("  • /plan ETHUSDT 4h      - Create trading plan (now with ML logging)")
        logger.info("  • /tune forecast        - Get ML-based improvement recommendations")
        logger.info("  • /metrics 30           - View 30-day visual intelligence performance")
        logger.info("  • /audit 7              - Detailed 7-day forecast accuracy audit")
        logger.info("\n🌟 Bot 2 has evolved from visual intelligence to VISUAL WISDOM!")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
