#!/usr/bin/env python3
"""
Test script for the Visual Forecast Chart System
Demonstrates the new chart generation and Telegram integration capabilities
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chart_generation():
    """Test the chart generation functionality"""
    try:
        from src.utils.chart_generator import chart_generator
        
        logger.info("🧪 Testing Chart Generation System...")
        
        # Generate mock market data
        timestamps = [datetime.now() - timedelta(hours=24-i) for i in range(24)]
        base_price = 50000
        
        market_data = {
            'timestamps': timestamps,
            'open': [base_price + np.random.normal(0, 100) for _ in range(24)],
            'high': [base_price + np.random.normal(200, 100) for _ in range(24)],
            'low': [base_price + np.random.normal(-200, 100) for _ in range(24)],
            'close': [base_price + np.random.normal(0, 100) for _ in range(24)],
            'volume': [np.random.uniform(1000000, 5000000) for _ in range(24)]
        }
        
        # Generate forecast data
        forecast_times = [datetime.now() + timedelta(hours=i) for i in range(1, 13)]
        forecast_data = {
            'timestamps': forecast_times,
            'predicted_prices': [base_price * (1 + 0.001 * i) for i in range(12)],
            'confidence_upper': [base_price * (1 + 0.001 * i + 0.01) for i in range(12)],
            'confidence_lower': [base_price * (1 + 0.001 * i - 0.01) for i in range(12)]
        }
        
        # Generate mock signal
        signal = {
            'symbol': 'BTCUSDT',
            'direction': 'LONG',
            'price': base_price,
            'profit_target': base_price * 1.02,
            'stop_loss': base_price * 0.98,
            'confidence': 87.5,
            'strategy_name': 'Supertrend ADX',
            'timestamp': datetime.now()
        }
        
        # Generate regime zones
        regime_zones = [
            {
                'start_time': datetime.now() - timedelta(hours=8),
                'end_time': datetime.now() - timedelta(hours=4),
                'type': 'bearish'
            },
            {
                'start_time': datetime.now() - timedelta(hours=4),
                'end_time': datetime.now(),
                'type': 'bullish'
            }
        ]
        
        # Test comprehensive chart generation
        logger.info("📊 Generating comprehensive forecast chart...")
        chart_data = chart_generator.generate_forecast_chart(
            data=market_data,
            forecast_data=forecast_data,
            signal=signal,
            regime_zones=regime_zones,
            confidence_score=87.5,
            timeframe="1h"
        )
        
        # Save chart to file for inspection
        with open('test_forecast_chart.png', 'wb') as f:
            f.write(chart_data)
        
        logger.info("✅ Comprehensive chart generated successfully! Saved as 'test_forecast_chart.png'")
        
        # Test simple chart generation
        logger.info("📈 Generating simple forecast chart...")
        simple_chart_data = chart_generator.generate_simple_chart(
            symbol="BTCUSDT",
            price=base_price,
            direction="LONG",
            confidence=87.5
        )
        
        # Save simple chart to file
        with open('test_simple_chart.png', 'wb') as f:
            f.write(simple_chart_data)
        
        logger.info("✅ Simple chart generated successfully! Saved as 'test_simple_chart.png'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chart generation test failed: {e}")
        return False

def test_telegram_commands():
    """Test the Telegram command functionality"""
    try:
        from src.integrations.telegram_commands import telegram_commands
        
        logger.info("🤖 Testing Telegram Command System...")
        
        # Test forecast command parsing
        logger.info("Testing /forecast command...")
        forecast_result = telegram_commands._cmd_forecast(['BTCUSDT', '1h'])
        logger.info(f"Forecast result type: {type(forecast_result)}")
        
        # Test plan command parsing
        logger.info("Testing /plan command...")
        plan_result = telegram_commands._cmd_plan(['ETHUSDT', '4h'])
        logger.info(f"Plan result type: {type(plan_result)}")
        
        # Test message formatting
        logger.info("Testing message formatting...")
        mock_signal = {
            'symbol': 'BTCUSDT',
            'direction': 'LONG',
            'price': 50000,
            'profit_target': 51000,
            'stop_loss': 49000,
            'strategy_name': 'Supertrend ADX'
        }
        
        forecast_msg = telegram_commands._format_forecast_message('BTCUSDT', '1h', mock_signal, 85.0)
        plan_msg = telegram_commands._format_plan_message('BTCUSDT', '1h', mock_signal, 85.0)
        
        logger.info("✅ Forecast message format:")
        logger.info(forecast_msg)
        logger.info("✅ Plan message format:")
        logger.info(plan_msg)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram command test failed: {e}")
        return False

def test_integration():
    """Test the full integration"""
    try:
        logger.info("🔗 Testing Full Integration...")
        
        # Test imports
        from src.utils.chart_generator import chart_generator
        from src.integrations.telegram_commands import telegram_commands
        from src.integrations.telegram import TelegramNotifier
        
        logger.info("✅ All imports successful")
        
        # Test chart generator availability
        if hasattr(telegram_commands, '_generate_visual_forecast'):
            logger.info("✅ Visual forecast method available")
        else:
            logger.warning("⚠️ Visual forecast method not found")
        
        # Test mock data generation
        mock_forecast = telegram_commands._generate_mock_forecast_data('BTCUSDT')
        mock_regimes = telegram_commands._generate_mock_regime_zones()
        
        logger.info(f"✅ Mock forecast data generated: {len(mock_forecast['timestamps'])} points")
        logger.info(f"✅ Mock regime zones generated: {len(mock_regimes)} zones")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Visual Forecast System Tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Chart Generation", test_chart_generation),
        ("Telegram Commands", test_telegram_commands),
        ("Full Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        logger.info("-" * 40)
        
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
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:20} : {status}")
    
    logger.info("-" * 60)
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Visual Forecast System is ready!")
        logger.info("\n🔥 Your trading bot now has LEGENDARY visual intelligence! 🔥")
        logger.info("\nTo use the new features:")
        logger.info("  • Send '/forecast BTCUSDT 1h' for visual market analysis")
        logger.info("  • Send '/plan ETHUSDT 4h' for comprehensive trading plans")
        logger.info("  • Charts will be automatically generated and sent!")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
