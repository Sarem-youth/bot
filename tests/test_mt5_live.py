import pytest
from datetime import datetime, timedelta
from mt5_adapter import MT5Config, MT5Environment, MT5Adapter

class TestMT5LiveTrading:
    @pytest.fixture
    def live_config(self):
        return MT5Config(
            environment="live",
            risk_limits={
                'max_daily_loss': 1000,
                'max_position_size': 1.0,
                'max_slippage': 3,
                'max_spread': 5
            }
        )
        
    def test_risk_limits(self, live_config):
        env = MT5Environment(live_config)
        
        # Test daily loss limit
        env.daily_stats['total_loss'] = 1100
        assert not env.check_trading_allowed()
        
        # Test error limit
        env.daily_stats['total_loss'] = 0
        env.daily_stats['errors'] = 6
        assert not env.check_trading_allowed()
        
    def test_order_validation(self, live_config):
        env = MT5Environment(live_config)
        
        # Test position size limit
        order = {
            'symbol': 'XAUUSD',
            'volume': 2.0,
            'type': 'BUY',
            'price': 1900.0,
            'sl': 1890.0,
            'tp': 1910.0
        }
        assert not env.validate_order(order)
        
        # Test valid order
        order['volume'] = 0.5
        with patch('mt5_adapter.mt5') as mock_mt5:
            mock_mt5.symbol_info_tick.return_value = Mock(ask=1900.0, bid=1899.5)
            mock_mt5.symbol_info.return_value = Mock(point=0.01)
            assert env.validate_order(order)
            
    @pytest.mark.live
    def test_live_connection(self, live_config):
        adapter = MT5Adapter(live_config)
        
        # Verify connection to live server
        assert adapter.connect()
        assert adapter.environment.config.environment == "live"
        
        # Test market data access
        tick = adapter.get_current_tick("XAUUSD")
        assert tick is not None
        assert tick['bid'] > 0
        assert tick['ask'] > 0
        
    @pytest.mark.live
    def test_weekend_trading_disabled(self, live_config):
        env = MT5Environment(live_config)
        
        # Simulate weekend
        with patch('datetime.datetime') as mock_date:
            mock_date.now.return_value = datetime(2024, 1, 13, 12, 0)  # Saturday
            assert not env.check_trading_allowed()
            
def run_live_test_sequence():
    """Run complete test sequence in demo before live"""
    config = MT5Config(environment="demo")
    adapter = MT5Adapter(config)
    
    # Test basic functionality
    assert adapter.connect()
    
    # Test market data
    symbol = "XAUUSD"
    tick = adapter.get_current_tick(symbol)
    assert tick is not None
    
    # Test order placement with small volume
    result = adapter.open_position(
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.01,
        price=tick['ask'],
        sl=tick['ask'] - 10,
        tp=tick['ask'] + 10
    )
    assert result
    
    # Monitor position
    time.sleep(60)
    positions = adapter.get_open_positions()
    
    # Close test position
    for pos in positions:
        adapter.close_position(pos['ticket'])
        
    # Verify cleanup
    assert len(adapter.get_open_positions()) == 0
    
if __name__ == "__main__":
    # Run test sequence
    run_live_test_sequence()
