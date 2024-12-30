import pytest
from datetime import datetime, timedelta
import time
from mt5_adapter import MT5Config, MT5Adapter
from backtester import MT5BacktestEngine

class DemoTester:
    def __init__(self, symbol: str = "XAUUSD"):
        self.config = MT5Config(
            environment="demo",
            risk_limits={
                'max_daily_loss': 100,  # Smaller limit for demo
                'max_position_size': 0.1,  # Micro lots for safety
                'max_slippage': 3,
                'max_spread': 5
            }
        )
        self.adapter = MT5Adapter(self.config)
        self.symbol = symbol
        self.test_results = []
        
    def run_demo_suite(self):
        """Run comprehensive demo testing suite"""
        tests = [
            self.test_market_data,
            self.test_order_execution,
            self.test_position_modification,
            self.test_error_handling,
            self.test_risk_limits
        ]
        
        for test in tests:
            try:
                result = test()
                self.test_results.append({
                    'test': test.__name__,
                    'result': result,
                    'status': 'passed' if result else 'failed'
                })
            except Exception as e:
                self.test_results.append({
                    'test': test.__name__,
                    'result': False,
                    'status': 'error',
                    'message': str(e)
                })
                
        return self.get_test_report()
        
    def test_market_data(self) -> bool:
        """Test market data retrieval"""
        tick = self.adapter.get_current_tick(self.symbol)
        if not tick:
            return False
            
        # Verify tick data structure
        required_fields = ['bid', 'ask', 'volume', 'time']
        return all(field in tick for field in required_fields)
        
    def test_order_execution(self) -> bool:
        """Test order execution with minimal volume"""
        tick = self.adapter.get_current_tick(self.symbol)
        if not tick:
            return False
            
        # Place test order
        result = self.adapter.open_position(
            symbol=self.symbol,
            order_type="BUY",
            volume=0.01,  # Minimum volume
            price=tick['ask'],
            sl=tick['ask'] - 10,
            tp=tick['ask'] + 10
        )
        
        if not result:
            return False
            
        # Verify position opened
        positions = self.adapter.get_open_positions()
        if not positions:
            return False
            
        # Close test position
        return all(self.adapter.close_position(pos['ticket']) for pos in positions)
        
    def test_position_modification(self) -> bool:
        """Test position modification"""
        # Open test position
        tick = self.adapter.get_current_tick(self.symbol)
        if not tick:
            return False
            
        result = self.adapter.open_position(
            symbol=self.symbol,
            order_type="BUY",
            volume=0.01,
            price=tick['ask'],
            sl=tick['ask'] - 10,
            tp=tick['ask'] + 10
        )
        
        if not result:
            return False
            
        positions = self.adapter.get_open_positions()
        if not positions:
            return False
            
        # Modify stop loss and take profit
        pos = positions[0]
        modify_result = self.adapter.modify_position(
            ticket=pos['ticket'],
            sl=tick['ask'] - 15,  # New stop loss
            tp=tick['ask'] + 15   # New take profit
        )
        
        # Clean up
        self.adapter.close_position(pos['ticket'])
        return modify_result
        
    def test_error_handling(self) -> bool:
        """Test error handling and recovery"""
        # Test invalid order parameters
        result = self.adapter.open_position(
            symbol=self.symbol,
            order_type="BUY",
            volume=100.0,  # Excessive volume
            price=0,  # Invalid price
            sl=0,
            tp=0
        )
        
        # Should return False due to risk limits
        if result:
            return False
            
        # Verify error was logged
        return self.adapter.environment.daily_stats['errors'] > 0
        
    def test_risk_limits(self) -> bool:
        """Test risk management limits"""
        # Test position size limit
        tick = self.adapter.get_current_tick(self.symbol)
        if not tick:
            return False
            
        # Try to exceed position size limit
        result = self.adapter.open_position(
            symbol=self.symbol,
            order_type="BUY",
            volume=self.config.risk_limits['max_position_size'] + 0.1,
            price=tick['ask'],
            sl=tick['ask'] - 10,
            tp=tick['ask'] + 10
        )
        
        # Should be rejected
        return not result
        
    def get_test_report(self) -> dict:
        """Generate test report"""
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['status'] == 'passed'])
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_results': self.test_results
        }

@pytest.mark.demo
class TestMT5Demo:
    @pytest.fixture
    def demo_tester(self):
        return DemoTester()
        
    def test_demo_suite(self, demo_tester):
        """Run full demo test suite"""
        report = demo_tester.run_demo_suite()
        
        # Assert test completion
        assert report['total_tests'] > 0
        
        # Check success rate
        assert report['success_rate'] >= 0.8, "Demo tests success rate below 80%"
        
        # Print detailed results
        for result in report['detailed_results']:
            print(f"\nTest: {result['test']}")
            print(f"Status: {result['status']}")
            if result.get('message'):
                print(f"Error: {result['message']}")

if __name__ == "__main__":
    # Run demo tests
    tester = DemoTester()
    report = tester.run_demo_suite()
    
    # Print results
    print("\nDemo Testing Report")
    print("==================")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed Tests: {report['passed_tests']}")
    print(f"Success Rate: {report['success_rate']:.2%}")
    
    # Print details of failed tests
    failed_tests = [t for t in report['detailed_results'] if t['status'] != 'passed']
    if failed_tests:
        print("\nFailed Tests:")
        for test in failed_tests:
            print(f"\n{test['test']}:")
            print(f"Status: {test['status']}")
            if 'message' in test:
                print(f"Error: {test['message']}")
