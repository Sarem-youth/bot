# Initialize components
backtester = BacktestEngine(initial_balance=10000)
validator = StrategyValidator(backtester, historical_data)

# Load historical data from multiple sources
data_processor = DataProcessor()
historical_data = data_processor.load_historical_data(
    timeframe='1h',
    start_date='2019-01-01',
    end_date='2023-12-31',
    sources=['local', 'api', 'broker']
)

# Validate and prepare data
historical_data = data_processor.validate_gold_data(historical_data)

# Optimize strategy parameters
param_space = {
    'atr_period': (10, 30),
    'volatility_threshold': (0.001, 0.01),
    'risk_per_trade': (0.01, 0.05)
}
best_params, score = validator.optimize_parameters(YourStrategy, param_space)

# Validate strategy
robustness = validator.validate_robustness(strategy)
monte_carlo = validator.monte_carlo_simulation(strategy)
walk_forward = validator.walk_forward_analysis(strategy)

# Run multi-period tests
results = backtester.run_multiperiod_test(strategy, historical_data, [
    'bull', 'bear', 'sideways'
])

# Analyze period-specific performance
for period, result in results.items():
    print(f"\nPerformance during {period} market:")
    print(f"Win rate: {result['metrics']['win_rate']:.2%}")
    print(f"Profit factor: {result['metrics']['profit_factor']:.2f}")
    print(f"Sharpe ratio: {result['metrics']['sharpe_ratio']:.2f}")

# Define stress test scenarios
stress_scenarios = [
    {
        'name': 'Volatility Shock',
        'type': 'volatility_shock',
        'magnitude': 0.05  # 5% increase in price ranges
    },
    {
        'name': 'Market Gap',
        'type': 'gap',
        'magnitude': -0.03  # 3% downward gap
    },
    {
        'name': 'Liquidity Crisis',
        'type': 'liquidity_crisis',
        'magnitude': 0.3  # 70% reduction in volume
    }
]

# Evaluate strategy under different market conditions
condition_results = backtester.evaluate_market_conditions(strategy, historical_data)

# Run stress tests
stress_test_results = backtester.stress_test_strategy(
    strategy, 
    historical_data,
    stress_scenarios
)

# Print condition-specific performance
print("\nMarket Condition Performance:")
for condition, metrics in condition_results.items():
    print(f"\n{condition.upper()}:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

# Print stress test results
print("\nStress Test Results:")
for scenario, results in stress_test_results.items():
    print(f"\n{scenario}:")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Recovery Time: {results['recovery_time']} bars")

# Generate performance report
backtester.plot_results('strategy_performance.png')

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester import BacktestEngine, MT5BacktestEngine
from mt5_adapter import MT5Config

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    data = pd.DataFrame({
        'time': dates,
        'open': np.random.normal(1800, 20, len(dates)),
        'high': np.random.normal(1810, 20, len(dates)),
        'low': np.random.normal(1790, 20, len(dates)),
        'close': np.random.normal(1800, 20, len(dates)),
        'volume': np.random.randint(100, 1000, len(dates))
    })
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    return data

@pytest.fixture
def mock_strategy():
    class Strategy:
        def __init__(self):
            self.symbol = "XAUUSD"
            self.timeframe = 60
            self.max_positions = 1
            
        def analyze_market(self, data):
            return {
                'type': 'BUY',
                'size': 1.0,
                'stop_loss': data['close'].iloc[-1] * 0.99,
                'take_profit': data['close'].iloc[-1] * 1.01
            }
    return Strategy()

class TestBacktestEngine:
    def test_initialization(self):
        engine = BacktestEngine(initial_balance=10000)
        assert engine.initial_balance == 10000
        assert len(engine.trades_history) == 0
        
    def test_run_backtest(self, sample_data, mock_strategy):
        engine = BacktestEngine()
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        results = engine.run_backtest(
            mock_strategy,
            sample_data,
            start_date,
            end_date
        )
        
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        assert len(engine.trades_history) > 0
        
    def test_stress_test(self, sample_data, mock_strategy):
        engine = BacktestEngine()
        scenarios = [{
            'name': 'High Volatility',
            'type': 'volatility_shock',
            'magnitude': 0.05
        }]
        
        results = engine.stress_test_strategy(
            mock_strategy,
            sample_data,
            scenarios
        )
        
        assert 'High Volatility' in results
        assert 'metrics' in results['High Volatility']
        
class TestMT5BacktestEngine:
    def test_mt5_mode(self, sample_data, mock_strategy):
        engine = MT5BacktestEngine()
        engine.set_mt5_mode(True)
        
        with patch('backtester.MT5TesterAdapter') as mock_adapter:
            mock_adapter.return_value.init_tester.return_value = True
            mock_adapter.return_value.run_test.return_value = {
                'trades': [],
                'results': Mock(
                    trades=10,
                    profit_trades=6,
                    profit_factor=1.5,
                    sharp_ratio=1.2,
                    max_drawdown=0.1,
                    profit=1000,
                    average_trade_length=120
                )
            }
            
            results = engine.run_backtest(
                mock_strategy,
                sample_data,
                datetime(2023, 1, 1),
                datetime(2023, 12, 31)
            )
            
            assert results['metrics']['total_trades'] == 10
            assert results['metrics']['win_rate'] == 0.6
            
    def test_optimization(self, mock_strategy):
        engine = MT5BacktestEngine()
        engine.set_mt5_mode(True)
        
        optimization_params = {
            'ma_period': (10, 1, 20),
            'stop_loss': (20, 5, 50)
        }
        
        engine.set_optimization_params(optimization_params)
        assert engine.optimization_params == optimization_params