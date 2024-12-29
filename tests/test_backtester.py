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