# Initialize components
backtester = BacktestEngine(initial_balance=10000)
validator = StrategyValidator(backtester, historical_data)

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

# Generate performance report
backtester.plot_results('strategy_performance.png')