# Walk-Forward Optimization

## Purpose
Walk-forward optimization helps prevent overfitting by validating strategy performance on unseen data.

## Process
1. **Split Data**: Use `TimeSeriesSplit` to divide data into multiple training/testing periods.
2. **Train and Test**: Train the strategy on each training window and test on the subsequent test window.
3. **Evaluate**: Aggregate performance metrics across all test periods to evaluate consistency.

## Implementation

### Step-by-Step Example

1. **Initialize Components**:
```python
from strategy_validator import StrategyValidator
from backtester import BacktestEngine 

# Initialize components
backtester = BacktestEngine(initial_balance=10000)
validator = StrategyValidator(backtester, historical_data)
```

2. **Run Walk-Forward Analysis**:
```python
# Run walk-forward analysis with 90-day windows
results = validator.walk_forward_analysis(strategy, window_size=90)
```

3. **Combine with Parameter Optimization**:
```python
# Define parameter space to optimize
param_space = {
    'atr_period': (10, 30),
    'volatility_threshold': (0.001, 0.01), 
    'risk_per_trade': (0.01, 0.05)
}

# Optimize parameters using Bayesian optimization
best_params, score = validator.optimize_parameters(
    strategy_class=YourStrategy,
    param_space=param_space,
    n_trials=100,
    method='bayesian'
)
```

4. **Evaluate Results**:
```python
print(results)
{
    'splits': [...],  # Results for each period
    'avg_return': 0.15,  # Average return across periods
    'std_return': 0.05,  # Standard deviation of returns
    'consistency': 0.8   # % of profitable periods
}
```

## Benefits
- Ensures strategy performs well on unseen data
- Validates robustness across different market conditions
- Reduces risk of overfitting to a single period

## Advanced Optimization Techniques

### 1. Grid Search
Systematic search through parameter space with defined intervals.

```python
param_space = {
    'atr_period': (10, 30),
    'volatility_threshold': (0.001, 0.01)
}

best_params, score = validator.optimize_parameters(
    strategy_class=YourStrategy,
    param_space=param_space,
    method='grid'
)
```

### 2. Genetic Algorithm
Evolutionary approach for complex parameter spaces.

```python
best_params, score = validator.optimize_parameters(
    strategy_class=YourStrategy,
    param_space=param_space,
    method='genetic',
    n_trials=50  # generations
)
```

### 3. Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| Bayesian | Efficient, learns from previous trials | May miss global optimum |
| Grid Search | Systematic, thorough | Computationally expensive |
| Genetic | Good for complex spaces | Requires tuning of GA parameters |

## Best Practices
1. Start with grid search for small parameter spaces
2. Use Bayesian for larger parameter spaces
3. Consider genetic algorithms for complex strategies
4. Validate results across different time periods
5. Monitor for overfitting
