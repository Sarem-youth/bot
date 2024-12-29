from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import optuna

class StrategyValidator:
    def __init__(self, backtester, data):
        self.backtester = backtester
        self.data = data
        self.n_splits = 5
        self.confidence_level = 0.95
        
    def monte_carlo_simulation(self, 
                             strategy: object, 
                             n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation of strategy performance"""
        base_results = self.backtester.run_backtest(strategy, self.data)
        returns = np.array([r['pnl'] for r in base_results['trades']])
        
        simulated_equity_curves = []
        for _ in range(n_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=len(returns))
            equity_curve = self.initial_balance + np.cumsum(sampled_returns)
            simulated_equity_curves.append(equity_curve)
            
        # Calculate confidence intervals
        percentiles = np.percentile(simulated_equity_curves, 
                                  [5, 25, 50, 75, 95], 
                                  axis=0)
        
        return {
            'percentiles': percentiles,
            'worst_case': min(curve[-1] for curve in simulated_equity_curves),
            'best_case': max(curve[-1] for curve in simulated_equity_curves),
            'expected_value': np.mean([curve[-1] for curve in simulated_equity_curves])
        }
        
    def walk_forward_analysis(self, 
                            strategy: object, 
                            window_size: int = 90) -> Dict:
        """Perform walk-forward analysis"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = []
        
        for train_idx, test_idx in tscv.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Train strategy on training data
            strategy.train(train_data)
            
            # Test on out-of-sample data
            test_results = self.backtester.run_backtest(strategy, test_data)
            results.append(test_results)
            
        return {
            'splits': results,
            'avg_return': np.mean([r['metrics']['total_return'] for r in results]),
            'std_return': np.std([r['metrics']['total_return'] for r in results]),
            'consistency': len([r for r in results if r['metrics']['total_return'] > 0]) / self.n_splits
        }
        
    def optimize_parameters(self, 
                          strategy_class, 
                          param_space: Dict,
                          n_trials: int = 100) -> Tuple[Dict, float]:
        """Optimize strategy parameters using Optuna"""
        def objective(trial):
            params = {
                k: trial.suggest_float(k, v[0], v[1]) 
                if isinstance(v[0], float) 
                else trial.suggest_int(k, v[0], v[1])
                for k, v in param_space.items()
            }
            
            strategy = strategy_class(**params)
            results = self.backtester.run_backtest(strategy, self.data)
            
            # Multi-objective optimization
            sharpe = results['metrics']['sharpe_ratio']
            drawdown = abs(results['metrics']['max_drawdown'])
            consistency = results['metrics']['win_rate']
            
            return -1 * (sharpe * 0.4 + (1-drawdown) * 0.3 + consistency * 0.3)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, -study.best_value
        
    def validate_robustness(self, strategy: object) -> Dict:
        """Validate strategy robustness"""
        # Test different market conditions
        market_conditions = self._identify_market_conditions()
        condition_results = {}
        
        for condition, data in market_conditions.items():
            results = self.backtester.run_backtest(strategy, data)
            condition_results[condition] = results['metrics']
            
        # Calculate strategy robustness score
        consistency_score = np.mean([
            r['win_rate'] > 0.5 and
            r['sharpe_ratio'] > 1.0 and
            r['max_drawdown'] > -0.2
            for r in condition_results.values()
        ])
        
        return {
            'condition_results': condition_results,
            'robustness_score': consistency_score,
            'market_adaptation': self._calculate_adaptation_score(condition_results)
        }
        
    def _identify_market_conditions(self) -> Dict:
        """Identify different market conditions in data"""
        conditions = {}
        
        # Trending markets
        volatility = self.data['close'].pct_change().std()
        conditions['high_volatility'] = self.data[self.data['close'].pct_change().abs() > volatility * 2]
        conditions['low_volatility'] = self.data[self.data['close'].pct_change().abs() < volatility * 0.5]
        
        # Range-bound markets
        sma = self.data['close'].rolling(20).mean()
        conditions['ranging'] = self.data[abs(self.data['close'] - sma) < volatility]
        
        return conditions
        
    def _calculate_adaptation_score(self, condition_results: Dict) -> float:
        """Calculate strategy's ability to adapt to different conditions"""
        baseline = np.mean([r['total_return'] for r in condition_results.values()])
        variations = np.std([r['total_return'] for r in condition_results.values()])
        
        return baseline / (1 + variations)  # Higher score = better adaptation