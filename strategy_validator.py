from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import optuna
from deap import base, creator, tools, algorithms  # Add this import

class StrategyValidator:
    def __init__(self, backtester, data):
        self.backtester = backtester
        self.data = data
        self.n_splits = 5
        self.confidence_level = 0.95
        
    def monte_carlo_simulation(self, 
                             strategy: object, 
                             n_simulations: int = 1000,
                             confidence_level: float = 0.95) -> Dict:
        """Run Monte Carlo simulation of strategy performance"""
        base_results = self.backtester.run_backtest(strategy, self.data)
        returns = np.array([r['pnl'] for r in base_results['trades']])
        
        simulated_metrics = {
            'final_equity': [],
            'max_drawdown': [],
            'sharpe_ratio': [],
            'win_rate': []
        }
        
        for _ in range(n_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=len(returns))
            equity_curve = self.initial_balance + np.cumsum(sampled_returns)
            
            # Calculate various metrics for this simulation
            max_dd = self._calculate_max_drawdown(equity_curve)
            sharpe = self._calculate_sharpe_ratio(sampled_returns)
            win_rate = len([r for r in sampled_returns if r > 0]) / len(sampled_returns)
            
            simulated_metrics['final_equity'].append(equity_curve[-1])
            simulated_metrics['max_drawdown'].append(max_dd)
            simulated_metrics['sharpe_ratio'].append(sharpe)
            simulated_metrics['win_rate'].append(win_rate)
        
        # Calculate confidence intervals for all metrics
        ci_lower = (1 - confidence_level) / 2
        ci_upper = 1 - ci_lower
        
        results = {}
        for metric, values in simulated_metrics.items():
            results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, ci_lower * 100),
                'ci_upper': np.percentile(values, ci_upper * 100),
                'worst': np.min(values),
                'best': np.max(values)
            }
            
        return results

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
        
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
                          n_trials: int = 100,
                          method: str = 'bayesian') -> Tuple[Dict, float]:
        """Optimize strategy parameters using various methods"""
        if method == 'bayesian':
            return self._bayesian_optimization(strategy_class, param_space, n_trials)
        elif method == 'grid':
            return self._grid_search(strategy_class, param_space)
        elif method == 'genetic':
            return self._genetic_optimization(strategy_class, param_space, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _bayesian_optimization(self, strategy_class, param_space: Dict, n_trials: int) -> Tuple[Dict, float]:
        """Bayesian optimization implementation"""
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

    def _grid_search(self, strategy_class, param_space: Dict) -> Tuple[Dict, float]:
        """Perform grid search optimization"""
        param_combinations = []
        for param, (start, end) in param_space.items():
            if isinstance(start, float):
                values = np.linspace(start, end, 10)
            else:
                values = range(start, end + 1)
            param_combinations.append((param, values))

        best_score = float('-inf')
        best_params = None

        for params in self._generate_grid_combinations(param_combinations):
            strategy = strategy_class(**params)
            results = self.backtester.run_backtest(strategy, self.data)
            score = self._calculate_optimization_score(results)
            
            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def _genetic_optimization(self, strategy_class, param_space: Dict, n_generations: int) -> Tuple[Dict, float]:
        """Perform genetic algorithm optimization"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        param_names = list(param_space.keys())

        def create_individual():
            return [np.random.uniform(param_space[p][0], param_space[p][1]) for p in param_names]

        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            params = dict(zip(param_names, individual))
            strategy = strategy_class(**params)
            results = self.backtester.run_backtest(strategy, self.data)
            return (self._calculate_optimization_score(results),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=50)
        result, logbook = algorithms.eaSimple(population, toolbox, 
                                            cxpb=0.7, mutpb=0.3, 
                                            ngen=n_generations,
                                            verbose=False)

        best_ind = tools.selBest(result, k=1)[0]
        best_params = dict(zip(param_names, best_ind))
        return best_params, best_ind.fitness.values[0]

    def _generate_grid_combinations(self, param_combinations: List) -> Iterator[Dict]:
        """Generate all possible parameter combinations for grid search"""
        if not param_combinations:
            yield {}
        else:
            param, values = param_combinations[0]
            for value in values:
                for next_comb in self._generate_grid_combinations(param_combinations[1:]):
                    yield {param: value, **next_comb}

    def _calculate_optimization_score(self, results: Dict) -> float:
        """Calculate optimization score using multiple metrics"""
        sharpe = results['metrics']['sharpe_ratio']
        drawdown = abs(results['metrics']['max_drawdown'])
        consistency = results['metrics']['win_rate']
        return sharpe * 0.4 + (1-drawdown) * 0.3 + consistency * 0.3
        
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
