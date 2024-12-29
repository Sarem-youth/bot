import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.trades_history = []
        self.equity_curve = []
        self.performance_metrics = {}
        self.max_drawdown = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.sharpe_ratio = 0
        self.trading_costs = {
            'spread': 0.03,  # Average gold spread in USD
            'commission': 0.0001,  # 0.01% commission
            'swap_long': -0.0002,  # Overnight swap rates
            'swap_short': -0.0001
        }
        self.tick_size = 0.01  # Minimum price movement for gold
        self.contract_size = 100  # Standard gold contract size (oz)
        self.market_conditions = {
            'volatility': {
                'high': [],
                'low': []
            },
            'trend': {
                'up': [],
                'down': [],
                'sideways': []
            },
            'events': {
                'crisis': [
                    ('2020-02-20', '2020-04-30', 'COVID-19 Crisis'),
                    ('2022-02-24', '2022-05-31', 'Russia-Ukraine Conflict')
                ],
                'rate_changes': [
                    ('2022-03-16', '2022-12-31', 'Fed Rate Hikes')
                ]
            }
        }
        
    def run_backtest(self, 
                     strategy: object, 
                     historical_data: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime,
                     parallel: bool = True) -> Dict:
        """Run backtest for a given strategy"""
        
        # Prepare data
        data = historical_data[(historical_data['time'] >= start_date) & 
                             (historical_data['time'] <= end_date)].copy()
        
        current_balance = self.initial_balance
        open_positions = []
        equity_points = []
        
        def process_bar(index, row):
            """Process individual price bar"""
            signals = strategy.analyze_market(data[:index+1])
            
            # Process open positions
            for pos in open_positions[:]:
                if self._check_position_exit(pos, row):
                    current_balance += self._calculate_pnl(pos, row['close'])
                    open_positions.remove(pos)
                    self.trades_history.append(pos)
                    
            # Open new positions if signals exist
            if signals and len(open_positions) < strategy.max_positions:
                new_position = self._create_position(signals, row)
                if new_position:
                    open_positions.append(new_position)
                    
            equity_points.append({
                'time': row['time'],
                'equity': current_balance + sum(self._calculate_floating_pnl(pos, row['close']) 
                                             for pos in open_positions)
            })
            
        # Run backtest - parallel or sequential
        if parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(lambda x: process_bar(*x), enumerate(data.itertuples()))
        else:
            for index, row in enumerate(data.itertuples()):
                process_bar(index, row)
                
        # Calculate performance metrics
        self.equity_curve = pd.DataFrame(equity_points)
        self._calculate_metrics()
        
        return self.get_results()
        
    def _calculate_metrics(self):
        """Calculate trading performance metrics"""
        if not self.trades_history:
            return
            
        trades_df = pd.DataFrame(self.trades_history)
        
        # Basic metrics
        self.win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        self.profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                               trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                               
        # Risk metrics
        returns = self.equity_curve['equity'].pct_change().dropna()
        self.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Maximum drawdown
        peak = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - peak) / peak
        self.max_drawdown = drawdown.min()
        
        # Store metrics
        self.performance_metrics = {
            'total_trades': len(trades_df),
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': (self.equity_curve['equity'].iloc[-1] / self.initial_balance - 1),
            'avg_trade_duration': trades_df['duration'].mean(),
            'avg_profit_per_trade': trades_df['pnl'].mean()
        }
        
    def plot_results(self, save_path: Optional[str] = None):
        """Generate performance visualization"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Equity curve
        self.equity_curve.set_index('time')['equity'].plot(ax=ax1, title='Equity Curve')
        ax1.set_ylabel('Account Value')
        
        # Drawdown
        peak = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - peak) / peak
        drawdown.plot(ax=ax2, title='Drawdown', color='red')
        ax2.set_ylabel('Drawdown %')
        
        # Trade distribution
        trades_df = pd.DataFrame(self.trades_history)
        sns.histplot(data=trades_df, x='pnl', ax=ax3, bins=50)
        ax3.set_title('Trade PnL Distribution')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def get_results(self) -> Dict:
        """Get backtest results summary"""
        return {
            'metrics': self.performance_metrics,
            'trades': self.trades_history,
            'equity_curve': self.equity_curve.to_dict('records')
        }
        
    def _create_position(self, signals: Dict, bar: pd.Series) -> Optional[Dict]:
        """Create new trading position"""
        return {
            'entry_time': bar['time'],
            'entry_price': bar['close'],
            'type': signals['type'],
            'size': signals['size'],
            'stop_loss': signals['stop_loss'],
            'take_profit': signals['take_profit']
        }
        
    def _check_position_exit(self, position: Dict, bar: pd.Series) -> bool:
        """Check if position should be closed"""
        if position['type'] == 'BUY':
            if bar['low'] <= position['stop_loss'] or bar['high'] >= position['take_profit']:
                return True
        else:  # SELL
            if bar['high'] >= position['stop_loss'] or bar['low'] <= position['take_profit']:
                return True
        return False
        
    def _calculate_pnl(self, position: Dict, close_price: float) -> float:
        """Calculate position profit/loss"""
        multiplier = 1 if position['type'] == 'BUY' else -1
        return (close_price - position['entry_price']) * position['size'] * multiplier
        
    def _calculate_floating_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized profit/loss"""
        multiplier = 1 if position['type'] == 'BUY' else -1
        return (current_price - position['entry_price']) * position['size'] * multiplier

    def _calculate_trading_costs(self, position: Dict, current_price: float) -> float:
        """Calculate total trading costs"""
        spread_cost = self.trading_costs['spread'] * position['size']
        commission = current_price * position['size'] * self.trading_costs['commission']
        
        # Calculate swap if position held overnight
        days_held = (datetime.now() - position['entry_time']).days
        swap_rate = self.trading_costs['swap_long'] if position['type'] == 'BUY' else self.trading_costs['swap_short']
        swap_cost = days_held * swap_rate * position['size'] * current_price
        
        return spread_cost + commission + swap_cost

    def run_multiperiod_test(self, 
                           strategy: object,
                           data: pd.DataFrame,
                           periods: List[str]) -> Dict:
        """Run backtest across different market periods"""
        results = {}
        
        # Define market periods (e.g., bull, bear, sideways)
        market_periods = {
            'bull': ('2019-01-01', '2020-08-01'),
            'bear': ('2020-08-01', '2021-03-01'),
            'sideways': ('2021-03-01', '2021-12-31')
        }
        
        for period_name, (start, end) in market_periods.items():
            period_data = data[(data['time'] >= start) & (data['time'] <= end)]
            results[period_name] = self.run_backtest(
                strategy,
                period_data,
                datetime.strptime(start, '%Y-%m-%d'),
                datetime.strptime(end, '%Y-%m-%d')
            )
            
        return results

    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Identify different market conditions in the data"""
        conditions = {}
        
        # Volatility analysis
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        vol_threshold = volatility.mean() + volatility.std()
        
        conditions['high_volatility'] = data[volatility > vol_threshold]
        conditions['low_volatility'] = data[volatility <= vol_threshold]
        
        # Trend analysis
        sma_short = data['close'].rolling(window=20).mean()
        sma_long = data['close'].rolling(window=50).mean()
        adx = self._calculate_adx(data, period=14)
        
        conditions['trending_up'] = data[(sma_short > sma_long) & (adx > 25)]
        conditions['trending_down'] = data[(sma_short < sma_long) & (adx > 25)]
        conditions['ranging'] = data[adx <= 25]
        
        return conditions

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        tr_smoothed = tr.rolling(period).sum()
        plus_dm_smoothed = plus_dm.clip(lower=0).rolling(period).sum()
        minus_dm_smoothed = minus_dm.clip(upper=0).abs().rolling(period).sum()
        
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        return dx.rolling(period).mean()

    def evaluate_market_conditions(self, 
                                strategy: object,
                                data: pd.DataFrame) -> Dict:
        """Evaluate strategy performance under different market conditions"""
        conditions = self.analyze_market_conditions(data)
        results = {}
        
        for condition, condition_data in conditions.items():
            if len(condition_data) > 0:
                results[condition] = self.run_backtest(
                    strategy,
                    condition_data,
                    condition_data.index[0],
                    condition_data.index[-1]
                )
                
        return self._analyze_condition_performance(results)

    def _analyze_condition_performance(self, results: Dict) -> Dict:
        """Analyze performance across different conditions"""
        analysis = {}
        
        for condition, result in results.items():
            metrics = result['metrics']
            analysis[condition] = {
                'sharpe_ratio': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades']
            }
            
        # Calculate condition-specific metrics
        analysis['volatility_stability'] = self._calculate_stability_score(
            analysis.get('high_volatility', {}),
            analysis.get('low_volatility', {})
        )
        
        analysis['trend_adaptation'] = self._calculate_adaptation_score(
            analysis.get('trending_up', {}),
            analysis.get('trending_down', {}),
            analysis.get('ranging', {})
        )
        
        return analysis

    def _calculate_stability_score(self, high_vol: Dict, low_vol: Dict) -> float:
        """Calculate strategy stability across volatility regimes"""
        if not high_vol or not low_vol:
            return 0.0
            
        return 1 - abs(
            high_vol.get('sharpe_ratio', 0) - 
            low_vol.get('sharpe_ratio', 0)
        ) / max(
            abs(high_vol.get('sharpe_ratio', 1)), 
            abs(low_vol.get('sharpe_ratio', 1))
        )

    def _calculate_adaptation_score(self, up: Dict, down: Dict, ranging: Dict) -> float:
        """Calculate strategy adaptation to different trends"""
        scores = []
        
        for condition in [up, down, ranging]:
            if condition:
                scores.append(
                    condition.get('win_rate', 0) * 
                    condition.get('profit_factor', 0)
                )
                
        return np.mean(scores) if scores else 0.0

    def stress_test_strategy(self, 
                           strategy: object,
                           data: pd.DataFrame,
                           scenarios: List[Dict]) -> Dict:
        """Stress test strategy under different scenarios"""
        results = {}
        
        for scenario in scenarios:
            # Apply scenario modifications to data
            scenario_data = self._apply_scenario(data.copy(), scenario)
            
            # Run backtest with modified data
            scenario_results = self.run_backtest(
                strategy,
                scenario_data,
                scenario_data.index[0],
                scenario_data.index[-1]
            )
            
            results[scenario['name']] = {
                'metrics': scenario_results['metrics'],
                'max_adverse_excursion': self._calculate_mae(scenario_results['trades']),
                'recovery_time': self._calculate_recovery_time(scenario_results['equity_curve'])
            }
            
        return results

    def _apply_scenario(self, data: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """Apply stress test scenario to data"""
        if scenario['type'] == 'volatility_shock':
            data['high'] += data['high'] * scenario['magnitude']
            data['low'] -= data['low'] * scenario['magnitude']
        elif scenario['type'] == 'gap':
            idx = data.index[len(data)//2]  # Apply gap in middle of data
            data.loc[idx:, ['open', 'high', 'low', 'close']] *= (1 + scenario['magnitude'])
        elif scenario['type'] == 'liquidity_crisis':
            data['volume'] *= scenario['magnitude']
            
        return data

class MT5BacktestEngine(BacktestEngine):
    def __init__(self, initial_balance: float = 10000):
        super().__init__(initial_balance)
        self.mt5_mode = False
        self.optimization_params = {}
        
    def set_mt5_mode(self, enabled: bool = True):
        """Enable/disable MT5 compatibility mode"""
        self.mt5_mode = enabled
        
    def set_optimization_params(self, params: Dict):
        """Set optimization parameters for MT5"""
        self.optimization_params = params
        
    def run_backtest(self, 
                     strategy: object, 
                     historical_data: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime,
                     parallel: bool = True) -> Dict:
        """Enhanced backtest with MT5 compatibility"""
        if self.mt5_mode:
            return self._run_mt5_backtest(strategy, start_date, end_date)
        return super().run_backtest(strategy, historical_data, start_date, end_date, parallel)
        
    def _run_mt5_backtest(self, strategy: object, start_date: datetime, end_date: datetime) -> Dict:
        """Run backtest using MT5's strategy tester"""
        # Convert strategy parameters for MT5
        mt5_params = self._convert_strategy_params(strategy)
        
        # Initialize MT5 tester
        if not hasattr(self, 'mt5_adapter'):
            self.mt5_adapter = MT5TesterAdapter(MT5Config())
            
        # Configure test
        self.mt5_adapter.init_tester(
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            testing=True,
            optimization=bool(self.optimization_params)
        )
        
        if self.optimization_params:
            self.mt5_adapter.set_optimization_inputs(self.optimization_params)
            
        # Run test
        results = asyncio.run(self.mt5_adapter.run_test(start_date, end_date))
        
        # Convert results to standard format
        return self._convert_mt5_results(results)
        
    def _convert_strategy_params(self, strategy: object) -> Dict:
        """Convert strategy parameters to MT5 format"""
        params = {}
        for param_name, param_value in strategy.__dict__.items():
            if not param_name.startswith('_'):
                params[param_name] = param_value
        return params
        
    def _convert_mt5_results(self, mt5_results: Dict) -> Dict:
        """Convert MT5 results to standard format"""
        if not mt5_results:
            return {}
            
        trades = []
        for trade in mt5_results['trades']:
            trades.append({
                'entry_time': trade.time,
                'entry_price': trade.price,
                'type': 'BUY' if trade.type == 0 else 'SELL',
                'size': trade.volume,
                'pnl': trade.profit,
                'duration': trade.time_exit - trade.time if trade.time_exit else 0
            })
            
        results = mt5_results['results']
        self.performance_metrics = {
            'total_trades': results.trades,
            'win_rate': results.profit_trades / results.trades if results.trades > 0 else 0,
            'profit_factor': results.profit_factor,
            'sharpe_ratio': results.sharp_ratio,
            'max_drawdown': results.max_drawdown,
            'total_return': results.profit,
            'avg_trade_duration': results.average_trade_length,
            'avg_profit_per_trade': results.profit / results.trades if results.trades > 0 else 0
        }
        
        return {
            'metrics': self.performance_metrics,
            'trades': trades,
            'optimization_results': mt5_results.get('optimization')
        }
