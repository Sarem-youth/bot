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
