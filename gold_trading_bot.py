import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from flask import Flask, render_template, jsonify, request
import threading
import queue
from io import BytesIO
import base64
import logging
import os

# Initialize Flask
app = Flask(__name__, template_folder='./')
signal_queue = queue.Queue()
analysis_results = {}

def load_config():
    with open('config.json') as f:
        return json.load(f)

def initialize_mt5():
    config = load_config()
    
    # Initialize MT5
    if not mt5.initialize(
        path=config['terminal_path'],
        login=config['login'],
        password=config['password'],
        server=config['server']
    ):
        print(f"Initialize failed. Error code: {mt5.last_error()}")
        mt5.shutdown()
        return False
    return True

def print_account_info():
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return
    
    print("\nAccount Information:")
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Profit: {account_info.profit}")
    print(f"Margin: {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")

def get_gold_historical_data(timeframe=mt5.TIMEFRAME_H1, num_bars=1000):
    """
    Fetch historical XAUUSD price data
    :param timeframe: MT5 timeframe (default: 1 hour)
    :param num_bars: Number of candles to fetch (default: 1000)
    :return: pandas DataFrame with OHLCV data
    """
    # Ensure MT5 is initialized
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return None

    # Fetch the historical data
    rates = mt5.copy_rates_from_pos("XAUUSD", timeframe, 0, num_bars)
    
    if rates is None:
        print(f"Failed to get XAUUSD data: {mt5.last_error()}")
        return None

    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    return df

def calculate_ma_signals(df):
    """
    Calculate moving averages and generate trading signals
    :param df: DataFrame with OHLCV data
    :return: DataFrame with moving averages and signals
    """
    # Calculate moving averages
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
    # Initialize signals column
    df['ma_signal'] = 0
    
    # Generate signals
    # Buy signal (1) when MA50 crosses above MA200
    # Sell signal (-1) when MA50 crosses below MA200
    df['ma_signal'] = np.where(
        (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1)),
        1,
        np.where(
            (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1)),
            -1,
            0
        )
    )
    
    return df

def calculate_support_resistance(df, window=20):
    """
    Calculate support and resistance levels using local min/max
    :param df: DataFrame with OHLCV data
    :param window: Window size for local min/max (default: 20)
    :return: DataFrame with support and resistance levels
    """
    df['support'] = df['low'].rolling(window=window, center=True).min()
    df['resistance'] = df['high'].rolling(window=window, center=True).max()
    return df

def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI)
    :param df: DataFrame with OHLCV data
    :param period: Period for RSI calculation (default: 14)
    :return: DataFrame with RSI values
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def combine_signals(df):
    """
    Generate signals from different strategies independently
    :param df: DataFrame with technical indicators
    :return: DataFrame with signals from each strategy
    """
    # MA Crossover signals
    df['ma_signal'] = np.where(
        (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1)),
        1,
        np.where(
            (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1)),
            -1,
            0
        )
    )
    
    # Support/Resistance signals
    df['sr_signal'] = np.where(
        (df['close'] > df['support']),
        1,
        np.where(
            (df['close'] < df['resistance']),
            -1,
            0
        )
    )
    
    # RSI signals
    df['rsi_signal'] = np.where(
        (df['RSI'] < 30),
        1,
        np.where(
            (df['RSI'] > 70),
            -1,
            0
        )
    )
    
    return df

def print_signals(df):
    """
    Print the latest trading signals
    :param df: DataFrame with signals
    """
    # Get only rows with signals
    ma_signals = df[df['ma_signal'] != 0].tail()
    sr_signals = df[df['sr_signal'] != 0].tail()
    rsi_signals = df[df['rsi_signal'] != 0].tail()
    
    if len(ma_signals) > 0:
        print("\nMA Crossover Signals:")
        for idx, row in ma_signals.iterrows():
            signal_type = "BUY" if row['ma_signal'] == 1 else "SELL"
            print(f"{idx}: {signal_type} - Price: {row['close']:.2f}")
    else:
        print("\nNo recent MA Crossover signals found")
    
    if len(sr_signals) > 0:
        print("\nSupport/Resistance Signals:")
        for idx, row in sr_signals.iterrows():
            signal_type = "BUY" if row['sr_signal'] == 1 else "SELL"
            print(f"{idx}: {signal_type} - Price: {row['close']:.2f}")
    else:
        print("\nNo recent Support/Resistance signals found")
    
    if len(rsi_signals) > 0:
        print("\nRSI Signals:")
        for idx, row in rsi_signals.iterrows():
            signal_type = "BUY" if row['rsi_signal'] == 1 else "SELL"
            print(f"{idx}: {signal_type} - Price: {row['close']:.2f}")
    else:
        print("\nNo recent RSI signals found")

def calculate_position_size(account_info, risk_percent: float, stop_loss_points: float) -> float:
    """
    Calculate position size based on account risk management
    """
    if stop_loss_points == 0:
        return 0.01  # Minimum lot size
    
    risk_amount = account_info.balance * (risk_percent / 100)
    point_value = mt5.symbol_info("XAUUSD").point
    
    # Calculate lot size based on risk
    lot_size = round(risk_amount / (stop_loss_points * point_value), 2)
    
    # Ensure lot size is within allowed limits
    symbol_info = mt5.symbol_info("XAUUSD")
    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
    
    return lot_size

def calculate_sl_tp(entry_price: float, signal_type: int, sl_points: float, tp_points: float) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit prices
    """
    point = mt5.symbol_info("XAUUSD").point
    
    if signal_type == 1:  # Buy
        sl = entry_price - (sl_points * point)
        tp = entry_price + (tp_points * point)
    else:  # Sell
        sl = entry_price + (sl_points * point)
        tp = entry_price - (tp_points * point)
    
    return round(sl, 2), round(tp, 2)

def execute_trade(signal_type: int, config: dict) -> bool:
    """
    Execute trade based on signal with stop loss and take profit
    """
    symbol = config['trading_params']['symbol']
    
    # Prepare the trade request
    price = mt5.symbol_info_tick(symbol).ask if signal_type == 1 else mt5.symbol_info_tick(symbol).bid
    
    # Calculate position size based on risk management
    account_info = mt5.account_info()
    lot_size = calculate_position_size(
        account_info,
        config['trading_params']['max_risk_percent'],
        config['trading_params']['stop_loss_pips']
    )
    
    # Calculate SL and TP levels
    sl, tp = calculate_sl_tp(
        price,
        signal_type,
        config['trading_params']['stop_loss_pips'],
        config['trading_params']['take_profit_pips']
    )
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if signal_type == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the trade request
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade execution failed. Error code: {result.retcode}")
        return False
    
    print(f"Trade executed successfully: {'BUY' if signal_type == 1 else 'SELL'} {lot_size} lots at {price}")
    print(f"Stop Loss: {sl}, Take Profit: {tp}")
    return True

def check_and_execute_signals(df: pd.DataFrame, config: dict):
    """
    Check latest signals and execute trades if conditions are met
    """
    latest = df.iloc[-1]
    
    # Check if we have any active trades
    positions = mt5.positions_get(symbol=config['trading_params']['symbol'])
    if positions:
        print("Skip trading - existing position open")
        return
    
    # Check signals from different strategies
    ma_signal = latest['ma_signal']
    rsi_signal = latest['rsi_signal']
    sr_signal = latest['sr_signal']
    
    # Execute trade if at least 2 strategies agree on the direction
    signals = [ma_signal, rsi_signal, sr_signal]
    buy_signals = sum(1 for s in signals if s == 1)
    sell_signals = sum(1 for s in signals if s == -1)
    
    if buy_signals >= 2:
        print("\nExecuting BUY trade - Multiple strategy confirmation")
        execute_trade(1, config)
    elif sell_signals >= 2:
        print("\nExecuting SELL trade - Multiple strategy confirmation")
        execute_trade(-1, config)

def run_backtest(df: pd.DataFrame, initial_balance: float = 10000.0, lot_size: float = 0.01) -> pd.DataFrame:
    """
    Run backtest on historical data
    """
    df = df.copy()
    
    # Initialize backtest columns
    df['position'] = 0
    df['equity'] = initial_balance
    df['returns'] = 0.0
    
    position = 0
    entry_price = 0
    balance = initial_balance
    
    # Calculate points value for XAUUSD
    point_value = mt5.symbol_info("XAUUSD").point * 100  # Approximate USD per point
    
    for i in range(1, len(df)):
        # Check for signal changes
        ma_signal = df.iloc[i]['ma_signal']
        rsi_signal = df.iloc[i]['rsi_signal']
        sr_signal = df.iloc[i]['sr_signal']
        
        signals = [ma_signal, rsi_signal, sr_signal]
        buy_signals = sum(1 for s in signals if s == 1)
        sell_signals = sum(1 for s in signals if s == -1)
        
        # Close existing position if opposite signals
        if position != 0:
            if (position == 1 and sell_signals >= 2) or (position == -1 and buy_signals >= 2):
                pnl = (df.iloc[i]['close'] - entry_price) * position * lot_size * point_value
                balance += pnl
                position = 0
        
        # Open new position
        if position == 0:
            if buy_signals >= 2:
                position = 1
                entry_price = df.iloc[i]['close']
            elif sell_signals >= 2:
                position = -1
                entry_price = df.iloc[i]['close']
        
        # Update positions and equity
        df.iloc[i, df.columns.get_loc('position')] = position
        if position != 0:
            unrealized_pnl = (df.iloc[i]['close'] - entry_price) * position * lot_size * point_value
            df.iloc[i, df.columns.get_loc('equity')] = balance + unrealized_pnl
        else:
            df.iloc[i, df.columns.get_loc('equity')] = balance
        
        # Calculate returns
        df.iloc[i, df.columns.get_loc('returns')] = df.iloc[i]['equity'] - df.iloc[i-1]['equity']
    
    return df

def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics
    """
    returns = df['returns'].dropna()
    
    total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0] * 100
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    max_drawdown = ((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max() * 100
    
    winning_trades = len(returns[returns > 0])
    losing_trades = len(returns[returns < 0])
    total_trades = winning_trades + losing_trades
    
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    return {
        'Total Return (%)': round(total_return, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Total Trades': total_trades
    }

def plot_equity_curve(df: pd.DataFrame, metrics: dict):
    """
    Plot equity curve and performance metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.plot(df.index, df['equity'], label='Equity Curve', color='blue')
    
    # Add metrics as text
    metrics_text = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    plt.text(0.02, 0.98, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Backtest Results - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

class Strategy:
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.performance = {}
        self.current_position = 0
        self.trades_history = []

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def backtest(self, df: pd.DataFrame, initial_balance: float = 10000.0, lot_size: float = 0.01) -> Dict:
        df = df.copy()
        position = 0
        entry_price = 0
        balance = initial_balance
        trades = []
        
        point_value = mt5.symbol_info("XAUUSD").point * 100
        
        for i in range(1, len(df)):
            signal = df.iloc[i].get('signal', 0)
            
            # Close existing position
            if position != 0 and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                pnl = (df.iloc[i]['close'] - entry_price) * position * lot_size * point_value
                balance += pnl
                position = 0
        
        # Open new position
        if position == 0:
            if buy_signals >= 2:
                position = 1
                entry_price = df.iloc[i]['close']
            elif sell_signals >= 2:
                position = -1
                entry_price = df.iloc[i]['close']
        
        # Update positions and equity
        df.iloc[i, df.columns.get_loc('position')] = position
        if position != 0:
            unrealized_pnl = (df.iloc[i]['close'] - entry_price) * position * lot_size * point_value
            df.iloc[i, df.columns.get_loc('equity')] = balance + unrealized_pnl
        else:
            df.iloc[i, df.columns.get_loc('equity')] = balance
        
        # Calculate returns
        df.iloc[i, df.columns.get_loc('returns')] = df.iloc[i]['equity'] - df.iloc[i-1]['equity']
    
    return df

class MAStrategy(Strategy):
    def __init__(self):
        super().__init__("Moving Average Crossover")

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        df['signal'] = np.where(
            (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1)),
            1,
            np.where(
                (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1)),
                -1,
                0
            )
        )
        return df

class RSIStrategy(Strategy):
    def __init__(self):
        super().__init__("RSI")

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ... RSI calculation code ...
        return df

class SRStrategy(Strategy):
    def __init__(self):
        super().__init__("Support/Resistance")

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ... Support/Resistance calculation code ...
        return df

class TradingBot:
    def __init__(self):
        self.strategies = {
            'ma': MAStrategy(),
            'rsi': RSIStrategy(),
            'sr': SRStrategy()
        }
        self.config = load_config()
        self.current_data = None
        self.is_running = False

    def fetch_data(self, timeframe=mt5.TIMEFRAME_H1, start_date=None, end_date=None):
        """
        Fetch historical data with proper error handling
        """
        try:
            # Convert timeframe string to MT5 timeframe
            timeframe_map = {
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Initialize MT5 if not already initialized
            if not mt5.initialize():
                raise Exception(f"Failed to initialize MT5: {mt5.last_error()}")
            
            # Default to last 1000 bars if no dates specified
            if start_date is None and end_date is None:
                self.current_data = get_gold_historical_data(timeframe=mt5_timeframe, num_bars=1000)
            else:
                # Convert dates to timestamps
                start_ts = pd.Timestamp(start_date).timestamp() if start_date else None
                end_ts = pd.Timestamp(end_date).timestamp() if end_date else None
                
                # Fetch data for date range
                rates = mt5.copy_rates_range("XAUUSD", mt5_timeframe, 
                                           pd.Timestamp(start_date).to_pydatetime() if start_date else None,
                                           pd.Timestamp(end_date).to_pydatetime() if end_date else None)
                
                if rates is not None:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    self.current_data = df
                else:
                    raise Exception(f"Failed to get XAUUSD data: {mt5.last_error()}")
            
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            self.current_data = None
            return False

    def analyze_all_strategies(self):
        if self.current_data is None:
            return {
                'error': 'No data available. Please fetch data first.'
            }
            
        results = {}
        try:
            for name, strategy in self.strategies.items():
                df = strategy.calculate_signals(self.current_data)
                backtest_results = strategy.backtest(df)
                results[name] = {
                    'signals': df.tail(10).to_dict('records'),
                    'performance': backtest_results
                }
        except Exception as e:
            results['error'] = f"Analysis failed: {str(e)}"
        
        return results

    def export_to_excel(self, filename: str):
        with pd.ExcelWriter(filename) as writer:
            # Write summary sheet
            summary = pd.DataFrame([s.performance for s in self.strategies.values()])
            summary.to_excel(writer, sheet_name='Summary')

            # Write detailed analysis for each strategy
            for name, strategy in self.strategies.items():
                df = strategy.calculate_signals(self.current_data)
                df.to_excel(writer, sheet_name=f'{name}_Analysis')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        timeframe = data.get('timeframe', 'H1')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        bot = TradingBot()
        if not bot.fetch_data(timeframe, start_date, end_date):
            return jsonify({'error': 'Failed to fetch data'}), 400
            
        results = bot.analyze_all_strategies()
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_analysis():
    # ... Excel export endpoint ...
    pass

def main():
    if not initialize_mt5():
        return
    
    config = load_config()
    print_account_info()
    
    # Fetch historical data for backtesting (longer period)
    gold_data = get_gold_historical_data(timeframe=mt5.TIMEFRAME_H1, num_bars=5000)
    
    if gold_data is not None:
        # Calculate signals
        gold_data = calculate_ma_signals(gold_data)
        gold_data = calculate_support_resistance(gold_data)
        gold_data = calculate_rsi(gold_data)
        gold_data = combine_signals(gold_data)
        
        # Run backtest
        print("\nRunning backtest...")
        backtest_results = run_backtest(
            gold_data,
            initial_balance=10000.0,
            lot_size=config['trading_params']['lot_size']
        )
        
        # Calculate and display performance metrics
        metrics = calculate_performance_metrics(backtest_results)
        print("\nBacktest Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        # Plot equity curve
        plot_equity_curve(backtest_results, metrics)
        
        # Check current signals for live trading
        print("\nChecking current trading signals...")
        print_signals(gold_data)
        check_and_execute_signals(gold_data, config)
    
    # Start Flask server
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False)).start()

if __name__ == "__main__":
    main()
