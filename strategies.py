import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
from collections import deque
from typing import Deque

class VolatilityStrategy:
    def __init__(self):
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        self.volatility_threshold = 0.001  # 0.1% minimum volatility
        self.breakout_threshold = 1.5  # Minimum ATR multiplier for breakout
        
    def calculate_atr(self, candles: List[Dict]) -> Tuple[float, List[float]]:
        """Calculate ATR and ATR values list"""
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        
        tr_values = []
        for i in range(1, len(candles)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
            
        atr_values = pd.Series(tr_values).rolling(window=self.atr_period).mean()
        current_atr = atr_values.iloc[-1]
        
        return current_atr, atr_values.tolist()
        
    def calculate_bollinger_bands(self, closes: List[float]) -> Tuple[pd.Series, pd.Series, float]:
        """Calculate Bollinger Bands with dynamic std dev"""
        series = pd.Series(closes)
        sma = series.rolling(window=self.bb_period).mean()
        std = series.rolling(window=self.bb_period).std()
        
        # Adjust BB width based on volatility
        volatility = std.iloc[-1] / sma.iloc[-1]
        bb_mult = self.bb_std * (1 + volatility)
        
        upper = sma + (std * bb_mult)
        lower = sma - (std * bb_mult)
        
        return upper, lower, volatility
        
    def analyze_volatility(self, candles: List[Dict]) -> Dict:
        """Analyze volatility patterns"""
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        
        # Calculate ATR
        current_atr, atr_values = self.calculate_atr(candles)
        
        # Calculate Bollinger Bands
        upper_band, lower_band, volatility = self.calculate_bollinger_bands(closes)
        
        # Check for volatility breakout
        signal = None
        strength = 0
        
        # Price distance from bands
        upper_distance = abs(closes[-1] - upper_band.iloc[-1])
        lower_distance = abs(closes[-1] - lower_band.iloc[-1])
        
        # Volatility breakout conditions
        if current_atr > 0:
            # Strong breakout above upper band
            if closes[-1] > upper_band.iloc[-1] and upper_distance > current_atr * self.breakout_threshold:
                signal = 'BUY'
                strength = 2
            # Strong breakout below lower band
            elif closes[-1] < lower_band.iloc[-1] and lower_distance > current_atr * self.breakout_threshold:
                signal = 'SELL'
                strength = 2
                
        return {
            'signal': signal,
            'strength': strength,
            'atr': current_atr,
            'volatility': volatility,
            'bb_width': (upper_band.iloc[-1] - lower_band.iloc[-1]) / closes[-1],
            'position_size': self.calculate_position_size(current_atr, closes[-1])
        }
        
    def calculate_position_size(self, atr: float, current_price: float) -> float:
        """Calculate dynamic position size based on ATR"""
        base_risk = 0.02  # 2% risk per trade
        account_size = 10000  # Example account size
        
        # Risk per pip calculation
        risk_amount = account_size * base_risk
        risk_pips = atr * 2  # Use 2x ATR for stop loss
        
        # Position size calculation
        position_size = risk_amount / risk_pips
        
        # Adjust for price volatility
        if atr / current_price > self.volatility_threshold:
            position_size *= 0.75  # Reduce position size in high volatility
            
        return position_size

class GoldTrendStrategy:
    def __init__(self):
        self.ama_period = 10
        self.ama_fast = 2
        self.ama_slow = 30
        self.adx_period = 14
        self.adx_threshold = 25
        self.volatility_lookback = 20
        self.volatility_strategy = VolatilityStrategy()
        self.scalping_strategy = ScalpingStrategy()
        
    def calculate_ama(self, prices: List[float]) -> List[float]:
        """Adaptive Moving Average calculation"""
        prices = np.array(prices)
        direction = abs(prices[-1] - prices[0])
        volatility = sum([abs(prices[i] - prices[i-1]) for i in range(1, len(prices))])
        er = direction / volatility if volatility != 0 else 0
        
        fast_sc = 2.0 / (self.ama_fast + 1)
        slow_sc = 2.0 / (self.ama_slow + 1)
        sc = er * (fast_sc - slow_sc) + slow_sc
        sc = sc * sc
        
        ama = [prices[0]]
        for i in range(1, len(prices)):
            ama.append(ama[-1] + sc * (prices[i] - ama[-1]))
        return ama
        
    def calculate_dynamic_ema_periods(self, prices: List[float]) -> Tuple[int, int]:
        """Calculate dynamic EMA periods based on volatility"""
        returns = pd.Series(prices).pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Adjust EMA periods based on volatility
        if volatility > 0.20:  # High volatility
            fast_period = 5
            slow_period = 15
        elif volatility > 0.10:  # Medium volatility
            fast_period = 9
            slow_period = 21
        else:  # Low volatility
            fast_period = 12
            slow_period = 26
            
        return fast_period, slow_period
        
    def calculate_adx(self, high: List[float], low: List[float], close: List[float]) -> List[float]:
        """Calculate Average Directional Index"""
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['dx'] = abs(df['high'] - df['high'].shift(1)) - abs(df['low'] - df['low'].shift(1))
        df['dmplus'] = np.where(df['dx'] > 0, df['dx'], 0)
        df['dmminus'] = np.where(df['dx'] < 0, -df['dx'], 0)
        
        df['diplus'] = 100 * df['dmplus'].rolling(self.adx_period).mean() / df['tr'].rolling(self.adx_period).mean()
        df['diminus'] = 100 * df['dmminus'].rolling(self.adx_period).mean() / df['tr'].rolling(self.adx_period).mean()
        df['dx'] = 100 * abs(df['diplus'] - df['diminus']) / (df['diplus'] + df['diminus'])
        df['adx'] = df['dx'].rolling(self.adx_period).mean()
        
        return df['adx'].fillna(0).tolist()
        
    def analyze_trend(self, candles: List[Dict]) -> Dict:
        """Analyze trend using combined strategies"""
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        
        # Calculate indicators
        ama = self.calculate_ama(closes)
        fast_period, slow_period = self.calculate_dynamic_ema_periods(closes)
        fast_ema = pd.Series(closes).ewm(span=fast_period, adjust=False).mean()
        slow_ema = pd.Series(closes).ewm(span=slow_period, adjust=False).mean()
        adx = self.calculate_adx(highs, lows, closes)
        
        # Calculate Bollinger Bands
        bb_period = 20
        std_dev = 2
        sma = pd.Series(closes).rolling(window=bb_period).mean()
        std = pd.Series(closes).rolling(window=bb_period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Generate signals
        trend_strength = 0
        signal_type = None
        
        # Combine signals with weights
        if ama[-1] > ama[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]:
            trend_strength += 1
        if adx[-1] > self.adx_threshold:
            trend_strength += 1
        if closes[-1] > upper_band.iloc[-1]:
            trend_strength += 1
        elif closes[-1] < lower_band.iloc[-1]:
            trend_strength -= 1
            
        signal_type = 'BUY' if trend_strength >= 2 else 'SELL' if trend_strength <= -2 else None
        
        # Add volatility analysis
        volatility_analysis = self.volatility_strategy.analyze_volatility(candles)
        
        # Add scalping analysis if available
        if hasattr(self, 'last_tick') and hasattr(self, 'order_book'):
            scalp_analysis = self.scalping_strategy.analyze_tick_data(self.last_tick)
            book_analysis = self.scalping_strategy.analyze_order_book(self.order_book)
            
            # Consider scalping signals in trend analysis
            if scalp_analysis['signal'] and book_analysis['signal'] == scalp_analysis['signal']:
                signal_type = scalp_analysis['signal']
                trend_strength += (scalp_analysis['strength'] + book_analysis['strength']) / 2
        
        return {
            'signal': signal_type,
            'strength': abs(trend_strength),
            'adx': adx[-1],
            'fast_period': fast_period,
            'slow_period': slow_period,
            'volatility': volatility_analysis['volatility'],
            'atr': volatility_analysis['atr'],
            'suggested_position_size': volatility_analysis['position_size']
        }

class RangeBoundStrategy:
    def __init__(self):
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
    def calculate_rsi_divergence(self, prices: List[float]) -> Optional[Dict]:
        """Calculate RSI divergence"""
        rsi_values = pd.Series(prices).diff()
        up = rsi_values.clip(lower=0)
        down = -1 * rsi_values.clip(upper=0)
        ma_up = up.rolling(self.rsi_period).mean()
        ma_down = down.rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        
        # Look for divergence
        price_highs = []
        rsi_highs = []
        price_lows = []
        rsi_lows = []
        
        for i in range(-20, -1):
            if i+2 < len(prices):
                # Bullish divergence
                if prices[i] < prices[i-1] and rsi[i] > rsi[i-1]:
                    price_lows.append(prices[i])
                    rsi_lows.append(rsi[i])
                # Bearish divergence    
                elif prices[i] > prices[i-1] and rsi[i] < rsi[i-1]:
                    price_highs.append(prices[i])
                    rsi_highs.append(rsi[i])
        
        if len(price_lows) >= 2 and rsi[-1] < self.rsi_oversold:
            return {'type': 'BUY', 'strategy': 'RSI_DIV', 'strength': 2}
        elif len(price_highs) >= 2 and rsi[-1] > self.rsi_overbought:
            return {'type': 'SELL', 'strategy': 'RSI_DIV', 'strength': 2}
            
        return None
        
    def calculate_stochastic(self, high: List[float], low: List[float], close: List[float]) -> Dict:
        """Calculate Stochastic Oscillator"""
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        
        # Calculate %K
        df['lowest_low'] = df['low'].rolling(window=self.stoch_k_period).min()
        df['highest_high'] = df['high'].rolling(window=self.stoch_k_period).max()
        df['%K'] = 100 * ((df['close'] - df['lowest_low']) / 
                         (df['highest_high'] - df['lowest_low']))
        
        # Calculate %D
        df['%D'] = df['%K'].rolling(window=self.stoch_d_period).mean()
        
        k = df['%K'].iloc[-1]
        d = df['%D'].iloc[-1]
        
        signal = None
        if k < self.stoch_oversold and d < self.stoch_oversold and k > d:
            signal = {'type': 'BUY', 'strategy': 'STOCH', 'strength': 1}
        elif k > self.stoch_overbought and d > self.stoch_overbought and k < d:
            signal = {'type': 'SELL', 'strategy': 'STOCH', 'strength': 1}
            
        return {
            'signal': signal,
            'k': k,
            'd': d
        }
        
    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        levels = {}
        for fib in self.fib_levels:
            if high > low:  # Uptrend retracement
                levels[fib] = high - (diff * fib)
            else:  # Downtrend retracement
                levels[fib] = low + (diff * fib)
        return levels
        
    def check_fib_breakout(self, price: float, fib_levels: Dict[float, float]) -> Optional[Dict]:
        """Check for breakouts from Fibonacci levels"""
        nearest_level = min(fib_levels.values(), key=lambda x: abs(x - price))
        distance = abs(price - nearest_level)
        
        # Check if price is breaking out of a Fibonacci level
        if distance < (price * 0.001):  # Within 0.1% of level
            if price > nearest_level:
                return {'type': 'BUY', 'strategy': 'FIB', 'strength': 2}
            else:
                return {'type': 'SELL', 'strategy': 'FIB', 'strength': 2}
        return None
        
    def analyze_range(self, candles: List[Dict]) -> Dict:
        """Analyze range-bound conditions"""
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        
        # RSI Divergence
        rsi_signal = self.calculate_rsi_divergence(closes)
        
        # Stochastic Oscillator
        stoch = self.calculate_stochastic(highs, lows, closes)
        
        # Fibonacci Analysis
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        fib_levels = self.calculate_fibonacci_levels(recent_high, recent_low)
        fib_signal = self.check_fib_breakout(closes[-1], fib_levels)
        
        signals = []
        if rsi_signal:
            signals.append(rsi_signal)
        if stoch['signal']:
            signals.append(stoch['signal'])
        if fib_signal:
            signals.append(fib_signal)
            
        # Combine signals
        signal_strength = sum(s['strength'] for s in signals)
        signal_type = signals[0]['type'] if signals else None
        
        return {
            'signal': signal_type,
            'strength': signal_strength,
            'stoch_k': stoch['k'],
            'stoch_d': stoch['d'],
            'fib_levels': fib_levels
        }

class EventStrategy:
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.economic_api_key = os.getenv('ECONOMIC_CALENDAR_API_KEY')
        self.sentiment_threshold = 0.2
        self.impact_keywords = {
            'high': ['fed', 'inflation', 'interest rate', 'monetary policy', 'fomc'],
            'medium': ['gdp', 'employment', 'trade', 'deficit', 'retail'],
            'low': ['market sentiment', 'technical analysis', 'forecast']
        }
        
    def get_recent_news(self) -> List[Dict]:
        """Fetch recent gold-related news"""
        endpoint = "https://newsapi.org/v2/everything"
        params = {
            'q': 'gold OR XAU/USD OR precious metals',
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.news_api_key,
            'from': (datetime.now() - timedelta(hours=24)).isoformat()
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()['articles']
        except Exception as e:
            print(f"News API error: {str(e)}")
        return []
        
    def analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment using TextBlob"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
        
    def check_economic_calendar(self) -> List[Dict]:
        """Check economic calendar for important events"""
        endpoint = "https://economic-calendar-api.example.com/events"  # Replace with actual API
        params = {
            'apikey': self.economic_api_key,
            'currencies': 'XAU,USD',
            'importance': 'high',
            'from': datetime.now().isoformat(),
            'to': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()['events']
        except Exception as e:
            print(f"Economic Calendar API error: {str(e)}")
        return []
        
    def calculate_impact_score(self, news_item: Dict) -> float:
        """Calculate news impact score based on keywords"""
        text = f"{news_item['title']} {news_item['description']}"
        score = 0
        
        for impact, keywords in self.impact_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    score += 1.0 if impact == 'high' else 0.5 if impact == 'medium' else 0.2
                    
        return score
        
    def analyze_events(self) -> Dict:
        """Analyze news and economic events"""
        news_items = self.get_recent_news()
        economic_events = self.check_economic_calendar()
        
        # Analyze news sentiment
        total_sentiment = 0
        total_impact = 0
        
        for news in news_items:
            sentiment = self.analyze_sentiment(f"{news['title']} {news['description']}")
            impact = self.calculate_impact_score(news)
            total_sentiment += sentiment * impact
            total_impact += impact
            
        avg_sentiment = total_sentiment / total_impact if total_impact > 0 else 0
        
        # Check economic events
        event_importance = 0
        for event in economic_events:
            if event['importance'] == 'high':
                event_importance += 1
                
        # Generate signal based on sentiment and events
        signal = None
        strength = 0
        
        if abs(avg_sentiment) > self.sentiment_threshold:
            signal = 'BUY' if avg_sentiment > 0 else 'SELL'
            strength = min(3, abs(avg_sentiment) * 2 + event_importance)
            
        return {
            'signal': signal,
            'strength': strength,
            'sentiment': avg_sentiment,
            'pending_events': event_importance,
            'impact': total_impact
        }

class ScalpingStrategy:
    def __init__(self):
        self.tick_buffer_size = 1000
        self.tick_buffer: Deque = deque(maxlen=self.tick_buffer_size)
        self.min_tick_threshold = 5
        self.price_threshold = 0.0001  # Minimum price movement to consider
        self.volume_threshold = 10.0   # Minimum volume to consider
        self.order_book_levels = 10    # Depth of order book to analyze
        self.scalp_timeout = 30        # Maximum seconds to hold position
        
    def analyze_tick_data(self, tick: Dict) -> Dict:
        """Analyze incoming tick data for scalping opportunities"""
        self.tick_buffer.append(tick)
        
        if len(self.tick_buffer) < self.min_tick_threshold:
            return {'signal': None}
            
        # Calculate micro-trend
        recent_ticks = list(self.tick_buffer)[-self.min_tick_threshold:]
        price_changes = [t['price'] - recent_ticks[i-1]['price'] 
                        for i, t in enumerate(recent_ticks[1:], 1)]
        
        # Volume analysis
        volume_weighted_price = sum(t['price'] * t['volume'] for t in recent_ticks) / \
                              sum(t['volume'] for t in recent_ticks)
        
        # Detect price acceleration
        acceleration = sum(1 for x in price_changes[1:] 
                         if abs(x) > abs(price_changes[price_changes.index(x)-1]))
        
        signal = None
        strength = 0
        
        # Generate scalping signals
        if acceleration >= 3 and sum(price_changes) > self.price_threshold:
            signal = 'BUY'
            strength = min(3, acceleration / 2)
        elif acceleration >= 3 and sum(price_changes) < -self.price_threshold:
            signal = 'SELL'
            strength = min(3, acceleration / 2)
            
        return {
            'signal': signal,
            'strength': strength,
            'vwap': volume_weighted_price,
            'acceleration': acceleration
        }
        
    def analyze_order_book(self, order_book: Dict) -> Dict:
        """Analyze order book for price pressure"""
        buy_volume = sum(level['volume'] for level in order_book['bids'][:self.order_book_levels])
        sell_volume = sum(level['volume'] for level in order_book['asks'][:self.order_book_levels])
        
        # Calculate price pressure
        volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # Calculate order book depth
        bid_ask_spread = order_book['asks'][0]['price'] - order_book['bids'][0]['price']
        
        signal = None
        strength = abs(volume_imbalance) * 2
        
        if volume_imbalance > 0.2:  # Strong buying pressure
            signal = 'BUY'
        elif volume_imbalance < -0.2:  # Strong selling pressure
            signal = 'SELL'
            
        return {
            'signal': signal,
            'strength': strength,
            'spread': bid_ask_spread,
            'imbalance': volume_imbalance
        }
        
    def get_optimal_scalp_levels(self, current_price: float, volatility: float) -> Dict:
        """Calculate optimal entry, stop loss, and take profit levels"""
        atr_multiplier = 0.5  # Tighter ranges for scalping
        stop_loss = atr_multiplier * volatility
        take_profit = stop_loss * 1.5  # 1.5:1 reward-to-risk ratio
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timeout': self.scalp_timeout
        }
