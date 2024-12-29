from ml_model import GoldPricePredictor
import numpy as np

class TradingBot:
    def __init__(self):
        # ...existing code...
        self.adaptive_periods = {
            'slow': 50,
            'medium': 21,
            'fast': 9
        }
        self.volatility_multiplier = 1.0
        self.adx_period = 14
        self.adx_threshold = 25
        self.tick_buffer = []
        self.tick_buffer_size = 1000
        self.ml_model = GoldPricePredictor()
        self.model_path = 'models/gold_predictor.joblib'
        self.training_data = []
        self.min_training_samples = 1000
        # ...existing code...

    def calculate_adaptive_ma(self, prices, volatility_index):
        """Calculate Adaptive Moving Average based on market volatility"""
        alpha = min(1.0, max(0.1, volatility_index / 100))
        ma = [prices[0]]
        for i in range(1, len(prices)):
            ma.append(alpha * prices[i] + (1 - alpha) * ma[i-1])
        return ma

    def check_adx_trend(self, candles):
        """Calculate ADX for trend strength confirmation"""
        highs = [candle[2] for candle in candles]
        lows = [candle[3] for candle in candles]
        closes = [candle[4] for candle in candles]
        
        # Calculate +DI and -DI
        plus_dm = np.zeros(len(candles))
        minus_dm = np.zeros(len(candles))
        
        for i in range(1, len(candles)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        tr = self.calculate_tr(highs, lows, closes)
        
        # Smooth the indicators
        smoothed_plus_dm = self.exponential_smooth(plus_dm, self.adx_period)
        smoothed_minus_dm = self.exponential_smooth(minus_dm, self.adx_period)
        smoothed_tr = self.exponential_smooth(tr, self.adx_period)
        
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = self.exponential_smooth(dx, self.adx_period)
        
        return adx[-1], plus_di[-1], minus_di[-1]

    def analyze_market(self, symbol):
        try:
            # ...existing code...
            
            # Enhanced market analysis with new strategies
            adx, plus_di, minus_di = self.check_adx_trend(m15_candles)
            
            # Add volatility-based adjustments
            atr = self.calculate_atr(m15_candles)
            self.volatility_multiplier = min(2.0, max(0.5, atr / self.gold_point))
            
            # Adjust strategy parameters based on volatility
            if self.volatility_multiplier > 1.5:
                self.rsi_buy_threshold = 35  # More aggressive in volatile markets
                self.rsi_sell_threshold = 65
            else:
                self.rsi_buy_threshold = 40  # More conservative in calm markets
                self.rsi_sell_threshold = 60
                
            # Apply adaptive moving averages
            closes = [candle[4] for candle in m15_candles]
            adaptive_ma = self.calculate_adaptive_ma(closes, self.volatility_multiplier * 100)
            
            # Enhanced signal validation
            signals = []
            if adx > self.adx_threshold:
                if plus_di > minus_di:
                    signals.append({'type': 'BUY', 'strategy': 'ADX', 'strength': 1.5})
                else:
                    signals.append({'type': 'SELL', 'strategy': 'ADX', 'strength': 1.5})
            
            # Add existing signal checks
            # ...existing code...
            
            # Machine learning prediction if model is available
            if self.ml_model:
                prediction = self.get_ml_prediction(m15_candles)
                if prediction is not None:
                    signals.append({
                        'type': 'BUY' if prediction > 0 else 'SELL',
                        'strategy': 'ML',
                        'strength': abs(prediction)
                    })
            
            # Collect training data
            self.collect_training_data(m15_candles[-1])
            
            # Get ML prediction if model is trained
            if self.ml_model.trained:
                prediction = self.get_ml_prediction(m15_candles)
                if prediction is not None:
                    signals.append({
                        'type': 'BUY' if prediction > 0 else 'SELL',
                        'strategy': 'ML',
                        'strength': abs(prediction)
                    })
            
            return self.validate_signals(signals, m15_candles)
            
        except Exception as e:
            print(f"Error in analyze_market: {str(e)}")
            return None

    def get_ml_prediction(self, candles):
        """Get prediction from machine learning model"""
        if not self.ml_model:
            return None
            
        # Prepare features
        features = self.prepare_ml_features(candles)
        
        try:
            prediction = self.ml_model.predict([features])[0]
            return prediction
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
            return None

    def prepare_ml_features(self, candles):
        """Prepare features for ML model"""
        features = []
        closes = [candle[4] for candle in candles]
        
        # Technical indicators
        rsi = self.calculate_rsi(closes)[-1]
        macd, signal = self.calculate_macd(closes)
        bb_upper, bb_lower = self.calculate_bollinger_bands(closes)
        
        features.extend([
            rsi,
            macd[-1],
            signal[-1],
            (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]),
            self.volatility_multiplier
        ])
        
        return features

    def train_ml_model(self):
        """Train or retrain the ML model with accumulated data"""
        if len(self.training_data) < self.min_training_samples:
            print(f"Insufficient training data: {len(self.training_data)}/{self.min_training_samples}")
            return False

        try:
            scores = self.ml_model.train(self.training_data)
            print(f"Model trained - Train score: {scores['train_score']:.2f}, Test score: {scores['test_score']:.2f}")
            self.ml_model.save_model(self.model_path)
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def collect_training_data(self, candle):
        """Collect and preprocess data for ML training"""
        if not candle:
            return

        data_point = {
            'close': candle[4],
            'rsi': self.calculate_rsi([c[4] for c in self.training_data[-14:] + [candle]])[-1],
            'macd': self.calculate_macd([c[4] for c in self.training_data[-26:] + [candle]])[0][-1],
            'bb_width': self.calculate_bb_width(candle),
            'volatility': self.calculate_volatility(candle)
        }
        self.training_data.append(data_point)

        # Train model when sufficient data is collected
        if len(self.training_data) == self.min_training_samples:
            self.train_ml_model()

    def calculate_bb_width(self, candle):
        """Calculate Bollinger Band width"""
        closes = [c[4] for c in self.training_data[-20:] + [candle]]
        upper, lower = self.calculate_bollinger_bands(closes)
        return (upper[-1] - lower[-1]) / closes[-1]

    def calculate_volatility(self, candle):
        """Calculate local volatility"""
        returns = [abs(c[4] - c[1]) / c[1] for c in self.training_data[-10:] + [candle]]
        return np.std(returns)
