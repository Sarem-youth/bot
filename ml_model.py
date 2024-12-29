import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

class GoldPricePredictor:
    def __init__(self):
        self.sequence_length = 60
        self.future_steps = 5
        self.scaler = MinMaxScaler()
        self.model = self._build_lstm_model()
        self.prediction_threshold = 0.6
        
    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for price prediction"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(self.future_steps)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def prepare_data(self, candles: list) -> tuple:
        """Prepare data for LSTM model"""
        df = pd.DataFrame(candles)
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length - self.future_steps):
            X.append(df[features].iloc[i:(i + self.sequence_length)].values)
            y.append(df['close'].iloc[(i + self.sequence_length):
                                    (i + self.sequence_length + self.future_steps)].values)
                                    
        X = np.array(X)
        y = np.array(y)
        
        # Scale data
        X_scaled = np.array([self.scaler.fit_transform(x) for x in X])
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        
        return train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
        
    def train(self, candles: list):
        """Train the LSTM model"""
        X_train, X_test, y_train, y_test = self.prepare_data(candles)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=50, batch_size=32, verbose=0)
        
    def predict(self, candles: list) -> dict:
        """Predict future price movements"""
        recent_data = np.array([candles[-self.sequence_length:]])
        scaled_data = np.array([self.scaler.fit_transform(recent_data[0])])
        prediction = self.model.predict(scaled_data)
        
        # Inverse transform predictions
        prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
        current_price = candles[-1][4]  # Current close price
        
        # Calculate prediction confidence and direction
        price_direction = 1 if prediction[-1] > current_price else -1
        confidence = abs(prediction[-1] - current_price) / current_price
        
        return {
            'direction': 'BUY' if price_direction > 0 else 'SELL',
            'confidence': confidence,
            'predicted_prices': prediction.flatten(),
            'signal': 'BUY' if price_direction > 0 and confidence > self.prediction_threshold else
                     'SELL' if price_direction < 0 and confidence > self.prediction_threshold else None
        }

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for Reinforcement Learning"""
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0
        self.positions = []
        self.balance = 10000
        self.window_size = 60
        
        # Action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: OHLCV data + technical indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 8), dtype=np.float32)
            
    def reset(self):
        self.current_step = self.window_size
        self.balance = 10000
        self.positions = []
        return self._get_observation()
        
    def _get_observation(self):
        """Get current market state"""
        obs = self.data[self.current_step-self.window_size:self.current_step]
        return obs
        
    def _calculate_reward(self, action):
        """Calculate reward based on action and price movement"""
        next_price = self.data[self.current_step+1]['close']
        current_price = self.data[self.current_step]['close']
        price_change = (next_price - current_price) / current_price
        
        if action == 1:  # Buy
            return price_change * 100
        elif action == 2:  # Sell
            return -price_change * 100
        return 0  # Hold
        
    def step(self, action):
        """Execute one trading step"""
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}

class RLTrader:
    def __init__(self):
        self.model = None
        self.env = None
        self.training_data = []
        
    def prepare_environment(self, candles: list):
        """Prepare trading environment with historical data"""
        self.training_data = self._prepare_data(candles)
        self.env = DummyVecEnv([lambda: TradingEnvironment(self.training_data)])
        self.model = PPO("MlpPolicy", self.env, verbose=0, learning_rate=0.0001)
        
    def _prepare_data(self, candles: list) -> list:
        """Prepare data with technical indicators"""
        df = pd.DataFrame(candles)
        
        # Add technical indicators
        df['sma'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['atr'] = self._calculate_atr(df)
        
        return df.dropna().to_dict('records')
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def train(self, total_timesteps: int = 100000):
        """Train the RL model"""
        if self.model is None:
            raise ValueError("Environment not prepared. Call prepare_environment first.")
            
        self.model.learn(total_timesteps=total_timesteps)
        
    def predict(self, current_state: list) -> dict:
        """Predict trading action based on current market state"""
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
            
        observation = self._prepare_data(current_state)[-self.env.observation_space.shape[0]:]
        action, _ = self.model.predict(observation)
        
        return {
            'action': action,
            'signal': 'BUY' if action == 1 else 'SELL' if action == 2 else None,
            'confidence': 0.7  # Fixed confidence for RL predictions
        }
