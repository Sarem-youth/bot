import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Union, Optional
from filterpy.kalman import KalmanFilter
from scipy.stats import zscore
import tensorflow as tf
from collections import deque
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch
from numba import jit
from sklearn.preprocessing import MinMaxScaler

@dataclass
class ProcessingConfig:
    kalman_process_var: float = 0.001
    kalman_measurement_var: float = 0.1
    ema_alpha: float = 0.1
    savgol_window: int = 7
    butterworth_cutoff: float = 0.1
    sequence_length: int = 60
    window_size: int = 60
    kalman_q: float = 0.001  # Process noise
    kalman_r: float = 0.1    # Measurement noise
    ema_alpha: float = 0.1   # EMA smoothing factor
    vwap_window: int = 20    # VWAP calculation window
    std_dev_window: int = 20
    price_decimals: int = 3
    batch_size: int = 32
    feature_columns: List[str] = None

class DataProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.kalman_filters = {}
        self.scalers = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.price_buffer = deque(maxlen=1000)
        self._initialize_filters()
        self.volume_buffer = deque(maxlen=1000)
        self._initialize_scalers()
        
    def _initialize_filters(self):
        """Initialize Kalman filters for different data types"""
        self.kalman_filters['price'] = self._create_kalman_filter()
        self.kalman_filters['volume'] = self._create_kalman_filter()
        
    def _create_kalman_filter(self) -> KalmanFilter:
        """Create Kalman filter with optimized parameters"""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0., 0.])
        kf.F = np.array([[1., 1.], [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.P *= 1000.
        kf.R = self.config.kalman_measurement_var
        kf.Q = self.config.kalman_process_var
        return kf
        
    def smooth_data(self, data: np.ndarray, method: str = 'kalman') -> np.ndarray:
        """Smooth data using specified method"""
        if method == 'kalman':
            return self._apply_kalman(data)
        elif method == 'ema':
            return self._apply_ema(data)
        elif method == 'savgol':
            return self._apply_savgol(data)
        elif method == 'butterworth':
            return self._apply_butterworth(data)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
            
    def _apply_kalman(self, data: np.ndarray) -> np.ndarray:
        """Apply Kalman filtering"""
        smoothed = np.zeros_like(data)
        kf = self.kalman_filters['price']
        
        for i, measurement in enumerate(data):
            kf.predict()
            kf.update(measurement)
            smoothed[i] = kf.x[0]
            
        return smoothed
        
    def _apply_ema(self, data: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average"""
        alpha = self.config.ema_alpha
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed
        
    def _apply_savgol(self, data: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter"""
        return savgol_filter(
            data, 
            window_length=self.config.savgol_window, 
            polyorder=3
        )
        
    def _apply_butterworth(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter"""
        b, a = butter(3, self.config.butterworth_cutoff, 'low')
        return filtfilt(b, a, data)
        
    async def process_tick_data(self, ticks: List[Dict]) -> pd.DataFrame:
        """Process tick data asynchronously"""
        loop = asyncio.get_running_loop()
        
        # Convert to DataFrame
        df = await loop.run_in_executor(
            self.executor,
            lambda: pd.DataFrame(ticks)
        )
        
        # Calculate mid prices
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Smooth prices using different methods
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.smooth_data,
                df['mid_price'].values,
                method
            )
            for method in ['kalman', 'ema']
        ]
        
        smooth_results = await asyncio.gather(*tasks)
        df['price_smooth_kalman'] = smooth_results[0]
        df['price_smooth_ema'] = smooth_results[1]
        
        return df
        
    def prepare_ml_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare features for ML models"""
        features = {}
        
        # Technical features
        features['price'] = self._normalize_feature(data['mid_price'].values, 'price')
        features['volume'] = self._normalize_feature(data['volume'].values, 'volume')
        features['spread'] = data['ask'] - data['bid']
        
        # Create sequences for time series models
        if len(data) >= self.config.sequence_length:
            sequences = []
            for i in range(len(data) - self.config.sequence_length):
                sequence = data.iloc[i:i+self.config.sequence_length]
                sequences.append(sequence[['mid_price', 'volume', 'spread']].values)
            features['sequences'] = np.array(sequences)
            
        return features
        
    def _normalize_feature(self, data: np.ndarray, name: str) -> np.ndarray:
        """Normalize features using robust scaling"""
        if name not in self.scalers:
            self.scalers[name] = RobustScaler()
            return self.scalers[name].fit_transform(data.reshape(-1, 1)).ravel()
        return self.scalers[name].transform(data.reshape(-1, 1)).ravel()
        
    def denormalize_feature(self, data: np.ndarray, name: str) -> np.ndarray:
        """Denormalize features"""
        if name in self.scalers:
            return self.scalers[name].inverse_transform(
                data.reshape(-1, 1)).ravel()
        return data
        
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp'])
        
        # Handle missing values
        data = data.interpolate(method='linear')
        
        # Remove outliers
        for col in ['mid_price', 'volume']:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > 3, col] = np.nan
                data[col] = data[col].interpolate(method='linear')
                
        return data

    def _initialize_scalers(self):
        """Initialize scalers for different features"""
        self.scalers = {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'technical': MinMaxScaler()
        }
        self.scalers['gold_specific'] = {
            'xau_usd_ratio': StandardScaler(),
            'real_rates': StandardScaler(),
            'dollar_index': StandardScaler()
        }

    def validate_gold_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate gold-specific data"""
        # Check for extreme price movements (more than 5% in a single period)
        price_changes = data['close'].pct_change()
        suspicious_moves = abs(price_changes) > 0.05
        
        if suspicious_moves.any():
            warnings.warn(f"Found {suspicious_moves.sum()} suspicious price movements")
            # Interpolate suspicious values
            data.loc[suspicious_moves, 'close'] = data['close'].interpolate(method='time')
        
        # Validate trading hours (gold trades 23/5)
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        invalid_hours = data[~data['hour'].between(0, 23)]
        if not invalid_hours.empty:
            warnings.warn(f"Found {len(invalid_hours)} records outside trading hours")
            data = data[data['hour'].between(0, 23)]
            
        # Check for minimum tick size (0.01 for gold)
        invalid_ticks = data['close'].apply(lambda x: len(str(x).split('.')[-1])) > 2
        if invalid_ticks.any():
            data.loc[invalid_ticks, 'close'] = data.loc[invalid_ticks, 'close'].round(2)
            
        return data.drop('hour', axis=1)

    def load_historical_data(self, timeframe: str, start_date: str, 
                           end_date: str, sources: List[str]) -> pd.DataFrame:
        """Load and combine historical data from multiple sources"""
        all_data = []
        
        for source in sources:
            try:
                if source == 'local':
                    data = self._load_local_data(timeframe, start_date, end_date)
                elif source == 'api':
                    data = self._load_api_data(timeframe, start_date, end_date)
                elif source == 'broker':
                    data = self._load_broker_data(timeframe, start_date, end_date)
                    
                if data is not None:
                    all_data.append(data)
            except Exception as e:
                warnings.warn(f"Error loading data from {source}: {str(e)}")
                
        if not all_data:
            raise ValueError("No data could be loaded from any source")
            
        # Combine and validate data
        combined_data = pd.concat(all_data).drop_duplicates()
        combined_data = self.validate_gold_data(combined_data)
        
        return combined_data

    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe"""
        # Map timeframe strings to pandas offset aliases
        timeframe_map = {
            'tick': '1ms',
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            'D': 'D'
        }
        
        if target_timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
            
        resampled = data.resample(
            timeframe_map[target_timeframe],
            on='timestamp'
        ).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled

    @jit(nopython=True)
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate VWAP with Numba optimization"""
        return np.sum(prices * volumes) / np.sum(volumes) if len(volumes) > 0 else 0

    def process_tick(self, tick: Dict) -> Dict:
        """Process individual tick data"""
        try:
            # Extract basic features
            price = (tick['bid'] + tick['ask']) / 2
            spread = tick['ask'] - tick['bid']
            
            # Update buffers
            self.price_buffer.append(price)
            self.volume_buffer.append(tick['volume'])
            
            # Calculate smoothed price
            smoothed_price = self._smooth_price(list(self.price_buffer))
            
            # Calculate VWAP
            vwap = self._calculate_vwap(
                np.array(list(self.price_buffer)[-self.config.vwap_window:]),
                np.array(list(self.volume_buffer)[-self.config.vwap_window:])
            )
            
            return {
                'timestamp': tick['time'],
                'price': round(price, self.config.price_decimals),
                'smoothed_price': round(smoothed_price, self.config.price_decimals),
                'spread': spread,
                'volume': tick['volume'],
                'vwap': round(vwap, self.config.price_decimals)
            }
        except Exception as e:
            warnings.warn(f"Error processing tick: {e}")
            return tick

    def prepare_ml_features(self, data: List[Dict], normalize: bool = True) -> np.ndarray:
        """Prepare features for ML models"""
        df = pd.DataFrame(data)
        
        # Calculate technical features
        features = self._calculate_technical_features(df)
        
        if normalize:
            # Normalize features by group
            for col in features.columns:
                if 'price' in col:
                    scaler = self.scalers['price']
                elif 'volume' in col:
                    scaler = self.scalers['volume']
                else:
                    scaler = self.scalers['technical']
                
                features[col] = scaler.fit_transform(
                    features[col].values.reshape(-1, 1)
                ).ravel()
        
        return features.values

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        features = pd.DataFrame()
        
        # Price features
        features['price_momentum'] = df['price'].diff()
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Volume features
        features['volume_momentum'] = df['volume'].diff()
        features['volume_ma'] = df['volume'].rolling(window=self.config.window_size).mean()
        
        # Volatility features
        features['volatility'] = df['price'].rolling(window=self.config.std_dev_window).std()
        features['spread_ma'] = df['spread'].rolling(window=self.config.window_size).mean()
        
        return features.fillna(0)

    def _smooth_price(self, prices: List[float]) -> float:
        """Apply price smoothing"""
        if len(prices) < 3:
            return prices[-1]
            
        # Apply Savitzky-Golay filter for smooth derivatives
        return savgol_filter(prices, min(len(prices), 7), 3)[-1]

    def create_sequences(self, features: np.ndarray, 
                        sequence_length: int) -> torch.Tensor:
        """Create sequences for time series models"""
        sequences = []
        for i in range(len(features) - sequence_length):
            sequence = features[i:(i + sequence_length)]
            sequences.append(sequence)
            
        return torch.FloatTensor(np.array(sequences))

    def process_batch(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process a batch of data for ML models"""
        features = self.prepare_ml_features(batch_data)
        sequences = self.create_sequences(features, self.config.window_size)
        
        return {
            'sequences': sequences,
            'features': torch.FloatTensor(features)
        }

class RealTimeProcessor:
    """Real-time data processing pipeline"""
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.pipeline = DataProcessor(config)
        self.tick_buffer = []
        self.batch_size = config.batch_size if config else 32

    async def process_streaming_data(self, tick: Dict) -> Optional[Dict]:
        """Process streaming data in real-time"""
        # Add to buffer
        self.tick_buffer.append(tick)
        
        # Process in batches for efficiency
        if len(self.tick_buffer) >= self.batch_size:
            batch = self.tick_buffer[-self.batch_size:]
            processed_batch = self.pipeline.process_batch(batch)
            self.tick_buffer = self.tick_buffer[-self.config.window_size:]
            return processed_batch
            
        return None

    def get_latest_features(self) -> np.ndarray:
        """Get latest processed features"""
        if len(self.tick_buffer) >= self.config.window_size:
            return self.pipeline.prepare_ml_features(
                self.tick_buffer[-self.config.window_size:]
            )
        return None

    async def process_streaming_data(self, tick_data):
        """Process new tick data in real-time"""
        processed_tick = self.calculate_real_time_features(tick_data)
        if self.validate_real_time_data(tick_data):
            # Add to batch and process if batch is ready
            self.tick_batch.append(processed_tick)
            if len(self.tick_batch) >= self.config.batch_size:
                return await self.process_tick_batch(self.tick_batch)
        return None

    def calculate_real_time_features(self, tick_data):
        processor = DataProcessor()
        processed_tick = processor.process_tick({
            'time': tick_data['timestamp'],
            'bid': tick_data['bid'],
            'ask': tick_data['ask'],
            'volume': tick_data['volume']
        })
        return processed_tick

    def validate_real_time_data(self, tick_data):
        if tick_data['bid'] <= 0 or tick_data['ask'] <= 0:
            return False
        spread = tick_data['ask'] - tick_data['bid']
        if spread > 1.0:
            return False
        if tick_data['timestamp'] < self.last_tick_time:
            return False
        return True

    async def process_tick_batch(self, tick_batch):
        try:
            features = self.prepare_ml_features(tick_batch)
            sequences = self.create_sequences(features, self.config.window_size)
            return {
                'features': features,
                'sequences': sequences
            }
        except Exception as e:
            logger.error(f"Error processing tick batch: {e}")
            return None
