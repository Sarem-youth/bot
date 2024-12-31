import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import sys
import psutil
from typing import Dict, Any, Optional, Tuple, ClassVar, List, Type
import requests
from datetime import datetime, timedelta
import asyncio
import aiohttp
import pytest
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Protocol

# Configuration and Constants
CONFIG = {
    "broker_server": "MetaQuotes-Demo",
    "account_number": 10005241552,
    "password": "3mUaUpK*",
    "symbol": "XAUUSD",
    "max_daily_loss": 1000,
    "max_position_size": 1.0,
    "allowed_trading_hours": {
        "start": "08:00",
        "end": "22:00"
    }
}

# Add new configuration settings after existing CONFIG
USER_PREFERENCES = {
    "platform": {
        "default_timeframe": mt5.TIMEFRAME_M5,
        "chart_timeframes": [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1],
        "enable_mobile_alerts": True,
        "enable_email_notifications": False
    },
    "strategy": {
        "indicators": {
            "sma": {"periods": [10, 20, 50]},
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "bollinger": {"period": 20, "deviation": 2}
        },
        "entry_rules": {
            "rsi_oversold": True,
            "moving_average_cross": True,
            "bollinger_bounce": False
        },
        "exit_rules": {
            "take_profit_pips": 50,
            "stop_loss_pips": 30,
            "trailing_stop": True
        }
    },
    "risk_management": {
        "max_position_size": 0.5,  # lots
        "max_daily_loss": 500,     # USD
        "max_trades_per_day": 5,
        "risk_reward_ratio": 2,
        "allow_overnight": False,
        "max_drawdown": 0.1        # 10% of account
    }
}

RESOURCE_REQUIREMENTS = {
    "min_ram_gb": 16,
    "min_cpu_cores": 4,
    "min_python_version": (3, 8),
    "required_packages": [
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "MetaTrader5"
    ]
}

# Add new API configuration after existing CONFIG blocks
API_CONFIG = {
    "market_data": {
        "mt5": {
            "timeout": 30,
            "max_retries": 3
        },
        "alpha_vantage": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://www.alphavantage.co/query",
            "timeout": 10
        }
    },
    "news": {
        "reuters": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://api.reuters.com/v2",
            "timeout": 5
        },
        "trading_central": {
            "api_key": "YOUR_API_KEY",
            "base_url": "https://api.tradingcentral.com",
            "timeout": 5
        }
    }
}

ORDER_BOOK_CONFIG = {
    'max_depth': 10,          # Maximum depth levels to analyze
    'update_interval': 0.1,   # Seconds between updates
    'imbalance_threshold': 0.2,  # Minimum imbalance to trigger signal
    'volume_levels': [0.25, 0.5, 0.75],  # Volume profile analysis levels
    'liquidity_threshold': 100  # Minimum liquidity required
}

class ComplianceCheck:
    def __init__(self):
        self.kyc_status = False
        self.trading_enabled = False
        self.compliance_logs = []
        
    def verify_kyc(self, user_id: int) -> bool:
        """Simulate KYC verification with broker"""
        # In real implementation, this would call broker's API
        self.kyc_status = True  # Simulated successful verification
        self.log_compliance_event("KYC verification completed")
        return self.kyc_status
    
    def check_trading_limits(self, position_size: float, account_info: Dict[str, Any]) -> bool:
        """Check if trade complies with position and loss limits"""
        daily_loss = self._calculate_daily_loss(account_info)
        if daily_loss > CONFIG["max_daily_loss"]:
            self.log_compliance_event("Daily loss limit exceeded")
            return False
        if position_size > CONFIG["max_position_size"]:
            self.log_compliance_event("Position size limit exceeded")
            return False
        return True
    
    def log_compliance_event(self, event: str):
        """Log compliance-related events"""
        timestamp = datetime.now().isoformat()
        self.compliance_logs.append({
            "timestamp": timestamp,
            "event": event
        })
        logging.info(f"Compliance event: {event}")

    def _calculate_daily_loss(self, account_info: Dict[str, Any]) -> float:
        """Calculate current day's trading loss"""
        # This would normally calculate from actual trading data
        return 0.0  # Placeholder return

class ResourceCheck:
    @staticmethod
    def check_system_resources() -> Tuple[bool, str]:
        """Verify system meets minimum requirements"""
        # Check Python version
        current_version = sys.version_info[:2]
        if current_version < RESOURCE_REQUIREMENTS["min_python_version"]:
            return False, f"Python version {current_version} not supported"
            
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < RESOURCE_REQUIREMENTS["min_ram_gb"]:
            return False, f"Insufficient RAM: {ram_gb:.1f}GB"
            
        # Check CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        if cpu_cores < RESOURCE_REQUIREMENTS["min_cpu_cores"]:
            return False, f"Insufficient CPU cores: {cpu_cores}"
            
        return True, "System requirements met"
    
    @staticmethod
    def check_packages() -> Tuple[bool, str]:
        """Verify required packages are installed"""
        missing_packages = []
        for package in RESOURCE_REQUIREMENTS["required_packages"]:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"
        return True, "All required packages found"

class UserPreferences:
    def __init__(self, default_preferences: Dict[str, Any] = USER_PREFERENCES):
        self.preferences = default_preferences.copy()
        self.load_saved_preferences()
    
    def load_saved_preferences(self):
        """Load user preferences from file if exists"""
        try:
            with open('user_preferences.json', 'r') as f:
                saved_prefs = json.load(f)
                self.preferences.update(saved_prefs)
        except FileNotFoundError:
            logging.info("No saved preferences found, using defaults")
    
    def save_preferences(self):
        """Save current preferences to file"""
        with open('user_preferences.json', 'w') as f:
            json.dump(self.preferences, f)
    
    def update_preference(self, category: str, setting: str, value: Any) -> bool:
        """Update specific preference setting"""
        try:
            if category in self.preferences and setting in self.preferences[category]:
                self.preferences[category][setting] = value
                self.save_preferences()
                return True
            return False
        except Exception as e:
            logging.error(f"Error updating preference: {str(e)}")
            return False

class TestSupport:
    """Test support functionality for development"""
    _instance: ClassVar[Optional['TestSupport']] = None
    
    def __init__(self):
        self.mock_data: Dict[str, Any] = {}
        self.is_test_mode = False
    
    @classmethod
    def get_instance(cls) -> 'TestSupport':
        if cls._instance is None:
            cls._instance = TestSupport()
        return cls._instance
    
    def enable_test_mode(self):
        self.is_test_mode = True
        logging.info("Test mode enabled")
    
    def set_mock_data(self, key: str, value: Any):
        self.mock_data[key] = value

class MarketDataAPI:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        
    async def get_gold_price(self, source: str = "mt5") -> Optional[float]:
        """Fetch gold price from specified source"""
        if source == "mt5":
            return await self._get_mt5_gold_price()
        elif source == "alpha_vantage":
            return await self._get_alphavantage_gold_price()
        return None
        
    async def _get_mt5_gold_price(self) -> Optional[float]:
        """Get gold price from MT5"""
        try:
            symbol_info = mt5.symbol_info("XAUUSD")
            if symbol_info is None:
                return None
            return symbol_info.bid
        except Exception as e:
            logging.error(f"MT5 gold price error: {str(e)}")
            return None
            
    async def _get_alphavantage_gold_price(self) -> Optional[float]:
        """Get gold price from Alpha Vantage"""
        url = f"{API_CONFIG['market_data']['alpha_vantage']['base_url']}"
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": "XAU",
            "to_currency": "USD",
            "apikey": API_CONFIG["market_data"]["alpha_vantage"]["api_key"]
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        except Exception as e:
            logging.error(f"Alpha Vantage API error: {str(e)}")
            return None

class NewsAPI:
    def __init__(self):
        self.news_cache = []
        self.last_news_update = None
        
    async def get_gold_news(self) -> List[Dict[str, Any]]:
        """Fetch gold-related news"""
        if not self._should_update_news():
            return self.news_cache
            
        reuters_news = await self._get_reuters_news()
        if reuters_news:
            self.news_cache = reuters_news
            self.last_news_update = datetime.now()
            
        return self.news_cache
        
    async def _get_reuters_news(self) -> List[Dict[str, Any]]:
        """Fetch news from Reuters API"""
        url = f"{API_CONFIG['news']['reuters']['base_url']}/news/search"
        headers = {"Authorization": f"Bearer {API_CONFIG['news']['reuters']['api_key']}"}
        params = {
            "keyword": "gold",
            "limit": 10,
            "sort": "newest"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    data = await response.json()
                    return data.get("results", [])
        except Exception as e:
            logging.error(f"Reuters API error: {str(e)}")
            return []
            
    def _should_update_news(self) -> bool:
        """Check if news cache should be updated"""
        if not self.last_news_update:
            return True
        return datetime.now() - self.last_news_update > timedelta(minutes=15)

@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    liquidity: float
    regime: str
    timestamp: datetime

class MarketRegime(Enum):
    TRENDING = auto()
    RANGING = auto()
    VOLATILE = auto()
    QUIET = auto()

class MarketAnalyzer:
    def __init__(self):
        self.conditions_history: List[MarketCondition] = []
        self.current_regime: MarketRegime = MarketRegime.QUIET
        self.volatility_threshold = 0.15
        
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            volatility = self._calculate_volatility(data)
            trend_strength = self._calculate_trend_strength(data)
            liquidity = self._analyze_liquidity(data)
            regime = self._determine_regime(volatility, trend_strength)
            
            condition = MarketCondition(
                volatility=volatility,
                trend_strength=trend_strength,
                liquidity=liquidity,
                regime=regime.name,
                timestamp=datetime.now()
            )
            
            self.conditions_history.append(condition)
            return condition
            
        except Exception as e:
            logging.error(f"Market analysis error: {str(e)}")
            return None
            
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        return data['close'].pct_change().std()
        
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        # Simplified ADX calculation
        return data['close'].rolling(14).std() / data['close'].mean()
        
    def _analyze_liquidity(self, data: pd.DataFrame) -> float:
        """Analyze market liquidity"""
        return data['volume'].mean() if 'volume' in data else 1.0
        
    def _determine_regime(self, volatility: float, trend_strength: float) -> MarketRegime:
        """Determine current market regime"""
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        elif trend_strength > 0.3:
            return MarketRegime.TRENDING
        elif trend_strength < 0.1:
            return MarketRegime.RANGING
        return MarketRegime.QUIET

class ScalabilityManager:
    def __init__(self):
        self.resource_usage = {
            'cpu': 0.0,
            'memory': 0.0,
            'network': 0.0
        }
        self.performance_metrics = {
            'processing_time': [],
            'api_latency': [],
            'queue_size': 0
        }
        
    async def monitor_resources(self):
        """Monitor system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.resource_usage.update({
                'cpu': cpu_percent,
                'memory': memory.percent,
                'network': len(asyncio.all_tasks())
            })
            
            if cpu_percent > 80 or memory.percent > 80:
                logging.warning("High resource usage detected")
                await self.optimize_resources()
                
        except Exception as e:
            logging.error(f"Resource monitoring error: {str(e)}")
            
    async def optimize_resources(self):
        """Optimize resource usage based on current conditions"""
        try:
            # Implement resource optimization logic
            if self.resource_usage['cpu'] > 80:
                await self.reduce_processing_load()
            if self.resource_usage['memory'] > 80:
                await self.clear_caches()
                
        except Exception as e:
            logging.error(f"Resource optimization error: {str(e)}")
            
    async def reduce_processing_load(self):
        """Reduce processing load during high usage"""
        # Implementation would include load reduction logic
        pass
        
    async def clear_caches(self):
        """Clear unnecessary caches"""
        # Implementation would include cache clearing logic
        pass

class Strategy(Protocol):
    """Protocol for trading strategy implementations"""
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]: ...
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def set_parameters(self, params: Dict[str, Any]) -> None: ...

class DataSource(Protocol):
    """Protocol for data source implementations"""
    def get_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame: ...
    def is_available(self) -> bool: ...

class BacktestEngine(ABC):
    """Abstract base class for backtesting engines"""
    @abstractmethod
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with given strategy and data"""
        pass
    
    @abstractmethod
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze backtest results"""
        pass

class PluginManager:
    """Manage bot plugins and extensions"""
    def __init__(self):
        self.strategies: Dict[str, Type[Strategy]] = {}
        self.data_sources: Dict[str, Type[DataSource]] = {}
        self.backtest_engines: Dict[str, Type[BacktestEngine]] = {}
        
    def register_strategy(self, name: str, strategy_class: Type[Strategy]):
        """Register a new trading strategy"""
        self.strategies[name] = strategy_class
        logging.info(f"Registered strategy: {name}")
        
    def register_data_source(self, name: str, source_class: Type[DataSource]):
        """Register a new data source"""
        self.data_sources[name] = source_class
        logging.info(f"Registered data source: {name}")
        
    def register_backtest_engine(self, name: str, engine_class: Type[BacktestEngine]):
        """Register a new backtest engine"""
        self.backtest_engines[name] = engine_class
        logging.info(f"Registered backtest engine: {name}")

class CustomizationManager:
    """Manage user customizations and configurations"""
    def __init__(self):
        self.custom_indicators: Dict[str, Any] = {}
        self.custom_rules: Dict[str, Any] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        
    def register_indicator(self, name: str, calculation_func: callable):
        """Register custom technical indicator"""
        self.custom_indicators[name] = calculation_func
        
    def register_rule(self, name: str, evaluation_func: callable):
        """Register custom trading rule"""
        self.custom_rules[name] = evaluation_func
        
    def save_template(self, name: str, config: Dict[str, Any]):
        """Save strategy template"""
        self.templates[name] = config.copy()
        
    def load_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Load strategy template"""
        return self.templates.get(name)

class OrderBookAnalyzer:
    """Analyze order book for price prediction"""
    def __init__(self):
        self.depth_cache = {
            'asks': {},  # price -> volume
            'bids': {},
            'timestamp': None
        }
        self.recent_imbalances = []
        self.vwap_levels = []
        
    def update_depth(self, depth_data: Dict[str, Dict[float, float]]):
        """Update order book depth cache"""
        if not depth_data.get('asks') or not depth_data.get('bids'):
            return
            
        self.depth_cache = {
            'asks': {float(p): float(v) for p, v in depth_data['asks'].items()},
            'bids': {float(p): float(v) for p, v in depth_data['bids'].items()},
            'timestamp': datetime.now()
        }
        
    def analyze_depth(self) -> Dict[str, Any]:
        """Analyze current order book state"""
        if not self._is_depth_valid():
            return {'valid': False}
            
        # Calculate order book imbalance
        imbalance = self._calculate_imbalance()
        
        # Analyze volume distribution
        volume_profile = self._analyze_volume_profile()
        
        # Calculate VWAP levels
        vwap = self._calculate_vwap_levels()
        
        # Detect potential walls
        walls = self._detect_price_walls()
        
        return {
            'valid': True,
            'imbalance': imbalance,
            'volume_profile': volume_profile,
            'vwap_levels': vwap,
            'walls': walls,
            'timestamp': self.depth_cache['timestamp']
        }
        
    def _is_depth_valid(self) -> bool:
        """Check if depth data is valid and recent"""
        if not self.depth_cache['timestamp']:
            return False
            
        age = (datetime.now() - self.depth_cache['timestamp']).total_seconds()
        return age < ORDER_BOOK_CONFIG['update_interval'] * 10
        
    def _calculate_imbalance(self) -> Dict[str, float]:
        """Calculate order book imbalance metrics"""
        bid_volume = sum(self.depth_cache['bids'].values())
        ask_volume = sum(self.depth_cache['asks'].values())
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return {'raw': 0.0, 'normalized': 0.0}
            
        raw_imbalance = (bid_volume - ask_volume) / total_volume
        
        # Track recent imbalances
        self.recent_imbalances.append(raw_imbalance)
        if len(self.recent_imbalances) > 20:
            self.recent_imbalances.pop(0)
            
        # Calculate normalized imbalance
        normalized_imbalance = (
            raw_imbalance / max(abs(min(self.recent_imbalances)),
                              abs(max(self.recent_imbalances)), 1)
        )
        
        return {
            'raw': raw_imbalance,
            'normalized': normalized_imbalance
        }
        
    def _analyze_volume_profile(self) -> Dict[str, Any]:
        """Analyze volume distribution across price levels"""
        all_volumes = sorted(
            list(self.depth_cache['asks'].values()) +
            list(self.depth_cache['bids'].values())
        )
        
        volume_levels = {}
        for percentile in ORDER_BOOK_CONFIG['volume_levels']:
            idx = int(len(all_volumes) * percentile)
            if idx < len(all_volumes):
                volume_levels[str(percentile)] = all_volumes[idx]
            
        return {
            'levels': volume_levels,
            'max_volume': max(all_volumes) if all_volumes else 0
        }
        
    def _calculate_vwap_levels(self) -> List[float]:
        """Calculate VWAP at different depth levels"""
        vwap_levels = []
        
        for depth in range(1, ORDER_BOOK_CONFIG['max_depth'] + 1):
            bid_vwap = self._calculate_side_vwap('bids', depth)
            ask_vwap = self._calculate_side_vwap('asks', depth)
            
            if bid_vwap and ask_vwap:
                mid_vwap = (bid_vwap + ask_vwap) / 2
                vwap_levels.append(mid_vwap)
                
        self.vwap_levels = vwap_levels
        return vwap_levels
        
    def _calculate_side_vwap(self, side: str, depth: int) -> Optional[float]:
        """Calculate VWAP for one side of the book"""
        orders = list(self.depth_cache[side].items())[:depth]
        if not orders:
            return None
            
        total_volume = sum(volume for _, volume in orders)
        if total_volume == 0:
            return None
            
        vwap = sum(price * volume for price, volume in orders) / total_volume
        return vwap
        
    def _detect_price_walls(self) -> List[Dict[str, Any]]:
        """Detect significant liquidity walls"""
        walls = []
        
        for side in ['bids', 'asks']:
            orders = sorted(
                self.depth_cache[side].items(),
                key=lambda x: float(x[1]),
                reverse=True
            )
            
            # Find price levels with significantly higher volume
            avg_volume = sum(v for _, v in orders) / len(orders) if orders else 0
            
            for price, volume in orders:
                if volume > avg_volume * 2:  # Significant wall
                    walls.append({
                        'side': side,
                        'price': price,
                        'volume': volume,
                        'strength': volume / avg_volume
                    })
                    
        return sorted(walls, key=lambda x: x['strength'], reverse=True)
        
    def predict_price_movement(self) -> Optional[Dict[str, Any]]:
        """Predict potential price movement based on order book"""
        if not self._is_depth_valid():
            return None
            
        analysis = self.analyze_depth()
        if not analysis['valid']:
            return None
            
        # Combined analysis for price prediction
        imbalance = analysis['imbalance']['normalized']
        walls = analysis['walls']
        
        prediction = {
            'direction': None,
            'strength': 0.0,
            'confidence': 0.0,
            'resistance': None,
            'support': None
        }
        
        # Predict based on imbalance
        if abs(imbalance) > ORDER_BOOK_CONFIG['imbalance_threshold']:
            prediction['direction'] = 'up' if imbalance > 0 else 'down'
            prediction['strength'] = abs(imbalance)
            
        # Identify support/resistance from walls
        for wall in walls:
            if wall['strength'] > 2:  # Significant wall
                if wall['side'] == 'asks':
                    prediction['resistance'] = wall['price']
                else:
                    prediction['support'] = wall['price']
                    
        # Calculate confidence
        signals = [
            abs(imbalance) > ORDER_BOOK_CONFIG['imbalance_threshold'],
            bool(prediction['resistance'] or prediction['support']),
            bool(analysis['volume_profile']['max_volume'] > 
                ORDER_BOOK_CONFIG['liquidity_threshold'])
        ]
        prediction['confidence'] = sum(signals) / len(signals)
        
        return prediction

class GoldTradingBot:
    def __init__(self):
        self.compliance = ComplianceCheck()
        self.initialized = False
        self.user_preferences = UserPreferences()
        self.resource_check = ResourceCheck()
        self.ml_models = {}
        self.test_support = TestSupport.get_instance()
        self.market_data_api = MarketDataAPI()
        self.news_api = NewsAPI()
        self.market_analyzer = MarketAnalyzer()
        self.scalability_manager = ScalabilityManager()
        self.plugin_manager = PluginManager()
        self.customization_manager = CustomizationManager()
        self.active_strategies: Dict[str, Strategy] = {}
        self.order_book = OrderBookAnalyzer()
        logging.basicConfig(level=logging.INFO)
        
        # Register AMA strategy
        self.plugin_manager.register_strategy(
            'adaptive_ma',
            AMAStrategy
        )
        
        # Register EMA Cross strategy
        self.plugin_manager.register_strategy(
            'ema_cross',
            EMACrossStrategy
        )
        
        # Register Bollinger Bands + ADX strategy
        self.plugin_manager.register_strategy(
            'bollinger_adx',
            BollingerADXStrategy
        )
        
        # Register Range Trading strategy
        self.plugin_manager.register_strategy(
            'range_trading',
            RangeStrategy
        )
        
        # Register Fibonacci strategy
        self.plugin_manager.register_strategy(
            'fibonacci',
            FibonacciStrategy
        )
        
        # Register Event-Driven strategy
        self.plugin_manager.register_strategy(
            'event_driven',
            EventDrivenStrategy
        )
        
        # Register Economic Events strategy
        self.plugin_manager.register_strategy(
            'economic_events',
            EconomicEventsStrategy
        )
        
        # Register Volatility Breakout strategy
        self.plugin_manager.register_strategy(
            'volatility_breakout',
            VolatilityBreakoutStrategy
        )
        
        # Register Scalping strategy
        self.plugin_manager.register_strategy(
            'scalping',
            ScalpingStrategy
        )
        
        # Add ATR Risk Manager
        self.risk_manager = ATRRiskManager()
        
    async def initialize(self) -> bool:
        """Initialize connection to MT5 and verify resources"""
        if self.test_support.is_test_mode:
            logging.info("Initializing in test mode")
            return True
            
        # Check system resources
        sys_check, sys_msg = ResourceCheck.check_system_resources()
        if not sys_check:
            logging.error(f"Resource check failed: {sys_msg}")
            return False
            
        # Check required packages
        pkg_check, pkg_msg = ResourceCheck.check_packages()
        if not pkg_check:
            logging.error(f"Package check failed: {pkg_msg}")
            return False
            
        # Initialize ML frameworks
        if not self._initialize_ml_components():
            return False
            
        # Verify API connections
        price = await self.market_data_api.get_gold_price()
        if price is None:
            logging.error("Failed to connect to market data APIs")
            return False
            
        news = await self.news_api.get_gold_news()
        if not news and not self.test_support.is_test_mode:
            logging.warning("News API connection failed")
            
        # Continue with existing initialization
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            return False
            
        if not self.compliance.verify_kyc(CONFIG["account_number"]):
            logging.error("KYC verification failed")
            return False
            
        # Initialize plugins
        await self._initialize_plugins()
        self.initialized = True
        return True
    
    def _initialize_ml_components(self) -> bool:
        """Initialize machine learning components"""
        try:
            import tensorflow as tf
            from sklearn.ensemble import RandomForestClassifier
            
            # Initialize ML models with basic configurations
            self.ml_models["trend_classifier"] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Set TensorFlow logging level
            tf.logging.set_verbosity(tf.logging.ERROR)
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize ML components: {str(e)}")
            return False
    
    async def _initialize_plugins(self):
        """Initialize registered plugins"""
        for name, strategy_class in self.plugin_manager.strategies.items():
            try:
                strategy = strategy_class()
                self.active_strategies[name] = strategy
                logging.info(f"Initialized strategy: {name}")
            except Exception as e:
                logging.error(f"Failed to initialize strategy {name}: {str(e)}")
    
    def get_gold_data(self) -> Optional[pd.DataFrame]:
        """Fetch current gold price data"""
        if not self.initialized:
            logging.error("Bot not initialized")
            return None
            
        if self.test_support.is_test_mode:
            mock_data = self.test_support.mock_data.get('gold_data')
            if mock_data is not None:
                return pd.DataFrame(mock_data)
                
        timeframe = self.user_preferences.preferences["platform"]["default_timeframe"]
        rates = mt5.copy_rates_from_pos(CONFIG["symbol"], timeframe, 0, 100)
        if rates is None:
            logging.error("Failed to fetch gold data")
            return None
            
        return pd.DataFrame(rates)
    
    def check_risk_limits(self, trade_size: float) -> bool:
        """Check if trade complies with user's risk preferences"""
        prefs = self.user_preferences.preferences["risk_management"]
        
        # Check position size
        if trade_size > prefs["max_position_size"]:
            logging.warning("Trade size exceeds user preference limit")
            return False
            
        # Check daily trade count
        if self._get_daily_trade_count() >= prefs["max_trades_per_day"]:
            logging.warning("Maximum daily trades reached")
            return False
            
        # Check if overnight trading is allowed
        if not prefs["allow_overnight"] and self._is_near_market_close():
            logging.warning("Overnight trading not allowed")
            return False
            
        return True
    
    def _get_daily_trade_count(self) -> int:
        """Get number of trades executed today"""
        # Implementation would track actual trades
        return 0
    
    def _is_near_market_close(self) -> bool:
        """Check if close to market closing time"""
        # Implementation would check actual market hours
        return False
    
    async def run_trading_cycle(self):
        """Execute trading cycle with market adaptation"""
        try:
            # Monitor system resources
            await self.scalability_manager.monitor_resources()
            
            # Get market data
            data = self.get_gold_data()
            if data is None:
                return
                
            # Analyze market conditions
            market_condition = self.market_analyzer.analyze_market_condition(data)
            if market_condition:
                # Adapt trading parameters based on market condition
                self._adapt_trading_parameters(market_condition)
                
            # Run active strategies
            for name, strategy in self.active_strategies.items():
                try:
                    analysis = strategy.analyze(data)
                    signals = strategy.generate_signals(analysis)
                    
                    for signal in signals:
                        if self._validate_signal(signal):
                            await self._execute_signal(signal)
                            
                except Exception as e:
                    logging.error(f"Strategy {name} error: {str(e)}")
            
            # Execute trading logic
            price = await self.market_data_api.get_gold_price()
            news = await self.news_api.get_gold_news()
            
            if price and self.check_risk_limits(1.0):
                position_size = self._calculate_adaptive_position_size(market_condition)
                logging.info(f"Adapted position size: {position_size}")
                # Implement trading logic here

            # Get order book data
            depth_data = mt5.market_book_get(CONFIG['symbol'])
            if depth_data:
                formatted_depth = {
                    'asks': {str(entry.price): entry.volume_real 
                            for entry in depth_data if entry.type == mt5.BOOK_TYPE_SELL},
                    'bids': {str(entry.price): entry.volume_real 
                            for entry in depth_data if entry.type == mt5.BOOK_TYPE_BUY}
                }
                self.order_book.update_depth(formatted_depth)
            
            # Get order book prediction
            prediction = self.order_book.predict_price_movement()
            if prediction and prediction['confidence'] > 0.7:
                logging.info(f"Order book prediction: {prediction}")
                # Adjust position sizing based on prediction confidence
                if self.position and prediction['direction']:
                    if ((self.position['direction'] == 'buy' and prediction['direction'] == 'down') or
                        (self.position['direction'] == 'sell' and prediction['direction'] == 'up')):
                        # Consider closing position
                        pass
                
        except Exception as e:
            logging.error(f"Trading cycle error: {str(e)}")
            
    def _adapt_trading_parameters(self, condition: MarketCondition):
        """Adapt trading parameters based on market conditions"""
        if condition.regime == MarketRegime.VOLATILE.name:
            self.user_preferences.preferences["risk_management"]["max_position_size"] *= 0.5
        elif condition.regime == MarketRegime.QUIET.name:
            self.user_preferences.preferences["risk_management"]["max_position_size"] *= 1.2
            
    def _calculate_adaptive_position_size(self, condition: MarketCondition) -> float:
        """Calculate position size based on market conditions"""
        base_size = self.user_preferences.preferences["risk_management"]["max_position_size"]
        volatility_factor = 1.0 / (1.0 + condition.volatility)
        liquidity_factor = min(1.0, condition.liquidity)
        
        return base_size * volatility_factor * liquidity_factor

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal with ATR-based risk management"""
        try:
            required_fields = ['direction', 'size', 'price']
            if not all(field in signal for field in required_fields):
                return False
                
            # Get current account balance
            account_balance = mt5.account_info().balance
            current_exposure = self._get_current_exposure()
            
            # Get latest market data
            data = self.get_gold_data()
            if data is None:
                return False
                
            # Calculate position parameters
            position_params = self.risk_manager.calculate_position_params(
                data=data,
                account_balance=account_balance,
                current_exposure=current_exposure
            )
            
            if position_params is None:
                return False
                
            # Adjust signal with ATR-based parameters
            signal['size'] = min(signal['size'], position_params['position_size'])
            signal['stop_loss'] = position_params['stop_loss']
            signal['take_profits'] = position_params['take_profits']
            signal['atr'] = position_params['atr']
            
            # Check risk limits
            if not self.check_risk_limits(signal['size']):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Signal validation error: {str(e)}")
            return False
    
    def _get_current_exposure(self) -> float:
        """Calculate current market exposure"""
        try:
            positions = mt5.positions_get(symbol=CONFIG['symbol'])
            if positions is None:
                return 0.0
                
            total_exposure = sum(pos.volume for pos in positions)
            return total_exposure
            
        except Exception as e:
            logging.error(f"Error calculating exposure: {str(e)}")
            return 0.0
    
    async def _execute_signal(self, signal: Dict[str, Any]):
        """Execute signal with ATR-based stops and targets"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": CONFIG['symbol'],
                "volume": signal['size'],
                "type": mt5.ORDER_TYPE_BUY if signal['direction'] == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": signal['price'],
                "sl": signal['stop_loss'],
                "tp": signal['take_profits'][0],  # Use first target
                "comment": f"ATR:{signal['atr']:.2f}"
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.comment}")
                return
                
            # Place additional take-profit orders if successful
            if len(signal['take_profits']) > 1:
                for tp_price in signal['take_profits'][1:]:
                    tp_request = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": CONFIG['symbol'],
                        "volume": signal['size'] * 0.5,  # Scale out
                        "type": mt5.ORDER_TYPE_SELL_LIMIT if signal['direction'] == 'buy' 
                               else mt5.ORDER_TYPE_BUY_LIMIT,
                        "price": tp_price,
                        "comment": f"TP_ATR:{signal['atr']:.2f}"
                    }
                    mt5.order_send(tp_request)
                    
        except Exception as e:
            logging.error(f"Signal execution error: {str(e)}")
            
    def run_backtest(self, strategy_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest for specified strategy"""
        try:
            strategy = self.active_strategies.get(strategy_name)
            if not strategy:
                raise ValueError(f"Strategy {strategy_name} not found")
                
            engine = list(self.plugin_manager.backtest_engines.values())[0]()
            data = self._get_historical_data(start_date, end_date)
            
            results = engine.run_backtest(strategy, data)
            analysis = engine.analyze_results(results)
            
            return analysis
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            return {}

    def shutdown(self):
        """Clean shutdown of bot"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logging.info("Bot shutdown completed")

# Development helper functions
def setup_development_env():
    """Configure development environment"""
    # Set up logging
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "debug.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logging.warning("python-dotenv not installed, skipping .env loading")

if __name__ == "__main__":
    setup_development_env()
    bot = GoldTradingBot()
    try:
        asyncio.run(bot.initialize())
        while True:
            asyncio.run(bot.run_trading_cycle())
            await asyncio.sleep(1)  # 1-second delay between cycles
    except KeyboardInterrupt:
        logging.info("Bot shutdown requested")
    except Exception as e:
        logging.error(f"Error during bot operation: {str(e)}")
    finally:
        bot.shutdown()

class AMAStrategy(Strategy):
    """Adaptive Moving Average strategy for gold trading"""
    def __init__(self):
        self.params = {
            'er_period': 10,        # Efficiency Ratio period
            'fast_period': 2,       # Fast EMA period
            'slow_period': 30,      # Slow EMA period
            'volatility_window': 20 # Window for volatility calculation
        }
        self.position = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data using AMA"""
        if len(data) < self.params['slow_period']:
            return {'valid': False}
            
        # Calculate price change and direction
        price_change = np.abs(data['close'].diff())
        direction = data['close'].diff()
        
        # Calculate Efficiency Ratio (ER)
        volatility = price_change.rolling(self.params['er_period']).sum()
        direction_movement = np.abs(data['close'].diff(self.params['er_period']))
        er = direction_movement / volatility
        er = er.fillna(0.0)
        
        # Calculate adaptive factor
        fast_sc = 2.0 / (self.params['fast_period'] + 1)
        slow_sc = 2.0 / (self.params['slow_period'] + 1)
        adaptive_factor = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate AMA
        ama = pd.Series(index=data.index, dtype=float)
        ama.iloc[0] = data['close'].iloc[0]
        
        for i in range(1, len(data)):
            ama.iloc[i] = ama.iloc[i-1] + adaptive_factor.iloc[i] * (
                data['close'].iloc[i] - ama.iloc[i-1]
            )
        
        # Calculate volatility
        volatility = data['close'].rolling(
            window=self.params['volatility_window']
        ).std()
        
        return {
            'valid': True,
            'ama': ama,
            'er': er,
            'volatility': volatility,
            'adaptive_factor': adaptive_factor,
            'last_close': data['close'].iloc[-1],
            'last_ama': ama.iloc[-1]
        }
    
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on AMA analysis"""
        if not analysis['valid']:
            return []
            
        signals = []
        last_close = analysis['last_close']
        last_ama = analysis['last_ama']
        last_volatility = analysis['volatility'].iloc[-1]
        
        # Dynamic position sizing based on volatility
        base_size = CONFIG['max_position_size']
        volatility_factor = 1.0 / (1.0 + last_volatility)
        position_size = base_size * volatility_factor
        
        # Generate signals based on price crossing AMA
        if self.position is None:  # No current position
            if last_close > last_ama:
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'price_above_ama'
                })
            elif last_close < last_ama:
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'price_below_ama'
                })
        else:  # Managing existing position
            if self.position['direction'] == 'buy' and last_close < last_ama:
                signals.append({
                    'direction': 'close_buy',
                    'size': self.position['size'],
                    'price': last_close,
                    'reason': 'price_below_ama'
                })
                self.position = None
            elif self.position['direction'] == 'sell' and last_close > last_ama:
                signals.append({
                    'direction': 'close_sell',
                    'size': self.position['size'],
                    'price': last_close,
                    'reason': 'price_above_ama'
                })
                self.position = None
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class EMACrossStrategy(Strategy):
    """Dynamic EMA Crossover strategy for gold trading"""
    def __init__(self):
        self.params = {
            'base_fast_period': 12,     # Base fast EMA period
            'base_slow_period': 26,     # Base slow EMA period
            'vol_window': 20,           # Volatility measurement window
            'vol_threshold': 0.15,      # Volatility threshold for adjustment
            'max_period_adjust': 0.5,   # Maximum period adjustment factor
            'min_period_adjust': -0.3   # Minimum period adjustment factor
        }
        self.position = None
        self.last_crossover = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data using dynamic EMA crossover"""
        if len(data) < self.params['base_slow_period'] + self.params['vol_window']:
            return {'valid': False}
            
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(
            window=self.params['vol_window']
        ).std()
        
        # Adjust EMA periods based on volatility
        vol_factor = self._calculate_volatility_factor(volatility.iloc[-1])
        fast_period = max(2, int(self.params['base_fast_period'] * (1 + vol_factor)))
        slow_period = max(5, int(self.params['base_slow_period'] * (1 + vol_factor)))
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(data['close'], fast_period)
        slow_ema = self._calculate_ema(data['close'], slow_period)
        
        # Detect crossovers
        crossover = self._detect_crossover(fast_ema, slow_ema)
        
        return {
            'valid': True,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'volatility': volatility,
            'crossover': crossover,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'last_close': data['close'].iloc[-1],
            'vol_factor': vol_factor
        }
        
    def _calculate_volatility_factor(self, current_vol: float) -> float:
        """Calculate period adjustment factor based on volatility"""
        if current_vol > self.params['vol_threshold']:
            # Higher volatility -> shorter periods
            return max(
                self.params['min_period_adjust'],
                -current_vol / self.params['vol_threshold']
            )
        else:
            # Lower volatility -> longer periods
            return min(
                self.params['max_period_adjust'],
                (self.params['vol_threshold'] - current_vol) / self.params['vol_threshold']
            )
            
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with specified period"""
        alpha = 2.0 / (period + 1)
        return data.ewm(alpha=alpha, adjust=False).mean()
        
    def _detect_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> Optional[str]:
        """Detect EMA crossovers"""
        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return None
            
        prev_diff = fast_ema.iloc[-2] - slow_ema.iloc[-2]
        curr_diff = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        
        if prev_diff < 0 and curr_diff > 0:
            return 'bullish'
        elif prev_diff > 0 and curr_diff < 0:
            return 'bearish'
        return None
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on EMA crossovers"""
        if not analysis['valid']:
            return []
            
        signals = []
        last_close = analysis['last_close']
        crossover = analysis['crossover']
        vol_factor = analysis['vol_factor']
        
        # Adjust position size based on volatility factor
        base_size = CONFIG['max_position_size']
        position_size = base_size * (1 - abs(vol_factor))  # Reduce size in extreme conditions
        
        # Generate signals based on crossovers
        if crossover and crossover != self.last_crossover:
            if crossover == 'bullish' and self.position is None:
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'ema_bullish_cross',
                    'fast_period': analysis['fast_period'],
                    'slow_period': analysis['slow_period']
                })
            elif crossover == 'bearish' and self.position is None:
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'ema_bearish_cross',
                    'fast_period': analysis['fast_period'],
                    'slow_period': analysis['slow_period']
                })
                
            # Close existing positions on opposite crossover
            elif self.position is not None:
                if (crossover == 'bearish' and self.position['direction'] == 'buy') or \
                   (crossover == 'bullish' and self.position['direction'] == 'sell'):
                    signals.append({
                        'direction': f"close_{self.position['direction']}",
                        'size': self.position['size'],
                        'price': last_close,
                        'reason': f'ema_{crossover}_cross'
                    })
                    self.position = None
                    
        self.last_crossover = crossover
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class BollingerADXStrategy(Strategy):
    """Bollinger Bands with ADX trend confirmation strategy"""
    def __init__(self):
        self.params = {
            'bb_period': 20,        # Bollinger Bands period
            'bb_std': 2,           # Number of standard deviations
            'adx_period': 14,      # ADX period
            'adx_threshold': 25,    # ADX trend strength threshold
            'pos_di_threshold': 20, # Positive DI threshold
            'neg_di_threshold': 20  # Negative DI threshold
        }
        self.position = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data using Bollinger Bands and ADX"""
        if len(data) < max(self.params['bb_period'], self.params['adx_period']):
            return {'valid': False}
            
        # Calculate Bollinger Bands
        middle_band = data['close'].rolling(window=self.params['bb_period']).mean()
        std_dev = data['close'].rolling(window=self.params['bb_period']).std()
        upper_band = middle_band + (std_dev * self.params['bb_std'])
        lower_band = middle_band - (std_dev * self.params['bb_std'])
        
        # Calculate ADX components
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.params['adx_period']).mean()
        
        # Calculate +DM and -DM
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth DM values
        plus_di = 100 * (plus_dm.rolling(window=self.params['adx_period']).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.params['adx_period']).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.params['adx_period']).mean()
        
        # Calculate %B (position within Bollinger Bands)
        percent_b = (data['close'] - lower_band) / (upper_band - lower_band)
        
        return {
            'valid': True,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'percent_b': percent_b,
            'last_close': data['close'].iloc[-1],
            'bb_width': (upper_band - lower_band) / middle_band
        }
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on Bollinger Bands and ADX"""
        if not analysis['valid']:
            return []
            
        signals = []
        last_close = analysis['last_close']
        last_adx = analysis['adx'].iloc[-1]
        last_plus_di = analysis['plus_di'].iloc[-1]
        last_minus_di = analysis['minus_di'].iloc[-1]
        last_percent_b = analysis['percent_b'].iloc[-1]
        bb_width = analysis['bb_width'].iloc[-1]
        
        # Calculate position size based on ADX strength and BB width
        base_size = CONFIG['max_position_size']
        adx_factor = min(1.0, last_adx / 50.0)  # Scale by ADX strength
        bb_factor = min(1.0, 1.0 / bb_width)    # Reduce size when bands widen
        position_size = base_size * adx_factor * bb_factor
        
        # Generate signals based on BB position and ADX confirmation
        if self.position is None:  # No current position
            if (last_percent_b < 0.1 and  # Price near lower band
                last_adx > self.params['adx_threshold'] and  # Strong trend
                last_plus_di > self.params['pos_di_threshold']):  # Bullish momentum
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'bb_adx_buy',
                    'adx': last_adx,
                    'percent_b': last_percent_b
                })
            elif (last_percent_b > 0.9 and  # Price near upper band
                  last_adx > self.params['adx_threshold'] and  # Strong trend
                  last_minus_di > self.params['neg_di_threshold']):  # Bearish momentum
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'bb_adx_sell',
                    'adx': last_adx,
                    'percent_b': last_percent_b
                })
        else:  # Managing existing position
            if self.position['direction'] == 'buy':
                if (last_percent_b > 0.8 or  # Price near upper band
                    last_minus_di > last_plus_di):  # Bearish crossover
                    signals.append({
                        'direction': 'close_buy',
                        'size': self.position['size'],
                        'price': last_close,
                        'reason': 'bb_adx_take_profit'
                    })
                    self.position = None
            elif self.position['direction'] == 'sell':
                if (last_percent_b < 0.2 or  # Price near lower band
                    last_plus_di > last_minus_di):  # Bullish crossover
                    signals.append({
                        'direction': 'close_sell',
                        'size': self.position['size'],
                        'price': last_close,
                        'reason': 'bb_adx_take_profit'
                    })
                    self.position = None
        
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class RangeStrategy(Strategy):
    """RSI Divergence and Stochastic Oscillator range trading strategy"""
    def __init__(self):
        self.params = {
            'rsi_period': 14,           # RSI calculation period
            'rsi_overbought': 70,       # RSI overbought threshold
            'rsi_oversold': 30,         # RSI oversold threshold
            'stoch_k_period': 14,       # Stochastic K period
            'stoch_d_period': 3,        # Stochastic D period
            'stoch_overbought': 80,     # Stochastic overbought threshold
            'stoch_oversold': 20,       # Stochastic oversold threshold
            'divergence_length': 5,     # Bars to check for divergence
            'confirmation_bars': 2      # Bars needed for confirmation
        }
        self.position = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data using RSI and Stochastic"""
        if len(data) < max(self.params['rsi_period'], 
                          self.params['stoch_k_period'] + self.params['stoch_d_period']):
            return {'valid': False}
            
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic Oscillator
        low_min = data['low'].rolling(window=self.params['stoch_k_period']).min()
        high_max = data['high'].rolling(window=self.params['stoch_k_period']).max()
        
        stoch_k = 100 * (data['close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=self.params['stoch_d_period']).mean()
        
        # Detect RSI divergence
        bullish_div = self._detect_bullish_divergence(data['close'], rsi)
        bearish_div = self._detect_bearish_divergence(data['close'], rsi)
        
        # Calculate range statistics
        volatility = data['close'].pct_change().std()
        avg_range = (data['high'] - data['low']).rolling(window=20).mean()
        
        return {
            'valid': True,
            'rsi': rsi,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'bullish_div': bullish_div,
            'bearish_div': bearish_div,
            'volatility': volatility,
            'avg_range': avg_range,
            'last_close': data['close'].iloc[-1],
            'last_rsi': rsi.iloc[-1],
            'last_stoch_k': stoch_k.iloc[-1],
            'last_stoch_d': stoch_d.iloc[-1]
        }
        
    def _detect_bullish_divergence(self, price: pd.Series, rsi: pd.Series) -> bool:
        """Detect bullish RSI divergence"""
        lookback = self.params['divergence_length']
        if len(price) < lookback or len(rsi) < lookback:
            return False
            
        # Check for lower lows in price but higher lows in RSI
        price_min = price.iloc[-lookback:].min()
        rsi_min = rsi.iloc[-lookback:].min()
        
        if (price.iloc[-1] < price_min and 
            rsi.iloc[-1] > rsi_min and 
            rsi.iloc[-1] < self.params['rsi_oversold']):
            return True
        return False
        
    def _detect_bearish_divergence(self, price: pd.Series, rsi: pd.Series) -> bool:
        """Detect bearish RSI divergence"""
        lookback = self.params['divergence_length']
        if len(price) < lookback or len(rsi) < lookback:
            return False
            
        # Check for higher highs in price but lower highs in RSI
        price_max = price.iloc[-lookback:].max()
        rsi_max = rsi.iloc[-lookback:].max()
        
        if (price.iloc[-1] > price_max and 
            rsi.iloc[-1] < rsi_max and 
            rsi.iloc[-1] > self.params['rsi_overbought']):
            return True
        return False
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on RSI divergence and Stochastic confirmation"""
        if not analysis['valid']:
            return []
            
        signals = []
        last_close = analysis['last_close']
        last_rsi = analysis['last_rsi']
        last_stoch_k = analysis['last_stoch_k']
        last_stoch_d = analysis['last_stoch_d']
        
        # Calculate position size based on volatility and average range
        base_size = CONFIG['max_position_size']
        vol_factor = 1.0 / (1.0 + analysis['volatility'])
        range_factor = min(1.0, analysis['avg_range'].iloc[-1] / last_close * 100)
        position_size = base_size * vol_factor * range_factor
        
        # Generate signals for range trading
        if self.position is None:  # No current position
            # Bullish signals
            if (analysis['bullish_div'] and  # RSI bullish divergence
                last_stoch_k < self.params['stoch_oversold'] and  # Stochastic oversold
                last_stoch_k > last_stoch_d):  # Stochastic bullish cross
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'range_bullish',
                    'rsi': last_rsi,
                    'stoch_k': last_stoch_k
                })
                
            # Bearish signals
            elif (analysis['bearish_div'] and  # RSI bearish divergence
                  last_stoch_k > self.params['stoch_overbought'] and  # Stochastic overbought
                  last_stoch_k < last_stoch_d):  # Stochastic bearish cross
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': last_close,
                    'reason': 'range_bearish',
                    'rsi': last_rsi,
                    'stoch_k': last_stoch_k
                })
                
        else:  # Managing existing position
            if self.position['direction'] == 'buy':
                if (last_stoch_k > self.params['stoch_overbought'] or  # Stochastic overbought
                    (last_rsi > self.params['rsi_overbought'] and  # RSI overbought
                     last_stoch_k < last_stoch_d)):  # Stochastic bearish cross
                    signals.append({
                        'direction': 'close_buy',
                        'size': self.position['size'],
                        'price': last_close,
                        'reason': 'range_take_profit'
                    })
                    self.position = None
                    
            elif self.position['direction'] == 'sell':
                if (last_stoch_k < self.params['stoch_oversold'] or  # Stochastic oversold
                    (last_rsi < self.params['rsi_oversold'] and  # RSI oversold
                     last_stoch_k > last_stoch_d)):  # Stochastic bullish cross
                    signals.append({
                        'direction': 'close_sell',
                        'size': self.position['size'],
                        'price': last_close,
                        'reason': 'range_take_profit'
                    })
                    self.position = None
        
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class FibonacciStrategy(Strategy):
    """Fibonacci retracement and breakout strategy"""
    def __init__(self):
        self.params = {
            'trend_period': 20,        # Period to identify trend
            'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786],  # Fibonacci levels
            'breakout_threshold': 0.002,  # 0.2% breakout confirmation
            'volume_factor': 1.5,      # Volume increase for breakout confirmation
            'sr_period': 50,           # Period for S/R identification
            'sr_threshold': 0.001      # 0.1% price cluster threshold
        }
        self.position = None
        self.fib_levels = None
        self.support_levels = []
        self.resistance_levels = []
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data using Fibonacci retracements and S/R levels"""
        if len(data) < max(self.params['trend_period'], self.params['sr_period']):
            return {'valid': False}
            
        # Calculate Fibonacci levels
        swing_high = data['high'].rolling(window=self.params['trend_period']).max()
        swing_low = data['low'].rolling(window=self.params['trend_period']).min()
        
        # Identify trend direction
        trend = 'up' if data['close'].iloc[-1] > data['close'].iloc[-self.params['trend_period']] else 'down'
        
        if trend == 'up':
            fib_range = swing_high.iloc[-1] - swing_low.iloc[-1]
            fib_levels = {level: swing_high.iloc[-1] - fib_range * level 
                         for level in self.params['fib_levels']}
        else:
            fib_range = swing_high.iloc[-1] - swing_low.iloc[-1]
            fib_levels = {level: swing_low.iloc[-1] + fib_range * level 
                         for level in self.params['fib_levels']}
        
        # Identify support and resistance levels
        self._update_support_resistance(data)
        
        # Check for breakouts
        breakouts = self._detect_breakouts(data, fib_levels)
        
        # Calculate price distance to nearest levels
        current_price = data['close'].iloc[-1]
        nearest_fib = self._find_nearest_level(current_price, list(fib_levels.values()))
        nearest_support = self._find_nearest_level(current_price, self.support_levels)
        nearest_resistance = self._find_nearest_level(current_price, self.resistance_levels)
        
        return {
            'valid': True,
            'trend': trend,
            'fib_levels': fib_levels,
            'breakouts': breakouts,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'nearest_fib': nearest_fib,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'last_close': current_price,
            'volume': data['tick_volume'].iloc[-1] if 'tick_volume' in data else None
        }
        
    def _update_support_resistance(self, data: pd.DataFrame):
        """Identify support and resistance levels using price clusters"""
        price_clusters = []
        window = self.params['sr_period']
        
        # Find price clusters
        for i in range(window, len(data)):
            subset = data.iloc[i-window:i]
            highs = subset['high'].value_counts().sort_index()
            lows = subset['low'].value_counts().sort_index()
            
            # Identify price levels with significant touches
            for prices in [highs, lows]:
                for price, count in prices.items():
                    if count >= window * 0.1:  # At least 10% of periods
                        if not any(abs(p - price) < self.params['sr_threshold'] for p in price_clusters):
                            price_clusters.append(price)
        
        current_price = data['close'].iloc[-1]
        self.support_levels = [p for p in price_clusters if p < current_price]
        self.resistance_levels = [p for p in price_clusters if p > current_price]
        
    def _detect_breakouts(self, data: pd.DataFrame, fib_levels: Dict[float, float]) -> List[Dict[str, Any]]:
        """Detect breakouts of Fibonacci and S/R levels"""
        breakouts = []
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        current_volume = data['tick_volume'].iloc[-1] if 'tick_volume' in data else None
        avg_volume = data['tick_volume'].rolling(20).mean().iloc[-1] if 'tick_volume' in data else None
        
        # Check Fibonacci level breakouts
        for level, price in fib_levels.items():
            if (prev_price < price < current_price or 
                prev_price > price > current_price):
                breakouts.append({
                    'type': 'fibonacci',
                    'level': level,
                    'price': price,
                    'direction': 'up' if current_price > price else 'down'
                })
        
        # Check S/R level breakouts
        for level in self.support_levels + self.resistance_levels:
            if (prev_price < level < current_price or 
                prev_price > level > current_price):
                # Confirm with volume if available
                volume_confirmed = (current_volume > avg_volume * self.params['volume_factor'] 
                                 if current_volume and avg_volume else True)
                if volume_confirmed:
                    breakouts.append({
                        'type': 'sr',
                        'price': level,
                        'direction': 'up' if current_price > level else 'down'
                    })
        
        return breakouts
        
    def _find_nearest_level(self, price: float, levels: List[float]) -> Optional[float]:
        """Find the nearest price level"""
        if not levels:
            return None
        return min(levels, key=lambda x: abs(x - price))
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on Fibonacci and breakout analysis"""
        if not analysis['valid']:
            return []
            
        signals = []
        current_price = analysis['last_close']
        
        # Calculate position size based on proximity to key levels
        base_size = CONFIG['max_position_size']
        level_proximity = min(
            abs(current_price - analysis['nearest_support']) / current_price if analysis['nearest_support'] else 1,
            abs(current_price - analysis['nearest_resistance']) / current_price if analysis['nearest_resistance'] else 1
        )
        position_size = base_size * (1 - level_proximity)  # Larger size when closer to levels
        
        # Generate signals based on breakouts
        if not self.position:  # No current position
            for breakout in analysis['breakouts']:
                if breakout['direction'] == 'up' and breakout['type'] == 'sr':
                    signals.append({
                        'direction': 'buy',
                        'size': position_size,
                        'price': current_price,
                        'reason': f'breakout_{breakout["type"]}',
                        'level': breakout['price']
                    })
                elif breakout['direction'] == 'down' and breakout['type'] == 'sr':
                    signals.append({
                        'direction': 'sell',
                        'size': position_size,
                        'price': current_price,
                        'reason': f'breakout_{breakout["type"]}',
                        'level': breakout['price']
                    })
        
        # Manage existing positions
        else:
            if self.position['direction'] == 'buy':
                # Close long if price breaks below support or Fibonacci level
                for breakout in analysis['breakouts']:
                    if breakout['direction'] == 'down':
                        signals.append({
                            'direction': 'close_buy',
                            'size': self.position['size'],
                            'price': current_price,
                            'reason': f'breakdown_{breakout["type"]}'
                        })
                        self.position = None
                        break
                        
            elif self.position['direction'] == 'sell':
                # Close short if price breaks above resistance or Fibonacci level
                for breakout in analysis['breakouts']:
                    if breakout['direction'] == 'up':
                        signals.append({
                            'direction': 'close_sell',
                            'size': self.position['size'],
                            'price': current_price,
                            'reason': f'breakout_{breakout["type"]}'
                        })
                        self.position = None
                        break
        
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class EventDrivenStrategy(Strategy):
    """Event-driven trading strategy using news and sentiment analysis"""
    def __init__(self):
        self.params = {
            'sentiment_threshold': 0.6,    # Minimum sentiment score to trigger
            'news_lookback': 30,          # Minutes to look back for news
            'volume_surge_factor': 2.0,    # Volume increase to confirm impact
            'position_hold_time': 60,      # Minutes to hold event-based positions
            'keywords': {                  # Keywords and their importance weights
                'inflation': 0.8,
                'fed': 0.9,
                'interest rate': 0.85,
                'central bank': 0.8,
                'gdp': 0.7,
                'recession': 0.75,
                'geopolitical': 0.7
            }
        }
        self.position = None
        self.last_news_check = None
        self.position_entry_time = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data and news sentiment"""
        if len(data) < 20:  # Minimum data requirement
            return {'valid': False}
            
        # Get recent news and calculate sentiment
        news_data = self._fetch_recent_news()
        sentiment_scores = self._analyze_sentiment(news_data)
        
        # Calculate market impact indicators
        volume_ma = data['tick_volume'].rolling(20).mean() if 'tick_volume' in data else None
        current_volume = data['tick_volume'].iloc[-1] if 'tick_volume' in data else None
        
        # Detect volatility surge
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        vol_surge = volatility > data['close'].pct_change().rolling(20).std().mean() * 2
        
        # Identify significant events
        significant_events = self._identify_significant_events(news_data, sentiment_scores)
        
        # Calculate event impact score
        impact_score = self._calculate_impact_score(
            significant_events,
            vol_surge,
            current_volume,
            volume_ma
        )
        
        return {
            'valid': True,
            'sentiment_scores': sentiment_scores,
            'significant_events': significant_events,
            'impact_score': impact_score,
            'volatility': volatility,
            'volume_surge': vol_surge,
            'last_close': data['close'].iloc[-1],
            'current_volume': current_volume,
            'avg_volume': volume_ma.iloc[-1] if volume_ma is not None else None
        }
        
    def _fetch_recent_news(self) -> List[Dict[str, Any]]:
        """Fetch recent news articles about gold"""
        if not self.last_news_check:
            self.last_news_check = datetime.now()
            return []
            
        # Filter news within lookback period
        lookback_time = datetime.now() - timedelta(minutes=self.params['news_lookback'])
        
        try:
            news_api = NewsAPI()  # Using existing NewsAPI class
            news = asyncio.get_event_loop().run_until_complete(news_api.get_gold_news())
            
            recent_news = [
                article for article in news
                if datetime.fromisoformat(article['published_at']) > lookback_time
            ]
            
            self.last_news_check = datetime.now()
            return recent_news
        except Exception as e:
            logging.error(f"Error fetching news: {str(e)}")
            return []
        
    def _analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Analyze sentiment of news articles"""
        try:
            from textblob import TextBlob
            
            sentiment_scores = []
            for article in news_data:
                # Analyze title and content
                title_blob = TextBlob(article['title'])
                content_blob = TextBlob(article['content']) if 'content' in article else TextBlob('')
                
                # Weight title more heavily than content
                sentiment = (title_blob.sentiment.polarity * 0.6 + 
                           content_blob.sentiment.polarity * 0.4)
                
                # Adjust sentiment based on keyword presence
                keyword_score = self._calculate_keyword_score(
                    article['title'] + ' ' + article.get('content', '')
                )
                
                sentiment_scores.append({
                    'article_id': article['id'],
                    'sentiment': sentiment,
                    'keyword_score': keyword_score,
                    'timestamp': article['published_at']
                })
                
            return sentiment_scores
        except Exception as e:
            logging.error(f"Sentiment analysis error: {str(e)}")
            return []
        
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate importance score based on keyword presence"""
        text = text.lower()
        score = 0.0
        matches = 0
        
        for keyword, weight in self.params['keywords'].items():
            if keyword in text:
                score += weight
                matches += 1
                
        return score / max(matches, 1)  # Average score if matches found
        
    def _identify_significant_events(self, news_data: List[Dict[str, Any]], 
                                   sentiment_scores: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify significant market-moving events"""
        significant_events = []
        
        for article, sentiment in zip(news_data, sentiment_scores):
            # Check if sentiment exceeds threshold
            if abs(sentiment['sentiment']) >= self.params['sentiment_threshold']:
                event = {
                    'id': article['id'],
                    'title': article['title'],
                    'sentiment': sentiment['sentiment'],
                    'keyword_score': sentiment['keyword_score'],
                    'timestamp': sentiment['timestamp'],
                    'impact': 'high' if abs(sentiment['sentiment']) > 0.8 else 'medium'
                }
                significant_events.append(event)
                
        return significant_events
        
    def _calculate_impact_score(self, events: List[Dict[str, Any]], vol_surge: bool,
                              current_volume: Optional[float], volume_ma: Optional[float]) -> float:
        """Calculate overall market impact score"""
        if not events:
            return 0.0
            
        # Base score from events
        event_score = sum(
            abs(event['sentiment']) * event['keyword_score']
            for event in events
        ) / len(events)
        
        # Adjust for market conditions
        if vol_surge:
            event_score *= 1.5
            
        if current_volume and volume_ma:
            volume_factor = current_volume / volume_ma
            if volume_factor > self.params['volume_surge_factor']:
                event_score *= volume_factor / self.params['volume_surge_factor']
                
        return min(1.0, event_score)  # Cap at 1.0
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on event analysis"""
        if not analysis['valid'] or not analysis['significant_events']:
            return []
            
        signals = []
        current_price = analysis['last_close']
        impact_score = analysis['impact_score']
        
        # Check if enough time has passed since last position
        if self.position and self.position_entry_time:
            hold_time = datetime.now() - self.position_entry_time
            if hold_time.total_seconds() / 60 < self.params['position_hold_time']:
                return []
                
        # Calculate position size based on impact score
        base_size = CONFIG['max_position_size']
        position_size = base_size * impact_score
        
        # Generate signals based on event analysis
        if not self.position:  # No current position
            net_sentiment = sum(
                event['sentiment'] for event in analysis['significant_events']
            ) / len(analysis['significant_events'])
            
            if net_sentiment > self.params['sentiment_threshold']:
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': current_price,
                    'reason': 'positive_event',
                    'impact_score': impact_score,
                    'events': [e['title'] for e in analysis['significant_events']]
                })
                self.position_entry_time = datetime.now()
                
            elif net_sentiment < -self.params['sentiment_threshold']:
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': current_price,
                    'reason': 'negative_event',
                    'impact_score': impact_score,
                    'events': [e['title'] for e in analysis['significant_events']]
                })
                self.position_entry_time = datetime.now()
                
        else:  # Managing existing position
            # Close position if sentiment reverses
            current_sentiment = sum(
                event['sentiment'] for event in analysis['significant_events']
            ) / len(analysis['significant_events'])
            
            if ((self.position['direction'] == 'buy' and 
                 current_sentiment < -self.params['sentiment_threshold']) or
                (self.position['direction'] == 'sell' and 
                 current_sentiment > self.params['sentiment_threshold'])):
                signals.append({
                    'direction': f"close_{self.position['direction']}",
                    'size': self.position['size'],
                    'price': current_price,
                    'reason': 'sentiment_reversal',
                    'impact_score': impact_score
                })
                self.position = None
                self.position_entry_time = None
                
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class EconomicEventsStrategy(Strategy):
    """Strategy for trading gold based on economic and geopolitical events"""
    def __init__(self):
        self.params = {
            'event_impact_window': 120,    # Minutes to monitor event impact
            'correlation_window': 30,      # Days for correlation analysis
            'min_correlation': 0.6,        # Minimum correlation threshold
            'risk_adjust_factor': 0.8,     # Position size adjustment for high-risk events
            'event_categories': {          # Event categories and their weights
                'monetary_policy': {
                    'weight': 1.0,
                    'keywords': ['fed', 'rate decision', 'central bank', 'monetary policy'],
                    'risk_level': 'high'
                },
                'inflation': {
                    'weight': 0.9,
                    'keywords': ['cpi', 'inflation', 'price index', 'deflation'],
                    'risk_level': 'high'
                },
                'geopolitical': {
                    'weight': 0.8,
                    'keywords': ['conflict', 'sanctions', 'war', 'treaty'],
                    'risk_level': 'high'
                },
                'economic_data': {
                    'weight': 0.7,
                    'keywords': ['gdp', 'employment', 'retail sales', 'trade balance'],
                    'risk_level': 'medium'
                },
                'market_sentiment': {
                    'weight': 0.6,
                    'keywords': ['risk appetite', 'safe haven', 'market fear'],
                    'risk_level': 'medium'
                }
            }
        }
        self.position = None
        self.active_events = []
        self.correlations = {}
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze economic events and their impact on gold"""
        if len(data) < 20:
            return {'valid': False}
            
        # Get economic calendar events
        events = self._fetch_economic_events()
        
        # Analyze event importance and potential impact
        event_analysis = self._analyze_events(events)
        
        # Calculate market correlations
        correlations = self._calculate_correlations(data)
        
        # Assess event impact on price
        impact_analysis = self._assess_event_impact(
            events=event_analysis,
            correlations=correlations,
            current_price=data['close'].iloc[-1],
            volatility=data['close'].pct_change().std()
        )
        
        return {
            'valid': True,
            'events': event_analysis,
            'correlations': correlations,
            'impact': impact_analysis,
            'last_close': data['close'].iloc[-1],
            'volatility': data['close'].pct_change().std() * np.sqrt(252)
        }
        
    def _fetch_economic_events(self) -> List[Dict[str, Any]]:
        """Fetch relevant economic and geopolitical events"""
        try:
            # Use existing NewsAPI to get events
            news_api = NewsAPI()
            news = asyncio.get_event_loop().run_until_complete(news_api.get_gold_news())
            
            # Filter and categorize events
            events = []
            for article in news:
                event_type = self._categorize_event(article['title'])
                if event_type:
                    events.append({
                        'id': article['id'],
                        'title': article['title'],
                        'type': event_type,
                        'timestamp': article['published_at'],
                        'category': self.params['event_categories'][event_type]
                    })
            
            return events
        except Exception as e:
            logging.error(f"Error fetching economic events: {str(e)}")
            return []
            
    def _categorize_event(self, text: str) -> Optional[str]:
        """Categorize event based on keywords"""
        text = text.lower()
        for category, info in self.params['event_categories'].items():
            if any(keyword in text for keyword in info['keywords']):
                return category
        return None
        
    def _analyze_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze events for trading significance"""
        analyzed_events = []
        for event in events:
            # Calculate event importance score
            category = self.params['event_categories'][event['type']]
            importance_score = category['weight']
            
            # Adjust for risk level
            if category['risk_level'] == 'high':
                importance_score *= self.params['risk_adjust_factor']
                
            analyzed_events.append({
                **event,
                'importance': importance_score,
                'risk_level': category['risk_level']
            })
            
        return analyzed_events
        
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with related markets"""
        try:
            # Example markets that affect gold
            correlations = {
                'usd_index': -0.7,  # Placeholder for actual correlation
                'treasury_yields': -0.5,
                'equity_markets': -0.3,
                'commodity_index': 0.4
            }
            
            # In real implementation, fetch data and calculate actual correlations
            return correlations
        except Exception as e:
            logging.error(f"Error calculating correlations: {str(e)}")
            return {}
            
    def _assess_event_impact(self, events: List[Dict[str, Any]], 
                           correlations: Dict[str, float],
                           current_price: float,
                           volatility: float) -> Dict[str, Any]:
        """Assess potential impact of events on gold price"""
        # Calculate aggregate impact score
        impact_score = sum(
            event['importance'] for event in events
        ) / max(len(events), 1)
        
        # Adjust for market conditions
        correlation_factor = abs(sum(correlations.values()) / len(correlations))
        volatility_factor = 1.0 / (1.0 + volatility)
        
        # Determine likely price direction
        bullish_events = sum(1 for e in events if e['type'] in ['geopolitical', 'market_sentiment'])
        bearish_events = sum(1 for e in events if e['type'] in ['monetary_policy'])
        
        direction = 'bullish' if bullish_events > bearish_events else 'bearish'
        
        return {
            'score': impact_score * correlation_factor * volatility_factor,
            'direction': direction,
            'confidence': correlation_factor,
            'risk_level': 'high' if impact_score > 0.7 else 'medium'
        }
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on economic events"""
        if not analysis['valid']:
            return []
            
        signals = []
        impact = analysis['impact']
        current_price = analysis['last_close']
        
        # Calculate position size based on impact and risk
        base_size = CONFIG['max_position_size']
        risk_factor = 0.7 if impact['risk_level'] == 'high' else 1.0
        position_size = base_size * impact['score'] * risk_factor
        
        # Generate entry signals
        if not self.position:
            if impact['direction'] == 'bullish' and impact['score'] > 0.6:
                signals.append({
                    'direction': 'buy',
                    'size': position_size,
                    'price': current_price,
                    'reason': 'economic_event_bullish',
                    'impact_score': impact['score'],
                    'confidence': impact['confidence']
                })
            elif impact['direction'] == 'bearish' and impact['score'] > 0.6:
                signals.append({
                    'direction': 'sell',
                    'size': position_size,
                    'price': current_price,
                    'reason': 'economic_event_bearish',
                    'impact_score': impact['score'],
                    'confidence': impact['confidence']
                })
                
        # Generate exit signals
        else:
            if ((self.position['direction'] == 'buy' and impact['direction'] == 'bearish') or
                (self.position['direction'] == 'sell' and impact['direction'] == 'bullish')):
                if impact['score'] > 0.5:  # Significant contrary event
                    signals.append({
                        'direction': f"close_{self.position['direction']}",
                        'size': self.position['size'],
                        'price': current_price,
                        'reason': 'event_reversal',
                        'impact_score': impact['score']
                    })
                    self.position = None
                    
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class ATRRiskManager:
    """ATR-based risk management for dynamic position sizing and stops"""
    def __init__(self):
        self.params = {
            'atr_period': 14,          # Period for ATR calculation
            'risk_per_trade': 0.02,    # 2% risk per trade
            'atr_multiplier': 2.0,     # Multiplier for stop distance
            'max_position_size': 1.0,  # Maximum allowed position size
            'min_stop_distance': 10,   # Minimum stop distance in points
            'pyramiding_levels': 3,    # Maximum number of scale-in entries
            'scale_out_levels': [0.5, 0.3, 0.2]  # Position exit proportions
        }
        
    def calculate_position_params(self, 
                                data: pd.DataFrame, 
                                account_balance: float,
                                current_exposure: float = 0.0) -> Dict[str, float]:
        """Calculate position size and stop levels using ATR"""
        try:
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=self.params['atr_period']).mean()
            
            current_atr = atr.iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Calculate stop distance
            stop_distance = max(
                current_atr * self.params['atr_multiplier'],
                self.params['min_stop_distance']
            )
            
            # Calculate position size based on risk
            risk_amount = account_balance * self.params['risk_per_trade']
            position_size = self._calculate_position_size(
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                current_price=current_price,
                current_exposure=current_exposure
            )
            
            # Calculate stop levels
            stop_loss = self._calculate_stop_level(
                current_price=current_price,
                stop_distance=stop_distance,
                position_type='long'
            )
            
            # Calculate take-profit levels
            take_profits = self._calculate_take_profits(
                entry_price=current_price,
                stop_distance=stop_distance,
                position_type='long'
            )
            
            return {
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'atr': current_atr,
                'risk_amount': risk_amount
            }
            
        except Exception as e:
            logging.error(f"Error calculating position parameters: {str(e)}")
            return None
            
    def _calculate_position_size(self,
                               risk_amount: float,
                               stop_distance: float,
                               current_price: float,
                               current_exposure: float) -> float:
        """Calculate safe position size based on risk parameters"""
        # Convert stop distance to price risk
        price_risk = stop_distance * current_price / 10000  # For gold points
        
        # Calculate base position size
        position_size = risk_amount / price_risk
        
        # Apply position size limits
        position_size = min(
            position_size,
            self.params['max_position_size'] - current_exposure
        )
        
        return max(0.0, position_size)
        
    def _calculate_stop_level(self,
                            current_price: float,
                            stop_distance: float,
                            position_type: str) -> float:
        """Calculate stop-loss level based on ATR"""
        if position_type == 'long':
            return current_price - stop_distance
        else:  # short position
            return current_price + stop_distance
            
    def _calculate_take_profits(self,
                              entry_price: float,
                              stop_distance: float,
                              position_type: str) -> List[float]:
        """Calculate multiple take-profit levels"""
        risk_reward_ratios = [1.5, 2.0, 3.0]  # Multiple R:R levels
        
        if position_type == 'long':
            return [
                entry_price + (stop_distance * ratio)
                for ratio in risk_reward_ratios
            ]
        else:  # short position
            return [
                entry_price - (stop_distance * ratio)
                for ratio in risk_reward_ratios
            ]
            
    def adjust_stops_for_volatility(self,
                                  current_stop: float,
                                  current_atr: float,
                                  price_data: pd.Series) -> float:
        """Dynamically adjust stop-loss based on volatility"""
        volatility_factor = current_atr / price_data.std()
        new_stop = current_stop
        
        if volatility_factor > 1.5:  # High volatility
            # Widen stop distance
            if current_stop < price_data.iloc[-1]:  # Long position
                new_stop -= current_atr * 0.5
            else:  # Short position
                new_stop += current_atr * 0.5
        
        return new_stop

class VolatilityBreakoutStrategy(Strategy):
    """Bollinger Bands volatility breakout strategy"""
    def __init__(self):
        self.params = {
            'bb_period': 20,           # Bollinger Bands period
            'bb_std': 2.0,            # Number of standard deviations
            'volume_trigger': 1.5,     # Volume increase to confirm breakout
            'breakout_threshold': 0.001, # 0.1% price movement confirmation
            'consolidation_periods': 5, # Periods of low volatility needed
            'volatility_threshold': 0.2, # Threshold for volatility squeeze
            'scale_out_levels': [0.5, 0.3, 0.2],  # Profit taking levels
            'stop_atr_multiple': 2.0   # ATR multiple for stop loss
        }
        self.position = None
        self.last_breakout = None
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price data for volatility breakout setups"""
        if len(data) < self.params['bb_period']:
            return {'valid': False}
            
        # Calculate Bollinger Bands
        middle_band = data['close'].rolling(window=self.params['bb_period']).mean()
        std_dev = data['close'].rolling(window=self.params['bb_period']).std()
        upper_band = middle_band + (std_dev * self.params['bb_std'])
        lower_band = middle_band - (std_dev * self.params['bb_std'])
        
        # Calculate band width and volatility
        band_width = (upper_band - lower_band) / middle_band
        volatility = data['close'].pct_change().rolling(window=self.params['bb_period']).std()
        
        # Detect volatility squeeze
        squeeze = self._detect_squeeze(band_width, volatility)
        
        # Calculate volume metrics
        volume_ma = data['tick_volume'].rolling(window=self.params['bb_period']).mean()
        volume_ratio = data['tick_volume'] / volume_ma
        
        # Detect breakouts
        breakouts = self._detect_breakouts(
            data=data,
            upper_band=upper_band,
            lower_band=lower_band,
            volume_ratio=volume_ratio,
            squeeze=squeeze
        )
        
        # Calculate ATR for stop loss
        atr = self._calculate_atr(data)
        
        return {
            'valid': True,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'band_width': band_width,
            'volatility': volatility,
            'squeeze': squeeze,
            'breakouts': breakouts,
            'volume_ratio': volume_ratio,
            'atr': atr,
            'last_close': data['close'].iloc[-1]
        }
        
    def _detect_squeeze(self, band_width: pd.Series, volatility: pd.Series) -> Dict[str, Any]:
        """Detect volatility squeeze conditions"""
        recent_width = band_width.iloc[-self.params['consolidation_periods']:]
        recent_vol = volatility.iloc[-self.params['consolidation_periods']:]
        
        is_squeezed = (
            (recent_width < recent_width.mean() * self.params['volatility_threshold']).all() and
            (recent_vol < recent_vol.mean() * self.params['volatility_threshold']).all()
        )
        
        return {
            'active': is_squeezed,
            'duration': self.params['consolidation_periods'] if is_squeezed else 0,
            'intensity': (1 - recent_width.mean() / band_width.mean()) if is_squeezed else 0
        }
        
    def _detect_breakouts(self, data: pd.DataFrame, upper_band: pd.Series,
                         lower_band: pd.Series, volume_ratio: pd.Series,
                         squeeze: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect valid breakout signals"""
        breakouts = []
        close = data['close']
        
        # Check for upper band breakout
        if (close.iloc[-1] > upper_band.iloc[-1] and
            close.iloc[-2] <= upper_band.iloc[-2]):
            
            # Confirm with volume and price movement
            if (volume_ratio.iloc[-1] > self.params['volume_trigger'] and
                (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] > 
                self.params['breakout_threshold']):
                
                breakouts.append({
                    'direction': 'up',
                    'price': close.iloc[-1],
                    'strength': volume_ratio.iloc[-1],
                    'squeeze_active': squeeze['active']
                })
                
        # Check for lower band breakout
        elif (close.iloc[-1] < lower_band.iloc[-1] and
              close.iloc[-2] >= lower_band.iloc[-2]):
              
            # Confirm with volume and price movement
            if (volume_ratio.iloc[-1] > self.params['volume_trigger'] and
                (close.iloc[-2] - close.iloc[-1]) / close.iloc[-2] >
                self.params['breakout_threshold']):
                
                breakouts.append({
                    'direction': 'down',
                    'price': close.iloc[-1],
                    'strength': volume_ratio.iloc[-1],
                    'squeeze_active': squeeze['active']
                })
                
        return breakouts
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on volatility breakouts"""
        if not analysis['valid'] or not analysis['breakouts']:
            return []
            
        signals = []
        current_price = analysis['last_close']
        
        # Calculate base position size and risk parameters
        base_size = CONFIG['max_position_size']
        atr = analysis['atr'].iloc[-1]
        stop_distance = atr * self.params['stop_atr_multiple']
        
        # Process each breakout
        for breakout in analysis['breakouts']:
            # Adjust position size based on squeeze and volume strength
            position_size = base_size
            if breakout['squeeze_active']:
                position_size *= 1.2  # Increase size for squeeze breakouts
            position_size *= min(1.0, breakout['strength'])  # Scale by volume strength
            
            if not self.position:  # No current position
                if breakout['direction'] == 'up':
                    signals.append({
                        'direction': 'buy',
                        'size': position_size,
                        'price': current_price,
                        'stop_loss': current_price - stop_distance,
                        'take_profits': [
                            current_price + (stop_distance * level)
                            for level in [1.5, 2.0, 3.0]
                        ],
                        'reason': 'volatility_breakout_long'
                    })
                else:  # down breakout
                    signals.append({
                        'direction': 'sell',
                        'size': position_size,
                        'price': current_price,
                        'stop_loss': current_price + stop_distance,
                        'take_profits': [
                            current_price - (stop_distance * level)
                            for level in [1.5, 2.0, 3.0]
                        ],
                        'reason': 'volatility_breakout_short'
                    })
                    
            else:  # Managing existing position
                if ((self.position['direction'] == 'buy' and breakout['direction'] == 'down') or
                    (self.position['direction'] == 'sell' and breakout['direction'] == 'up')):
                    
                    signals.append({
                        'direction': f"close_{self.position['direction']}",
                        'size': self.position['size'],
                        'price': current_price,
                        'reason': 'volatility_reversal'
                    })
                    self.position = None
                    
        return signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)

class ScalpingStrategy(Strategy):
    """High-frequency scalping strategy for gold trading"""
    def __init__(self):
        self.params = {
            'tick_window': 100,        # Number of ticks to analyze
            'price_threshold': 0.0001,  # Minimum price movement
            'volume_threshold': 1.5,    # Volume surge multiplier
            'max_spread': 0.00015,     # Maximum allowed spread
            'min_tick_sequence': 3,     # Minimum ticks in same direction
            'take_profit_ticks': 5,     # Take profit in ticks
            'stop_loss_ticks': 3,       # Stop loss in ticks
            'max_hold_time': 30,        # Maximum position hold time (seconds)
            'min_tick_volume': 10       # Minimum tick volume
        }
        self.position = None
        self.position_time = None
        self.tick_buffer = []
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tick data for scalping opportunities"""
        if len(data) < self.params['tick_window']:
            return {'valid': False}
            
        # Calculate tick-by-tick metrics
        tick_changes = data['ask'].diff()
        tick_volumes = data['tick_volume']
        spreads = data['ask'] - data['bid']
        
        # Detect price momentum
        momentum = self._analyze_momentum(tick_changes)
        
        # Analyze volume profile
        volume_profile = self._analyze_volume(tick_volumes)
        
        # Check orderbook imbalance (if available)
        imbalance = self._check_orderbook_imbalance(data)
        
        # Calculate tick volatility
        volatility = self._calculate_tick_volatility(tick_changes)
        
        return {
            'valid': True,
            'momentum': momentum,
            'volume_profile': volume_profile,
            'imbalance': imbalance,
            'volatility': volatility,
            'current_spread': spreads.iloc[-1],
            'last_ask': data['ask'].iloc[-1],
            'last_bid': data['bid'].iloc[-1],
            'tick_volume': tick_volumes.iloc[-1]
        }
        
    def _analyze_momentum(self, tick_changes: pd.Series) -> Dict[str, Any]:
        """Analyze price momentum in recent ticks"""
        recent_changes = tick_changes.tail(self.params['min_tick_sequence'])
        
        # Count consecutive moves
        up_ticks = (recent_changes > self.params['price_threshold']).sum()
        down_ticks = (recent_changes < -self.params['price_threshold']).sum()
        
        # Calculate momentum strength
        momentum_strength = abs(up_ticks - down_ticks) / self.params['min_tick_sequence']
        direction = 'up' if up_ticks > down_ticks else 'down'
        
        return {
            'direction': direction,
            'strength': momentum_strength,
            'consecutive_moves': max(up_ticks, down_ticks)
        }
        
    def _analyze_volume(self, tick_volumes: pd.Series) -> Dict[str, Any]:
        """Analyze recent volume profile"""
        recent_volume = tick_volumes.tail(20)  # Last 20 ticks
        avg_volume = tick_volumes.mean()
        
        return {
            'current_volume': recent_volume.iloc[-1],
            'volume_surge': recent_volume.iloc[-1] / avg_volume,
            'rising_volume': recent_volume.is_monotonic_increasing
        }
        
    def _check_orderbook_imbalance(self, data: pd.DataFrame) -> float:
        """Check order book imbalance if depth data available"""
        if 'ask_volume' in data and 'bid_volume' in data:
            ask_volume = data['ask_volume'].iloc[-1]
            bid_volume = data['bid_volume'].iloc[-1]
            total_volume = ask_volume + bid_volume
            
            if total_volume > 0:
                return (bid_volume - ask_volume) / total_volume
        return 0.0
        
    def _calculate_tick_volatility(self, tick_changes: pd.Series) -> float:
        """Calculate recent tick volatility"""
        return tick_changes.tail(50).std()
        
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scalping signals based on tick analysis"""
        if not analysis['valid']:
            return []
            
        signals = []
        momentum = analysis['momentum']
        volume = analysis['volume_profile']
        
        # Check if spread is acceptable
        if analysis['current_spread'] > self.params['max_spread']:
            return []
            
        # Calculate scalping position size
        base_size = CONFIG['max_position_size'] * 0.1  # Smaller size for scalping
        position_size = self._calculate_scalp_size(
            base_size=base_size,
            momentum_strength=momentum['strength'],
            volume_ratio=volume['volume_surge']
        )
        
        # Generate entry signals
        if not self.position:
            if self._check_entry_conditions(analysis):
                entry_price = analysis['last_ask'] if momentum['direction'] == 'up' else analysis['last_bid']
                
                signals.append({
                    'direction': 'buy' if momentum['direction'] == 'up' else 'sell',
                    'size': position_size,
                    'price': entry_price,
                    'stop_loss': self._calculate_stop_level(
                        entry_price,
                        momentum['direction'],
                        analysis['volatility']
                    ),
                    'take_profit': self._calculate_target_level(
                        entry_price,
                        momentum['direction'],
                        analysis['volatility']
                    ),
                    'reason': f"scalp_{momentum['direction']}"
                })
                self.position_time = datetime.now()
                
        # Manage existing position
        else:
            if self._check_exit_conditions(analysis):
                exit_price = analysis['last_bid'] if self.position['direction'] == 'buy' else analysis['last_ask']
                
                signals.append({
                    'direction': f"close_{self.position['direction']}",
                    'size': self.position['size'],
                    'price': exit_price,
                    'reason': 'scalp_exit'
                })
                self.position = None
                self.position_time = None
                
        return signals
        
    def _calculate_scalp_size(self, base_size: float, 
                            momentum_strength: float,
                            volume_ratio: float) -> float:
        """Calculate position size for scalp trade"""
        # Adjust size based on setup quality
        momentum_factor = min(1.0, momentum_strength * 2)
        volume_factor = min(1.0, volume_ratio / self.params['volume_threshold'])
        
        return base_size * momentum_factor * volume_factor
        
    def _check_entry_conditions(self, analysis: Dict[str, Any]) -> bool:
        """Check if conditions are right for scalp entry"""
        momentum = analysis['momentum']
        volume = analysis['volume_profile']
        
        return (
            momentum['consecutive_moves'] >= self.params['min_tick_sequence'] and
            volume['volume_surge'] >= self.params['volume_threshold'] and
            analysis['tick_volume'] >= self.params['min_tick_volume'] and
            abs(analysis['imbalance']) >= 0.2  # Significant order book imbalance
        )
        
    def _check_exit_conditions(self, analysis: Dict[str, Any]) -> bool:
        """Check if position should be closed"""
        if not self.position_time:
            return False
            
        # Check hold time
        hold_time = (datetime.now() - self.position_time).total_seconds()
        if hold_time > self.params['max_hold_time']:
            return True
            
        # Check momentum reversal
        momentum = analysis['momentum']
        if self.position['direction'] == 'buy' and momentum['direction'] == 'down':
            if momentum['consecutive_moves'] >= 2:  # Quick reversal
                return True
        elif self.position['direction'] == 'sell' and momentum['direction'] == 'up':
            if momentum['consecutive_moves'] >= 2:
                return True
                
        return False
        
    def _calculate_stop_level(self, entry_price: float, 
                            direction: str, volatility: float) -> float:
        """Calculate adaptive stop-loss level"""
        base_stop = self.params['stop_loss_ticks'] * 0.0001  # Base stop in pips
        volatility_adjustment = min(0.0001, volatility * 2)  # Cap volatility adjustment
        
        if direction == 'up':
            return entry_price - (base_stop + volatility_adjustment)
        else:
            return entry_price + (base_stop + volatility_adjustment)
            
    def _calculate_target_level(self, entry_price: float,
                              direction: str, volatility: float) -> float:
        """Calculate adaptive take-profit level"""
        base_target = self.params['take_profit_ticks'] * 0.0001
        volatility_adjustment = min(0.0001, volatility * 2)
        
        if direction == 'up':
            return entry_price + (base_target + volatility_adjustment)
        else:
            return entry_price - (base_target + volatility_adjustment)
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(params)