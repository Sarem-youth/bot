from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

@dataclass
class BacktestPlatformConfig:
    name: str
    data_format: str
    timeframes: List[str]
    api_key: Optional[str] = None
    connection_params: Optional[Dict] = None

class BacktestPlatformManager:
    def __init__(self):
        self.platforms = {
            'metatrader5': {
                'name': 'MetaTrader 5',
                'data_format': 'mt5',
                'timeframes': ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'],
                'features': [
                    'Built-in Strategy Tester',
                    'Forward Testing',
                    'Multi-currency testing',
                    'Real tick data',
                    'Visual testing mode'
                ],
                'connection_params': {
                    'login': None,
                    'server': None,
                    'password': None
                }
            },
            'backtrader': {
                'name': 'Backtrader',
                'data_format': 'csv',
                'timeframes': ['minutes', 'days', 'weeks'],
                'features': [
                    'Multiple data feeds',
                    'Custom indicators',
                    'Plot visualization',
                    'Live trading support'
                ]
            },
            'vectorbt': {
                'name': 'VectorBT',
                'data_format': 'pandas',
                'timeframes': ['any'],
                'features': [
                    'Vectorized backtesting',
                    'High performance',
                    'Portfolio optimization',
                    'Advanced analytics'
                ]
            },
            'ccxt': {
                'name': 'CCXT',
                'data_format': 'json',
                'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
                'features': [
                    'Multiple exchange support',
                    'Historical data access',
                    'Unified API',
                    'Real-time data'
                ]
            }
        }
        
    def get_recommended_platform(self, requirements: Dict) -> str:
        """Get recommended platform based on requirements"""
        scores = {}
        
        for platform, config in self.platforms.items():
            score = 0
            
            # Check data format compatibility
            if requirements.get('data_format') == config['data_format']:
                score += 2
                
            # Check timeframe availability
            if requirements.get('timeframe') in config['timeframes']:
                score += 2
                
            # Check features
            required_features = requirements.get('features', [])
            for feature in required_features:
                if feature in config['features']:
                    score += 1
                    
            scores[platform] = score
            
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def export_platform_config(self, platform: str, filepath: str):
        """Export platform configuration to YAML"""
        if platform in self.platforms:
            with open(filepath, 'w') as f:
                yaml.dump(self.platforms[platform], f)
                
    def get_platform_features(self, platform: str) -> List[str]:
        """Get features for specific platform"""
        return self.platforms.get(platform, {}).get('features', [])

    def get_setup_instructions(self, platform: str) -> str:
        """Get setup instructions for platform"""
        instructions = {
            'metatrader5': """
                1. Download and install MetaTrader 5
                2. Set up demo account
                3. Install Python MT5 package: pip install MetaTrader5
                4. Configure login credentials
                5. Enable Expert Advisors
                """,
            'backtrader': """
                1. Install Backtrader: pip install backtrader
                2. Optional: pip install backtrader[plotting]
                3. Prepare data in CSV format
                4. Create strategy class inheriting from backtrader.Strategy
                """,
            'vectorbt': """
                1. Install VectorBT: pip install vectorbt
                2. Install dependencies: numpy, pandas
                3. Import market data
                4. Configure vectorized operations
                """,
            'ccxt': """
                1. Install CCXT: pip install ccxt
                2. Create exchange instance
                3. Configure API credentials if needed
                4. Use fetch_ohlcv for historical data
                """
        }
        return instructions.get(platform, "Instructions not available")
