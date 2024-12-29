# Advanced Gold Trading Bot

A sophisticated algorithmic trading system specialized for gold (XAUUSD) trading, featuring multiple strategies, machine learning integration, and real-time market analysis.

## Features

- Multiple Trading Strategies:
  - Trend Following
  - Range Trading
  - Event-Driven Trading
  - Volatility-Based Trading
  - High-Frequency Scalping
  - Machine Learning Predictions

- Advanced Analysis:
  - Technical Indicators
  - Market Sentiment Analysis
  - Order Book Analysis
  - Economic Calendar Integration
  - Real-time News Processing

- Risk Management:
  - Dynamic Position Sizing
  - Adaptive Stop Losses
  - Multi-timeframe Analysis
  - Volatility Adjustments

- Machine Learning:
  - LSTM Price Prediction
  - Reinforcement Learning
  - Sentiment Analysis
  - Pattern Recognition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Usage

### Basic Running
```bash
python trading_bot.py
```

### Backtesting
```bash
python backtester.py --strategy trend --start 2023-01-01 --end 2023-12-31
```

### Strategy Optimization
```bash
python strategy_validator.py --optimize --strategy range
```

## Project Structure

```
trading_bot/
├── trading_bot.py      # Main trading bot implementation
├── strategies.py       # Trading strategies
├── ml_model.py        # Machine learning models
├── backtester.py      # Backtesting engine
├── strategy_validator.py # Strategy validation
├── requirements.txt    # Project dependencies
├── Dockerfile         # Container configuration
├── docker-compose.yml # Service orchestration
└── .env              # Configuration file
```

## Configuration

Key configuration parameters in `.env`:

- MT5_LOGIN: MetaTrader 5 account login
- MT5_PASSWORD: MetaTrader 5 password
- NEWS_API_KEY: News API credentials
- MAX_DAILY_LOSS: Maximum daily loss limit
- RISK_PER_TRADE: Risk per trade (default 2%)

## Deployment

Using Docker:
```bash
docker-compose up -d
```

## Monitoring

Access monitoring interfaces:
- Trading Dashboard: http://localhost:8080
- RabbitMQ Management: http://localhost:15672
- Redis Commander: http://localhost:8081

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Testing

Run tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_strategies.py -v
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Past performance does not guarantee future results.