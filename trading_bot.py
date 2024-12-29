import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import MetaTrader5 as mt5
from order_execution import LowLatencyExecutor
from mt5_adapter import MT5Communication

logger = logging.getLogger(__name__)

from strategies import GoldTrendStrategy, RangeBoundStrategy, EventStrategy, VolatilityStrategy
from ml_model import GoldPricePredictor, RLTrader
from data_feeds import RealTimeDataManager

class TradingBot:
    def __init__(self):
        try:
            # Initialize MT5 communication
            self.mt5_comm = MT5Communication("mt5_bridge")
            
            # Core components
            self.trend_strategy = GoldTrendStrategy()
            self.range_strategy = RangeBoundStrategy()
            self.event_strategy = EventStrategy()
            self.volatility_strategy = VolatilityStrategy()
            
            # Machine learning
            self.price_predictor = GoldPricePredictor()
            self.rl_trader = RLTrader()
            
            # Data management
            self.data_manager = RealTimeDataManager()
            
            # Trading parameters
            self.volatility_multiplier = 1.0
            self.tick_enabled = True
            self.ml_enabled = True
            self.gold_point = 0.1  # 1 point for gold
            self.max_lot = float(os.getenv('MAX_LOT', '1.0'))
            self.min_lot = float(os.getenv('MIN_LOT', '0.01'))
            
            # Operational parameters
            self.model_retrain_interval = 24 * 60 * 60  # 24 hours
            self.event_check_interval = 300  # 5 minutes
            self.min_tick_interval = 0.1  # 100ms
            
            # State management
            self.last_train_time = None
            self.last_event_check = None
            self.last_tick_time = None
            self.tick_buffer = []
            self.scalp_positions = []
            
            # Add risk management parameters
            self.max_risk_per_trade = 0.02  # 2% max risk per trade
            self.max_daily_risk = 0.06      # 6% max daily risk
            self.daily_loss_limit = 0.05    # 5% max daily loss
            self.total_daily_risk = 0.0
            self.daily_loss = 0.0
            self.last_reset_day = datetime.now().date()
            
            self.setup_data_handlers()
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing trading bot: {str(e)}")
            raise

    async def setup(self):
        """Setup MT5 communication and data handlers"""
        if not await self.mt5_comm.connect():
            raise Exception("Failed to establish MT5 communication")
        self.setup_data_handlers()
        logger.info("Trading bot setup completed")

    def setup_data_handlers(self):
        """Setup real-time data handlers"""
        self.data_manager.register_price_callback(self.handle_price_update)
        self.data_manager.register_news_callback(self.handle_news_update)
        
    async def handle_price_update(self, tick_data: Dict):
        """Process real-time price updates"""
        if self.tick_enabled:
            await self.process_tick(tick_data)
            
        # Update order book if available
        if hasattr(self, 'order_book'):
            self.order_book['bids'][0]['price'] = tick_data['bid']
            self.order_book['asks'][0]['price'] = tick_data['ask']
            await self.process_order_book(self.order_book)
            
    async def handle_news_update(self, news_data: Dict):
        """Process real-time news updates"""
        if news_data['sentiment']['label'] != 'NEUTRAL':
            # Adjust trading parameters based on sentiment
            if news_data['sentiment']['score'] > 0.8:
                self.risk_multiplier = 0.5  # Reduce risk during high-impact news
                
        # Check for upcoming economic events
        upcoming_events = self.data_manager.get_upcoming_events(hours=1)
        if upcoming_events:
            self.max_lot *= 0.5  # Reduce position size before events
            
    async def run(self, symbol: str):
        """Main trading loop with MT5 communication"""
        try:
            await self.setup()
            logger.info(f"Starting trading bot for {symbol}")
            
            while True:
                try:
                    # Get market data
                    tick_data = await self.mt5_comm.receive_data()
                    if tick_data and tick_data.startswith("TICK"):
                        await self.handle_tick_data(tick_data)
                    
                    # Regular market analysis
                    analysis_result = await self.analyze_market(symbol)
                    if analysis_result:
                        await self.execute_trade(analysis_result)
                        
                    await asyncio.sleep(0.1)  # Prevent CPU overload
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Critical error in trading bot: {e}")
        finally:
            self.mt5_comm.close()

    async def handle_tick_data(self, tick_data: str):
        """Handle incoming tick data"""
        try:
            parts = tick_data.split(',')
            if len(parts) >= 6:
                tick = {
                    'symbol': parts[1],
                    'bid': float(parts[2]),
                    'ask': float(parts[3]),
                    'volume': int(parts[4]),
                    'time': int(parts[5])
                }
                await self.process_tick(tick)
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")

    async def execute_trade(self, trade: Dict) -> bool:
        """Execute trade with risk management and validation"""
        try:
            # Reset daily metrics if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_day:
                self.total_daily_risk = 0.0
                self.daily_loss = 0.0
                self.last_reset_day = current_date

            # Validate risk limits
            trade_risk = self.calculate_trade_risk(trade)
            if not self.validate_risk_limits(trade_risk):
                logger.warning("Trade rejected: Risk limits exceeded")
                return False

            # Calculate optimal position size
            trade['size'] = self.calculate_position_size(trade)

            # Add execution validation
            if not self.validate_execution_conditions(trade):
                logger.warning("Trade rejected: Invalid execution conditions")
                return False

            # Execute through low latency executor
            executor = LowLatencyExecutor()
            result = await executor.submit_order({
                'symbol': trade['symbol'],
                'volume': trade['size'],
                'type': mt5.ORDER_TYPE_BUY if trade['type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': trade['entry'],
                'sl': trade['stop_loss'],
                'tp': trade['take_profit']
            })

            if result:
                self.total_daily_risk += trade_risk
                logger.info(f"Trade executed: {trade}, risk: {trade_risk:.2%}")
                return True
                
            return False

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    async def close_trade(self, trade: Dict) -> bool:
        """Close trade using MT5 communication"""
        try:
            result = await self.mt5_comm.send_command(f"CLOSE,{trade['ticket']}")
            if result:
                logger.info(f"Trade closed successfully: {trade['ticket']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False

    def process_tick(self, tick_data: Dict):
        """Process incoming tick data"""
        current_time = time.time()
        
        # Rate limiting for tick processing
        if (self.last_tick_time and 
            current_time - self.last_tick_time < self.min_tick_interval):
            return
            
        self.last_tick_time = current_time
        self.trend_strategy.last_tick = tick_data
        
        # Quick scalping analysis
        if self.tick_enabled:
            scalp_analysis = self.trend_strategy.scalping_strategy.analyze_tick_data(tick_data)
            if scalp_analysis['signal']:
                self.execute_scalp_trade(scalp_analysis)
                
    def process_order_book(self, order_book: Dict):
        """Process order book updates"""
        self.trend_strategy.order_book = order_book
        
        if self.tick_enabled:
            book_analysis = self.trend_strategy.scalping_strategy.analyze_order_book(order_book)
            # Update active scalp positions
            self.update_scalp_positions(book_analysis)
            
    def execute_scalp_trade(self, analysis: Dict):
        """Execute a scalping trade"""
        if len(self.scalp_positions) >= 3:  # Maximum concurrent scalp trades
            return
            
        # Get current price and volatility
        current_price = self.get_current_price()
        volatility = self.volatility_strategy.analyze_volatility(
            self.get_recent_candles())['atr']
            
        # Calculate trade levels
        levels = self.trend_strategy.scalping_strategy.get_optimal_scalp_levels(
            current_price, volatility)
            
        # Execute trade with tight stops
        if analysis['signal'] == 'BUY':
            stop_loss = current_price - levels['stop_loss']
            take_profit = current_price + levels['take_profit']
        else:
            stop_loss = current_price + levels['stop_loss']
            take_profit = current_price - levels['take_profit']
            
        # Use smaller position size for scalping
        position_size = self.max_lot * 0.3
        
        trade = {
            'type': analysis['signal'],
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'start_time': time.time(),
            'timeout': levels['timeout']
        }
        
        self.scalp_positions.append(trade)
        self.execute_trade(trade)
        
    def update_scalp_positions(self, book_analysis: Dict):
        """Update and manage scalping positions"""
        current_time = time.time()
        current_price = self.get_current_price()
        
        for position in self.scalp_positions[:]:  # Copy list for safe removal
            # Check timeout
            if current_time - position['start_time'] > position['timeout']:
                self.close_trade(position)
                self.scalp_positions.remove(position)
                continue
                
            # Dynamic exit based on order book pressure
            if (position['type'] == 'BUY' and book_analysis['imbalance'] < -0.3 or
                position['type'] == 'SELL' and book_analysis['imbalance'] > 0.3):
                self.close_trade(position)
                self.scalp_positions.remove(position)
                
    def analyze_market(self, symbol):
        try:
            # Initialize signals list
            signals = []

            # Check if models need retraining
            if (self.ml_enabled and 
                (self.last_train_time is None or 
                 time.time() - self.last_train_time > self.model_retrain_interval)):
                historical_data = self.get_historical_data(symbol, mt5.TIMEFRAME_H1, 0, 1000)
                self.train_models(historical_data)
            
            # Get current candles
            m15_candles = self.get_historical_data(symbol, mt5.TIMEFRAME_M15, 0, 100)
            if not m15_candles:
                return None

            # Check events periodically
            current_time = time.time()
            if (self.last_event_check is None or 
                current_time - self.last_event_check > self.event_check_interval):
                event_analysis = self.event_strategy.analyze_events()
                self.last_event_check = current_time
                
                # Adjust position sizing based on pending events
                if event_analysis['pending_events'] > 0:
                    self.max_lot *= 0.5  # Reduce position size before major events
                    
                # Add event signals
                if event_analysis['signal']:
                    signals.append({
                        'type': event_analysis['signal'],
                        'strategy': 'EVENT',
                        'strength': event_analysis['strength']
                    })
                    
                    # Adjust stop loss for event-based trades
                    if abs(event_analysis['sentiment']) > 0.5:
                        self.min_pip_movement *= 1.5  # Wider stops for high-impact events
            
            # ...existing code until signals list...
            
            # Add trend strategy signals
            trend_analysis = self.trend_strategy.analyze_trend(m15_candles)
            if trend_analysis['signal']:
                signals.append({
                    'type': trend_analysis['signal'],
                    'strategy': 'TREND',
                    'strength': trend_analysis['strength']
                })
                
                # Adjust stop loss and take profit based on volatility
                self.volatility_multiplier = max(1.0, trend_analysis['volatility'] * 2)
                
            # Add range analysis
            range_analysis = self.range_strategy.analyze_range(m15_candles)
            if range_analysis['signal']:
                signals.append({
                    'type': range_analysis['signal'],
                    'strategy': 'RANGE',
                    'strength': range_analysis['strength']
                })
                
                # Adjust stop loss for range trades
                if range_analysis['signal'] and trend_analysis['adx'] < 20:
                    self.min_pip_movement = 10  # Tighter stop loss for range trades
            
            # Add volatility analysis
            volatility_analysis = self.volatility_strategy.analyze_volatility(m15_candles)
            
            # Adjust position sizing and stops based on volatility
            current_atr = volatility_analysis['atr']
            if current_atr > 0:
                self.min_pip_movement = max(self.min_pip_movement, current_atr * 0.5)
                self.max_lot = min(self.max_lot, 
                                 volatility_analysis['suggested_position_size'])
                
            # Add volatility breakout signals
            if volatility_analysis['signal']:
                signals.append({
                    'type': volatility_analysis['signal'],
                    'strategy': 'VOLATILITY',
                    'strength': volatility_analysis['strength']
                })
                
            # Add scalping signals if enabled
            if self.tick_enabled and hasattr(self.trend_strategy, 'last_tick'):
                scalp_analysis = self.trend_strategy.scalping_strategy.analyze_tick_data(
                    self.trend_strategy.last_tick)
                if scalp_analysis['signal']:
                    signals.append({
                        'type': scalp_analysis['signal'],
                        'strategy': 'SCALP',
                        'strength': scalp_analysis['strength']
                    })
                    
            # Add ML predictions
            if self.ml_enabled:
                # LSTM price prediction
                price_prediction = self.price_predictor.predict(m15_candles)
                if price_prediction['signal']:
                    signals.append({
                        'type': price_prediction['signal'],
                        'strategy': 'ML_LSTM',
                        'strength': price_prediction['confidence'] * 3
                    })
                    
                # RL trading signal
                rl_prediction = self.rl_trader.predict(m15_candles)
                if rl_prediction['signal']:
                    signals.append({
                        'type': rl_prediction['signal'],
                        'strategy': 'ML_RL',
                        'strength': rl_prediction['confidence'] * 2
                    })
                    
            # Modify signal validation
            if signals:
                potential_trade = self.validate_signals(signals, m15_candles)
                if potential_trade:
                    # Adjust stop loss and take profit based on volatility multiplier
                    sl_pips = max(self.min_pip_movement, 
                                self.min_pip_movement * self.volatility_multiplier)
                    tp_pips = sl_pips * (2.0 if trend_analysis['adx'] > 25 else 1.5)
                    
                    # Dynamic stop loss based on ATR
                    sl_pips = max(self.min_pip_movement, current_atr * 2)
                    tp_pips = sl_pips * (2.5 if volatility_analysis['volatility'] < 0.1 else 1.5)
                    
                    potential_trade['stop_loss'] = (current_price - 
                        (sl_pips * self.gold_point if potential_trade['type'] == 'BUY' 
                         else -sl_pips * self.gold_point))
                    potential_trade['take_profit'] = (current_price + 
                        (tp_pips * self.gold_point if potential_trade['type'] == 'BUY' 
                         else -tp_pips * self.gold_point))
                    
                    return potential_trade
            
            # ...rest of existing code...

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return None
            
    def validate_signals(self, signals, candles):
        # ...existing code...
        
        # Add trend strength consideration
        trend_signals = [s for s in signals if s['strategy'] == 'TREND']
        if trend_signals:
            trend_strength = trend_signals[0]['strength']
            base_lot = min(max(self.min_lot, 
                             self.min_lot * (1 + trend_strength * 0.1)), 
                         self.max_lot)
        
        # Consider both trend and range signals
        range_signals = [s for s in signals if s['strategy'] == 'RANGE']
        
        # Prioritize range signals in low ADX conditions
        if range_signals and self.trend_strategy.calculate_adx(
            [c[2] for c in candles],
            [c[3] for c in candles],
            [c[4] for c in candles]
        )[-1] < 20:
            base_lot *= 0.8  # Reduce position size for range trades
        
        # Consider event signals
        event_signals = [s for s in signals if s['strategy'] == 'EVENT']
        if event_signals and event_signals[0]['strength'] >= 2:
            # Prioritize event signals with high strength
            base_lot = min(base_lot * 1.2, self.max_lot)  # Increase position size
            
        # Consider volatility signals
        volatility_signals = [s for s in signals if s['strategy'] == 'VOLATILITY']
        if volatility_signals:
            volatility_analysis = self.volatility_strategy.analyze_volatility(candles)
            # Adjust position size based on volatility
            base_lot *= (1 - volatility_analysis['volatility'])
            
        # Consider ML signals
        ml_signals = [s for s in signals if s['strategy'].startswith('ML_')]
        if ml_signals:
            # Increase position size if ML models agree with other signals
            ml_consensus = all(s['type'] == signals[0]['type'] for s in ml_signals)
            if ml_consensus:
                base_lot *= 1.2
                
        # ...rest of existing code...
        
    def check_economic_calendar(self):
        """Enhanced economic calendar check"""
        if not self.event_strategy:
            return True
            
        events = self.event_strategy.check_economic_calendar()
        for event in events:
            if (event['importance'] == 'high' and 
                abs(datetime.now() - datetime.fromisoformat(event['time'])) < timedelta(minutes=30)):
                return False
        return True
        
    def train_models(self, historical_data: list):
        """Train ML models with historical data"""
        print("Training ML models...")
        self.price_predictor.train(historical_data)
        self.rl_trader.prepare_environment(historical_data)
        self.rl_trader.train()
        self.last_train_time = time.time()

    def validate_risk_limits(self, trade_risk: float) -> bool:
        """Validate trade against risk management rules"""
        # Check individual trade risk
        if trade_risk > self.max_risk_per_trade:
            return False
            
        # Check daily risk limits
        if (self.total_daily_risk + trade_risk > self.max_daily_risk or
            self.daily_loss > self.daily_loss_limit):
            return False
            
        return True

    def calculate_trade_risk(self, trade: Dict) -> float:
        """Calculate risk for a trade"""
        entry = trade['entry']
        stop_loss = trade['stop_loss']
        position_size = trade['size']
        
        risk_amount = abs(entry - stop_loss) * position_size
        account_value = mt5.account_info().balance
        
        return risk_amount / account_value

    def calculate_position_size(self, trade: Dict) -> float:
        """Calculate position size based on risk management"""
        entry = trade['entry']
        stop_loss = trade['stop_loss']
        risk_amount = self.max_risk_per_trade * mt5.account_info().balance
        
        pip_value = self.gold_point
        stop_distance = abs(entry - stop_loss) / pip_value
        
        position_size = risk_amount / stop_distance
        return min(position_size, self.max_lot)

    def validate_execution_conditions(self, trade: Dict) -> bool:
        """Validate market conditions for trade execution"""
        # Check spread
        symbol_info = mt5.symbol_info(trade['symbol'])
        if symbol_info.spread > symbol_info.spread_float * 2:
            return False
            
        # Check market volatility
        volatility = self.volatility_strategy.analyze_volatility(
            self.get_recent_candles())['atr']
        if volatility > symbol_info.trade_tick_size * 10:
            return False
            
        # Check economic calendar
        if not self.check_economic_calendar():
            return False
            
        return True
