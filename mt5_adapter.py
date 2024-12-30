import zmq
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
import threading
from queue import Queue
import time
import asyncio
import MetaTrader5 as mt5
import win32pipe
import win32file
import pywintypes

logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    pull_port: str = "5555"
    push_port: str = "5556"
    host: str = "localhost"
    recv_timeout: int = 1000
    magic_number: int = 234000
    environment: str = "demo"  # demo, live
    risk_limits: Dict = field(default_factory=lambda: {
        'max_daily_loss': 1000,
        'max_position_size': 1.0,
        'max_slippage': 3,
        'max_spread': 5
    })
    monitoring: Dict = field(default_factory=lambda: {
        'log_level': 'INFO',
        'save_ticks': True,
        'performance_report': True
    })

class MT5Environment:
    """Environment-specific configuration and safety checks"""
    def __init__(self, config: MT5Config):
        self.config = config
        self.daily_stats = {
            'total_loss': 0,
            'total_trades': 0,
            'errors': 0
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger(f"mt5_ea_{self.config.environment}")
        handler = logging.FileHandler(f"ea_{self.config.environment}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.monitoring['log_level'])
        return logger
        
    def check_trading_allowed(self) -> bool:
        """Verify if trading is allowed based on current conditions"""
        if self.daily_stats['total_loss'] >= self.config.risk_limits['max_daily_loss']:
            self.logger.warning("Daily loss limit reached")
            return False
            
        if self.daily_stats['errors'] > 5:
            self.logger.warning("Too many errors, stopping trading")
            return False
            
        return True
        
    def validate_order(self, order: Dict) -> bool:
        """Validate order parameters"""
        if order['volume'] > self.config.risk_limits['max_position_size']:
            self.logger.warning(f"Position size {order['volume']} exceeds limit")
            return False
            
        current_spread = self._get_current_spread(order['symbol'])
        if current_spread > self.config.risk_limits['max_spread']:
            self.logger.warning(f"Current spread {current_spread} exceeds limit")
            return False
            
        return True
        
    def _get_current_spread(self, symbol: str) -> float:
        """Get current symbol spread"""
        tick = mt5.symbol_info_tick(symbol)
        return (tick.ask - tick.bid) / mt5.symbol_info(symbol).point

class MT5Communication:
    def __init__(self, pipe_name: str = "mt5_bridge"):
        self.pipe_name = rf"\\.\pipe\{pipe_name}"
        self.pipe = None
        self.buffer_size = 1024
        self.connected = False
        
    async def connect(self):
        """Connect to MT5 via named pipe"""
        try:
            self.pipe = win32pipe.CreateNamedPipe(
                self.pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, 65536, 65536,
                0,
                None
            )
            
            # Wait for MT5 to connect
            win32pipe.ConnectNamedPipe(self.pipe, None)
            self.connected = True
            logger.info("MT5 communication established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish MT5 communication: {e}")
            return False
            
    async def send_command(self, command: str) -> bool:
        """Send command to MT5"""
        if not self.connected:
            return False
            
        try:
            win32file.WriteFile(self.pipe, command.encode())
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
            
    async def receive_data(self) -> Optional[str]:
        """Receive data from MT5"""
        if not self.connected:
            return None
            
        try:
            result, data = win32file.ReadFile(self.pipe, self.buffer_size)
            return data.decode().strip()
        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            return None
            
    def close(self):
        """Close communication"""
        if self.pipe:
            try:
                win32file.CloseHandle(self.pipe)
            except:
                pass
        self.connected = False

class MT5Adapter:
    def __init__(self, config: MT5Config):
        self.config = config
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.pull_socket = self.context.socket(zmq.PULL)
        self.running = False
        self.message_queue = Queue()
        self.tick_handlers = []
        self.environment = MT5Environment(config)
        
        # Configure sockets
        self.push_socket.connect(f"tcp://{config.host}:{config.pull_port}")
        self.pull_socket.connect(f"tcp://{config.host}:{config.push_port}")
        self.pull_socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout)
        
        # Start message processing thread
        self.start_processing()
        
    def start_processing(self):
        """Start the message processing thread"""
        self.running = True
        self.process_thread = threading.Thread(target=self._process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def _process_messages(self):
        """Process incoming messages from MT5"""
        while self.running:
            try:
                message = self.pull_socket.recv_string()
                if message:
                    parts = message.split(",")
                    if parts[0] == "TICK":
                        self._handle_tick(parts[1:])
            except zmq.error.Again:
                continue
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                
    def _handle_tick(self, tick_data: list):
        """Process incoming tick data"""
        tick = {
            'symbol': tick_data[0],
            'bid': float(tick_data[1]),
            'ask': float(tick_data[2]),
            'volume': int(tick_data[3]),
            'time': int(tick_data[4]),
            'flags': int(tick_data[5])
        }
        
        # Notify tick handlers
        for handler in self.tick_handlers:
            try:
                handler(tick)
            except Exception as e:
                logging.error(f"Error in tick handler: {e}")
                
    def open_position(self, symbol: str, order_type: str, volume: float,
                     price: float, sl: float, tp: float) -> bool:
        """Enhanced position opening with safety checks"""
        if not self.environment.check_trading_allowed():
            return False
            
        order = {
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'sl': sl,
            'tp': tp
        }
        
        if not self.environment.validate_order(order):
            return False
            
        command = f"OPEN,{symbol},{order_type},{volume},{price},{sl},{tp}"
        return self._send_command(command)
        
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        command = f"CLOSE,{ticket}"
        return self._send_command(command)
        
    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify an existing position"""
        command = f"MODIFY,{ticket},{sl},{tp}"
        return self._send_command(command)
        
    def _send_command(self, command: str) -> bool:
        """Send command to MT5"""
        try:
            self.push_socket.send_string(command)
            return True
        except Exception as e:
            logging.error(f"Error sending command: {e}")
            return False
            
    def register_tick_handler(self, handler):
        """Register a function to handle tick data"""
        self.tick_handlers.append(handler)
        
    def shutdown(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()

    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """Get current market tick data"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'volume': tick.volume,
                    'time': tick.time,
                    'last': tick.last,
                    'flags': tick.flags
                }
        except Exception as e:
            self.environment.logger.error(f"Error getting tick data: {e}")
            self.environment.daily_stats['errors'] += 1
        return None

    def get_open_positions(self) -> List[Dict]:
        """Get currently open positions"""
        try:
            positions = mt5.positions_get()
            if positions:
                return [{
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit
                } for pos in positions]
        except Exception as e:
            self.environment.logger.error(f"Error getting positions: {e}")
            self.environment.daily_stats['errors'] += 1
        return []

class MT5TesterAdapter:
    def __init__(self, config: MT5Config):
        self.config = config
        self.is_testing = False
        self.is_optimization = False
        self.test_symbol = ""
        self.test_period = 0
        self.optimization_inputs = {}
        
    def init_tester(self, symbol: str, timeframe: int, testing: bool = False, optimization: bool = False):
        """Initialize MT5 Strategy Tester"""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
            
        self.is_testing = testing
        self.is_optimization = optimization
        self.test_symbol = symbol
        self.test_period = timeframe
        
        if testing:
            # Configure tester settings
            tester_settings = {
                "symbol": symbol,
                "period": timeframe,
                "spread": 2,  # Default spread in points
                "model": mt5.TERMINAL_TESTER_MODEL_EVERY_TICK,  # Every tick mode
                "use_spread": True
            }
            
            if not mt5.tester_set_parameters(**tester_settings):
                logger.error("Failed to set tester parameters")
                return False
                
        return True
        
    def set_optimization_inputs(self, inputs: Dict):
        """Set optimization parameters"""
        self.optimization_inputs = inputs
        if self.is_optimization:
            for param, value in inputs.items():
                if isinstance(value, tuple):
                    # Format: (start, step, stop)
                    mt5.tester_set_parameter(param, *value)
                else:
                    mt5.tester_set_parameter(param, value)
                    
    async def run_test(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run strategy test"""
        if not self.is_testing:
            return {}
            
        result = mt5.tester_run(
            start_date=start_date,
            end_date=end_date,
            symbol=self.test_symbol,
            period=self.test_period
        )
        
        if not result:
            logger.error("Strategy test failed")
            return {}
            
        return {
            'trades': mt5.tester_get_trades(),
            'results': mt5.tester_get_results(),
            'optimization': mt5.tester_get_optimization() if self.is_optimization else None
        }
