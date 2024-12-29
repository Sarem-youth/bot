import zmq
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass
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
        """Open a new position"""
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
