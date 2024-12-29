import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from textblob import TextBlob
from transformers import pipeline
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import MetaTrader5 as mt5
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import redis
from threading import Lock
import orjson  # Faster JSON processing
from collections import deque  # More efficient than list for FIFO
from data_processing import DataProcessor, RealTimeProcessor, ProcessingConfig

class LowLatencyDataManager:
    def __init__(self):
        # Use deque with maxlen for efficient buffer management
        self.tick_buffer = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = None
        self.websockets = {}
        
        # Pre-allocate connection pools
        self.connection_pools = {
            'mt5': aiohttp.TCPConnector(limit=50, ttl_dns_cache=300),
            'news': aiohttp.TCPConnector(limit=20, ttl_dns_cache=300),
            'events': aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        }
        
        # Initialize data processor with config
        config = ProcessingConfig(
            window_size=60,
            kalman_q=0.001,
            kalman_r=0.1,
            batch_size=32
        )
        self.processor = RealTimeProcessor(config)
        
    async def setup(self):
        """Initialize connections and sessions"""
        # Create persistent aiohttp session
        self.session = aiohttp.ClientSession(
            json_serialize=orjson.dumps,
            connector=self.connection_pools['mt5'],
            timeout=aiohttp.ClientTimeout(total=5)
        )
        
        # Initialize MT5 with optimal settings
        mt5.initialize(
            server="BrokerServerName",  # Use closest server
            timeout=100,  # Milliseconds
            portable=False  # Desktop mode for better performance
        )
        
    async def connect_websocket(self, url: str, name: str):
        """Establish WebSocket connection with keep-alive"""
        while True:
            try:
                async with self.session.ws_connect(
                    url,
                    heartbeat=15,
                    compress=True,
                    ssl=False  # If using direct connection to broker
                ) as ws:
                    self.websockets[name] = ws
                    await self._handle_websocket(ws, name)
            except Exception as e:
                print(f"WebSocket error {name}: {e}")
                await asyncio.sleep(1)  # Quick reconnect

    async def _handle_websocket(self, ws, name: str):
        """Handle WebSocket messages efficiently"""
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                # Process binary data directly
                await self.process_binary_message(msg.data, name)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                # Parse JSON only if needed
                data = orjson.loads(msg.data)
                await self.process_text_message(data, name)

    def optimize_mt5_settings(self):
        """Optimize MT5 connection settings"""
        mt5.symbol_select("XAUUSD", True)
        
        # Set up symbol properties for faster access
        symbol_info = mt5.symbol_info("XAUUSD")._asdict()
        self.tick_size = symbol_info['trade_tick_size']
        self.point = symbol_info['point']
        
        # Reduce network calls by caching common values
        self.cached_values = {
            'spread': symbol_info['spread'],
            'contract_size': symbol_info['trade_contract_size'],
            'volume_min': symbol_info['volume_min']
        }

    async def get_ticks_fast(self, symbol: str, count: int = 100):
        """Optimized tick data retrieval"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: mt5.copy_ticks_from(
                symbol,
                mt5.symbol_info_tick(symbol).time,
                count,
                mt5.COPY_TICKS_ALL
            )
        )

    def process_tick_binary(self, binary_data: bytes) -> Dict:
        """Process binary tick data format"""
        # Assuming a fixed binary format for tick data
        # Format: timestamp(8) + bid(8) + ask(8) + volume(4) = 28 bytes
        import struct
        timestamp, bid, ask, volume = struct.unpack('!dddi', binary_data)
        return {
            'timestamp': timestamp,
            'bid': bid,
            'ask': ask,
            'volume': volume
        }

    async def subscribe_market_data(self, symbol: str):
        """Subscribe to market data with optimized settings"""
        try:
            request = {
                'symbol': symbol,
                'type': 'subscribe',
                'heartbeat': True,
                'frequency': 'tick'  # Request raw tick data
            }
            ws = self.websockets.get('market')
            if ws:
                await ws.send_bytes(orjson.dumps(request))
        except Exception as e:
            print(f"Subscription error: {e}")

    def preprocess_tick(self, tick: Dict) -> Dict:
        """Pre-process tick data in memory"""
        return {
            'timestamp': tick['time'],
            'price': (tick['bid'] + tick['ask']) / 2,
            'spread': tick['ask'] - tick['bid']
        }

    async def process_tick(self, tick_data: Dict):
        """Process tick data with optimized pipeline"""
        try:
            # Process tick through pipeline
            processed_data = await self.processor.process_streaming_data(tick_data)
            
            if processed_data:
                # Notify callbacks with processed data
                for callback in self.price_callbacks:
                    await callback(processed_data)
                    
        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}")

    async def run_optimized(self):
        """Run data manager with optimized settings"""
        await self.setup()
        self.optimize_mt5_settings()
        
        # Start websocket connections
        ws_tasks = [
            self.connect_websocket("wss://broker/market", "market"),
            self.connect_websocket("wss://broker/events", "events")
        ]
        
        # Use gather for concurrent connections
        await asyncio.gather(*ws_tasks)

    async def cleanup(self):
        """Clean up connections"""
        for ws in self.websockets.values():
            await ws.close()
        
        if self.session:
            await self.session.close()
            
        for pool in self.connection_pools.values():
            await pool.close()
            
        mt5.shutdown()
