import asyncio
import MetaTrader5 as mt5
from typing import Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor

class LowLatencyExecutor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.order_queue = asyncio.Queue()
        self.last_order_time = 0
        self.min_order_interval = 0.1  # 100ms minimum between orders
        
    async def submit_order(self, order: Dict) -> Optional[Dict]:
        """Submit order with minimal latency"""
        current_time = time.time()
        
        # Rate limiting check
        if current_time - self.last_order_time < self.min_order_interval:
            await asyncio.sleep(self.min_order_interval)
            
        try:
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order['symbol'],
                "volume": order['volume'],
                "type": order['type'],
                "price": order['price'],
                "deviation": 5,  # Minimal deviation
                "magic": 234000,
                "comment": "v2",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }
            
            # Execute in thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: mt5.order_send(request)
            )
            
            self.last_order_time = time.time()
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    'ticket': result.order,
                    'executed_price': result.price,
                    'volume': result.volume
                }
            
            return None
            
        except Exception as e:
            print(f"Order execution error: {e}")
            return None
            
    async def process_order_queue(self):
        """Process queued orders"""
        while True:
            order = await self.order_queue.get()
            result = await self.submit_order(order)
            
            if not result:
                # Requeue failed orders with backoff
                await asyncio.sleep(0.5)
                await self.order_queue.put(order)
                
            self.order_queue.task_done()
