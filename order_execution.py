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
        self.max_retries = 3
        self.retry_delay = 0.2  # 200ms between retries
        self.order_monitor = OrderMonitor()
        
    async def submit_order(self, order: Dict) -> Optional[Dict]:
        """Submit order with retries and monitoring"""
        for attempt in range(self.max_retries):
            try:
                result = await self._execute_order(order)
                if result:
                    # Start monitoring the order
                    await self.order_monitor.monitor_order(result['ticket'])
                    return result

                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return None
                    
                await asyncio.sleep(self.retry_delay)

    async def _execute_order(self, order: Dict) -> Optional[Dict]:
        """Execute single order attempt with validation"""
        # Verify market is open
        if not mt5.symbol_info(order['symbol']).trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            return None
            
        # Check account has sufficient margin
        account_info = mt5.account_info()
        if account_info.margin_free < account_info.margin_initial * 1.5:
            return None
            
        # Execute order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order['symbol'],
            "volume": order['volume'],
            "type": order['type'],
            "price": order['price'],
            "sl": order['sl'],
            "tp": order['tp'],
            "deviation": 5,
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

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                'ticket': result.order,
                'executed_price': result.price,
                'volume': result.volume,
                'sl': order['sl'],
                'tp': order['tp']
            }

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

class OrderMonitor:
    """Monitor and manage open orders"""
    def __init__(self):
        self.monitored_orders = {}
        self.check_interval = 0.1  # 100ms check interval

    async def monitor_order(self, ticket: int):
        """Start monitoring an order"""
        self.monitored_orders[ticket] = {
            'start_time': time.time(),
            'last_check': time.time(),
            'status': 'ACTIVE'
        }
        
        asyncio.create_task(self._monitor_loop(ticket))

    async def _monitor_loop(self, ticket: int):
        """Monitor single order status"""
        while ticket in self.monitored_orders:
            try:
                # Get order status
                order = mt5.positions_get(ticket=ticket)
                if not order:
                    # Order closed or invalid
                    del self.monitored_orders[ticket]
                    break

                # Update stop loss and take profit if needed
                await self._update_order_protection(ticket, order[0])
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring order {ticket}: {e}")
                await asyncio.sleep(1)

    async def _update_order_protection(self, ticket: int, order):
        """Update stop loss and take profit based on market conditions"""
        try:
            current_price = mt5.symbol_info_tick(order.symbol).last
            
            # Calculate trailing stop
            if order.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (current_price - order.price_open) * 0.5
                if new_sl > order.sl:
                    self._modify_order(ticket, sl=new_sl)
            else:
                new_sl = current_price + (order.price_open - current_price) * 0.5
                if new_sl < order.sl:
                    self._modify_order(ticket, sl=new_sl)
                    
        except Exception as e:
            logger.error(f"Error updating order protection: {e}")

    def _modify_order(self, ticket: int, sl: float = None, tp: float = None):
        """Modify order stop loss or take profit"""
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order modification failed: {result.comment}")
