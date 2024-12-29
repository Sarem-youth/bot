import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
import MetaTrader5 as mt5
from trading_bot import TradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize MT5 connection
        if not mt5.initialize(
            login=int(os.getenv('MT5_LOGIN')),
            server=os.getenv('MT5_SERVER'),
            password=os.getenv('MT5_PASSWORD')
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return
            
        logger.info("MT5 connection established")
        
        # Create and run trading bot
        bot = TradingBot()
        await bot.run("XAUUSD")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
