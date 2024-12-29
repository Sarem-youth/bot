import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
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
        
        # Create and run trading bot
        bot = TradingBot()
        await bot.run("XAUUSD")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
