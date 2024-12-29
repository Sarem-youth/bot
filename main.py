
from data_processing import RealTimeProcessor, ProcessingConfig

# Initialize configuration
config = ProcessingConfig(
    window_size=60,
    kalman_q=0.001,
    kalman_r=0.1,
    batch_size=32
)

# Create processor instance
processor = RealTimeProcessor(config)

# In your trading bot's main loop
async def handle_tick_data(tick_data):
    processed_data = await processor.process_streaming_data(tick_data)
    if processed_data:
        sequences = processed_data['sequences']
        features = processed_data['features']
        await update_trading_signals(processed_data)