import time
from typing import Dict, List
import numpy as np
from collections import deque
import logging

class DataMonitor:
    def __init__(self):
        self.latency_buffer = deque(maxlen=1000)
        self.error_counts = {}
        self.data_stats = {}
        self.logger = logging.getLogger(__name__)
        
    def record_latency(self, start_time: float):
        """Record processing latency"""
        latency = time.time() - start_time
        self.latency_buffer.append(latency)
        
        if len(self.latency_buffer) >= 100:
            avg_latency = np.mean(self.latency_buffer)
            if avg_latency > 0.1:  # Alert if average latency > 100ms
                self.logger.warning(f"High latency detected: {avg_latency*1000:.2f}ms")
                
    def track_data_quality(self, data: Dict):
        """Monitor data quality metrics"""
        if 'price' in data:
            self.data_stats['price_gaps'] = self._check_price_gaps(data['price'])
            self.data_stats['price_jumps'] = self._check_price_jumps(data['price'])
            
        if 'timestamp' in data:
            self.data_stats['timestamp_gaps'] = self._check_timestamp_gaps(
                data['timestamp'])
                
    def _check_price_gaps(self, prices: np.ndarray) -> int:
        """Check for price gaps"""
        return np.sum(np.abs(np.diff(prices)) > 0.1)
        
    def _check_price_jumps(self, prices: np.ndarray) -> int:
        """Check for abnormal price jumps"""
        returns = np.diff(prices) / prices[:-1]
        return np.sum(np.abs(returns) > 0.01)
        
    def _check_timestamp_gaps(self, timestamps: np.ndarray) -> int:
        """Check for timestamp gaps"""
        gaps = np.diff(timestamps)
        return np.sum(gaps > 1.0)  # Gaps larger than 1 second
        
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'avg_latency': np.mean(self.latency_buffer) if self.latency_buffer else 0,
            'max_latency': np.max(self.latency_buffer) if self.latency_buffer else 0,
            'error_counts': self.error_counts,
            'data_stats': self.data_stats
        }
