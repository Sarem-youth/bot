import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
from mt5_adapter import MT5Config, MT5Communication, MT5Adapter, MT5TesterAdapter

@pytest.fixture
def mt5_config():
    return MT5Config(
        pull_port="5555",
        push_port="5556",
        host="localhost",
        recv_timeout=1000,
        magic_number=234000
    )

@pytest.fixture
def mock_mt5():
    with patch('mt5_adapter.mt5') as mock:
        mock.initialize.return_value = True
        mock.TERMINAL_TESTER_MODEL_EVERY_TICK = 1
        mock.tester_set_parameters.return_value = True
        mock.tester_run.return_value = True
        yield mock

class TestMT5Communication:
    @pytest.mark.asyncio
    async def test_connect(self):
        with patch('mt5_adapter.win32pipe') as mock_pipe:
            comm = MT5Communication()
            mock_pipe.CreateNamedPipe.return_value = Mock()
            
            result = await comm.connect()
            assert result == True
            assert comm.connected == True
            
    @pytest.mark.asyncio
    async def test_send_command(self):
        with patch('mt5_adapter.win32file') as mock_file:
            comm = MT5Communication()
            comm.connected = True
            comm.pipe = Mock()
            
            result = await comm.send_command("TEST")
            assert result == True
            mock_file.WriteFile.assert_called_once()

class TestMT5TesterAdapter:
    def test_init_tester(self, mt5_config, mock_mt5):
        adapter = MT5TesterAdapter(mt5_config)
        result = adapter.init_tester("XAUUSD", 15, testing=True)
        
        assert result == True
        mock_mt5.initialize.assert_called_once()
        mock_mt5.tester_set_parameters.assert_called_once()
        
    def test_set_optimization_inputs(self, mt5_config, mock_mt5):
        adapter = MT5TesterAdapter(mt5_config)
        adapter.is_optimization = True
        
        test_inputs = {
            "param1": (10, 1, 20),
            "param2": 15
        }
        
        adapter.set_optimization_inputs(test_inputs)
        assert adapter.optimization_inputs == test_inputs
        assert mock_mt5.tester_set_parameter.call_count == 2
        
    @pytest.mark.asyncio
    async def test_run_test(self, mt5_config, mock_mt5):
        adapter = MT5TesterAdapter(mt5_config)
        adapter.is_testing = True
        adapter.test_symbol = "XAUUSD"
        adapter.test_period = 15
        
        mock_mt5.tester_get_trades.return_value = []
        mock_mt5.tester_get_results.return_value = Mock(
            trades=10,
            profit_trades=6,
            profit_factor=1.5,
            sharp_ratio=1.2,
            max_drawdown=0.1,
            profit=1000,
            average_trade_length=120
        )
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        result = await adapter.run_test(start_date, end_date)
        
        assert 'trades' in result
        assert 'results' in result
        mock_mt5.tester_run.assert_called_once_with(
            start_date=start_date,
            end_date=end_date,
            symbol="XAUUSD",
            period=15
        )
