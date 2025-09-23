import ccxt
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from config import API_KEY, SECRET_KEY, PASSPHRASE, FLAG

class DataFetcher:
    def __init__(self):
        # 初始化交易所API
        self.exchange = ccxt.okx({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'password': PASSPHRASE,
            'options': {
                'defaultType': 'swap'
            }
        })
        
        # 设置沙盒模式
        if FLAG == "1":
            self.exchange.set_sandbox_mode(True)
            logger.info("OKX sandbox mode enabled.")

    def fetch_ohlcv_all(self, symbol, timeframe='4h', days=120):
        """获取历史K线数据"""
        since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Fetched {len(df)} rows of {symbol} {timeframe} data")
        return df

if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv_all("BTC-USDT-SWAP")
    print(df.tail())