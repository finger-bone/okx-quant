from data_fetcher import DataFetcher
from features import FeatureEngineer
from model import MLModel
from config import SYMBOLS, ORDER_USDT_BASE, API_KEY, SECRET_KEY, PASSPHRASE, FLAG
from loguru import logger
import ccxt
import numpy as np

class LiveTrader:
    def __init__(self):
        self.fetcher = DataFetcher()
        # 设置交易所API密钥
        self.fetcher.exchange.apiKey = API_KEY
        self.fetcher.exchange.secret = SECRET_KEY
        self.fetcher.exchange.password = PASSPHRASE
        self.fetcher.exchange.options['defaultType'] = 'swap'
        if FLAG == "1":
            self.fetcher.exchange.set_sandbox_mode(True)
        
        self.fe = FeatureEngineer()
        self.ml_model = MLModel()
        self.position_tracker = {}  # 跟踪当前仓位
        self.last_signal = {}  # 跟踪上一次信号

    def execute_trade(self, symbol, signal, price, current_position=0):
        """
        执行交易
        symbol: 交易对
        signal: 交易信号
        price: 当前价格
        current_position: 当前仓位
        """
        # 检查是否与上一次信号相同，避免重复交易
        last_signal = self.last_signal.get(symbol, 0)
        if signal == last_signal:
            logger.info(f"信号未变化，不执行交易: {symbol} 信号={signal}")
            return
            
        # 记录当前信号
        self.last_signal[symbol] = signal
            
        # 计算目标仓位（与回测保持一致）
        if signal == 1:
            target_position = 0.5  # 最大仓位50%，降低风险（与回测中最大仓位限制一致）
        elif signal == -1:
            target_position = 0  # 清仓
        else:
            target_position = current_position  # 保持当前仓位
            
        # 计算需要调整的仓位
        position_change = target_position - current_position
        
        # 只有当仓位变化超过5%时才交易，避免频繁小额交易（与回测保持一致）
        if abs(position_change) > 0.05:
            amount = ORDER_USDT_BASE * abs(position_change) / price
            
            try:
                if position_change > 0:
                    # 买入
                    logger.info(f"买入 {symbol}: 信号={signal}, 价格={price:.4f}, 数量={amount:.6f}")
                    self.fetcher.exchange.create_market_buy_order(symbol, amount)
                elif position_change < 0:
                    # 卖出
                    logger.info(f"卖出 {symbol}: 信号={signal}, 价格={price:.4f}, 数量={amount:.6f}")
                    self.fetcher.exchange.create_market_sell_order(symbol, amount)
                    
                # 更新仓位跟踪器
                self.position_tracker[symbol] = target_position
                
            except ccxt.BaseError as e:
                logger.error(f"交易执行失败 {symbol}: {e}")
        else:
            logger.info(f"仓位变化过小，不执行交易: {symbol} 信号={signal}")

    def run(self):
        for symbol in SYMBOLS:
            try:
                # 获取当前仓位（实际环境中应该从交易所获取）
                current_position = self.position_tracker.get(symbol, 0)
                
                # 获取数据
                df = self.fetcher.fetch_ohlcv_all(symbol)
                df = self.fe.add_features(df)
                
                # 加载模型（如果尚未加载）
                model_path = f"model_{symbol.replace('/', '_').replace('-', '_')}.pkl"
                try:
                    self.ml_model.load(model_path)
                except FileNotFoundError:
                    logger.warning(f"模型文件 {model_path} 不存在，跳过 {symbol}")
                    continue
                    
                df = self.ml_model.predict(df)  # 做预测
                
                # 获取最新信号
                latest_signal = df.iloc[-1]["signal"]
                latest_price = df.iloc[-1]["close"]
                
                logger.info(f"{symbol} 最新信号: {latest_signal}")
                
                # 执行交易
                self.execute_trade(symbol, latest_signal, latest_price, current_position)
                
            except Exception as e:
                logger.error(f"处理 {symbol} 时出错: {e}")

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()