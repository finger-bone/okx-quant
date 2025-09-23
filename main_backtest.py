from data_fetcher import DataFetcher
from features import FeatureEngineer
from model import MLModel
from backtester import Backtester
from config import SYMBOLS
from loguru import logger

if __name__ == "__main__":
    fetcher = DataFetcher()
    fe = FeatureEngineer()
    # 调整参数以适应不同市场环境
    backtester = Backtester(initial_capital=10000, risk_per_trade=0.02, atr_multiplier=2, transaction_cost=0.001)
    ml_model = MLModel()

    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}")
        df = fetcher.fetch_ohlcv_all(symbol)
        df = fe.add_features(df)
        
        # Split data for training and testing (70/30 split to have more training data)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Train model only on training data
        train_df = ml_model.train(train_df)
        
        # Predict on test data
        test_df = ml_model.predict(test_df)
        
        # Run backtest on test data only
        result_df = backtester.run(test_df)
        backtester.plot(result_df, symbol)
        logger.info(f"{symbol} backtest done.")
        
        # 输出详细统计信息
        total_return = (result_df["equity_curve"].iloc[-1] / backtester.initial_capital) - 1
        buy_and_hold_return = (result_df["close"].iloc[-1] / result_df["close"].iloc[0]) - 1
        logger.info(f"{symbol} Strategy Total Return: {total_return:.2%}")
        logger.info(f"{symbol} Buy and Hold Return: {buy_and_hold_return:.2%}")
        logger.info(f"{symbol} Strategy vs Buy and Hold: {total_return - buy_and_hold_return:.2%}")
        
        # 保存模型
        ml_model.save(f"model_{symbol.replace('/', '_').replace('-', '_')}.pkl")