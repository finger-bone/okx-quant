from data_fetcher import DataFetcher
from features import FeatureEngineer
from model import MLModel
from config import SYMBOLS
from loguru import logger

if __name__ == "__main__":
    fetcher = DataFetcher()
    fe = FeatureEngineer()
    ml_model = MLModel()

    for symbol in SYMBOLS:
        logger.info(f"Training model for {symbol}")
        df = fetcher.fetch_ohlcv_all(symbol)
        df = fe.add_features(df)
        df = ml_model.train(df)
        
        model_filename = f"model_{symbol.replace('/', '_').replace('-', '_')}.pkl"
        ml_model.save(model_filename)
        
        logger.info(f"Model for {symbol} saved as {model_filename}")