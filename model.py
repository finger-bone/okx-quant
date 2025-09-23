import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from loguru import logger
import numpy as np

class MLModel:
    def __init__(self):
        # 增加模型复杂度以提高预测能力，同时保持一定的正则化防止过拟合
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=6, 
            random_state=42, 
            min_samples_split=20, 
            min_samples_leaf=10,
            max_features='sqrt'
        )
        self.signal_threshold = 0.0003  # 降低阈值以增加交易机会 (0.03%)

    def train(self, df: pd.DataFrame):
        """训练模型并生成预测信号"""
        # 增加更多有效特征以提高预测能力
        features = [
            "ma5", "ma10", "ma20", "ma50",
            "rsi", "rsi_7", "rsi_21",
            "return_1h", "return_6h", "return_12h", "return_24h", "return_7d",
            "volatility", "volatility_10",
            "volume_ratio", "volume_change",
            "bb_position", "bb_width",
            "macd_histogram",
            "adx",
            "close_to_ma5", "close_to_ma10", "close_to_ma20", "close_to_ma50",
            "price_percentile",
            "ma5_gt_ma10", "ma10_gt_ma20", "ma20_gt_ma50"
        ]
        df = df.copy()
        df["future_return"] = df["close"].pct_change().shift(-1)  # 下一K线收益
        
        # 添加市场状态特征
        df["trend_strength"] = df["adx"] / 100  # 趋势强度
        df["volatility_regime"] = df["volatility"] > df["volatility"].rolling(50).mean()  # 波动率状态
        
        df.dropna(inplace=True)
        X = df[features]
        y = df["future_return"]
        
        # Proper train/test split to avoid overfitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        logger.info(f"Model R² - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Predict on all data
        df["pred_return"] = self.model.predict(X)
        df = self._generate_signal(df)
        logger.info("ML model training completed")
        return df

    def predict(self, df: pd.DataFrame):
        """加载训练好的模型进行预测"""
        # 使用更多有效特征以提高预测能力
        features = [
            "ma5", "ma10", "ma20", "ma50",
            "rsi", "rsi_7", "rsi_21",
            "return_1h", "return_6h", "return_12h", "return_24h", "return_7d",
            "volatility", "volatility_10",
            "volume_ratio", "volume_change",
            "bb_position", "bb_width",
            "macd_histogram",
            "adx",
            "close_to_ma5", "close_to_ma10", "close_to_ma20", "close_to_ma50",
            "price_percentile",
            "ma5_gt_ma10", "ma10_gt_ma20", "ma20_gt_ma50"
        ]
        df = df.copy()
        
        # 添加市场状态特征
        df["trend_strength"] = df["adx"] / 100  # 趋势强度
        df["volatility_regime"] = df["volatility"] > df["volatility"].rolling(50).mean()  # 波动率状态
        
        df.dropna(inplace=True)
        df["pred_return"] = self.model.predict(df[features])
        df = self._generate_signal(df)
        return df

    def _generate_signal(self, df: pd.DataFrame):
        """
        根据预测值和市场状态生成最终预测结果
        1: 买入, -1: 卖出, 0: 持有
        """
        df = df.copy()
        
        # 使用机器学习预测值结合前一天的数值作为最终预测
        # alpha系数用于调整前一天数值的影响程度
        alpha = 0.2  # 减少历史值影响，增加对当前预测的敏感度
        
        # 计算结合前一天数值的预测值
        df["combined_pred"] = (1 - alpha) * df["pred_return"] + alpha * df["pred_return"].shift(1)
        
        # 根据市场状态调整信号阈值
        # 在趋势强的市场中降低阈值，在震荡市场中提高阈值
        trend_threshold = np.where(
            df["trend_strength"] > 0.25,  # 趋势较强(ADX > 25)
            self.signal_threshold * 0.7,  # 降低阈值，增加交易机会
            np.where(
                df["trend_strength"] < 0.2,  # 趋势较弱(ADX < 20)
                self.signal_threshold * 1.5,  # 提高阈值，减少交易
                self.signal_threshold  # 正常阈值
            )
        )
        
        # 基于组合预测值和市场状态生成信号
        df["signal"] = 0
        df.loc[df["combined_pred"] > trend_threshold, "signal"] = 1   # 仅在预测收益超过阈值时买入
        df.loc[df["combined_pred"] < -trend_threshold, "signal"] = -1 # 仅在预测损失超过阈值时卖出
        
        return df

    def save(self, path="model.pkl"):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path="model.pkl"):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")