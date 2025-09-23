import pandas as pd
import numpy as np

class FeatureEngineer:
    def add_features(self, df: pd.DataFrame):
        df = df.copy()
        # 移动平均 - 增加多个时间周期的均线以捕捉不同趋势
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma50"] = df["close"].rolling(50).mean()
        
        # 移动平均相对位置
        df["close_to_ma5"] = df["close"] / df["ma5"] - 1
        df["close_to_ma10"] = df["close"] / df["ma10"] - 1
        df["close_to_ma20"] = df["close"] / df["ma20"] - 1
        df["close_to_ma50"] = df["close"] / df["ma50"] - 1
        
        # 多均线排列特征
        df["ma5_gt_ma10"] = (df["ma5"] > df["ma10"]).astype(int)
        df["ma10_gt_ma20"] = (df["ma10"] > df["ma20"]).astype(int)
        df["ma20_gt_ma50"] = (df["ma20"] > df["ma50"]).astype(int)
        
        # 价格动量 - 增加多个时间窗口以捕捉不同周期的动量
        df["return_1h"] = df["close"].pct_change()
        df["return_6h"] = df["close"].pct_change(6)
        df["return_12h"] = df["close"].pct_change(12)
        df["return_24h"] = df["close"].pct_change(24)
        df["return_7d"] = df["close"].pct_change(24*7)
        
        # RSI - 增加不同周期的RSI以捕捉超买超卖信号
        df["rsi"] = self.rsi(df["close"], 14)
        df["rsi_7"] = self.rsi(df["close"], 7)   # 短期RSI
        df["rsi_21"] = self.rsi(df["close"], 21) # 长期RSI
        
        # 波动率 - 增加不同周期的波动率
        df["volatility"] = df["return_1h"].rolling(20).std()
        df["volatility_10"] = df["return_1h"].rolling(10).std()  # 短期波动率
        df["volatility_50"] = df["return_1h"].rolling(50).std()  # 长期波动率
        
        # 成交量特征
        df["volume_sma"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["volume_change"] = df["volume"].pct_change()  # 成交量变化率
        
        # 价格波动范围
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["price_range_sma"] = df["price_range"].rolling(20).mean()
        
        # 布林带特征
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + bb_std * 2
        df["bb_lower"] = df["bb_middle"] - bb_std * 2
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]  # 布林带宽度
        
        # MACD
        df["macd_line"], df["macd_signal"] = self.macd(df["close"])
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        
        # 趋势强度指标
        df["adx"] = self.adx(df, 14)  # 平均趋向指数
        
        # 价格位置指标
        df["price_percentile"] = df["close"].rolling(100).rank(pct=True)  # 价格在过去100期的分位数
        
        # 超买超卖指标
        df["willr"] = self.williams_r(df, 14)  # 威廉指标
        
        # 动量指标
        df["roc_10"] = self.rate_of_change(df["close"], 10)  # 10周期价格变化率
        df["roc_30"] = self.rate_of_change(df["close"], 30)  # 30周期价格变化率
        
        df.dropna(inplace=True)
        return df

    def rsi(self, series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = -delta.clip(upper=0).rolling(period).mean()
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    def macd(self, series, fast=12, slow=26, signal=9):
        """
        计算MACD指标
        """
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def williams_r(self, df, period=14):
        """
        计算威廉指标
        """
        highest_high = df["high"].rolling(period).max()
        lowest_low = df["low"].rolling(period).min()
        willr = (highest_high - df["close"]) / (highest_high - lowest_low) * -100
        return willr
    
    def adx(self, df, period=14):
        """
        计算平均趋向指数(ADX)
        """
        # 计算+DM和-DM
        up_move = df["high"].diff()
        down_move = df["low"].diff()
        
        # 计算+DI和-DI
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
        
        # 计算真实波幅(TR)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算+DI和-DI
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
        
        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx
    
    def rate_of_change(self, series, period):
        """
        计算价格变化率(ROC)
        """
        return series.pct_change(period)