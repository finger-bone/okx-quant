import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import pandas as pd

class Backtester:
    def __init__(self, initial_capital=10000, transaction_cost=0.001, risk_per_trade=0.02, atr_multiplier=2):
        """
        Initializes the backtester.
        
        :param initial_capital: Starting capital.
        :param transaction_cost: Transaction cost per trade (e.g., 0.001 = 0.1%).
        :param risk_per_trade: Maximum percentage of capital to risk per trade (e.g., 0.02 = 2%).
        :param atr_multiplier: Multiplier for the ATR to set the stop-loss level.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier

    def run(self, df):
        """
        Runs the backtest simulation.
        
        :param df: DataFrame containing price data and signals.
        :return: DataFrame with backtest results, including equity curve and position.
        """
        df = df.copy()

        # 1. Calculate technical indicators (ATR and returns)
        df["tr"] = np.maximum(np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift())), abs(df['low'] - df['close'].shift()))
        df['atr'] = df['tr'].rolling(14).mean()
        df["return"] = df["close"].pct_change()

        # Initialize columns for equity and position
        df["equity_curve"] = self.initial_capital
        df["position"] = 0.0
        df["strategy_return"] = 0.0
        
        # Keep track of state variables for the loop
        current_position = 0.0
        entry_price = 0
        stop_loss_price = 0

        # 2. Iterate through the DataFrame to simulate trading day by day
        for i in range(1, len(df)):
            # Get data for the current time step
            signal = df["signal"].iloc[i]
            close_price = df["close"].iloc[i]
            atr = df["atr"].iloc[i]
            
            # Use previous day's equity for today's calculations
            current_equity = df["equity_curve"].iloc[i-1]

            # --- Determine the target position for the current day ---
            target_position = current_position
            
            # Check if stop-loss is triggered for long positions
            if current_position > 0 and close_price <= stop_loss_price and not pd.isna(stop_loss_price) and stop_loss_price > 0:
                logger.debug(f"Stop-loss triggered at {close_price:.2f}. Entry price was {entry_price:.2f}.")
                target_position = 0
            
            # Check for entry signals only when not in position
            elif signal != 0:  # A buy or sell signal
                if signal == 1:  # Buy signal
                    # Calculate stop-loss based on ATR
                    if not pd.isna(atr) and atr > 0:
                        stop_loss_price = close_price - self.atr_multiplier * atr
                        # Make sure stop loss price is positive
                        stop_loss_price = max(stop_loss_price, 0.01)
                        stop_loss_amount = close_price - stop_loss_price
                        
                        if stop_loss_amount > 0:
                            # Calculate position size based on risk management
                            risk_amount = current_equity * self.risk_per_trade
                            # Position size in terms of number of contracts/shares
                            position_size = risk_amount / stop_loss_amount
                            # 根据市场波动率调整仓位，在低波动市场中增加仓位，在高波动市场中减少仓位
                            volatility_adjustment = df["volatility"].iloc[i] / df["volatility"].rolling(50).mean().iloc[i]
                            max_position_ratio = np.clip(0.5 / volatility_adjustment, 0.3, 0.5)  # 与实盘最大仓位保持一致
                            # Limit position size based on adjusted volatility
                            position_size = min(position_size, (current_equity * max_position_ratio) / close_price)
                            target_position = position_size
                            
                            # Update state variables
                            entry_price = close_price
                            
                            logger.debug(f"Entry signal at {close_price:.2f}. Position: {target_position:.2f}")
                elif signal == -1 and current_position > 0:  # Sell signal and we're in a long position
                    target_position = 0
                    logger.debug(f"Exit signal at {close_price:.2f}")
            
            # Store today's position in the DataFrame
            df.iloc[i, df.columns.get_loc("position")] = target_position
            
            # Calculate today's return and transaction cost
            prev_position = current_position
            current_position = target_position
            
            # Calculate strategy return
            # If we have a position, calculate the return based on price change
            if prev_position > 0:
                # Return is based on price change and existing position
                price_return = (close_price / df["close"].iloc[i-1]) - 1
                strategy_return_day = prev_position * price_return
            else:
                strategy_return_day = 0
            
            # Calculate transaction costs when changing position
            transaction_cost = 0
            position_change = abs(current_position - prev_position)
            if position_change > 0 and i > 1:  # Avoid transaction costs on the first day
                # Transaction cost is based on the value of the position change
                transaction_cost = position_change * close_price * self.transaction_cost
            
            # Store the strategy return (without transaction costs)
            df.iloc[i, df.columns.get_loc("strategy_return")] = strategy_return_day
            
            # Update the equity curve: current equity + return - transaction costs
            df.iloc[i, df.columns.get_loc("equity_curve")] = current_equity * (1 + strategy_return_day) - transaction_cost

        self.calculate_metrics(df)

        logger.info(f"Backtest completed. Final equity: {df['equity_curve'].iloc[-1]:.2f}")
        return df

    def calculate_metrics(self, df):
        """Calculates and logs key performance metrics."""
        total_return = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1
        num_years = len(df) / (365 * 4)  # Assuming 6h data
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

        strategy_returns = df["strategy_return"].dropna()
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365 * 4) if strategy_returns.std() > 0 else 0

        rolling_max = df["equity_curve"].expanding().max()
        drawdown = (df["equity_curve"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        position_changes = df["position"].diff().abs()
        num_trades = (position_changes > 0).sum()
        
        # Calculate win rate based on individual trades
        if num_trades > 0:
            positive_returns = (df["strategy_return"] > 0).sum()
            win_rate = positive_returns / num_trades
        else:
            win_rate = 0
        
        # Calculate MSE if prediction columns exist
        if "pred_return" in df.columns and "future_return" in df.columns:
            # Remove NaN values
            valid_data = df.dropna(subset=["pred_return", "future_return"])
            if len(valid_data) > 0:
                mse = ((valid_data["pred_return"] - valid_data["future_return"]) ** 2).mean()
                logger.info(f"Prediction MSE: {mse:.6f}")
        
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")

    def plot(self, df, symbol="BTC-USDT"):
        """Plots the backtest results."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df["timestamp"], df["equity_curve"], label="Strategy")
        buy_and_hold = self.initial_capital * (1 + df["return"]).cumprod()
        plt.plot(df["timestamp"], buy_and_hold, label="Buy & Hold")
        plt.title(f"Backtest Result - {symbol} (6h)")
        plt.legend()
        plt.ylabel("Equity")
        
        plt.subplot(2, 1, 2)
        plt.plot(df["timestamp"], df["close"], label="Close Price", color="orange")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()