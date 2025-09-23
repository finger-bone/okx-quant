import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
SECRET_KEY = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
FLAG = os.getenv("OKX_FLAG", "1")  # 1 for demo, 0 for live

SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT").split(",")
WEIGHTS = list(map(float, os.getenv("WEIGHTS", "0.5,0.5").split(",")))

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 120))
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
ORDER_USDT_BASE = float(os.getenv("ORDER_USDT_BASE", 100))
