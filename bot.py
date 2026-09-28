import datetime
import math
import time
import pandas as pd
import numpy as np
import yfinance as yf

class IndianStockIntradayAgent:
    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        capital: float = 100000.0,
        risk_per_trade: float = 0.01,       # 1% capital risk per trade
        stop_loss_pct: float = 0.008,       # 0.8% tight intraday stop-loss
        target_pct: float = 0.016,          # 1.6% target (1:2 Risk-to-Reward)
        market_start: str = "09:15",
        square_off_time: str = "15:15"      # Mandatory intraday square-off (IST)
    ):
        self.symbol = symbol.upper()
        self.capital = capital
        self.cash = capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        
        self.market_start = datetime.datetime.strptime(market_start, "%H:%M").time()
        self.square_off_time = datetime.datetime.strptime(square_off_time, "%H:%M").time()

        # Position tracking
        self.position = None  # None, 'LONG', or 'SHORT'
        self.position_qty = 0
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.target_price = 0.0
        self.total_pnl = 0.0

    def log(self, message: str):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] {message}")

    def fetch_data(self) -> pd.DataFrame:
        """Fetch 5-minute candles for the last 5 days to ensure ample indicator history."""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period="5d", interval="5m")
            if df.empty or len(df) < 30:
                return pd.DataFrame()

            # 1. Fast & Slow Exponential Moving Averages
            df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
            df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

            # 2. Relative Strength Index (RSI - 14)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df["RSI"] = 100 - (100 / (1 + rs))

            # 3. Average True Range (ATR - 10) for volatility
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(window=10).mean()

            return df
        except Exception as e:
            self.log(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def get_position_size(self, price: float) -> int:
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = price * self.stop_loss_pct
        if risk_per_share <= 0:
            return 0
        qty = math.floor(risk_amount / risk_per_share)
        return max(1, qty)

    def print_action_box(self, action: str, price: float, reason: str):
        border = "=" * 60
        print(f"\n{border}")
        print(f"📢 SIGNAL ALERT: {action} on {self.symbol}")
        print(f"Price: ₹{price:,.2f} | Reason: {reason}")
        if self.position == "LONG":
            print(f"Target (TP): ₹{self.target_price:,.2f} (+{self.target_pct*100:.1f}%)")
            print(f"Stop-Loss (SL): ₹{self.stop_loss_price:,.2f} (-{self.stop_loss_pct*100:.1f}%)")
            print(f"Suggested Qty: {self.position_qty} shares")
        print(f"{border}\n")

    def run_check(self):
        now = datetime.datetime.now().time()
        df = self.fetch_data()

        if df.empty:
            self.log("Waiting for market data connection...")
            return

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(curr["Close"])
        ema9 = float(curr["EMA9"])
        ema21 = float(curr["EMA21"])
        rsi = float(curr["RSI"])

        # Status Line
        status = f"IN TRADE ({self.position} @ ₹{self.entry_price:.2f})" if self.position else "WATCHING / FLAT"
        self.log(f"{self.symbol} | LTP: ₹{price:.2f} | 9 EMA: ₹{ema9:.2f} | 21 EMA: ₹{ema21:.2f} | RSI: {rsi:.1f} | {status}")

        # Rule 1: Intraday Mandatory Square-Off (15:15 IST)
        if now >= self.square_off_time:
            if self.position is not None:
                pnl = (price - self.entry_price) * self.position_qty
                self.print_action_box("MANDATORY CLOSE (SELL)", price, "End-of-day square-off time (15:15 IST)")
                self.total_pnl += pnl
                self.position = None
            return

        # Rule 2: Active Trade Management (Exit triggers)
        if self.position == "LONG":
            if price <= self.stop_loss_price:
                pnl = (price - self.entry_price) * self.position_qty
                self.total_pnl += pnl
                self.print_action_box("EXIT (STOP LOSS HIT)", price, f"Loss: ₹{pnl:,.2f}")
                self.position = None
                return

            elif price >= self.target_price:
                pnl = (price - self.entry_price) * self.position_qty
                self.total_pnl += pnl
                self.print_action_box("EXIT (TARGET ACHIEVED)", price, f"Profit: ₹{pnl:,.2f}")
                self.position = None
                return

            # Technical Exit: EMA Bearish Cross
            elif prev["EMA9"] >= prev["EMA21"] and curr["EMA9"] < curr["EMA21"]:
                pnl = (price - self.entry_price) * self.position_qty
                self.total_pnl += pnl
                self.print_action_box("EXIT (MOMENTUM WEAKENED)", price, f"9 EMA crossed below 21 EMA. P&L: ₹{pnl:,.2f}")
                self.position = None
                return

        # Rule 3: Entry Trigger (Only when Flat)
        elif self.position is None:
            # Bullish Condition: 9 EMA crosses above 21 EMA AND RSI is healthy (above 50, below overbought 70)
            bullish_cross = (prev["EMA9"] <= prev["EMA21"]) and (curr["EMA9"] > curr["EMA21"])
            rsi_confirmation = 50 < rsi < 70

            if bullish_cross and rsi_confirmation:
                self.position = "LONG"
                self.position_qty = self.get_position_size(price)
                self.entry_price = price
                self.stop_loss_price = round(price * (1.0 - self.stop_loss_pct), 2)
                self.target_price = round(price * (1.0 + self.target_pct), 2)
                self.print_action_box("BUY ENTRY", price, "Bullish 9/21 EMA crossover + RSI > 50")


if __name__ == "__main__":
    # Choose any Indian stock: "RELIANCE.NS", "TCS.NS", "INFY.NS", "TATAMOTORS.NS"
    STOCK = "RELIANCE.NS"
    agent = IndianStockIntradayAgent(symbol=STOCK, capital=100000.0)

    print("=" * 65)
    print(f"  🇮🇳 INTRADAY BUY/SELL SIGNAL AGENT - {agent.symbol}")
    print(f"  Virtual Capital : ₹{agent.capital:,.2f}")
    print(f"  Risk Setup      : SL: 0.8% | Target: 1.6% (1:2 Risk/Reward)")
    print(f"  Polling Rate    : Checking live 5-minute ticks every 30 seconds")
    print("=" * 65)

    try:
        while True:
            agent.run_check()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nAgent stopped by user.")
