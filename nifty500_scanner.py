import json
import time
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Top liquid NSE Universe
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "TATAMOTORS.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "NTPC.NS", "M&M.NS"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_strike_step(price):
    if price < 250:
        return 2.5
    elif price < 500:
        return 5
    elif price < 1500:
        return 10
    elif price < 3000:
        return 20
    else:
        return 50

def run_scanner():
    print("Fetching live market data from NSE...")
    ist = pytz.timezone('Asia/Kolkata')
    current_market_time = datetime.now(ist).strftime("%H:%M:%S")

    results = []

    for sym in TICKERS:
        try:
            clean_sym = sym.replace(".NS", "")
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5d", interval="5m")

            if df.empty or len(df) < 25:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]

            ltp = float(close.iloc[-1])
            ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            rsi_series = calculate_rsi(close, 14)
            rsi = float(rsi_series.iloc[-1])

            # ATR calculation (14 periods)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if np.isnan(atr) or atr == 0:
                atr = ltp * 0.008

            # Decision Logic
            is_bullish = (ltp > ema9 > ema21) and (rsi >= 55)
            is_bearish = (ltp < ema9 < ema21) and (rsi <= 45)

            if is_bullish:
                signal = "BUY"
                win_prob = min(88, int(55 + (rsi - 50) * 1.5))
                risk = "LOW" if rsi < 70 else "MODERATE"
                tp = round(ltp + (1.5 * atr), 2)
                sl = round(ltp - (1.0 * atr), 2)
                step = get_strike_step(ltp)
                rounded_strike = round(ltp / step) * step
                strike = f"{int(rounded_strike) if rounded_strike % 1 == 0 else rounded_strike} CE"
            elif is_bearish:
                signal = "SELL"
                win_prob = min(88, int(55 + (50 - rsi) * 1.5))
                risk = "LOW" if rsi > 30 else "MODERATE"
                tp = round(ltp - (1.5 * atr), 2)
                sl = round(ltp + (1.0 * atr), 2)
                step = get_strike_step(ltp)
                rounded_strike = round(ltp / step) * step
                strike = f"{int(rounded_strike) if rounded_strike % 1 == 0 else rounded_strike} PE"
            else:
                signal = "BULLISH BIAS" if ltp > ema21 else "NEUTRAL"
                win_prob = 50
                risk = "MODERATE"
                tp = round(ltp * 1.015, 2)
                sl = round(ltp * 0.992, 2)
                step = get_strike_step(ltp)
                rounded_strike = round(ltp / step) * step
                strike = f"{int(rounded_strike) if rounded_strike % 1 == 0 else rounded_strike} CE"

            results.append({
                "symbol": clean_sym,
                "ltp": round(ltp, 2),
                "ema_9": round(ema9, 2),
                "ema_21": round(ema21, 2),
                "rsi": round(rsi, 1),
                "signal": signal,
                "win_prob": win_prob,
                "risk": risk,
                "strike": strike,
                "tp": tp,
                "sl": sl
            })
        except Exception as e:
            print(f"Error parsing {sym}: {e}")
            continue

    output = {
        "timestamp": current_market_time,
        "signals": results
    }

    with open("signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"[{current_market_time} IST] Successfully updated signals.json with {len(results)} symbols.")

if __name__ == "__main__":
    run_scanner()