import datetime
import time
import json
import concurrent.futures
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate

# Top liquid NSE market leaders
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    "KOTAKBANK.NS", "NTPC.NS", "ONGC.NS", "AXISBANK.NS", "TITAN.NS"
]

def fetch_macro_regime():
    regime = {
        "nifty_trend": "NEUTRAL",
        "nifty_change_pct": 0.0,
        "vix": 14.0,
        "volatility_state": "NORMAL",
        "dynamic_sl_pct": 0.008,
        "dynamic_tp_pct": 0.016
    }
    try:
        nifty = yf.Ticker("^NSEI").history(period="2d", interval="5m")
        if not nifty.empty and len(nifty) >= 10:
            nifty_close = nifty["Close"].iloc[-1]
            nifty_open = nifty["Open"].iloc[0]
            nifty_ema = nifty["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
            change = ((nifty_close - nifty_open) / nifty_open) * 100
            regime["nifty_change_pct"] = round(change, 2)
            regime["nifty_trend"] = "BULLISH" if nifty_close > nifty_ema else "BEARISH"

        vix = yf.Ticker("^INDIAVIX").history(period="1d", interval="5m")
        if not vix.empty:
            vix_val = float(vix["Close"].iloc[-1])
            regime["vix"] = round(vix_val, 2)
            if vix_val >= 18.0:
                regime["volatility_state"] = "HIGH"
                regime["dynamic_sl_pct"] = 0.012
                regime["dynamic_tp_pct"] = 0.024
            elif vix_val <= 12.5:
                regime["volatility_state"] = "LOW"
                regime["dynamic_sl_pct"] = 0.006
                regime["dynamic_tp_pct"] = 0.012
    except Exception as e:
        print(f"Macro fetch notice: {e}")

    return regime

def calculate_option_strike(price: float) -> int:
    step = 50 if price > 2000 else (20 if price > 1000 else (10 if price > 500 else 5))
    return int(round(price / step) * step)

def analyze_stock(symbol: str, macro: dict):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="5m")
        if df.empty or len(df) < 25:
            return None

        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(curr["Close"])
        ema9 = float(curr["EMA9"])
        ema21 = float(curr["EMA21"])
        rsi = float(curr["RSI"])
        vol = int(curr["Volume"])

        bullish_cross = (prev["EMA9"] <= prev["EMA21"]) and (ema9 > ema21)
        bearish_cross = (prev["EMA9"] >= prev["EMA21"]) and (ema9 < ema21)
        clean_sym = symbol.replace(".NS", "")
        strike = calculate_option_strike(price)

        base_prob = 50
        signal = "NEUTRAL"
        risk_factor = "MODERATE"
        rec_option = "HOLD"

        if bullish_cross and (50 <= rsi <= 68):
            signal = "BUY"
            base_prob = int(min(92, 65 + (rsi - 50) * 1.5))
            rec_option = f"{strike} CE"
        elif ema9 > ema21 and rsi > 54:
            signal = "BULLISH BIAS"
            base_prob = 62
            rec_option = f"{strike} CE"
        elif bearish_cross and (32 <= rsi <= 50):
            signal = "SELL"
            base_prob = int(min(90, 65 + (50 - rsi) * 1.5))
            rec_option = f"{strike} PE"
        elif ema9 < ema21 and rsi < 46:
            signal = "BEARISH BIAS"
            base_prob = 60
            rec_option = f"{strike} PE"

        if "BUY" in signal or "BULLISH" in signal:
            if macro["nifty_trend"] == "BULLISH":
                base_prob += 6
                risk_factor = "LOW" if rsi < 62 else "MODERATE"
            else:
                base_prob -= 12
                risk_factor = "HIGH"
        elif "SELL" in signal or "BEARISH" in signal:
            if macro["nifty_trend"] == "BEARISH":
                base_prob += 6
                risk_factor = "LOW" if rsi > 38 else "MODERATE"
            else:
                base_prob -= 12
                risk_factor = "HIGH"

        base_prob = max(35, min(95, base_prob))
        sl_pct = macro["dynamic_sl_pct"]
        tp_pct = macro["dynamic_tp_pct"]

        is_long = "BUY" in signal or "BULLISH" in signal
        stop_loss = round(price * (1 - sl_pct), 2) if is_long else round(price * (1 + sl_pct), 2)
        target = round(price * (1 + tp_pct), 2) if is_long else round(price * (1 - tp_pct), 2)

        return {
            "symbol": clean_sym,
            "price": round(price, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "rsi": round(rsi, 1),
            "volume": vol,
            "signal": signal,
            "probability": f"{base_prob}%",
            "risk": risk_factor,
            "option_strike": rec_option,
            "target": target,
            "stop_loss": stop_loss,
            "updated_at": datetime.datetime.now().strftime("%H:%M:%S")
        }
    except Exception:
        return None

def run_scanner():
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    macro = fetch_macro_regime()

    print("\n" + "=" * 80)
    print(f"🌍 MACRO STATUS: NIFTY 50 [{macro['nifty_trend']}] ({macro['nifty_change_pct']:+.2f}%) | "
          f"INDIA VIX: {macro['vix']} ({macro['volatility_state']})")
    print(f"⚙️ ADAPTIVE ENGINE: Dynamic SL {macro['dynamic_sl_pct']*100:.1f}% | TP {macro['dynamic_tp_pct']*100:.1f}%")
    print(f"🚀 SCANNING {len(WATCHLIST)} SYMBOLS AT {now_time}")
    print("=" * 80)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(analyze_stock, sym, macro): sym for sym in WATCHLIST}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            if res:
                results.append(res)

    if not results:
        print("No market data returned.")
        return

    payload = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "macro": macro,
        "total_scanned": len(results),
        "stocks": results
    }

    with open("signals.json", "w") as f:
        json.dump(payload, f, indent=4)

    df_res = pd.DataFrame(results)
    table_cols = ["symbol", "price", "rsi", "signal", "probability", "risk", "option_strike"]
    print(tabulate(df_res[table_cols], headers=["Symbol", "Price (₹)", "RSI", "Signal", "Win Prob", "Risk", "Option"],
                   tablefmt="fancy_grid", showindex=False))

    print(f"\n[✓] signals.json updated ({len(results)} symbols)")

if __name__ == "__main__":
    INTERVAL_SECONDS = 180  # 3 minutes

    print("=" * 80)
    print(f"  🇮🇳 NIFTY INTRADAY SCANNER (INTERVAL: {INTERVAL_SECONDS // 60} MINUTES)")
    print("  Sync Target: signals.json")
    print("=" * 80)

    try:
        while True:
            run_scanner()
            print(f"\nSleeping {INTERVAL_SECONDS // 60} minutes until next cycle...")
            for remaining in range(INTERVAL_SECONDS, 0, -10):
                mins, secs = divmod(remaining, 60)
                print(f"\rNext scan in: {mins:02d}:{secs:02d} | Press Ctrl+C to stop", end="", flush=True)
                time.sleep(10)
            print("\n")
    except KeyboardInterrupt:
        print("\n\nScanner halted.")