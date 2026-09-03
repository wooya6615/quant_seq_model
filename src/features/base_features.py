"""
BASE feature (모멘텀/변동성/거래량/상대강도 13개) + label 생성.
quant_xgboost/src/feature_engineering.py 로직을 그대로 이식함 (자기완결적 레포 유지 목적,
quant_cnn_chart와 동일한 원칙). 피처 정의 자체는 절대 바꾸지 않음 -- 이 레포의 실험 변수는
"모델이 이 피처들의 시간적 배열을 보는지"뿐이라, 피처가 달라지면 비교가 오염됨.
"""

import numpy as np
import pandas as pd
import yfinance as yf

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]


# ------------------------------------------------------------------
# 1. 데이터 수집
# ------------------------------------------------------------------
def load_data(ticker: str, benchmark: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    bench = yf.download(benchmark, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)

    df = df.dropna()
    bench = bench.dropna()
    return df, bench


# ------------------------------------------------------------------
# 2. 모멘텀 feature
# ------------------------------------------------------------------
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    for n in [5, 10, 20]:
        df[f"return_{n}d"] = close.pct_change(n)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line

    return df


# ------------------------------------------------------------------
# 3. 변동성 feature
# ------------------------------------------------------------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    daily_ret = close.pct_change()
    df["hist_vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_width"] = (upper - lower) / ma20
    df["bb_position"] = (close - lower) / (upper - lower)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    return df


# ------------------------------------------------------------------
# 4. 거래량 feature
# ------------------------------------------------------------------
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio_20d"] = volume / vol_ma20

    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()
    df["obv_change_20d"] = df["obv"].pct_change(20)

    return df


# ------------------------------------------------------------------
# 5. 상대강도 feature (벤치마크 대비)
# ------------------------------------------------------------------
def add_relative_strength_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    stock_ret = df["Close"].pct_change()
    bench_ret = bench["Close"].pct_change()

    aligned = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()

    for n in [5, 20]:
        stock_cum = (1 + aligned["stock"]).rolling(n).apply(lambda x: x.prod() - 1, raw=True)
        bench_cum = (1 + aligned["bench"]).rolling(n).apply(lambda x: x.prod() - 1, raw=True)
        df[f"excess_return_{n}d"] = (stock_cum - bench_cum).reindex(df.index)

    return df


# ------------------------------------------------------------------
# 6. Label 생성 (N일 후 방향성, 거래비용 반영)
# ------------------------------------------------------------------
def add_label(df: pd.DataFrame, horizon: int = 10, cost_threshold: float = 0.005) -> pd.DataFrame:
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["future_return"] = future_return
    df["label"] = (future_return > cost_threshold).astype(int)
    df.loc[df.index[-horizon:], "label"] = np.nan
    df.loc[df.index[-horizon:], "future_return"] = np.nan
    return df


# ------------------------------------------------------------------
# 실행 -- 단일 종목 BASE feature 데이터셋 생성
# ------------------------------------------------------------------
def build_feature_dataset(
    ticker: str,
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 10,
    cost_threshold: float = 0.005,
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)

    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)
    df = add_label(df, horizon=horizon, cost_threshold=cost_threshold)

    feature_cols = ["Close", "Volume", *FEATURE_COLS_BASE, "future_return", "label"]
    result = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return result