"""
검증된 3종목(현대로템/한전기술/모트렉스) BASE feature를 WINDOW일치 시퀀스로 만들어 풀링.
외부 데이터 소스는 기존 quant_xgboost와 동일(yfinance) -- 새로 조회할 것 없음.

⚠️ 사전 등록 (실험 전 고정):
- WINDOW = 20 거래일 (오늘 포함 과거 20일)
- HORIZON = 10, cost_threshold = 0.005 (기존 XGBoost 실험 기본값과 동일해서 비교 가능하게)
- 118990(모트렉스)은 코스닥이라 yfinance 티커가 .KQ (KRX 코드가 아니라 yfinance 접미사 규칙)
- 정규화(z-score)는 여기서 하지 않음 -- fold별 train 구간 통계로만 정규화해야 누수가
  없어서, 정규화는 run_seq_experiment.py의 fold 루프 안에서 처리함
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.base_features import build_feature_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    ("064350.KS", "064350"),  # 현대로템
    ("052690.KS", "052690"),  # 한전기술
    ("118990.KQ", "118990"),  # 모트렉스 (코스닥)
]
WINDOW = 20
HORIZON = 10
COST_THRESHOLD = 0.005


# ------------------------------------------------------------------
# 1. 슬라이딩 윈도우 생성
# ------------------------------------------------------------------
def build_windows(df: pd.DataFrame, feature_cols: list, window: int):
    """
    df: 날짜순 정렬된 단일 종목 feature 데이터프레임 (build_feature_dataset 결과)
    반환: X(n, window, n_features), y(n,), dates(n,) -- 각 샘플의 날짜는 윈도우 마지막 날(오늘)
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    dates = df.index.to_numpy()

    X, y, out_dates = [], [], []
    for i in range(window - 1, len(df)):
        X.append(values[i - window + 1: i + 1])
        y.append(labels[i])
        out_dates.append(dates[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(out_dates)


# ------------------------------------------------------------------
# 2. 3종목 풀링 + 저장
# ------------------------------------------------------------------
def build_pooled_windows(
    tickers: list = None,
    window: int = WINDOW,
    horizon: int = HORIZON,
    cost_threshold: float = COST_THRESHOLD,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    save: bool = True,
):
    tickers = tickers if tickers is not None else TICKERS

    all_X, all_y, all_dates, all_tickers = [], [], [], []
    for yf_ticker, krx in tickers:
        df = build_feature_dataset(
            ticker=yf_ticker, start=start, end=end,
            horizon=horizon, cost_threshold=cost_threshold,
        )
        X, y, dates = build_windows(df, FEATURE_COLS_BASE, window)
        print(f"{krx}: {X.shape[0]}개 윈도우 생성 ({df.index.min().date()} ~ {df.index.max().date()})")

        all_X.append(X)
        all_y.append(y)
        all_dates.append(dates)
        all_tickers.extend([krx] * len(y))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    dates = np.concatenate(all_dates, axis=0)
    tickers_arr = np.array(all_tickers)

    print(f"\n풀링 완료: 총 {X.shape[0]}개 윈도우, shape={X.shape[1:]} (window={window}, n_features={len(FEATURE_COLS_BASE)})")
    print(f"라벨 분포: {y.mean():.3f} (1의 비율)")

    if save:
        out_path = DATA_DIR / "pooled_windows_3tickers.npz"
        np.savez(out_path, X=X, y=y, dates=dates, tickers=tickers_arr)
        print(f"저장 완료: {out_path}")

    return X, y, dates, tickers_arr


def load_pooled_windows(filename: str = "pooled_windows_3tickers.npz"):
    data = np.load(DATA_DIR / filename, allow_pickle=True)
    return data["X"], data["y"], data["dates"], data["tickers"]


if __name__ == "__main__":
    build_pooled_windows()