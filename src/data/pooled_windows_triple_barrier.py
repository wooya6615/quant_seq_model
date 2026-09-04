"""
Triple-barrier 라벨(pt_sl=(2,1), num_days=30 -- 실전 배포 모델과 동일 설정)로
20일 윈도우 시퀀스 데이터셋 생성.

⚠️ 이 프레임에서는 052690/118990이 검증 과정에서 탈락하고 064350 단독만
production 후보로 남았음(quant_xgboost PROJECT_SUMMARY.md 7절) -- 그래서
3종목 풀링이 아니라 064350 단독으로만 진행함. 데이터량이 단일 종목이라
3종목 풀링 실험 때보다도 적다는(오히려 더 불리한 조건) 점을 감안할 것.

⚠️ 라벨링 로직(체결지연 1일, 장중 High/Low 터치 판정, 거래일 기준 수직배리어)은
여기서 새로 계산하지 않음 -- quant_xgboost/src/labeling_triple_barrier.py가
이미 여러 차례 버그를 잡으며 검증한 로직이라, 재구현하면 미묘한 버그가 재발할
위험이 큼. 검증된 산출물 CSV를 그대로 재사용함.

사전 준비:
    quant_xgboost/data/064350_features_triple_barrier_pt2sl1_nd30_hl_base.csv
    파일을 이 레포의 data/ 폴더로 복사해둘 것.
    (quant_xgboost에서 compute_pbo_num_days_064350.py를 실행한 적이 있으면
    이미 생성돼 있을 것. 없으면 quant_xgboost에서 먼저:
        from feature_engineering_triple_barrier import build_triple_barrier_dataset
        df = build_triple_barrier_dataset(ticker="064350.KS", pt_sl=(2,1), num_days=30)
        df.to_csv("data/064350_features_triple_barrier_pt2sl1_nd30_hl_base.csv")
    )

⚠️ 사전 등록: WINDOW=20 (모델이 보는 과거 피처 구간 -- triple-barrier의
num_days=30(라벨의 최대 보유기간)과는 다른 개념이니 혼동하지 말 것).
pt_sl/num_days는 production 설정 그대로 고정.

사용법 (레포 루트에서):
    python -m src.data.pooled_windows_triple_barrier
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TICKER_KRX = "064350"
CONFIG_LABEL = "pt2sl1_nd30_hl"
WINDOW = 20

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]


def load_triple_barrier_dataset(ticker_krx: str = TICKER_KRX, config_label: str = CONFIG_LABEL) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{config_label}_base.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. quant_xgboost/data/{path.name}을 이 레포의 data/ 폴더로 복사할 것."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


def build_windows(df: pd.DataFrame, feature_cols: list, window: int = WINDOW):
    """
    반환: X(n, window, n_features), y(n,), dates(n,), ret_tb(n,), holding(n,)
    -- ret_tb/holding은 accuracy가 아니라 실제 백테스트(거래비용 반영, 가변
    보유기간)를 재현할 때 씀. 오늘 확인한 vs_base_rate 아티팩트 교훈을 반영해서
    이번엔 accuracy 기반 지표만으로 판정하지 않을 것.
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["label_tb_binary"].to_numpy(dtype=np.int64)
    ret_tb = df["ret_tb"].to_numpy(dtype=np.float32)
    holding = df["holding_rows_tb"].to_numpy(dtype=np.int64)
    dates = df.index.to_numpy()

    X, y, out_dates, out_ret, out_holding = [], [], [], [], []
    for i in range(window - 1, len(df)):
        X.append(values[i - window + 1: i + 1])
        y.append(labels[i])
        out_dates.append(dates[i])
        out_ret.append(ret_tb[i])
        out_holding.append(holding[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(out_dates),
        np.array(out_ret, dtype=np.float32),
        np.array(out_holding, dtype=np.int64),
    )


def build_and_save(ticker_krx: str = TICKER_KRX, window: int = WINDOW, save: bool = True):
    df = load_triple_barrier_dataset(ticker_krx)
    X, y, dates, ret_tb, holding = build_windows(df, FEATURE_COLS_BASE, window)

    print(f"{ticker_krx}: {X.shape[0]}개 윈도우, shape={X.shape[1:]}")
    print(f"기간: {pd.DatetimeIndex(dates).min().date()} ~ {pd.DatetimeIndex(dates).max().date()}")
    print(f"라벨 분포: {y.mean():.3f} (익절 비율)")

    if save:
        out_path = DATA_DIR / f"{ticker_krx}_windows_triple_barrier_{CONFIG_LABEL}.npz"
        np.savez(out_path, X=X, y=y, dates=dates, ret_tb=ret_tb, holding=holding)
        print(f"저장 완료: {out_path}")

    return X, y, dates, ret_tb, holding


def load_windows(ticker_krx: str = TICKER_KRX, filename: str = None):
    filename = filename or f"{ticker_krx}_windows_triple_barrier_{CONFIG_LABEL}.npz"
    data = np.load(DATA_DIR / filename, allow_pickle=True)
    return data["X"], data["y"], data["dates"], data["ret_tb"], data["holding"]


if __name__ == "__main__":
    build_and_save()