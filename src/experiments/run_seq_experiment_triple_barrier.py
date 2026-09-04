"""
Triple-barrier 프레임(pt_sl=(2,1), num_days=30 -- 실전 배포 모델과 동일 설정)으로
GRU/TCN 재시도.

오늘 확인한 교훈 반영: vs_base_rate는 fold별 test_base_rate 드리프트 때문에
불공정한 지표였음 -- 이번엔 처음부터 accuracy 기반 지표를 아예 쓰지 않고,
바로 실제 백테스트(거래비용 반영, 고정 B&H 구간 대비)로 판정함.

⚠️ 사전 등록: WINDOW=20, pt_sl=(2,1), num_days=30(=embargo), TRAIN=300/TEST=60/
STEP=60, threshold=[0.50,0.55,0.60,0.65,0.70](production 검증 때와 동일 후보군),
5-seed(42/1/7/123/2024), ROUND_TRIP_COST=0.002. 064350 단독(이 프레임에서 유일하게
production까지 간 종목).

방법: proba >= threshold인 날 진입, holding_rows_tb(가변, triple-barrier 배리어
도달까지 걸린 실제 거래일)만큼 보유 후 ret_tb로 청산, 다음 진입은 그만큼 건너뜀
(중복 방지). 고정 B&H 구간(전체 out-of-sample span)과 비교.

⚠️ 최적화: threshold는 거래 추출 단계에서만 쓰이고 모델 학습에는 영향 없으므로,
(model, seed) 조합당 학습은 1번만 하고 fold별 (test_idx, proba)를 캐싱해서
5개 threshold 각각은 재학습 없이 거래 추출만 반복함 (compute_pbo_joint_pt_sl_
threshold_064350.py와 동일한 방식).

사용법 (레포 루트에서):
    python -m src.experiments.run_seq_experiment_triple_barrier
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.pooled_windows_triple_barrier import load_windows, load_triple_barrier_dataset
from src.models.gru_classifier import GRUClassifier
from src.models.tcn_classifier import TCNClassifier

TRAIN_DAYS = 300
TEST_DAYS = 60
STEP_DAYS = 60
NUM_DAYS = 30
EMBARGO_DAYS = NUM_DAYS
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64
SEEDS = (42, 1, 7, 123, 2024)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
ROUND_TRIP_COST = 0.002

MODELS = ["GRU", "TCN"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_BUILDERS = {
    "GRU": lambda n_features: GRUClassifier(n_features=n_features),
    "TCN": lambda n_features: TCNClassifier(n_features=n_features),
}


# ------------------------------------------------------------------
# 1. Walk-Forward 분할 (position 기준)
# ------------------------------------------------------------------
def walk_forward_splits_by_position(n_rows: int, train_days: int, test_days: int, step_days: int, embargo_days: int):
    splits = []
    start = 0
    while start + train_days + embargo_days + test_days <= n_rows:
        train_idx = list(range(start, start + train_days))
        test_start = start + train_days + embargo_days
        test_idx = list(range(test_start, test_start + test_days))
        splits.append((train_idx, test_idx))
        start += step_days
    return splits


# ------------------------------------------------------------------
# 2. 모델 학습 (fold 하나)
# ------------------------------------------------------------------
def train_model(X: np.ndarray, y: np.ndarray, train_idx: list, model_name: str, random_state: int, n_features: int):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_train_raw = X[train_idx]
    y_train = y[train_idx]

    mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
    std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train_raw - mean) / std

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model = MODEL_BUILDERS[model_name](n_features=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n_train = len(X_train_t)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_train_t[idx].to(DEVICE)
            yb = y_train_t[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model, mean, std


# ------------------------------------------------------------------
# 3. (model, seed)당 1번만 학습 -- fold별 (test_idx, proba) 캐싱
# ------------------------------------------------------------------
def compute_fold_probas(X: np.ndarray, y: np.ndarray, model_name: str, random_state: int):
    n_features = X.shape[2]
    splits = walk_forward_splits_by_position(len(X), TRAIN_DAYS, TEST_DAYS, STEP_DAYS, EMBARGO_DAYS)

    fold_results = []
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            continue

        model, mean, std = train_model(X, y, train_idx, model_name, random_state, n_features)

        X_test = (X[test_idx] - mean) / std
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(X_test_t), dim=1)[:, 1].cpu().numpy()

        fold_results.append((test_idx, proba))

    return fold_results


# ------------------------------------------------------------------
# 4. threshold별 거래 추출 (재학습 없이 캐싱된 proba로만)
# ------------------------------------------------------------------
def extract_trades(fold_results: list, dates: np.ndarray, ret_tb: np.ndarray, holding: np.ndarray, threshold: float) -> pd.DataFrame:
    trades = []
    for test_idx, proba in fold_results:
        i = 0
        while i < len(test_idx):
            if proba[i] >= threshold:
                row_idx = test_idx[i]
                trades.append({
                    "entry_date": dates[row_idx],
                    "proba": proba[i],
                    "gross_return": ret_tb[row_idx],
                    "net_return": ret_tb[row_idx] - ROUND_TRIP_COST,
                })
                i += max(int(holding[row_idx]), 1)
            else:
                i += 1
    return pd.DataFrame(trades)


# ------------------------------------------------------------------
# 5. 고정 B&H 구간 (fold별 seed 무관, 전체 out-of-sample span 하나로 고정)
# ------------------------------------------------------------------
def get_full_test_period(n_rows: int, dates: np.ndarray):
    splits = walk_forward_splits_by_position(n_rows, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, EMBARGO_DAYS)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return pd.Timestamp(dates[first_test_idx]), pd.Timestamp(dates[last_test_idx])


def buy_and_hold_return(price_df: pd.DataFrame, start_date, end_date) -> float:
    period = price_df.loc[start_date:end_date]
    if len(period) < 2:
        return np.nan
    return period["Close"].iloc[-1] / period["Close"].iloc[0] - 1


def summarize_trades(trades: pd.DataFrame, bh_return: float) -> dict:
    if trades.empty:
        return {
            "n_trades": 0, "total_net_return": np.nan, "win_rate": np.nan,
            "bh_return": bh_return, "excess_vs_bh": np.nan,
        }
    total_net_return = (1 + trades["net_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()
    return {
        "n_trades": len(trades),
        "total_net_return": total_net_return,
        "win_rate": win_rate,
        "bh_return": bh_return,
        "excess_vs_bh": total_net_return - bh_return,
    }


if __name__ == "__main__":
    if DEVICE.type == "cuda":
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU를 못 찾아서 CPU로 돌아감\n")

    X, y, dates, ret_tb, holding = load_windows()
    price_df = load_triple_barrier_dataset()[["Close"]]

    full_start, full_end = get_full_test_period(len(X), dates)
    fixed_bh_return = buy_and_hold_return(price_df, full_start, full_end)
    print(f"윈도우: {X.shape[0]}개, shape={X.shape[1:]}")
    print(f"고정 B&H 비교 구간: {full_start.date()} ~ {full_end.date()} (수익률 {fixed_bh_return:+.2%})\n")

    rows = []
    for model_name in MODELS:
        for seed in SEEDS:
            t0 = time.time()
            fold_results = compute_fold_probas(X, y, model_name, seed)
            print(f"[{model_name}] seed={seed} 학습 완료 ({time.time() - t0:.1f}s), threshold별 거래 추출 중...")

            for threshold in THRESHOLDS:
                trades = extract_trades(fold_results, dates, ret_tb, holding, threshold)
                summary = summarize_trades(trades, fixed_bh_return)
                summary.update({"model": model_name, "threshold": threshold, "seed": seed})
                rows.append(summary)

                if summary["n_trades"]:
                    print(f"  threshold={threshold}: {summary['n_trades']}건, "
                          f"순수익률={summary['total_net_return']:+.2%}, "
                          f"승률={summary['win_rate']:.1%}, "
                          f"초과수익={summary['excess_vs_bh']:+.2%}")
                else:
                    print(f"  threshold={threshold}: 거래 없음")

    result_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("=== model x threshold별 5-seed 요약 ===")
    print("=" * 70)
    grouped = result_df.groupby(["model", "threshold"]).agg(
        mean_n_trades=("n_trades", "mean"),
        mean_net_return=("total_net_return", "mean"),
        mean_excess_vs_bh=("excess_vs_bh", "mean"),
        seeds_beat_bh=("excess_vs_bh", lambda s: (s > 0).sum()),
    )
    print(grouped.round(4).to_string())

    print("\n5-seed 전부(5/5) excess_vs_bh > 0인 (model, threshold) 조합이 있으면 -- ")
    print("triple-barrier 프레임에서 시퀀스 모델이 XGBoost(production, threshold=0.60)")
    print("대비 우위가 있는지 다음 단계로 비교. 하나도 없으면 -- 시퀀스 모델링 자체를")
    print("이 종목/프레임에서는 접는 게 맞음.")