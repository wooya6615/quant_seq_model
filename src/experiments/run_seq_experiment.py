"""
GRU vs TCN vs 동일가중(=base rate), 날짜 기준 walk-forward, 5-seed 비교.
quant_cnn_chart의 run_cnn_experiment.py와 동일한 1차 검증 철학 --
동일가중조차 못 이기면 XGBoost(BASE) 비교는 의미 없으므로, 이 단계부터 통과해야
다음(진짜 XGBoost-BASE 대비 비교, KOSPI200 확장 등)으로 넘어갈 근거가 생김.

⚠️ 사전 등록 (실험 전 고정, 결과 보고 나서 바꾸지 않음):
- WINDOW=20, HORIZON=10, 대상 3종목(064350/052690/118990) 풀링
  (pooled_windows.py에서 이미 고정됨)
- 정규화(z-score)는 매 fold의 train 구간 평균/표준편차로만 계산 (train->test 누수 방지)
- TRAIN_DAYS=300, TEST_DAYS=60, STEP_DAYS=60, EMBARGO_DAYS=10(=horizon)
  -- quant_xgboost/quant_cnn_chart의 walk-forward 파라미터와 동일한 스케일
- 5 seed: 42/1/7/123/2024
- EPOCHS=10, LR=1e-3, BATCH_SIZE=64 -- ChartCNN 실험처럼 "일단 하나의 설정으로
  고정하고 seed 안정성부터 확인"하는 게 우선. 하이퍼파라미터 탐색은 여기서 신호가
  확인된 다음 단계.

사용법 (레포 루트에서):
    python -m src.experiments.run_seq_experiment

⚠️ 이 레포는 src/data, src/features, src/models처럼 하위 폴더가 나뉜 구조라
   (quant_xgboost의 flat 구조와 다르게) 모듈 경로 임포트(from src.xxx import ...)를
   써야 해서 -m 실행이 필요함.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data.pooled_windows import load_pooled_windows
from src.models.gru_classifier import GRUClassifier
from src.models.tcn_classifier import TCNClassifier

TRAIN_DAYS = 300
TEST_DAYS = 60
STEP_DAYS = 60
EMBARGO_DAYS = 10  # horizon과 동일
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64
SEEDS = (42, 1, 7, 123, 2024)

# 3종목 -> KOSPI200 50종목 pilot -> 전체 200종목 순으로 이 값만 바꿔서 실행하면 됨
# ("pooled_windows_kospi200_pilot.npz" / "pooled_windows_kospi200_full.npz")
POOLED_WINDOWS_FILE = "pooled_windows_kospi200_pilot.npz"

# 시간 오래 걸리면 여기서 하나만 남겨서 따로 돌리면 됨 (예: MODELS = ["TCN"])
MODELS = ["TCN"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_BUILDERS = {
    "GRU": lambda n_features: GRUClassifier(n_features=n_features),
    "TCN": lambda n_features: TCNClassifier(n_features=n_features),
}


# ------------------------------------------------------------------
# 1. 날짜 기준 Walk-Forward 분할 (quant_cnn_chart와 동일 패턴)
# ------------------------------------------------------------------
def walk_forward_splits_by_date(dates, train_days: int, test_days: int, step_days: int, embargo_days: int):
    unique_dates = pd.DatetimeIndex(dates).unique().sort_values()
    n = len(unique_dates)

    splits = []
    start = 0
    while start + train_days + embargo_days + test_days <= n:
        train_dates = unique_dates[start: start + train_days]
        test_start = start + train_days + embargo_days
        test_dates = unique_dates[test_start: test_start + test_days]
        splits.append((train_dates, test_dates))
        start += step_days
    return splits


# ------------------------------------------------------------------
# 2. fold별 정규화 (train 통계로만 -- 누수 방지)
# ------------------------------------------------------------------
def normalize_fold(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ------------------------------------------------------------------
# 3. fold 하나 학습 + 평가
# ------------------------------------------------------------------
def train_and_eval_fold(X, y, dates, train_dates, test_dates, model_name: str, random_state: int):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    date_index = pd.DatetimeIndex(dates)
    train_mask = date_index.isin(train_dates)
    test_mask = date_index.isin(test_dates)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    X_train, X_test = normalize_fold(X_train, X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    model = MODEL_BUILDERS[model_name](n_features=X.shape[2]).to(DEVICE)
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

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X_test_t), dim=1)[:, 1].cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    base_rate = y_test.mean()

    return {
        "auc": roc_auc_score(y_test, proba),
        "vs_base_rate": accuracy_score(y_test, pred) - max(base_rate, 1 - base_rate),
        "n_test": len(y_test),
    }


# ------------------------------------------------------------------
# 4. 5-seed 검증
# ------------------------------------------------------------------
def run_walk_forward(X, y, dates, model_name: str, random_state: int):
    splits = walk_forward_splits_by_date(dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, EMBARGO_DAYS)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요. TRAIN_DAYS/TEST_DAYS를 줄이세요.")

    rows = []
    for train_dates, test_dates in splits:
        result = train_and_eval_fold(X, y, dates, train_dates, test_dates, model_name, random_state)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def run_multi_seed(X, y, dates, model_name: str, seeds=SEEDS):
    rows = []
    for seed in seeds:
        t0 = time.time()
        fold_df = run_walk_forward(X, y, dates, model_name, random_state=seed)
        rows.append({
            "seed": seed,
            "mean_auc": fold_df["auc"].mean(),
            "mean_vs_base_rate": fold_df["vs_base_rate"].mean(),
            "win_folds": int((fold_df["vs_base_rate"] > 0).sum()),
            "n_folds": len(fold_df),
        })
        print(f"  [{model_name}] seed={seed}: AUC={rows[-1]['mean_auc']:.4f}, "
              f"vs_base_rate={rows[-1]['mean_vs_base_rate']:+.4f} "
              f"({rows[-1]['win_folds']}/{rows[-1]['n_folds']} fold 승, {time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if DEVICE.type == "cuda":
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU를 못 찾아서 CPU로 돌아감\n")

    X, y, dates, tickers_arr = load_pooled_windows(filename=POOLED_WINDOWS_FILE)
    print(f"윈도우: {X.shape[0]}개, shape={X.shape[1:]}, 종목: {len(np.unique(tickers_arr))}개")
    print(f"기간: {pd.DatetimeIndex(dates).min().date()} ~ {pd.DatetimeIndex(dates).max().date()}")
    print(f"라벨 분포: {y.mean():.3f} (1의 비율)\n")

    all_results = {}
    for model_name in MODELS:
        print(f"=== {model_name} vs 동일가중 (5-seed walk-forward) ===")
        seed_df = run_multi_seed(X, y, dates, model_name)
        print("\n" + seed_df.round(4).to_string(index=False))
        wins = int((seed_df["mean_vs_base_rate"] > 0).sum())
        print(f"\n{model_name}: vs_base_rate 5/5 양수 {wins}/5 {'(통과)' if wins == 5 else '(일관성 실패)'}\n")
        all_results[model_name] = seed_df

    print("=" * 60)
    print("여기서 둘 다 5/5 통과 못하면 -- lag feature 실험과 동일한 기준 --")
    print("시퀀스 모델링 자체를 이 데이터량(3종목 풀링)으로는 접거나, quant_cnn_chart처럼")
    print("KOSPI200 규모로 풀링을 넓혀서 데이터량부터 늘려볼 것.")
    print("둘 중 하나라도 통과하면 다음 단계: 같은 3종목 XGBoost(BASE) baseline과 정면 비교.")