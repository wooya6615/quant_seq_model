"""
GRU/TCN 실험에서 threshold=0.5 고정이 문제였는지 확인.

직전 실험에서 GRU/TCN의 vs_base_rate(-8.9~-9.3%p)가 lag feature 실험의 XGBoost
BASE vs_base_rate(-9.6~-10.4%p)와 거의 같은 폭으로 나온 게 이상했음 -- 모델 구조가
완전히 다른데 같은 패턴이 반복된다는 건, 모델이 아니라 "threshold=0.5로 딱 자르는
평가 방식" 자체가 문제일 가능성을 시사함. 이 실험은 그 가설을 확인함.

방법: 각 fold의 train 구간(300일)을 train_fit(앞 80%, 240일)/train_val(뒤 20%, 60일)로
다시 쪼개서, train_fit으로만 모델을 학습하고 train_val에서 Youden's J(TPR-FPR 최대화)로
최적 threshold를 찾은 다음, 그 threshold를 test에 적용.
-- test 라벨은 threshold 선택에 전혀 안 쓰임 (train_val은 여전히 train 구간 안이라
   미래 정보 누출 없음). 같은 fold, 같은 학습된 모델로 threshold=0.5 결과와 최적
   threshold 결과를 나란히 비교함.

⚠️ 사전 등록:
- VAL_FRAC = 0.2 (train 300일 중 뒤쪽 60일을 threshold 검증용으로 뗌)
- 그 외 WINDOW/HORIZON/모델 구조/walk-forward 파라미터는 run_seq_experiment.py와
  100% 동일 -- 그래야 성능 차이가 순수하게 "threshold 선택 방식" 효과인지 확인 가능
- ⚠️ 트레이드오프: 실제 모델 학습에 쓰는 데이터가 300일 -> 240일로 줄어듦. 여기서
  성능이 나빠지면 "threshold 문제"가 아니라 "학습 데이터가 줄어든 효과"와 섞여있을
  수 있다는 점을 해석할 때 감안할 것.

사용법 (레포 루트에서):
    python -m src.experiments.run_seq_experiment_opt_threshold
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from src.data.pooled_windows import load_pooled_windows
from src.models.gru_classifier import GRUClassifier
from src.models.tcn_classifier import TCNClassifier

TRAIN_DAYS = 300
TEST_DAYS = 60
STEP_DAYS = 60
EMBARGO_DAYS = 10
VAL_FRAC = 0.2
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 64
SEEDS = (42, 1, 7, 123, 2024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_BUILDERS = {
    "GRU": lambda n_features: GRUClassifier(n_features=n_features),
    "TCN": lambda n_features: TCNClassifier(n_features=n_features),
}


# ------------------------------------------------------------------
# 1. 날짜 기준 Walk-Forward 분할 (run_seq_experiment.py와 동일)
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


def split_train_val(train_dates, val_frac: float = VAL_FRAC):
    n_val = max(1, int(len(train_dates) * val_frac))
    return train_dates[:-n_val], train_dates[-n_val:]


def find_optimal_threshold(y_val, proba_val) -> float:
    """Youden's J = TPR - FPR를 최대화하는 threshold. 검증셋이 단일 클래스면 0.5로 폴백."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_val, proba_val)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ------------------------------------------------------------------
# 2. fold 하나 학습 + 평가 (threshold=0.5 / 최적 threshold 둘 다 계산)
# ------------------------------------------------------------------
def train_and_eval_fold(X, y, dates, train_dates, test_dates, model_name: str, random_state: int):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    date_index = pd.DatetimeIndex(dates)
    train_fit_dates, train_val_dates = split_train_val(train_dates)

    train_mask = date_index.isin(train_dates)      # 정규화 통계용 (train 전체)
    fit_mask = date_index.isin(train_fit_dates)     # 실제 학습용 (앞 80%)
    val_mask = date_index.isin(train_val_dates)     # threshold 선택용 (뒤 20%)
    test_mask = date_index.isin(test_dates)

    X_train_all = X[train_mask]
    X_fit, y_fit = X[fit_mask], y[fit_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_test)) < 2 or len(y_val) == 0:
        return None

    # 정규화 통계는 train 전체(fit+val)로 계산 -- run_seq_experiment.py와 동일 기준
    mean = X_train_all.mean(axis=(0, 1), keepdims=True)
    std = X_train_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_fit = (X_fit - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    model = MODEL_BUILDERS[model_name](n_features=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n_fit = len(X_fit_t)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n_fit)
        for i in range(0, n_fit, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_fit_t[idx].to(DEVICE)
            yb = y_fit_t[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        proba_val = torch.softmax(model(X_val_t), dim=1)[:, 1].cpu().numpy()
        proba_test = torch.softmax(model(X_test_t), dim=1)[:, 1].cpu().numpy()

    opt_threshold = find_optimal_threshold(y_val, proba_val)

    pred_fixed = (proba_test >= 0.5).astype(int)
    pred_opt = (proba_test >= opt_threshold).astype(int)
    base_rate = y_test.mean()
    naive_acc = max(base_rate, 1 - base_rate)

    return {
        "auc": roc_auc_score(y_test, proba_test),
        "opt_threshold": opt_threshold,
        "vs_base_rate_fixed": accuracy_score(y_test, pred_fixed) - naive_acc,
        "vs_base_rate_opt": accuracy_score(y_test, pred_opt) - naive_acc,
        "n_test": len(y_test),
    }


# ------------------------------------------------------------------
# 3. 5-seed 검증
# ------------------------------------------------------------------
def run_walk_forward(X, y, dates, model_name: str, random_state: int):
    splits = walk_forward_splits_by_date(dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, EMBARGO_DAYS)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

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
            "mean_threshold": fold_df["opt_threshold"].mean(),
            "vs_base_rate_fixed": fold_df["vs_base_rate_fixed"].mean(),
            "vs_base_rate_opt": fold_df["vs_base_rate_opt"].mean(),
            "win_folds_fixed": int((fold_df["vs_base_rate_fixed"] > 0).sum()),
            "win_folds_opt": int((fold_df["vs_base_rate_opt"] > 0).sum()),
            "n_folds": len(fold_df),
        })
        r = rows[-1]
        print(f"  [{model_name}] seed={seed}: AUC={r['mean_auc']:.4f}, thr(avg)={r['mean_threshold']:.3f}, "
              f"fixed={r['vs_base_rate_fixed']:+.4f}({r['win_folds_fixed']}/{r['n_folds']}), "
              f"opt={r['vs_base_rate_opt']:+.4f}({r['win_folds_opt']}/{r['n_folds']}), "
              f"{time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if DEVICE.type == "cuda":
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU를 못 찾아서 CPU로 돌아감\n")

    X, y, dates, tickers_arr = load_pooled_windows()
    print(f"윈도우: {X.shape[0]}개, shape={X.shape[1:]}, 종목: {len(np.unique(tickers_arr))}개\n")

    for model_name in ["GRU", "TCN"]:
        print(f"=== {model_name}: threshold=0.5 고정 vs fold별 최적 threshold(Youden's J) ===")
        seed_df = run_multi_seed(X, y, dates, model_name)
        print("\n" + seed_df.round(4).to_string(index=False))
        fixed_wins = int((seed_df["vs_base_rate_fixed"] > 0).sum())
        opt_wins = int((seed_df["vs_base_rate_opt"] > 0).sum())
        print(f"\n{model_name}: 고정 threshold 5/5 통과 {fixed_wins}/5, "
              f"최적 threshold 5/5 통과 {opt_wins}/5\n")

    print("=" * 60)
    print("최적 threshold에서도 5/5 실패면 -- 평가 방식 문제가 아니라 진짜 신호 부족.")
    print("최적 threshold에서 5/5 통과하면 -- 원래 [실패] 판정을 threshold=0.5 아티팩트로")
    print("재해석해야 하고, 다음 단계(XGBoost BASE 비교 등)에도 동일한 최적 threshold")
    print("방식을 적용해야 함 (지금 XGBoost 쪽 스크립트들은 전부 threshold=0.5 고정임).")