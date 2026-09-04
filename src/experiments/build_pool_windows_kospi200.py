"""
KOSPI200 구성종목을 풀링해서 20일 윈도우 시퀀스 데이터셋 생성.
3종목 풀링(7688개 윈도우)으로는 GRU/TCN 둘 다 [실패], threshold 아티팩트도 아님이
확인됨 -- 남은 가설은 "데이터량 부족"이라 quant_cnn_chart가 이미 겪은 것과 같은
KOSPI200 풀링 인프라(종목 리스트는 pykrx, 가격은 yfinance)를 그대로 재사용.

핵심 설계 (quant_cnn_chart의 build_pool_images.py와 동일한 단계적 접근):
    - 종목 리스트 조회(get_kospi200_tickers)는 1회만 호출
    - 먼저 N_TICKERS_PILOT=50개로 파이프라인부터 검증 (yfinance 실패/데이터 부족
      종목 스킵 로직이 제대로 도는지), 문제 없으면 전체 200종목으로 확장
    - BASE 13개 feature는 pykrx 수급/밸류에이션 데이터 불필요, 종목 리스트 조회만
      pykrx 사용

⚠️ 사전 등록: WINDOW/HORIZON/cost_threshold는 3종목 실험(pooled_windows.py)과 완전히
동일하게 유지 -- 그래야 성능 차이가 순수하게 "데이터량 확장" 효과인지 확인 가능.

⚠️ 예상 소요시간: 3종목(7688개 윈도우) 기준 GRU 5-seed가 CPU로 ~7분 걸렸음. 50종목이면
   윈도우 수가 대략 16~17배로 늘어나서 학습 시간도 비슷한 비율로 늘어날 가능성이 큼
   (모델·seed당 수십 분 단위). 데이터 생성(이 스크립트) 자체는 yfinance 호출 50번 +
   딜레이라 몇 분이면 끝나지만, 그 다음 run_seq_experiment.py 돌릴 때는 SEEDS를
   1~2개로 줄여서 먼저 시간을 재보는 걸 권장.

사용법 (레포 루트에서):
    python -m src.experiments.build_pool_windows_kospi200
"""

import time
from pathlib import Path

import numpy as np

from src.data.kospi200_universe import get_kospi200_tickers
from src.data.pooled_windows import build_windows, WINDOW, HORIZON, COST_THRESHOLD
from src.features.base_features import build_feature_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_TICKERS_PILOT = 50
BENCHMARK = "^KS11"
MIN_ROWS = 100 + WINDOW  # 윈도우 만들고도 최소 100개 샘플은 남아야 의미 있음


def build_pooled_windows_kospi200(
    n_tickers: int = N_TICKERS_PILOT,   # None이면 전체 KOSPI200
    window: int = WINDOW,
    horizon: int = HORIZON,
    cost_threshold: float = COST_THRESHOLD,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    sleep_sec: float = 0.3,              # yfinance 연속 호출 딜레이 -- 과도한 요청 차단 방지
    save: bool = True,
):
    tickers = get_kospi200_tickers()
    if n_tickers is not None:
        tickers = tickers[:n_tickers]
        print(f"-> 서브셋 {len(tickers)}개로 진행 (n_tickers={n_tickers})")

    all_X, all_y, all_dates, all_tickers = [], [], [], []
    failed = []
    for i, code in enumerate(tickers):
        yf_ticker = f"{code}.KS"
        try:
            df = build_feature_dataset(
                ticker=yf_ticker, benchmark=BENCHMARK, start=start, end=end,
                horizon=horizon, cost_threshold=cost_threshold,
            )
            if len(df) < MIN_ROWS:
                print(f"  [{i + 1}/{len(tickers)}] {code}: 행 수 부족({len(df)}행)으로 건너뜀")
                failed.append(code)
                continue

            X, y, dates = build_windows(df, FEATURE_COLS_BASE, window)
            all_X.append(X)
            all_y.append(y)
            all_dates.append(dates)
            all_tickers.extend([code] * len(y))
            print(f"  [{i + 1}/{len(tickers)}] {code}: {X.shape[0]}개 윈도우")
        except Exception as e:
            print(f"  [{i + 1}/{len(tickers)}] {code}: 실패, 건너뜀 ({e})")
            failed.append(code)
        time.sleep(sleep_sec)

    if not all_X:
        raise RuntimeError("모든 종목 다운로드/윈도우 생성에 실패했습니다. 네트워크/yfinance 상태를 확인하세요.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    dates = np.concatenate(all_dates, axis=0)
    tickers_arr = np.array(all_tickers)

    print(f"\n풀링 완료: {len(all_X)}종목 성공 / {len(failed)}종목 실패")
    if failed:
        print(f"제외된 종목: {failed}")
    print(f"총 {X.shape[0]}개 윈도우, shape={X.shape[1:]}")
    print(f"라벨 분포: {y.mean():.3f} (1의 비율)")

    if save:
        suffix = "pilot" if n_tickers is not None else "full"
        out_path = DATA_DIR / f"pooled_windows_kospi200_{suffix}.npz"
        np.savez(out_path, X=X, y=y, dates=dates, tickers=tickers_arr)
        print(f"저장 완료: {out_path}")

    return X, y, dates, tickers_arr


if __name__ == "__main__":
    X, y, dates, tickers_arr = build_pooled_windows_kospi200(n_tickers=N_TICKERS_PILOT)

    print("\n[다음 단계] 이 pilot 파일이 잘 나왔으면:")
    print("  1) run_seq_experiment.py 상단의 POOLED_WINDOWS_FILE을")
    print("     'pooled_windows_kospi200_pilot.npz'로 바꿔서 실행")
    print("     (처음엔 SEEDS를 1~2개로 줄여서 실제 소요시간부터 재볼 것)")
    print("  2) 문제 없으면 n_tickers=None으로 바꿔서 전체 200종목 재실행")