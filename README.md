# quant_seq_model

`quant_xgboost`는 트리 기반이라 각 날짜(row)를 독립 샘플로 취급 -- 어제/그제 값을
직접 보지 못함. lag feature를 나란히 붙이는 우회(`quant_xgboost/train_xgboost_ablation_lag.py`)는
[실패]로 끝났고 (AUC는 5/5 개선했지만 vs_base_rate는 5/5 악화), 이 레포는 그 다음
단계 -- 진짜 시퀀스 모델(GRU, TCN)이 과거 20일치 피처 궤적에서 신호를 뽑아내는지 확인.

## 배경 문제

- BASE 13개 피처를 XGBoost에 그대로 넣으면 순서 정보가 없음
- lag feature(과거 값을 컬럼으로 나란히 붙이기)는 실패 -- 트리 분기 구조가 시퀀스
  관계를 못 잡는 것으로 추정
- GRU/TCN은 애초에 시퀀스를 위해 설계된 구조라 다른 결과가 나올 수 있는지 확인

## 방법론

- **입력**: BASE 13개 피처 x 과거 20일 윈도우 (오늘 포함), quant_xgboost와 동일한
  피처 정의를 그대로 이식해서 재사용 (`src/features/base_features.py`)
- **대상**: 검증된 3종목 풀링 -- 현대로템(064350), 한전기술(052690), 모트렉스(118990)
  (단일 종목은 데이터량이 너무 적어서 GRU/TCN 오버피팅 위험이 큼)
- **라벨**: `quant_xgboost`와 동일 -- horizon(10일)일 후 수익률이 거래비용 이상이면 1
- **모델**:
  - `src/models/gru_classifier.py` -- 2-layer GRU, hidden=32 (LSTM보다 파라미터
    적어서 이 데이터량에서 오버피팅 덜함)
  - `src/models/tcn_classifier.py` -- causal conv1d 4블록, dilation 1/2/4/8
    (receptive field 31 >= window 20, 윈도우 전체 커버)
- **1차 baseline 비교**: quant_cnn_chart와 동일한 철학 -- GRU/TCN vs 동일가중(=base
  rate)만 먼저 확인. 여기서 못 이기면 XGBoost 비교는 의미 없음.
- **검증**: 날짜 기준 walk-forward(같은 날짜 여러 종목이 train/test로 안 쪼개지게),
  5-seed(42/1/7/123/2024), fold별 정규화(train 통계로만, 누수 방지)

## 구조

```
src/
  features/
    base_features.py       # BASE 13개 피처 생성 (quant_xgboost 로직 이식)
  data/
    pooled_windows.py      # 3종목 20일 윈도우 생성 + 풀링, npz로 저장
  models/
    gru_classifier.py      # PyTorch GRU
    tcn_classifier.py      # PyTorch TCN (causal conv1d)
  experiments/
    run_seq_experiment.py  # GRU vs TCN vs 동일가중, 5-seed 비교
data/                       # 생성되는 npz (gitignore 처리)
docs/
  PROJECT_SUMMARY.md        # 실험 진행하며 채울 예정
```

## 셋업

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install torch pandas numpy yfinance scikit-learn tqdm
```

## 실행 순서

```bash
python -m src.data.pooled_windows          # 3종목 20일 윈도우 생성 (data/pooled_windows_3tickers.npz)
python -m src.experiments.run_seq_experiment  # GRU vs TCN vs 동일가중, 5-seed 비교
```

이 레포는 `src/data`, `src/features`, `src/models`처럼 하위 폴더가 나뉜 구조라서
(quant_xgboost의 flat `src/` 구조와 다르게) `from src.xxx import ...` 임포트를 쓰고,
그래서 `-m` 실행이 필요함.

**확인할 것**
- `vs_base_rate`가 GRU/TCN 둘 다 5/5 음수면 -- lag feature 실험과 같은 기준으로
  -- 이 방향은 접고 3종목 풀링을 KOSPI200 규모로 넓히는 걸 고려할 것 (quant_cnn_chart가
  이미 KOSPI200 풀링 인프라를 갖고 있으니 재사용 가능)
- 둘 중 하나라도 5/5 통과하면 다음 단계: 같은 3종목으로 XGBoost(BASE) baseline과
  정면 비교해서 "시퀀스 모델링 자체가 XGBoost보다 나은지" 확인