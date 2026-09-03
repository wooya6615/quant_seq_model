"""
GRU 기반 이진분류기. LSTM보다 파라미터가 적어서(게이트 3개 vs 4개) 종목 3개 풀링
정도의 적은 데이터에서 오버피팅 위험이 상대적으로 덜함 -- 그래서 LSTM 대신 GRU를 먼저 시도.
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int = 13, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # 마지막 레이어의 최종 시점 hidden state, (batch, hidden_size)
        return self.classifier(last_hidden)