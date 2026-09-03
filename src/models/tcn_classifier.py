"""
TCN(Temporal Convolutional Network) 이진분류기.
causal(인과) conv1d를 dilation을 늘려가며 쌓아서, 미래 정보 누출 없이 과거 윈도우
전체를 receptive field 안에 담음.

⚠️ receptive field 계산: kernel_size=3, dilation=(1,2,4,8) 4개 블록
   -> field = 1 + 2*(1+2+4+8) = 31 >= WINDOW(20), 윈도우 전체를 커버함.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """왼쪽에만 패딩을 줘서 미래 시점을 안 보게 만든 conv1d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        # 채널 수가 바뀌는 블록은 residual을 더하기 전에 1x1 conv로 맞춰줌
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNClassifier(nn.Module):
    def __init__(self, n_features: int = 13, channels: tuple = (32, 64, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features) -> conv1d는 (batch, channels, seq_len)을 기대
        x = x.transpose(1, 2)
        out = self.blocks(x)
        last_step = out[:, :, -1]  # 가장 최근 시점 -- causal이라 전체 window 정보가 누적돼 있음
        return self.classifier(last_step)