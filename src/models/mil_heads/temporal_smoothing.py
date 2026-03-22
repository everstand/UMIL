import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSmoothingPrior(nn.Module):
    """
    时间平滑先验模块 (Temporal Smoothing Prior).
    机制: 利用一维平均池化抹平单帧的 Logit 尖峰，促使模型关注具有时间跨度的连续片段。
    输入/输出约定: [Batch, Time, Channels]
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, logits):
        # [B, T, C] -> [B, C, T] 以适配 F.avg_pool1d
        logits_transposed = logits.transpose(1, 2)
        
        smoothed_transposed = F.avg_pool1d(
            logits_transposed, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
        
        # 还原为 [B, T, C]
        return smoothed_transposed.transpose(1, 2)