import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationPrior(nn.Module):
    """
    [Paper Core Contribution] 视觉代表性先验分支 (Representativeness Branch - R_t)
    机制: 通过计算单片段视觉特征与全局视频上下文的余弦相似度，
          评估其在无监督意义下的可摘要性（Summarizability）。
    """
    def __init__(self):
        super().__init__()

    def forward(self, visual_features):
        """
        Args:
            visual_features: 模型输出的纯视觉特征，Shape: [Batch, T, Dim]
                             (T 在这里即 N_clips)
        Returns:
            r_scores: 归一化后的代表性分数，Shape: [Batch, T]
        """
        # 1. 确保特征被 L2 归一化
        features_norm = F.normalize(visual_features, p=2, dim=-1) # [1, T, D]
        
        # 2. 计算视频全局特征重心 (Global Context)
        global_feature = features_norm.mean(dim=1, keepdim=True) # [1, 1, D]
        global_feature = F.normalize(global_feature, p=2, dim=-1)
        
        # 3. 计算每个片段与全局重心的余弦相似度
        # 越接近重心的片段，越能代表整个视频的基调
        sim = torch.sum(features_norm * global_feature, dim=-1) # [1, T]
        
        # 4. Min-Max 归一化到 [0, 1] 空间，使其能够与 P_t 平滑融合
        sim_min = sim.min(dim=1, keepdim=True)[0]
        sim_max = sim.max(dim=1, keepdim=True)[0]
        
        # 加 1e-8 防止分母为 0
        r_scores = (sim - sim_min) / (sim_max - sim_min + 1e-8)
        
        return r_scores