import torch

def fix_text(model):
    """
    冻结文本编码器的物理契约
    遍历模型参数，将属于 text 端（或被明确要求冻结）的参数梯度截断
    """
    for name, param in model.named_parameters():
        if "token_embedding" in name or "transformer" in name or "text_projection" in name or "ln_final" in name or "positional_embedding" in name:
            param.requires_grad = False

def build_optimizer(config, model):
    """
    基于绝对物理属性 (requires_grad) 的极简优化器组装
    """
    # 处理 DDP/DP 模型的 module 嵌套封装
    base_model = model.module if hasattr(model, 'module') else model

    # 1. 严格执行冻结契约 (如果配置要求冻结文本端)
    if getattr(config.MODEL, 'FIX_TEXT', True):
        fix_text(base_model)

    # 2. 探针：仅收集计算图中真正需要求导的活跃节点
    active_params = [p for p in base_model.parameters() if p.requires_grad]
    
    # 3. 组装极简 AdamW 引擎
    optimizer = torch.optim.AdamW(
        active_params,
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    
    return optimizer