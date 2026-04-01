import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

def fix_text(model):
    text_prefixes = (
        "token_embedding.",
        "transformer.",
        "ln_final.",
    )
    text_exact = {
        "text_projection",
        "positional_embedding",
    }

    for name, param in model.named_parameters():
        if name.startswith(text_prefixes) or name in text_exact:
            param.requires_grad = False

def build_optimizer(config, model):
    """
    基于绝对物理属性 (requires_grad) 的极简优化器组装
    """
    # 处理 DDP/DP 模型的 module 嵌套封装
    base_model = model.module if hasattr(model, 'module') else model

    # 1. 严格执行冻结契约 (如果配置要求冻结文本端)
    active_params = [p for p in base_model.parameters() if p.requires_grad]
    if len(active_params) == 0:
        raise ValueError("No trainable parameters remain after freezing.")
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
    
    # 匹配 train.py 中的解包契约: optimizer, _ = build_optimizer(...)
    return optimizer, None

def build_scheduler(config, optimizer, n_iter_per_epoch):
    """
    构建余弦退火学习率调度器
    """
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(getattr(config.TRAIN, 'WARMUP_EPOCHS', 5) * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.LR / 100,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler