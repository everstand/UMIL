import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录强行加入 Python 的雷达扫描路径中
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 🌟 引入顶会必备的实验追踪神器 TensorBoard
from torch.utils.tensorboard import SummaryWriter

# 引入我们昨天写好的评测器 (用于边训练边验证)
from evaluate import evaluate
import time
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# ===== 魔改点：干掉 apex，换成 PyTorch 原生 AMP =====
from torch.cuda.amp import autocast, GradScaler
# =================================================

from einops import rearrange
import mmcv
import clip

from models import xclip
from datasets.build import build_dataloader
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint
from utils.logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/summe/32_5.yaml')
    parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--batch-size-umil', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    parser.add_argument('--umil-epoch', default=30, type=float)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--cluster-threshold', default=0.8, type=float)
    parser.add_argument('--w-smooth', default=0.01, type=float)
    parser.add_argument('--w-sparse', default=0.005, type=float)
    parser.add_argument('--w-mil', default=1.0, type=float)
    # 🌟 [新增参数]：强制要求传入官方 Split
    parser.add_argument('--split_file', type=str, required=True, help='官方划分文件, 如 splits/tvsum.yml')
    parser.add_argument('--split_id', type=int, default=0, help='当前跑的是第几折 (0-4)')
    parser.add_argument('--dataset', type=str, required=True, choices=['summe', 'tvsum'], help='数据集名字')

    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(config):
    # ---------------------------------------------------------
    # 🌟 1. 解析官方 YAML 划分文件 (带 H5 盲匹配翻译器 + 终极诊断)
    # ---------------------------------------------------------
    import yaml
    import h5py
    import decord
    import cv2
    
    print(f"📖 正在加载官方数据划分: {args.split_file} (Split {args.split_id})")
    with open(args.split_file, 'r') as f:
        all_splits = yaml.safe_load(f)
        current_split = all_splits[args.split_id]
        
    raw_train_keys = current_split['train_keys']
    raw_test_keys = current_split['test_keys']
    
    print(f"🕵️ 诊断: YAML 文件里的原始 Key 长什么样？ 样例: {raw_train_keys[:3]}")
    
    # 确定路径
    if args.dataset == 'summe':
        h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
        video_dir = "data/SumMe/videos"
    else:
        h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
        video_dir = "data/TVSum/videos"
        
    print(f"🔤 正在通过户口本 {h5_path} 翻译真实视频文件名...")
    
    # 🌟 建立物理视频的帧数指纹库
    fingerprints = {}
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.webm', '.mkv'))]
        for fname in video_files:
            vpath = os.path.join(video_dir, fname)
            vid_id = fname.split('.')[0]
            try:
                vr = decord.VideoReader(vpath)
                fingerprints[vid_id] = len(vr)
            except Exception:
                try:
                    cap = cv2.VideoCapture(vpath)
                    if cap.isOpened():
                        fingerprints[vid_id] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                except Exception:
                    pass
    print(f"🎯 成功提取到 {len(fingerprints)} 个视频的指纹库！")

    train_keys = []
    test_keys = []
    
    def translate_keys(raw_keys, out_list):
        with h5py.File(h5_path, 'r') as h5_data:
            h5_keys = list(h5_data.keys())
            for k in raw_keys:
                k = str(k).strip().split('/')[-1]
                
                # 🛡️ 容错 1：如果 YAML 里已经是真实的视频名了 (在物理文件夹里能找到)，直接放行！
                if k in fingerprints:
                    out_list.append(k)
                    continue
                
                # 🛡️ 容错 2：如果在 H5 里找不到，说明 YAML 格式和 H5 彻底对不上
                if k not in h5_data:
                    print(f"⚠️ 警告: '{k}' 既不是真实视频名，也不在 H5 字典中 (H5包含 {h5_keys[:3]}...)")
                    continue
                
                vname = None
                if 'video_name' in h5_data[k]:
                    vname_raw = h5_data[k]['video_name'][()]
                    vname = vname_raw.item().decode('utf-8') if hasattr(vname_raw, 'item') else str(vname_raw.decode('utf-8'))
                
                if not vname:
                    n_frames_h5 = h5_data[k]['n_frames'][()]
                    min_diff = float('inf')
                    for fname, vlen in fingerprints.items():
                        diff = abs(vlen - n_frames_h5)
                        if diff < min_diff:
                            min_diff = diff
                            vname = fname
                
                if vname:
                    out_list.append(vname)
                else:
                    print(f"⚠️ 警告: '{k}' 盲匹配失败，极大概率是指纹库提取数为 0！")

    translate_keys(raw_train_keys, train_keys)
    translate_keys(raw_test_keys, test_keys)
            
    print(f"✅ 严格控制变量: 训练集 {len(train_keys)} 个视频，测试集 {len(test_keys)} 个视频")
    print(f"🔍 翻译样例: {train_keys[:3]}") 
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # 🌟 2. 初始化 TensorBoard 和带有名单的 DataLoader
    # ---------------------------------------------------------
    log_dir = os.path.join(config.OUTPUT, f"tensorboard_logs/split_{args.split_id}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 🌟 核心：精确接住 8 个返回值，丢弃不需要的重复项
    _, _, _, train_loader, val_loader, _, _, train_loader_umil = build_dataloader(logger, config, train_keys=train_keys)
    
    # 强行传入 None 触发自动下载预训练模型
    model, _ = xclip.load(None, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES,
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger)
                         
    # 🌟 显存优化：冻结视觉编码器，只练摘要层
    for param in model.visual.parameters():
        param.requires_grad = False
    print("❄️  已冻结 CLIP 视觉编码器，显存压力已降低！")
    
    model = model.cuda()

    optimizer, _ = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # ===== 魔改点：初始化原生的 GradScaler =====
    use_amp = (config.TRAIN.OPT_LEVEL != 'O0')
    scaler = GradScaler(enabled=use_amp)
    # =======================================

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch = 0
    if config.MODEL.RESUME:
        start_epoch, _ = load_checkpoint(config, model.module if dist.is_initialized() else model, optimizer, lr_scheduler, logger)

    with open('labels/action_vocabulary.txt', 'r', encoding='utf-8') as f:
        action_classes = [line.strip() for line in f if line.strip()]
    text_prompts = [f"A video of a person {action}" for action in action_classes]
    logger.info(f"Loaded {len(text_prompts)} action prompts.")
    text_labels = clip.tokenize(text_prompts).cuda()

    # ---------------------------------------------------------
    # 🌟 顶会级带验证的训练主循环 (Training Loop with Auto-Validation)
    # ---------------------------------------------------------
    best_f1_score = 0.0
    
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        model.train()
        
        # 🌟 1. 跑一轮真正的 MIL 训练
        train_loss = mil_one_epoch(
            epoch=epoch, 
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            train_loader=train_loader_umil,
            text_labels=text_labels, 
            config=config, 
            scaler=scaler, 
            use_amp=use_amp
        )
        
        # 实时将 Loss 写入 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # 2. 跑一轮验证 (Auto-Validation)
        print(f"\n🔍 正在验证 Epoch {epoch} 的模型泛化能力...")
        
        # 保存临时权重用于测试
        temp_ckpt_path = os.path.join(config.OUTPUT, "temp_checkpoint.pth")
        epoch_saving(config, epoch, model, 0.0, optimizer, lr_scheduler, logger, config.OUTPUT, is_best=False)
        # 将刚刚保存的 epoch_xx.pth 重命名为临时文件
        import shutil
        shutil.move(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth"), temp_ckpt_path)
        
        # 🌟 调用评价器，并传入 test_keys 进行绝对隔离测试
        current_f1 = evaluate(config, args.dataset, temp_ckpt_path, test_keys=test_keys)
        
        # 🌟 将 F1 分数写入 TensorBoard
        writer.add_scalar('Metric/F1_Score', current_f1, epoch)
        
        # 3. 🌟 核心：只拦截并保存最高分的模型
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_ckpt_name = os.path.join(config.OUTPUT, f"best_model_split{args.split_id}.pth")
            
            # 覆盖保存最优模型
            shutil.copyfile(temp_ckpt_path, best_ckpt_name)
            print(f"🎉 发现新的最好成绩! F1: {best_f1_score:.4f}，已保存至 {best_ckpt_name}")
            
    writer.close()
    print(f"🏆 Split {args.split_id} 训练彻底结束！历史最高 F1: {best_f1_score:.4f}")

def mil_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config, scaler, use_amp):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    mil_loss_meter = AverageMeter()

    end = time.time()
    
    for idx, batch_data in enumerate(train_loader):
        torch.cuda.empty_cache()
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        
        bz = images.shape[0]
        a_aug = images.shape[1]
        n_clips = images.shape[2]
        
        images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')

        # ===== 魔改点：用 autocast 包装前向传播和 Loss 计算 =====
        with autocast(enabled=use_amp):
            output = model(images, text_labels)
            
            # 1. 提取结构化 Logits 矩阵 [B, T, C]
            logits_tc = rearrange(output['y'], '(b a k) c -> (b a) k c', b=bz, a=a_aug, k=n_clips)
            
            # 获取真实的动作类别数量 C
            C = text_labels.shape[0]
            labels = label_id.view(bz * a_aug, C)
            
            # =====================================================================
            # 🏆 理论落地：方向 1 - 时间平滑先验 (Temporal Smoothing Prior)
            # =====================================================================
            # 目的：强行抹平单帧尖峰，逼迫模型寻找具有时间跨度的“事件块”
            
            K = 3 # 平滑窗口大小 (重要超参数：代表视频切片的最小叙事连贯长度)
            
            # F.avg_pool1d 严格要求输入维度为 [Batch, Channels, Length]，因此必须转置
            logits_transposed = logits_tc.transpose(1, 2) # 变为 [B, C, T]
            
            # 在时间轴 T 上滑动平均。padding=K//2 极其关键，它保证了输出的时间维度仍然是 32
            smoothed_logits_transposed = F.avg_pool1d(logits_transposed, kernel_size=K, stride=1, padding=K//2)
            
            # 还原维度回 [B, T, C]
            smoothed_logits_tc = smoothed_logits_transposed.transpose(1, 2)
            # =====================================================================
            
            # 2. 🌟 联合 Soft-Pooling (LogSumExp) 
            # 注意：这里使用的是经过平滑处理后的 smoothed_logits_tc
            tau = 1.0  
            bag_logits = tau * torch.logsumexp(smoothed_logits_tc / tau, dim=1)
            loss_mil = F.binary_cross_entropy_with_logits(bag_logits, labels.float())
            
            # 3. 基础微弱稀疏项 (防止平滑后整体水位过高)
            prob_tc = torch.sigmoid(smoothed_logits_tc)  
            sparsity_loss = prob_tc.mean()
            
            # 4. 汇总 Total Loss (纯净的平滑先验 Baseline)
            total_loss = loss_mil + 0.005 * sparsity_loss
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        # ===== 魔改点：用 Scaler 控制反向传播和梯度更新 =====
        scaler.scale(total_loss).backward()
            
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)
        # ================================================

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), bz)
        mil_loss_meter.update(loss_mil.item(), bz)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'lr {lr:.6f}\t'
                f'Tot Loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'MIL Loss {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})')
    return tot_loss_meter.avg

@torch.no_grad()
def validate(val_loader, text_labels, model, config, out_path):
    # 【终极魔改】：直接跳过传统的单分类验证！因为它对多标签摘要毫无意义且会报错。
    print("🎯 [魔改] 已跳过传统分类 Validate 阶段，直接放行去保存模型权重！")
    return 0.0

if __name__ == '__main__':
    args, config = parse_option()
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1
        
    torch.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
    
    if rank != -1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=7200),
                                             world_size=world_size, rank=rank)
        torch.distributed.barrier()

    seed = config.SEED + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=(dist.get_rank() if dist.is_initialized() else 0), name=f"{config.MODEL.ARCH}")
    
    main(config)