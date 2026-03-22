import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
THIRD_PARTY_DIR = os.path.join(BASE_DIR, 'third_party')
if THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, THIRD_PARTY_DIR)

import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import clip

from umil.datasets.splits import load_split
from umil.datasets.metadata.adapter import build_identity_maps
from umil.engine.trainer import VideoTrainer

from models.builder import build_umil_model
from datasets.build import build_dataloader
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import load_checkpoint
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
    parser.add_argument('--split_file', type=str, required=True, help='官方划分文件')
    parser.add_argument('--split_id', type=int, default=0, help='当前跑的是第几折 (0-4)')
    parser.add_argument('--dataset', type=str, required=True, choices=['summe', 'tvsum'], help='数据集名字')

    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(config, args):
    # ---------------------------------------------------------
    # 1. 统一数据划分与身份防腐层
    # ---------------------------------------------------------
    print(f"📖 正在通过统一接口加载数据划分: {args.split_file} (Split {args.split_id})")
    train_h5_keys, test_h5_keys = load_split(args.split_file, args.split_id)

    if args.dataset == 'summe':
        h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
    else:
        h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"

    h5_to_real, real_to_h5 = build_identity_maps(args.dataset, h5_path)
    print(f"✅ 统一解析完毕: 训练集 {len(train_h5_keys)} 个视频，测试集 {len(test_h5_keys)} 个视频")

    # ---------------------------------------------------------
    # 2. 初始化 TensorBoard 和 DataLoader
    # ---------------------------------------------------------
    log_dir = os.path.join(config.OUTPUT, f"tensorboard_logs/split_{args.split_id}")
    writer = SummaryWriter(log_dir=log_dir)

    _, _, _, train_loader, val_loader, _, _, train_loader_umil = build_dataloader(
        logger, config, train_keys=train_h5_keys, real_to_h5_map=real_to_h5,
    )

    # ---------------------------------------------------------
    # 3. 构建模型
    # ---------------------------------------------------------
    model = build_umil_model(config, is_training=True, logger=logger)
                         
    for param in model.visual.parameters():
        param.requires_grad = False
    print("❄️  已冻结 CLIP 视觉编码器，显存压力已降低！")
    
    model = model.cuda()

    optimizer, _ = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    use_amp = (config.TRAIN.OPT_LEVEL != 'O0')
    scaler = GradScaler(enabled=use_amp)

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
    # 4. 实例化引擎并执行训练
    # ---------------------------------------------------------
    trainer = VideoTrainer(
        config=config, 
        args=args, 
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        train_loader=train_loader_umil, 
        text_labels=text_labels, 
        scaler=scaler, 
        logger=logger, 
        writer=writer, 
        test_h5_keys=test_h5_keys
    )
    
    trainer.run(start_epoch=start_epoch)


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
    
    # 显式将 args 传入 main 以避免对全局变量的依赖
    main(config, args)