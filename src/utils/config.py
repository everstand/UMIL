import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'kinetics400'
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_CLIPS = 16
_C.DATA.NUM_FRAMES = 5
_C.DATA.FRAME_INTERVAL = 6
_C.DATA.NUM_CLASSES = 400
_C.DATA.LABEL_LIST = 'labels/action_vocabulary.txt' # 指向动态生成的伪标签词表
_C.DATA.FILENAME_TMPL = 'img_{:08}.jpg'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.BATCH_SIZE_UMIL = 4
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.0
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.0
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False

# -----------------------------------------------------------------------------
# Evaluation & Ablation settings (🌟 新增的消融实验引擎节点)
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.ABLATION_MODE = 'E3' # 可选: 'E0', 'E1', 'E2', 'E3'
_C.EVAL.TOP_K = 3            # 语义聚合的并发动作假设数量
_C.EVAL.ALPHA = 0.5          # S_t = αP_t + (1-α)R_t 中的融合权重
_C.EVAL.REP_SPACE = 'raw'
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 20
_C.SEED = 1024


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if getattr(args, 'batch_size', None):
        config.TRAIN.BATCH_SIZE = args.batch_size
    if getattr(args, 'pretrained', None):
        config.MODEL.PRETRAINED = args.pretrained
    if getattr(args, 'resume', None):
        config.MODEL.RESUME = args.resume
    if getattr(args, 'accumulation_steps', None):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if getattr(args, 'output', None):
        config.OUTPUT = args.output
    if getattr(args, 'only_test', None):
        config.TEST.ONLY_TEST = True
    
    # 防御性读取 local_rank，兼容非分布式运行环境
    config.LOCAL_RANK = getattr(args, 'local_rank', 0) 
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)
    return config