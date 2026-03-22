import logging
from models import xclip

def build_umil_model(config, state_dict=None, is_training=False, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    T = config.DATA.NUM_FRAMES
    droppath = getattr(config.MODEL, 'DROP_PATH_RATE', 0.0)
    use_checkpoint = getattr(config.TRAIN, 'USE_CHECKPOINT', False)
    use_cache = getattr(config.MODEL, 'FIX_TEXT', True)

    mit_layers = getattr(config.MODEL, 'MIT_LAYERS', 1)
    prompts_layers = getattr(config.MODEL, 'PROMPTS_LAYERS', 2)
    prompts_alpha = getattr(config.MODEL, 'PROMPTS_ALPHA', 1e-1)

    arch = config.MODEL.ARCH
    pretrained_path = getattr(config.MODEL, 'PRETRAINED', None)

    logger.info(
        f"==> Building UMIL Model | Is_Train: {is_training} | T: {T} | "
        f"MIT: {mit_layers} | Prompts: {prompts_layers} | Alpha: {prompts_alpha}"
    )

    if is_training:
        model, _ = xclip.load(
            pretrained_path, arch,
            device="cpu", jit=False,
            T=T, droppath=droppath,
            use_checkpoint=use_checkpoint, use_cache=use_cache,
            logger=logger,
            mit_layers=mit_layers,
            prompts_alpha=prompts_alpha,
            prompts_layers=prompts_layers,
        )
        if state_dict is not None:
            state_dict = state_dict['model'] if isinstance(state_dict, dict) and 'model' in state_dict else state_dict
            model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = state_dict['model'] if isinstance(state_dict, dict) and 'model' in state_dict else state_dict
        model = xclip.build_model(
            state_dict=state_dict,
            T=T, droppath=droppath,
            use_checkpoint=use_checkpoint, use_cache=use_cache,
            logger=logger,
            mit_layers=mit_layers,
            prompts_alpha=prompts_alpha,
            prompts_layers=prompts_layers,
        )

    return model