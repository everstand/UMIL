import h5py
import logging
from functools import lru_cache

from umil.datasets.metadata.tvsum_metadata import TVSUM_STATIC_MAP

logger = logging.getLogger(__name__)


def _decode_h5_string(value):
    """
    统一解码 H5 中可能出现的多种字符串表示：
    - bytes / numpy.bytes_
    - 已经是 str
    - 标量数组包裹
    """
    # 先尽量剥掉 numpy 标量 / object 容器
    try:
        value = value.item()
    except Exception:
        pass

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value

    # 最后兜底
    return str(value)


@lru_cache(maxsize=None)
def build_identity_maps(dataset_name, h5_path=None):
    """
    构建全项目统一的双向身份映射。

    Args:
        dataset_name (str): 'tvsum' or 'summe'
        h5_path (str|None): SumMe 需要显式提供 H5 路径；TVSum 可忽略

    Returns:
        h5_to_real (dict): {'video_35': 'AwmHb44_ouw', ...}
        real_to_h5 (dict): {'AwmHb44_ouw': 'video_35', ...}

    设计原则：
        1. H5 key 是系统内部主身份
        2. 真实视频名只在 Adapter 层出现
        3. 严禁静默覆盖重复映射
    """
    dataset_name = dataset_name.lower()
    h5_to_real = {}
    real_to_h5 = {}

    if dataset_name == "tvsum":
        h5_to_real = TVSUM_STATIC_MAP.copy()

        for h5_key, real_name in h5_to_real.items():
            if real_name in real_to_h5:
                raise ValueError(
                    f"TVSum 身份映射冲突：真实视频名 {real_name} "
                    f"同时对应 {real_to_h5[real_name]} 和 {h5_key}"
                )
            real_to_h5[real_name] = h5_key

    elif dataset_name == "summe":
        if not h5_path:
            raise ValueError("构建 SumMe 映射字典必须显式传入 h5_path")

        with h5py.File(h5_path, "r") as h5_data:
            for h5_key in h5_data.keys():
                if "video_name" not in h5_data[h5_key]:
                    raise KeyError(f"SumMe H5 键 {h5_key} 缺少 video_name 字段")

                raw_value = h5_data[h5_key]["video_name"][()]
                real_name = _decode_h5_string(raw_value)

                if h5_key in h5_to_real:
                    raise ValueError(f"🚨 重复的 H5 key: {h5_key}")

                if real_name in real_to_h5:
                    raise ValueError(
                        f"SumMe 身份映射冲突：真实视频名 {real_name} "
                        f"同时对应 {real_to_h5[real_name]} 和 {h5_key}"
                    )

                h5_to_real[h5_key] = real_name
                real_to_h5[real_name] = h5_key

    else:
        raise ValueError(f"身份适配器尚未支持该数据集: {dataset_name}")

    logger.info(
        f"==> 成功构建 {dataset_name.upper()} 身份双向字典 | 容量: {len(h5_to_real)} 项"
    )

    return h5_to_real, real_to_h5