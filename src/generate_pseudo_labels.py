import os
import cv2
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel

# =========================================================
# 1) TVSum: 保持动作/事件导向
# =========================================================
TVSUM_ACTIONS = [
    "cutting food", "spreading butter or jam", "making a sandwich",
    "pouring liquid", "washing hands or dishes", "chewing and swallowing food",
    "using hand tools", "operating a car jack", "tightening or loosening tire nuts",
    "handling heavy tires", "pushing a stuck car", "digging dirt or snow",
    "attaching a tow rope", "driving a car", "stepping on car pedals", "waiting in a car",
    "grooming a pet", "washing a pet with water", "applying pet shampoo",
    "drying a pet with a towel", "petting an animal", "walking a dog on a leash",
    "feeding a pet", "a dog jumping through a hoop", "a dog sitting on command",
    "running or sprinting", "parkour jumping over obstacles", "climbing a wall",
    "rolling on the ground", "riding a BMX bike", "balancing on a bike",
    "falling down", "deploying a parachute", "wearing extreme sports gear",
    "scuba diving underwater", "swimming", "aiming and shooting a paintball gun",
    "swinging a mallet in bike polo", "sliding down a water slide",
    "marching in a parade", "dancing in a flash mob", "playing a musical instrument",
    "waving to a crowd", "holding a flag or banner", "clapping and cheering",
    "standing in a crowd", "talking to someone", "setting up dominoes", "lighting a fire",
    "walking around a tourist attraction", "looking up at a monument",
    "taking a picture or recording a video", "posing for a photo",
    "children playing and chasing", "climbing playground equipment",
    "operating airplane controls", "looking out an airplane window",
    "operating an excavator", "digging dirt with an excavator", "dumping dirt or rocks"
]
TVSUM_PROMPTS = [f"a video segment showing {x}" for x in TVSUM_ACTIONS]

# =========================================================
# 2) SumMe: 去掉纯抽象 prompt，改成物理视觉锚点更强的 prompt
#    并做静态语义分组去冗余
# =========================================================
SUMME_PROMPTS = [
    # G0: 极限运动 / 高刺激
    "a person performing an extreme sport or stunt",
    "a moment of free fall, jumping, or flying through the air",
    "a person doing water sports, surfing, or scuba diving",
    "navigating a challenging physical obstacle or terrain",

    # G1: 第一人称 / 镜头动态
    "an exciting first-person view of an activity",
    "a fast-moving or shaky camera capturing an intense moment",
    "a person looking directly into the camera and reacting",

    # G2: 日常实体交互 / 生活特写
    "a close-up shot of preparing food or an interesting object",
    "a sudden physical impact or collision",
    "a group of people working together on a task",
    "children or animals in an interesting activity",

    # G3: 地标 / 环境奇观
    "a clear view of a famous landmark or monument",
    "a scenic outdoor view or impressive landscape",

    # G4: 群体活动 / 表演 / 聚会
    "a group of people gathering or celebrating",
    "a public performance or event",
]

# 每个组里只保留一个代表 prompt，防止近义 prompt 重复置正
SUMME_GROUPS = {
    "extreme_sport": [0, 1, 2, 3],
    "camera_dynamic": [4, 5, 6],
    "daily_interaction": [7, 8, 9, 10],
    "landmark_scenery": [11, 12],
    "social_event": [13, 14],
}

DATASETS = {
    "SumMe": {
        "video_dir": "data/SumMe/videos",
        "output_npy": "labels/summe_multihot_labels.npy",
        "vocab_file": "labels/summe_prompt_vocabulary.txt",
        "prompts": SUMME_PROMPTS,
        "groups": SUMME_GROUPS,
        "margin": 1.5,
        "min_keep": 1,
        "max_keep": 5,
    },
    "TVSum": {
        "video_dir": "data/TVSum/videos",
        "output_npy": "labels/tvsum_multihot_labels.npy",
        "vocab_file": "labels/tvsum_prompt_vocabulary.txt",
        "prompts": TVSUM_PROMPTS,
        "groups": None,
        "margin": 1.5,
        "min_keep": 1,
        "max_keep": 5,
    }
}


# =========================================================
# 3) 稳健抽帧
# =========================================================
def _extract_single_clip_cv2(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()

    return frames


def _extract_all_clips_cv2(video_path, num_clips=16, clip_len=16, min_valid_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return []

    clip_centers = np.linspace(0, max(total_frames - 1, 0), num_clips).astype(int)
    clips = []
    last_valid_clip = None

    for center in clip_centers:
        start = max(0, int(center) - clip_len // 2)
        end = min(total_frames - 1, start + clip_len - 1)
        idxs = np.linspace(start, end, clip_len).astype(int)

        clip_frames = _extract_single_clip_cv2(video_path, idxs)

        if len(clip_frames) > 0:
            while len(clip_frames) < clip_len:
                clip_frames.append(clip_frames[-1])

        if len(clip_frames) < min_valid_frames:
            if last_valid_clip is not None:
                clip_frames = list(last_valid_clip)
            else:
                return []

        clip_frames = clip_frames[:clip_len]
        last_valid_clip = clip_frames
        clips.append(clip_frames)

    if len(clips) != num_clips:
        return []

    return clips


def extract_clip_frames(video_path, num_clips=16, clip_len=16, min_valid_frames=8):
    """
    decord 主解码，失败时回退到 cv2
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            return _extract_all_clips_cv2(video_path, num_clips, clip_len, min_valid_frames)

        clip_centers = np.linspace(0, max(total_frames - 1, 0), num_clips).astype(int)
        clips = []
        last_valid_clip = None

        for center in clip_centers:
            start = max(0, int(center) - clip_len // 2)
            end = min(total_frames - 1, start + clip_len - 1)
            idxs = np.linspace(start, end, clip_len).astype(int)

            clip_frames = []
            try:
                arr = vr.get_batch(idxs.tolist()).asnumpy()
                for frame in arr:
                    clip_frames.append(Image.fromarray(frame))
            except Exception:
                if last_valid_clip is not None:
                    clip_frames = list(last_valid_clip)
                else:
                    clip_frames = _extract_single_clip_cv2(video_path, idxs)

            if len(clip_frames) > 0:
                while len(clip_frames) < clip_len:
                    clip_frames.append(clip_frames[-1])

            if len(clip_frames) < min_valid_frames:
                if last_valid_clip is not None:
                    clip_frames = list(last_valid_clip)
                else:
                    return _extract_all_clips_cv2(video_path, num_clips, clip_len, min_valid_frames)

            clip_frames = clip_frames[:clip_len]
            last_valid_clip = clip_frames
            clips.append(clip_frames)

        if len(clips) != num_clips:
            return _extract_all_clips_cv2(video_path, num_clips, clip_len, min_valid_frames)

        return clips

    except Exception:
        return _extract_all_clips_cv2(video_path, num_clips, clip_len, min_valid_frames)


# =========================================================
# 4) 选择逻辑
# =========================================================
def select_active_indices(scores, min_keep=1, max_keep=5, margin=1.5):
    """
    通用 margin 选择逻辑
    """
    max_score = torch.max(scores)
    keep_mask = scores >= (max_score - margin)
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        keep_indices = torch.topk(scores, k=max_keep).indices

    return keep_indices


def select_summe_prompts_with_groups(video_level_scores, groups, min_keep=1, max_keep=5, margin=1.5):
    """
    SumMe 专用：
    1. 先算组分数（组内 max）
    2. 再做组级选择
    3. 每个被选中的组，只激活该组内得分最高的一个 prompt
    """
    group_names = list(groups.keys())
    group_scores = []
    group_best_prompt_idx = []

    for gname in group_names:
        member_idx = groups[gname]
        member_scores = video_level_scores[member_idx]
        best_local = torch.argmax(member_scores).item()
        best_prompt_idx = member_idx[best_local]
        best_score = member_scores[best_local]

        group_scores.append(best_score)
        group_best_prompt_idx.append(best_prompt_idx)

    group_scores = torch.stack(group_scores, dim=0)
    selected_group_idx = select_active_indices(
        group_scores,
        min_keep=min_keep,
        max_keep=max_keep,
        margin=margin
    )

    selected_prompt_idx = [group_best_prompt_idx[i] for i in selected_group_idx.cpu().tolist()]
    return selected_prompt_idx


# =========================================================
# 5) 单数据集处理
# =========================================================
def process_dataset(dataset_name, cfg, model, processor, device):
    video_dir = cfg["video_dir"]
    output_npy = cfg["output_npy"]
    vocab_file = cfg["vocab_file"]
    prompts = cfg["prompts"]
    groups = cfg["groups"]
    margin = cfg["margin"]
    min_keep = cfg["min_keep"]
    max_keep = cfg["max_keep"]

    print("\n" + "=" * 60)
    print(f"开始处理数据集: {dataset_name}")
    print("=" * 60)

    if not os.path.exists(video_dir):
        print(f"[跳过] 找不到目录: {video_dir}")
        return

    # 写入 prompt 词表文件（训练/评估会直接读取）
    with open(vocab_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")
    print(f"[OK] 已写入 prompt 词表: {vocab_file}")

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".webm"))])
    print(f"找到视频数: {len(video_files)}")

    video_labels = {}

    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        print(f"处理 {video_name} ...")

        clips = extract_clip_frames(video_path, num_clips=16, clip_len=16)
        if len(clips) == 0:
            print(f"  -> skip broken video: {video_name}")
            continue

        # 每个 clip 对所有 prompt 打分，再视频级平均
        clip_scores_all = []

        for frames in clips:
            inputs = processor(
                text=prompts,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image.float()   # [T, C]
                clip_scores = logits_per_image.mean(dim=0)            # [C]

            clip_scores_all.append(clip_scores)

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        video_level_scores = torch.stack(clip_scores_all, dim=0).mean(dim=0)   # [C]

        # TVSum 直接按 prompt 选；SumMe 先组内去冗余
        if groups is None:
            selected_prompt_idx = select_active_indices(
                video_level_scores,
                min_keep=min_keep,
                max_keep=max_keep,
                margin=margin
            ).cpu().tolist()
        else:
            selected_prompt_idx = select_summe_prompts_with_groups(
                video_level_scores,
                groups=groups,
                min_keep=min_keep,
                max_keep=max_keep,
                margin=margin
            )

        label_vector = np.zeros(len(prompts), dtype=np.float32)
        label_vector[selected_prompt_idx] = 1.0

        video_id = os.path.splitext(video_name)[0]
        # 与 build.py 的 dict_key 规则保持一致
        dict_key = video_id.replace('_fixed', '') if video_id.endswith('_fixed') else video_id
        video_labels[dict_key] = label_vector

        top5_vals, top5_idx = torch.topk(video_level_scores, k=min(5, len(prompts)))
        top5_prompts = [prompts[i] for i in top5_idx.cpu().tolist()]
        selected_prompts = [prompts[i] for i in selected_prompt_idx]

        print(f"  -> 选中标签数: {int(label_vector.sum())}")
        print(f"  -> 选中 prompt: {selected_prompts}")
        print("  -> Top5 raw scores:")
        for p, s in zip(top5_prompts, top5_vals.cpu().tolist()):
            print(f"     {p}: {s:.4f}")

    np.save(output_npy, video_labels)
    print(f"[OK] 已保存多热标签: {output_npy}")
    print(f"[OK] 视频条目数: {len(video_labels)}")


# =========================================================
# 6) 主程序
# =========================================================
def main():
    os.makedirs("labels", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")
    print("正在加载 CLIP 模型...")

    local_model_path = "./clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(local_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(local_model_path)

    for dataset_name, cfg in DATASETS.items():
        process_dataset(dataset_name, cfg, model, processor, device)


if __name__ == "__main__":
    main()