import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from decord import VideoReader, cpu

# =====================================================================
# 核心配置区
# =====================================================================
DATASETS = {
    "SumMe": {
        "video_dir": "data/SumMe/videos",
        "output_npy": "labels/summe_multihot_labels.npy"
    },
    "TVSum": {
        "video_dir": "data/TVSum/videos",
        "output_npy": "labels/tvsum_multihot_labels.npy"
    }
}
# =====================================================================

candidate_actions = [
    "scenic landscape or beautiful view",
    "landmark or sightseeing moment",
    "crowd gathering or group scene",
    "people interacting or talking",
    "celebration or festive moment",
    "music or dance performance",
    "sports action or highlight play",
    "fast motion or stunt",
    "animal or pet activity",
    "children playing",
    "food preparation or eating",
    "travel or transportation scene",
    "vehicle activity",
    "water activity or swimming",
    "outdoor adventure",
    "object manipulation or hands-on activity",
    "interesting close-up object",
    "taking photos or recording video",
    "posing for the camera",
    "emotional or expressive moment",
    "surprising or unusual event",
    "playful interaction",
    "group activity or game",
    "ceremony or public event",
    "visually distinctive scene",
    "high visual motion",
    "scene transition or viewpoint change",
    "important interaction moment",
    "memorable highlight moment",
    "rare or attention-grabbing event"
]

text_prompts = [f"A video segment showing {action}" for action in candidate_actions]


def extract_frames_cv2_fallback(video_path, num_frames=16, min_valid_ratio=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    target_indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
    target_ptr = 0
    current_idx = 0
    frames = []
    last_valid = None

    try:
        while target_ptr < len(target_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx >= target_indices[target_ptr]:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    frames.append(pil_img)
                    last_valid = pil_img
                except Exception:
                    if last_valid is not None:
                        frames.append(last_valid)

                target_ptr += 1

            current_idx += 1
    finally:
        cap.release()

    while len(frames) < num_frames and last_valid is not None:
        frames.append(last_valid)

    if len(frames) < max(1, int(num_frames * min_valid_ratio)):
        return []

    return frames[:num_frames]


def extract_frames(video_path, num_frames=16, min_valid_ratio=0.5):
    """
    更稳版本：
    1. decord 作为主解码器
    2. 单帧失败时用上一帧兜底
    3. 整体失败时回退到 OpenCV 顺序解码
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            return []

        target_indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)

        frames = []
        last_valid = None

        for idx in target_indices:
            try:
                frame = vr[idx].asnumpy()
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)
                last_valid = pil_img
            except Exception:
                if last_valid is not None:
                    frames.append(last_valid)

        while len(frames) < num_frames and last_valid is not None:
            frames.append(last_valid)

        if len(frames) < max(1, int(num_frames * min_valid_ratio)):
            return extract_frames_cv2_fallback(video_path, num_frames, min_valid_ratio)

        return frames[:num_frames]

    except Exception:
        return extract_frames_cv2_fallback(video_path, num_frames, min_valid_ratio)


def select_pseudo_labels(video_level_scores, min_keep=1, max_keep=5, margin=1.5):
    """
    不再强制 top-3。
    规则：
    1. 先对 raw logits 做视频级平均，不先 softmax
    2. 保留所有 >= (max_score - margin) 的类别
    3. 至少保留 1 个，最多保留 5 个
    """
    max_score = torch.max(video_level_scores)
    keep_mask = video_level_scores >= (max_score - margin)
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(video_level_scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        keep_indices = torch.topk(video_level_scores, k=max_keep).indices

    return keep_indices


def process_dataset(dataset_name, config, model, processor, device, margin=1.5):
    video_dir = config["video_dir"]
    output_npy = config["output_npy"]

    print("\n" + "=" * 50)
    print(f"开始处理数据集: {dataset_name}")
    print("=" * 50)

    if not os.path.exists(video_dir):
        print(f"错误: 找不到文件夹 '{video_dir}'，跳过 {dataset_name}")
        return

    video_labels = {}
    video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".webm"))]
    print(f"找到 {len(video_files)} 个视频文件")

    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        print(f"  处理 {video_name} ...")

        frames = extract_frames(video_path, num_frames=16, min_valid_ratio=0.5)
        if len(frames) == 0:
            print("  -> 解码失败或有效帧过少，跳过")
            continue

        inputs = processor(
            text=text_prompts,
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            # 关键修改：不先 softmax，直接对 raw logits 做视频级平均
            logits_per_image = outputs.logits_per_image.float()   # [T, C]
            video_level_scores = logits_per_image.mean(dim=0)     # [C]

            keep_indices = select_pseudo_labels(
                video_level_scores,
                min_keep=1,
                max_keep=5,
                margin=margin,
            )

        label_vector = np.zeros(len(candidate_actions), dtype=np.float32)
        label_vector[keep_indices.cpu().numpy()] = 1.0

        video_id = os.path.splitext(video_name)[0]
        video_labels[video_id] = label_vector

        top5_vals, top5_idx = torch.topk(video_level_scores, k=min(5, len(candidate_actions)))
        top5_actions = [candidate_actions[i] for i in top5_idx.cpu().tolist()]
        selected_actions = [candidate_actions[i] for i in keep_indices.cpu().tolist()]

        print(f"  -> 选中标签: {selected_actions}")
        print("  -> Top5 raw scores:")
        for a, s in zip(top5_actions, top5_vals.cpu().tolist()):
            print(f"     {a}: {s:.4f}")

        del inputs, outputs, logits_per_image, video_level_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save(output_npy, video_labels)
    print(f"{dataset_name} 处理完毕，已保存到: {output_npy}")


def main():
    os.makedirs("labels", exist_ok=True)

    with open("labels/action_vocabulary.txt", "w", encoding="utf-8") as f:
        for action in candidate_actions:
            f.write(action + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("正在加载 CLIP 模型...")

    local_model_path = "./clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(local_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(local_model_path)

    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config, model, processor, device, margin=1.5)


if __name__ == "__main__":
    main()