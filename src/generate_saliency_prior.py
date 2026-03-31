# 先离线生成每个视频每个窗口的显著性先验
# 不是当标签，只当 prior

import os
import cv2
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel


DATASETS = {
    "SumMe": {
        "video_dir": "data/SumMe/videos",
        "output_npy": "labels/summe_saliency_priors.npy"
    },
    "TVSum": {
        "video_dir": "data/TVSum/videos",
        "output_npy": "labels/tvsum_saliency_priors.npy"
    }
}

saliency_prompts = [
    "a video segment that is a memorable highlight",
    "a video segment worth including in a summary",
    "a video segment containing a key event",
    "a video segment that is visually distinctive",
    "a video segment with strong motion or action",
    "a video segment showing an important interaction",
    "a video segment with emotional expression",
    "a video segment with a surprising or unusual event",
    "a video segment showing a crowd, gathering, or celebration",
    "a video segment with a scenic or visually impressive view",
    "a video segment that looks like a climax or peak moment",
    "a video segment that is more important than the rest of the video",
]

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
    更稳版本：
    1. decord 主解码
    2. 单个 clip 失败时，先用上一段兜底
    3. decord 全局失败时，整体回退到 cv2
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

def score_clip_saliency(model, processor, device, frames):
    inputs = processor(
        text=saliency_prompts,
        images=frames,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.float()   # [T, P]
        clip_scores = logits.mean(dim=0)            # [P]
        score = torch.topk(clip_scores, k=min(3, clip_scores.numel())).values.mean().item()          # 一个窗口一个标量
    return score

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = "./clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(local_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(local_model_path)

    os.makedirs("labels", exist_ok=True)

    for dataset_name, cfg in DATASETS.items():
        priors = {}
        video_dir = cfg["video_dir"]

        for video_name in os.listdir(video_dir):
            if not video_name.endswith((".mp4", ".avi", ".webm")):
                continue

            video_path = os.path.join(video_dir, video_name)
            clips = extract_clip_frames(video_path, num_clips=16, clip_len=16)
            if len(clips) == 0:
                print(f"skip broken video: {video_name}")
                continue

            scores = []
            for frames in clips:
                s = score_clip_saliency(model, processor, device, frames)
                scores.append(s)

            scores = np.array(scores, dtype=np.float32)

            # 标准化到 [0,1]
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = np.ones_like(scores, dtype=np.float32)

            video_id = os.path.splitext(video_name)[0]
            priors[video_id] = scores

        np.save(cfg["output_npy"], priors)
        print(f"saved: {cfg['output_npy']}")

if __name__ == "__main__":
    main()