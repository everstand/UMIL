import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# =====================================================================
# 🌟 核心配置区
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

text_prompts = [f"A video of a person {action}" for action in candidate_actions]

def extract_frames(video_path, num_frames=16):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
        
        step = max(total_frames // num_frames, 1)
        frames = []
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)
        cap.release()
        return frames
    except Exception as e:
        print(f"  -> 🚨 物理层损坏跳过: {e}")
        return []

def process_dataset(dataset_name, config, model, processor, device):
    video_dir = config["video_dir"]
    output_npy = config["output_npy"]
    
    print(f"\n" + "="*50)
    print(f"🚀 开始提取数据集: {dataset_name}")
    print("="*50)

    if not os.path.exists(video_dir):
        print(f"🚨 错误: 找不到文件夹 '{video_dir}'，跳过 {dataset_name}。")
        return

    video_labels = {}
    top_k = 3 
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.webm'))]
    print(f"🔍 找到 {len(video_files)} 个视频文件。")
    
    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        print(f"  ⏳ 提取 {video_name}...")
        
        frames = extract_frames(video_path, num_frames=16)
        if len(frames) == 0:
            print(f"  -> ⚠️ 警告: 无法抽取画面，已跳过")
            continue
            
        inputs = processor(text=text_prompts, images=frames, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1)
            video_level_scores = probs.mean(dim=0)
            
        top_indices = video_level_scores.topk(top_k)[1].cpu().numpy()
        
        # 构建 Multi-hot 向量
        num_classes = len(candidate_actions)
        label_vector = np.zeros(num_classes, dtype=np.float32)
        label_vector[top_indices] = 1.0
        
        video_id = os.path.splitext(video_name)[0]
        video_labels[video_id] = label_vector
        
        extracted_actions = [candidate_actions[idx] for idx in top_indices]
        print(f"  -> ✅ 匹配动作: {extracted_actions}")
        
        del inputs, outputs, logits_per_image, probs, video_level_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save(output_npy, video_labels)
    print(f"🎉 {dataset_name} 处理完毕！真实特征已保存至: {output_npy}")

def main():
    os.makedirs("labels", exist_ok=True)
    output_txt = "labels/action_vocabulary.txt"
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        for action in candidate_actions:
            f.write(action + '\n')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧠 正在从本地文件夹加载 CLIP 模型...")
    
    # 直接读取当前目录下的 clip-vit-base-patch32 文件夹！
    local_model_path = "./clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(local_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(local_model_path)
    
    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config, model, processor, device)

if __name__ == "__main__":
    main()