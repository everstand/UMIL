import os

video_dir = "data/SumMe/videos"
output_file = "labels/summe_train_list.txt"

with open(output_file, 'w') as f:
    for video_name in os.listdir(video_dir):
        if video_name.endswith(('.mp4', '.avi', '.webm')):
            # 写入相对路径
            f.write(f"{os.path.join(video_dir, video_name)}\n")
            
print(f"✅ 成功生成 SumMe 视频清单: {output_file}")