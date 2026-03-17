import os

# 把路径换成 TVSum 的文件夹
video_dir = "data/TVSum/videos"
output_file = "labels/tvsum_train_list.txt"

with open(output_file, 'w') as f:
    for video_name in os.listdir(video_dir):
        if video_name.endswith(('.mp4', '.avi', '.webm')):
            # 写入相对路径
            f.write(f"{os.path.join(video_dir, video_name)}\n")
            
print(f"✅ 成功生成 TVSum 视频清单: {output_file}")