import os
import argparse

def generate_video_list(dataset_name):
    """动态生成数据集的训练/测试视频路径列表"""
    
    if dataset_name == 'summe':
        video_dir = 'data/SumMe/videos'
        output_file = 'labels/summe_train_list.txt'
    elif dataset_name == 'tvsum':
        video_dir = 'data/TVSum/videos'
        output_file = 'labels/tvsum_train_list.txt'
    else:
        raise ValueError(f"🚨 不支持的数据集: {dataset_name}")

    print(f"🔍 正在扫描 {dataset_name.upper()} 视频目录: {video_dir}")
    
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"找不到目录 {video_dir}，请检查原始数据是否解压正确！")

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.webm', '.mkv'))]
    video_files.sort()  # 保证顺序一致性
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for vid in video_files:
            vpath = os.path.join(video_dir, vid)
            f.write(f"{vpath}\n")
            
    print(f"✅ 成功生成 {len(video_files)} 个视频路径，已保存至: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="统一生成数据集路径列表脚本")
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum', 'all'], help="指定要生成列表的数据集")
    args = parser.parse_args()

    if args.dataset == 'all':
        generate_video_list('summe')
        generate_video_list('tvsum')
    else:
        generate_video_list(args.dataset)