import h5py
import os
import decord

# 配置路径
h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
video_dir = "data/TVSum/videos"
output_file = "tvsum_static_map.py"

def build_audit_map():
    print(f"🔍 1. 扫描物理视频指纹: {video_dir}")
    fingerprints = {}
    if os.path.exists(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(('.mp4', '.avi', '.webm', '.mkv')):
                vpath = os.path.join(video_dir, fname)
                try:
                    fingerprints[os.path.basename(vpath).split('.')[0]] = len(decord.VideoReader(vpath))
                except Exception:
                    pass
    
    print(f"📖 2. 解析 H5 帧数: {h5_path}")
    h5_frames = {}
    with h5py.File(h5_path, 'r') as h5_data:
        for k in h5_data.keys():
            h5_frames[k] = h5_data[k]['n_frames'][()]

    print(f"⚖️ 3. 计算全局距离矩阵与风险分级...")
    
    # 记录每个物理视频被谁作为了 Top-1 候选，用于检测碰撞
    video_to_h5_claims = {vname: [] for vname in fingerprints.keys()}
    
    results = {}
    for h5_key, h5_len in h5_frames.items():
        # 计算该 H5 视频与所有物理视频的误差
        distances = []
        for vname, vlen in fingerprints.items():
            distances.append((vname, vlen, abs(vlen - h5_len)))
        
        # 按误差从小到大排序
        distances.sort(key=lambda x: x[2])
        top3 = distances[:3]
        
        best_vname, best_vlen, best_diff = top3[0]
        video_to_h5_claims[best_vname].append(h5_key)
        
        results[h5_key] = {
            'h5_len': h5_len,
            'top3': top3,
            'status': 'SAFE', # 默认安全，后续修正
            'notes': []
        }

    # 风险评级与碰撞检测
    for h5_key, data in results.items():
        top3 = data['top3']
        best_vname, best_vlen, best_diff = top3[0]
        
        # 风险 1: 基础误差大于 0
        if best_diff > 0:
            data['status'] = 'REVIEW'
            data['notes'].append(f"帧数不完全匹配 (误差 {best_diff} 帧)")
            
        # 风险 2: 候选歧义 (Top-1 和 Top-2 帧数极其接近)
        if len(top3) > 1:
            second_diff = top3[1][2]
            if (second_diff - best_diff) <= 2:
                data['status'] = 'REVIEW'
                data['notes'].append(f"存在高度近似的备胎候选 ({top3[1][0]} 误差仅为 {second_diff})")
                
        # 风险 3: 致命碰撞 (多个 H5 key 抢夺同一个物理视频)
        if len(video_to_h5_claims[best_vname]) > 1:
            data['status'] = 'BAD'
            conflicts = ", ".join(video_to_h5_claims[best_vname])
            data['notes'].append(f"致命碰撞: 与 [{conflicts}] 竞争同一个视频 {best_vname}")

    # 4. 生成 Python 文件
    print(f"📝 4. 正在生成静态映射与审计文件: {output_file}")
    
    # 将 key 排序，保证输出美观
    sorted_keys = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('自动生成的 TVSum 静态映射表与风险审计报告。\n')
        f.write('请人工复核 REVIEW 和 BAD 级别的映射。\n')
        f.write('"""\n\n')
        
        f.write('TVSUM_STATIC_MAP = {\n')
        for k in sorted_keys:
            best_vname = results[k]['top3'][0][0]
            if results[k]['status'] == 'SAFE':
                f.write(f"    '{k}': '{best_vname}',  # [SAFE] exact match\n")
            else:
                # 把有问题的先注释掉，强迫人工核对后再解开
                f.write(f"    # '{k}': '{best_vname}',  # [{results[k]['status']}] 需复核!\n")
        f.write('}\n\n')
        
        f.write('TVSUM_REVIEW_NOTES = {\n')
        for k in sorted_keys:
            if results[k]['status'] != 'SAFE':
                f.write(f"    '{k}': {{\n")
                f.write(f"        'h5_len': {results[k]['h5_len']},\n")
                f.write(f"        'issues': {results[k]['notes']},\n")
                f.write(f"        'top3_candidates': {results[k]['top3']}  # (vname, vlen, diff)\n")
                f.write(f"    }},\n")
        f.write('}\n')
        
    print("🎉 审计文件生成完毕！")

if __name__ == "__main__":
    build_audit_map()