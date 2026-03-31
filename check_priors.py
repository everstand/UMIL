import numpy as np
import os

def check_priors():
    file_path = "labels/summe_saliency_priors.npy"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return

    priors = np.load(file_path, allow_pickle=True).item()
    print(f"Total videos in prior file: {len(priors)}")
    print("-" * 40)
    
    for i, (k, v) in enumerate(priors.items()):
        v_arr = np.array(v)
        print(f"Vid: {k} | Shape: {v_arr.shape} | Min: {np.min(v_arr):.4f} | Max: {np.max(v_arr):.4f}")
        if i == 2:
            break

if __name__ == "__main__":
    check_priors()