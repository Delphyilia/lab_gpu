import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
import srgan_origin

# =====================================================================
# --- 確認したい画像のインデックスを指定 ---
# (リストで複数指定すると、横に並べて比較できます)
# =====================================================================
TARGET_INDICES = [1,2,7,42] 


# --- データの準備 (評価コードと全く同じ分割を行い、インデックスを一致させます) ---
full_train_dataset = srgan_origin.STL10SRDataset(root='./data', split='train')
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

_, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)  # ※ここを42で固定することが重要です
)

def preview_validation_images(dataset, indices):
    """ 指定されたインデックスのLR画像とHR画像をプレビュー表示する """
    num_images = len(indices)
    
    # 画像数に合わせて描画領域を調整
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    
    # 1枚だけ指定された場合、axesが1次元になるのを防ぐための処理
    if num_images == 1:
        axes = np.array([axes]).T

    print(f"全検証データ数: {len(dataset)}枚")
    print("画像のプレビューを描画します...")

    for i, idx in enumerate(indices):
        # 範囲外のインデックスが指定された場合の安全対策
        if idx < 0 or idx >= len(dataset):
            print(f"警告: インデックス {idx} は範囲外です (0 ～ {len(dataset)-1} の範囲で指定してください)")
            continue

        lr_img_tensor, hr_img_tensor = dataset[idx]
        
        # PyTorchのTensor (C, H, W) を Matplotlib用の NumPy配列 (H, W, C) に変換
        lr_img = lr_img_tensor.cpu().permute(1, 2, 0).numpy()
        hr_img = hr_img_tensor.cpu().permute(1, 2, 0).numpy()
        
        # 1行目: 低解像度 (LR)
        axes[0, i].imshow(np.clip(lr_img, 0, 1))
        axes[0, i].set_title(f"Index: {idx}\nLow Res (LR)")
        axes[0, i].axis('off')
        
        # 2行目: 高解像度 (HR)
        axes[1, i].imshow(np.clip(hr_img, 0, 1))
        axes[1, i].set_title(f"Index: {idx}\nHigh Res (HR)")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

# 実行
if __name__ == "__main__":
    preview_validation_images(val_dataset, TARGET_INDICES)