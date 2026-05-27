import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import lpips
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split

import srgan_pretrain

# =====================================================================
# --- ユーザー設定欄 (実験条件や表示のカスタマイズ) ---
# =====================================================================
TARGET_VAL_INDEX = 2  # 図示およびLPIPS計算に使用する検証データのインデックス (0 ～ 999)

# グラフのY軸の表示範囲 (他の実験とスケールを固定するため)
YLIM_LOSS = (0.0, 4.0)   # Lossグラフの範囲 (None にすると自動調整)
YLIM_SSIM = (0.0, 1.0)   # SSIMグラフの範囲
YLIM_LPIPS = (0.0, 0.6)  # LPIPSグラフの範囲
# =====================================================================

# --- 学習時と同じ設定の読み込み ---
BATCH_SIZE = srgan_pretrain.BATCH_SIZE
EPOCHS = srgan_pretrain.EPOCHS
LEARNING_RATE = srgan_pretrain.LEARNING_RATE
RESIDUAL_BLOCKS = srgan_pretrain.RESIDUAL_BLOCKS
GAN_LOSS_WEIGHT = srgan_pretrain.GAN_LOSS_WEIGHT
OTHERINFO = srgan_pretrain.OTHERINFO

suffix = OTHERINFO if OTHERINFO else "BASIC"
FILENAME = f"{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE}_{RESIDUAL_BLOCKS}_{GAN_LOSS_WEIGHT}_{suffix}"

JSON_PATH = f"training_history_{FILENAME}.json"
WEIGHTS_DIR = f"weights_{FILENAME}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化
GENERATOR = srgan_pretrain.Generator().to(DEVICE)
DISCRIMINATOR = srgan_pretrain.Discriminator().to(DEVICE)

# LPIPSモデルの初期化 (VGGネットワークを使用)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)

# --- データの準備 ---
full_train_dataset = srgan_pretrain.STL10SRDataset(root='./data', split='train')
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)  # 再現性のためにシードを固定
)

# 履歴データの読み込み
with open(JSON_PATH, "r", encoding="utf-8") as f:
    history = json.load(f)

# =====================================================================
# --- プロット・評価関数 ---
# =====================================================================

def plot_losses(history, ylim=None):
    g_loss = history["g_loss"]
    d_loss = history["d_loss"]
    epochs = range(1, len(g_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, g_loss, label="Generator Loss", color="blue")
    plt.plot(epochs, d_loss, label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss over Epochs")
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ssim(history, ylim=None):
    val_ssim = history["val_ssim"]
    epochs = range(1, len(val_ssim) + 1)
    
    best_epoch_idx = np.argmax(val_ssim)
    best_epoch = best_epoch_idx + 1
    best_value = val_ssim[best_epoch_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_ssim, label="Validation SSIM", color="green")
    plt.scatter(best_epoch, best_value, color="red", zorder=5, 
                label=f"Best SSIM: {best_value:.4f} at Epoch {best_epoch}")
    
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM over Epochs")
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_epoch


def calculate_and_plot_lpips(lr_img_tensor, hr_img_tensor, total_epochs, ylim=None):
    """ 指定された1枚の画像ペアに対して各エポックのLPIPSを計算・プロットする """
    generator = GENERATOR
    lpips_scores = []
    
    # LPIPSは [-1, 1] の入力を期待するためスケーリング
    hr_img_lpips = hr_img_tensor * 2 - 1

    print(f"Calculating LPIPS for val_dataset[{TARGET_VAL_INDEX}] across all epochs...")
    for epoch in range(1, total_epochs + 1):
        weight_path = f"{WEIGHTS_DIR}/generator_epoch_{epoch:03d}.pth"
        if not os.path.exists(weight_path):
            lpips_scores.append(float('inf'))
            continue
            
        generator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        generator.eval()
        
        with torch.no_grad():
            gen_img = generator(lr_img_tensor)
            gen_img_lpips = gen_img * 2 - 1
            # 1枚のペアに対するLPIPSスコアを計算
            score = loss_fn_vgg(gen_img_lpips, hr_img_lpips).item()
            lpips_scores.append(score)
            
    epochs = range(1, len(lpips_scores) + 1)
    best_epoch_idx = np.argmin(lpips_scores)
    best_epoch = best_epoch_idx + 1
    best_value = lpips_scores[best_epoch_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lpips_scores, label="Validation LPIPS (Single Image)", color="purple")
    plt.scatter(best_epoch, best_value, color="red", zorder=5, 
                label=f"Best LPIPS: {best_value:.4f} at Epoch {best_epoch}")
    
    plt.xlabel("Epoch")
    plt.ylabel("LPIPS (Lower is better)")
    plt.title(f"Validation LPIPS over Epochs (Image Index: {TARGET_VAL_INDEX})")
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_epoch


def display_best_images(lr_img_tensor, hr_img_tensor, best_ssim_epoch, best_lpips_epoch):
    """ 指定された画像ペアを用いて、SSIMとLPIPSそれぞれのベストエポックでの生成結果を表示する """
    generator = GENERATOR
    
    # 表示用にNumPy配列（H, W, C）に変換
    hr_img_np = hr_img_tensor[0].cpu().permute(1, 2, 0).numpy()
    
    # 低解像度画像を表示用にバイキュービックリサイズ
    lr_img_display = TF.resize(lr_img_tensor, size=[96, 96], 
                               interpolation=TF.InterpolationMode.BICUBIC)[0].cpu().permute(1, 2, 0).numpy()

    # --- SSIMベストモデルでの生成 ---
    generator.load_state_dict(torch.load(f"{WEIGHTS_DIR}/generator_epoch_{best_ssim_epoch:03d}.pth", map_location=DEVICE))
    generator.eval()
    with torch.no_grad():
        sr_img_ssim = generator(lr_img_tensor)[0].cpu().permute(1, 2, 0).numpy()

    # --- LPIPSベストモデルでの生成 ---
    generator.load_state_dict(torch.load(f"{WEIGHTS_DIR}/generator_epoch_{best_lpips_epoch:03d}.pth", map_location=DEVICE))
    generator.eval()
    with torch.no_grad():
        sr_img_lpips = generator(lr_img_tensor)[0].cpu().permute(1, 2, 0).numpy()

    # --- 画像のプロット (2行3列) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1行目: SSIMが最も良かったモデル
    axes[0, 0].imshow(np.clip(lr_img_display, 0, 1))
    axes[0, 0].set_title("Low Resolution (Bicubic)")
    axes[0, 1].imshow(np.clip(sr_img_ssim, 0, 1))
    axes[0, 1].set_title(f"SR (Best SSIM: Ep {best_ssim_epoch})")
    axes[0, 2].imshow(np.clip(hr_img_np, 0, 1))
    axes[0, 2].set_title("High Resolution (GT)")

    # 2行目: LPIPSが最も良かったモデル
    axes[1, 0].imshow(np.clip(lr_img_display, 0, 1))
    axes[1, 0].set_title("Low Resolution (Bicubic)")
    axes[1, 1].imshow(np.clip(sr_img_lpips, 0, 1))
    axes[1, 1].set_title(f"SR (Best LPIPS: Ep {best_lpips_epoch})")
    axes[1, 2].imshow(np.clip(hr_img_np, 0, 1))
    axes[1, 2].set_title("High Resolution (GT)")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# =====================================================================
# --- メイン処理の実行 ---
# =====================================================================
if __name__ == "__main__":
    # 1. 損失関数のプロット
    plot_losses(history, ylim=YLIM_LOSS)

    # 2. SSIMのプロットとベストエポックの取得
    best_ssim_epoch = plot_ssim(history, ylim=YLIM_SSIM)
    print(f"SSIMが最も良かったエポック: {best_ssim_epoch}")

    # 3. 指定されたインデックスのデータを取得し、ミニバッチ化 (1, C, H, W) してGPUへ
    lr_img, hr_img = val_dataset[TARGET_VAL_INDEX]
    lr_img_tensor = lr_img.unsqueeze(0).to(DEVICE)
    hr_img_tensor = hr_img.unsqueeze(0).to(DEVICE)

    # 4. 指定画像に対するLPIPSの計算とプロット
    best_lpips_epoch = calculate_and_plot_lpips(lr_img_tensor, hr_img_tensor, len(history["g_loss"]), ylim=YLIM_LPIPS)
    print(f"LPIPSが最も良かったエポック: {best_lpips_epoch}")

    # 5. 画像の比較表示
    display_best_images(lr_img_tensor, hr_img_tensor, best_ssim_epoch, best_lpips_epoch)