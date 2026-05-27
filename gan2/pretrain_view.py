import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import random_split

# 1つ目の事前学習スクリプトから設定とクラスをインポート
import pretrain

# =====================================================================
# --- 設定欄 ---
# =====================================================================
# 確認したい検証データのインデックス（複数指定可能）
TARGET_INDICES = [2] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前学習スクリプトのDEFINEからファイル名と重みパスを再構築
suffix = pretrain.DEFINE["OTHERINFO"] if pretrain.DEFINE["OTHERINFO"] else "BASIC"
FILENAME = f"PRETRAIN_{pretrain.DEFINE['BATCH_SIZE']}_{pretrain.DEFINE['EPOCHS']}_{pretrain.DEFINE['LEARNING_RATE']}_{pretrain.DEFINE['RESIDUAL_BLOCKS']}_{suffix}"
WEIGHT_PATH = f"weights_{FILENAME}/generator_best.pth"
# =====================================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading weights from: {WEIGHT_PATH}")

    # 1. Generatorの初期化と重みのロード
    generator = pretrain.Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        print("Successfully loaded pre-trained weights.")
    except FileNotFoundError:
        print(f"Error: 重みファイルが見つかりません ({WEIGHT_PATH})。学習が完了しているか確認してください。")
        return
    
    # 推論モードへ切り替え
    generator.eval()

    # 2. データの準備 (学習時と同じ分割で検証データを取得)
    full_train_dataset = pretrain.STL10SRDataset(root='./data', split='train')
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    _, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # 学習時とシードを統一
    )

    # 3. 画像の推論と描画準備
    num_images = len(TARGET_INDICES)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    
    # 1枚だけ指定した場合、axesが1次元になるため2次元に整形
    if num_images == 1:
        axes = [axes]

    # 勾配計算を無効化して推論
    with torch.no_grad():
        for i, idx in enumerate(TARGET_INDICES):
            # 検証データセットから画像を1ペア取得
            lr_img, hr_img = val_dataset[idx]
            
            # モデルに入力するためにバッチ次元を追加: (C, H, W) -> (1, C, H, W)
            lr_img_tensor = lr_img.unsqueeze(0).to(DEVICE)
            
            # 推論の実行
            sr_img_tensor = generator(lr_img_tensor)

            # 表示用にNumPy配列 (H, W, C) に変換
            hr_img_np = hr_img.permute(1, 2, 0).numpy()
            sr_img_np = sr_img_tensor[0].cpu().permute(1, 2, 0).numpy()

            # 低解像度画像（LR）はそのまま表示すると小さすぎるため、表示用にBicubicで拡大
            lr_img_display = TF.resize(
                lr_img, 
                size=[pretrain.DEFINE["OUTPUT_SIZE"], pretrain.DEFINE["OUTPUT_SIZE"]], 
                interpolation=TF.InterpolationMode.BICUBIC
            ).permute(1, 2, 0).numpy()

            # --- プロット ---
            # 左: 入力画像 (Bicubic拡大)
            axes[i][0].imshow(np.clip(lr_img_display, 0, 1))
            axes[i][0].set_title(f"Low Res (Bicubic)\nVal Index: {idx}")
            axes[i][0].axis('off')

            # 中央: 事前学習モデルによる超解像
            axes[i][1].imshow(np.clip(sr_img_np, 0, 1))
            axes[i][1].set_title("Pre-trained SR\n(MSE Only)")
            axes[i][1].axis('off')

            # 右: 正解の元画像
            axes[i][2].imshow(np.clip(hr_img_np, 0, 1))
            axes[i][2].set_title("High Res (Ground Truth)")
            axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()