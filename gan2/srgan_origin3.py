import os
from torchvision.utils import save_image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import InterpolationMode, Normalize
import numpy as np
import numpy.typing as npt
from torch.utils.data import random_split
from typing import Tuple, List, Dict
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------------------------------------------------
# Hyperparameters & Configurations
# ---------------------------------------------------------
INPUT_SIZE: int = 24
OUTPUT_SIZE: int = 96
BATCH_SIZE: int = 16
EPOCHS: int = 200
LEARNING_RATE: float = 1e-4
PATIENCE: int = 1000000000  # Early stoppingの忍耐エポック数
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
RESIDUAL_BLOCKS: int = 16  # SRResNetのResidual Blockの数 (元論文は16ですが軽量化のため5としています)
GAN_LOSS_WEIGHT: float = 1e-3  # Generatorの損失に対するGAN Lossの重み (Content Lossとのバランスを取るために小さめに設定)

OTHERINFO = ""

suffix = OTHERINFO if OTHERINFO else "BASIC"
FILENAME = f"{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE}_{RESIDUAL_BLOCKS}_{GAN_LOSS_WEIGHT}_{suffix}"

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class STL10SRDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    r""":class:`Dataset` を表す抽象クラスです。

    キーからデータサンプルへのマッピングを表すすべてのデータセットは、
    このクラスをサブクラス化（継承）する必要があります。
    すべてのサブクラスは :meth:`__getitem__` をオーバーライドし、指定されたキーに対するデータサンプルの取得をサポートする必要があります。
    また、サブクラスはオプションで :meth:`__len__` をオーバーライドすることもできます。
    これは、多くの :class:`~torch.utils.data.Sampler` の実装や :class:`~torch.utils.data.DataLoader` のデフォルト設定において、データセットのサイズを返すことが期待されているためです。
    さらに、バッチサンプルの読み込みを高速化するために、サブクラスで任意に :meth:`__getitems__` を実装することもできます。
    このメソッドは、バッチを構成するサンプルのインデックスのリストを受け取り、サンプルのリストを返します。

    .. note::
      :class:`~torch.utils.data.DataLoader` はデフォルトで、整数のインデックスを生成（yield）するインデックスサンプラーを構築します。
      整数以外のインデックス/キーを持つマップスタイルのデータセットで動作させるには、カスタムサンプラーを提供する必要があります。(STL10は整数インデックスなので、ここでは必要ありません)
    """
    def __init__(self, root: str, split: str = 'train') -> None:
        """
        STL10データセットを読み込み、低解像度と高解像度のペアを生成するカスタムデータセットクラスです。
        Args:
            root (str): データセットのルートディレクトリ。
            split (str): データセットの分割（'train' または 'test'）。

        Returns:
            None    
        """
        super().__init__()
        self.dataset = datasets.STL10(root=root, split=split, download=True)
        
        # 高解像度画像 (HR: 96x96) 用の変換
        self.transform_hr = transforms.Compose([
            # [0, 1]にスケーリング
            transforms.ToTensor(), 
        ])
        
        # 低解像度画像 (LR: 24x24) 用の変換
        self.transform_lr = transforms.Compose([
            # バイキュービック補間で24x24にリサイズ
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            # [高さ、幅、チャンネル] -> [チャンネル、高さ、幅] に変換
            # さらに[0, 1]にスケーリング
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """
        データセットのサンプル数を返します。
        Returns:
            int: データセットのサンプル数。
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスに対するサンプルを取得します。
        Args:
            index (int): データセット内のサンプルのインデックス。
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 低解像度画像と高解像度画像のタプル。
        """
        img, _ = self.dataset[index]
        lr_img: torch.Tensor = self.transform_lr(img)
        hr_img: torch.Tensor = self.transform_hr(img)
        return lr_img, hr_img

# ---------------------------------------------------------
# Models: Generator (SRResNet)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    SRResNetのResidual Blockを表すクラスです。
    Args:
        channels (int): ブロック内の畳み込み層のチャンネル数。
    
    Returns:
        None"""

    def __init__(self, channels: int) -> None:
        """
        conv1 -> BatchNorm -> PReLU -> conv2 -> BatchNorm -> Skip Connection
        Args:
            channels (int): ブロック内の畳み込み層のチャンネル数。
        Returns:
            None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        実際の順伝播処理を定義します。
        スキップ接続を使用して、入力を出力に直接加算します。
        Args:
            x (torch.Tensor): ブロックへの入力テンソル。
        Returns:
            torch.Tensor: ブロックの出力テンソル。
        """
        res: torch.Tensor = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class Generator(nn.Module):
    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # 5つのResidual Blocks (元論文は16ですが軽量化のため5としています)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(RESIDUAL_BLOCKS)])
        
        # Skip Connectionの後の畳み込み層
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling (24 -> 48 -> 96)
        upsampling: List[nn.Module] = []
        for _ in range(scale_factor // 2):
            upsampling.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ])
        self.upsampling = nn.Sequential(*upsampling)
        
        # 最後の畳み込み層でRGB画像を生成
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Sigmoid() # [0, 1]の出力に正規化
        )

    # Generatorの順伝播処理を定義します。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1: torch.Tensor = self.conv1(x)
        out: torch.Tensor = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

# ---------------------------------------------------------
# Models: Discriminator
# ---------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def discriminator_block(in_filters: int, out_filters: int, stride: int, normalize: bool) -> List[nn.Module]:
            # 畳み込み -> (オプションでBatchNorm) -> LeakyReLUのブロックを定義するユーティリティ関数
            layers: List[nn.Module] = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Discriminatorの畳み込みブロックを定義します。96x96を入力とし、最終的に1x1の特徴マップになるように設計します。
        self.model = nn.Sequential(
            *discriminator_block(3, 64, stride=1, normalize=False),
            *discriminator_block(64, 64, stride=2, normalize=True),
            *discriminator_block(64, 128, stride=1, normalize=True),
            *discriminator_block(128, 128, stride=2, normalize=True),
            *discriminator_block(128, 256, stride=1, normalize=True),
            *discriminator_block(256, 256, stride=2, normalize=True),
            *discriminator_block(256, 512, stride=1, normalize=True),
            *discriminator_block(512, 512, stride=2, normalize=True),
        )
        
        # 96x96を入力とし、stride=2が4回 -> 96 / 16 = 6
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
            # BCEWithLogitsLossを使用するため、Sigmoidは不要
        )

    # Discriminatorの順伝播処理を定義します。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# Feature Extractor (Perceptual Loss)
# ---------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # VGG19の事前学習済みモデルをロードします。ここでは、ImageNetで学習された重みを使用します。
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)#ここの引数は何 -> ImageNet1K V1の重みを指定しています。
        # Activation前までの特徴量マップを取得
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # ImageNetの平均と標準偏差で正規化する層を追加 (Perceptual Loss用)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 1]の入力テンソルをImageNet基準に正規化
        x = self.normalize(x)
        # VGG19の特徴量マップを抽出します。これをContent Lossの計算に使用します。
        out: torch.Tensor = self.feature_extractor(x)
        return out



class EarlyStopping:
    def __init__(self, patience: int = 15) -> None:
        self.patience: int = patience
        self.counter: int = 0
        self.best_score: float = -1.0
        self.early_stop: bool = False

    def __call__(self, current_score: float) -> None:
        if current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------

def train() -> None:
    # 保存用ディレクトリの作成
    os.makedirs(f"weights_{FILENAME}", exist_ok=True)
    os.makedirs(f"saved_images_{FILENAME}", exist_ok=True)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    # 1. STL10の 'train' データセット全体を取得
    full_train_dataset = STL10SRDataset(root='./data', split='train')
    
    # 2. trainデータを 8割(Train) と 2割(Validation) に分割する
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # 再現性のためにシードを固定
    )

    # 3. Train用とValidation用のDataLoaderを作成
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )
    val_dataloader = DataLoader( # これを学習ループ内の Validation Phase で使用する
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    # 4. STL10の 'test' は最終評価用 (Test) として温存する
    # ※ 学習ループ (EPOCHSのfor文) の中では絶対にアクセスしない！
    test_dataset = STL10SRDataset(root='./data', split='test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    # 経過観察用の固定画像は validation データから取得するように変更
    val_fixed_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    fixed_lr_imgs, fixed_hr_imgs = next(iter(val_fixed_dataloader))
    fixed_lr_imgs = fixed_lr_imgs.to(DEVICE)
    save_image(fixed_hr_imgs, f"saved_images_{FILENAME}/real_hr_samples.png")

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    feature_extractor = FeatureExtractor().to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss().to(DEVICE)
    criterion_content = nn.MSELoss().to(DEVICE)

    early_stopping = EarlyStopping(patience=PATIENCE)

    history: Dict[str, List[float]] = {
        "g_loss": [],
        "d_loss": [],
        "val_ssim": [] # 訓練時のssimから、検証時のssimに変更
    }

    best_ssim: float = -1.0 # 最高SSIM記録用

    print("Training Started...")
    for epoch in range(1, EPOCHS + 1):
        gen_loss_epoch: float = 0.0
        disc_loss_epoch: float = 0.0

        # --- Training Phase ---
        generator.train()
        discriminator.train()

        for lr_imgs, hr_imgs in train_dataloader:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            valid_labels: torch.Tensor = torch.ones((BATCH_SIZE, 1), device=DEVICE, dtype=torch.float32)
            fake_labels: torch.Tensor = torch.zeros((BATCH_SIZE, 1), device=DEVICE, dtype=torch.float32)

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            gen_imgs: torch.Tensor = generator(lr_imgs)

            pred_fake: torch.Tensor = discriminator(gen_imgs)
            loss_GAN: torch.Tensor = criterion_GAN(pred_fake, valid_labels)

            gen_features: torch.Tensor = feature_extractor(gen_imgs)
            real_features: torch.Tensor = feature_extractor(hr_imgs).detach()
            loss_content: torch.Tensor = criterion_content(gen_features, real_features)

            loss_G: torch.Tensor = loss_content + GAN_LOSS_WEIGHT * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real: torch.Tensor = discriminator(hr_imgs)
            loss_real: torch.Tensor = criterion_GAN(pred_real, valid_labels)

            pred_fake = discriminator(gen_imgs.detach())
            loss_fake: torch.Tensor = criterion_GAN(pred_fake, fake_labels)

            loss_D: torch.Tensor = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss_epoch += float(loss_G.item())
            disc_loss_epoch += float(loss_D.item())

        # --- Validation Phase ---
        # 検証データでSSIMを計算し、Early Stoppingなどの判定基準にする
        # --- Validation Phase ---
        generator.eval()
        with torch.no_grad():
            for val_lr_imgs, val_hr_imgs in val_dataloader:
                val_lr_imgs = val_lr_imgs.to(DEVICE)
                val_hr_imgs = val_hr_imgs.to(DEVICE)
                
                val_gen_imgs = generator(val_lr_imgs)
                # バッチごとのデータを蓄積
                ssim_metric.update(val_gen_imgs, val_hr_imgs)

        # エポック全体の正確なSSIMを計算
        avg_val_ssim: float = float(ssim_metric.compute().item())
        
        # 次のエポックのために内部状態をリセット（重要）
        ssim_metric.reset()


        # エポックごとの平均値を計算
        avg_g_loss: float = gen_loss_epoch / len(train_dataloader)
        avg_d_loss: float = disc_loss_epoch / len(train_dataloader)

        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch:03d}/{EPOCHS}] [G loss: {avg_g_loss:.4f}] [D loss: {avg_d_loss:.4f}] [Val SSIM: {avg_val_ssim:.4f}]")

        # ---------------------------------------------------------
        # モデルと画像の保存処理
        # ---------------------------------------------------------
        # 1. 毎エポックの重みを保存
        torch.save(generator.state_dict(), f"weights_{FILENAME}/generator_epoch_{epoch:03d}.pth")

        # 2. 検証用SSIMが過去最高を更新したらベストモデルとして保存
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save(generator.state_dict(), f"weights_{FILENAME}/generator_best.pth")
            print(f"  --> Best SSIM updated! Saved to weights_{FILENAME}/generator_best.pth")

        # 3. エポックごとの生成画像を保存 (経過確認用)
        with torch.no_grad():
            gen_imgs_fixed = generator(fixed_lr_imgs)
            save_image(gen_imgs_fixed, f"saved_images_{FILENAME}/epoch_{epoch:03d}.png")
        # ---------------------------------------------------------

        # 検証データ上のSSIMの向上に基づいてEarly Stoppingを判定
        early_stopping(avg_val_ssim)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in Validation SSIM.")
            break

    print("\nTraining completed. Evaluating on Test Dataset using the best model...")
    
    # 保存された最高性能の重みをロード
    best_weight_path = f"weights_{FILENAME}/generator_best.pth"
    if os.path.exists(best_weight_path):
        generator.load_state_dict(torch.load(best_weight_path, map_location=DEVICE))
        print(f"Loaded best weights from {best_weight_path}")
    
    generator.eval()
    ssim_metric.reset() # 念のためリセット
    
    with torch.no_grad():
        for test_lr_imgs, test_hr_imgs in test_dataloader:
            test_lr_imgs = test_lr_imgs.to(DEVICE)
            test_hr_imgs = test_hr_imgs.to(DEVICE)
            
            test_gen_imgs = generator(test_lr_imgs)
            ssim_metric.update(test_gen_imgs, test_hr_imgs)
            
    final_test_ssim: float = float(ssim_metric.compute().item())
    ssim_metric.reset() # 使用後のリセット
    
    print(f"==> Final Test SSIM: {final_test_ssim:.4f}")
    
    # 履歴の辞書に単一の数値として追加
    history["final_test_ssim"] = final_test_ssim

    # 損失とSSIMの履歴をJSONとして保存 (既存のコード)
    with open(f"training_history_{FILENAME}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to training_history_{FILENAME}.json")

if __name__ == "__main__":
    train()