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
from torch.utils.data import random_split
from typing import Tuple, List, Dict
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------------------------------------------------
# Hyperparameters & Configurations (C言語のdefine風)
# ---------------------------------------------------------
DEFINE = {
    "INPUT_SIZE": 24,
    "OUTPUT_SIZE": 96,
    "BATCH_SIZE": 16,
    "EPOCHS": 200,
    "LEARNING_RATE": 1e-4,
    "PATIENCE": 1000000000,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "RESIDUAL_BLOCKS": 4,
    "GAN_LOSS_WEIGHT": 1e-3,
    "OTHERINFO": "SRGAN",
    # 読み込む事前学習済みの重みパスを指定してください（なければ最初から学習します）
    "PRETRAIN_WEIGHT_PATH": "./weights_PRETRAIN_16_100_0.0001_4_PRETRAIN/generator_best.pth" 
}

print(f"Using device: {DEFINE['DEVICE']}")

suffix = DEFINE["OTHERINFO"] if DEFINE["OTHERINFO"] else "BASIC"
FILENAME = f"GAN_{DEFINE['BATCH_SIZE']}_{DEFINE['EPOCHS']}_{DEFINE['LEARNING_RATE']}_{DEFINE['RESIDUAL_BLOCKS']}_{DEFINE['GAN_LOSS_WEIGHT']}_{suffix}"

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class STL10SRDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: str, split: str = 'train') -> None:
        super().__init__()
        self.dataset = datasets.STL10(root=root, split=split, download=True)
        
        self.transform_hr = transforms.Compose([
            transforms.ToTensor(), 
        ])
        
        self.transform_lr = transforms.Compose([
            transforms.Resize((DEFINE["INPUT_SIZE"], DEFINE["INPUT_SIZE"]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[index]
        lr_img: torch.Tensor = self.transform_lr(img)
        hr_img: torch.Tensor = self.transform_hr(img)
        return lr_img, hr_img

# ---------------------------------------------------------
# Models: Generator (SRResNet)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(DEFINE["RESIDUAL_BLOCKS"])])
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        upsampling: List[nn.Module] = []
        for _ in range(scale_factor // 2):
            upsampling.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ])
        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Sigmoid()
        )

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
            layers: List[nn.Module] = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

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
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

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
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
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
    os.makedirs(f"weights_{FILENAME}", exist_ok=True)
    os.makedirs(f"saved_images_{FILENAME}", exist_ok=True)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEFINE["DEVICE"])
    
    full_train_dataset = STL10SRDataset(root='./data', split='train')
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=DEFINE["BATCH_SIZE"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=DEFINE["BATCH_SIZE"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    test_dataset = STL10SRDataset(root='./data', split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=DEFINE["BATCH_SIZE"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    val_fixed_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    fixed_lr_imgs, fixed_hr_imgs = next(iter(val_fixed_dataloader))
    fixed_lr_imgs = fixed_lr_imgs.to(DEFINE["DEVICE"])
    save_image(fixed_hr_imgs, f"saved_images_{FILENAME}/real_hr_samples.png")

    generator = Generator().to(DEFINE["DEVICE"])
    discriminator = Discriminator().to(DEFINE["DEVICE"])
    feature_extractor = FeatureExtractor().to(DEFINE["DEVICE"])

    # ---------------------------------------------------------
    # 【追加機能】事前学習済み重みのロード
    # ---------------------------------------------------------
    if os.path.exists(DEFINE["PRETRAIN_WEIGHT_PATH"]):
        # pretrainの重みをロード
        generator.load_state_dict(torch.load(DEFINE["PRETRAIN_WEIGHT_PATH"], map_location=DEFINE["DEVICE"]))
        print(f"✅ Successfully loaded pretrained weights from: {DEFINE['PRETRAIN_WEIGHT_PATH']}")
    else:
        print(f"⚠️ Warning: Pretrained weights not found. Starting from scratch.")
    # ---------------------------------------------------------

    optimizer_G = optim.Adam(generator.parameters(), lr=DEFINE["LEARNING_RATE"], betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=DEFINE["LEARNING_RATE"], betas=(0.9, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss().to(DEFINE["DEVICE"])
    criterion_content = nn.MSELoss().to(DEFINE["DEVICE"])

    early_stopping = EarlyStopping(patience=DEFINE["PATIENCE"])

    # 元のコード通り、history辞書を初期化
    history: Dict[str, List[float]] = {
        "g_loss": [],
        "d_loss": [],
        "val_ssim": []
    }

    best_ssim: float = -1.0

    print("Training Started...")
    for epoch in range(1, DEFINE["EPOCHS"] + 1):
        gen_loss_epoch: float = 0.0
        disc_loss_epoch: float = 0.0

        # --- Training Phase ---
        generator.train()
        discriminator.train()

        for lr_imgs, hr_imgs in train_dataloader:
            lr_imgs = lr_imgs.to(DEFINE["DEVICE"])
            hr_imgs = hr_imgs.to(DEFINE["DEVICE"])

            valid_labels: torch.Tensor = torch.ones((DEFINE["BATCH_SIZE"], 1), device=DEFINE["DEVICE"], dtype=torch.float32)
            fake_labels: torch.Tensor = torch.zeros((DEFINE["BATCH_SIZE"], 1), device=DEFINE["DEVICE"], dtype=torch.float32)

            # Train Generator
            optimizer_G.zero_grad()
            gen_imgs: torch.Tensor = generator(lr_imgs)

            pred_fake: torch.Tensor = discriminator(gen_imgs)
            loss_GAN: torch.Tensor = criterion_GAN(pred_fake, valid_labels)

            gen_features: torch.Tensor = feature_extractor(gen_imgs)
            real_features: torch.Tensor = feature_extractor(hr_imgs).detach()
            loss_content: torch.Tensor = criterion_content(gen_features, real_features)

            loss_G: torch.Tensor = loss_content + DEFINE["GAN_LOSS_WEIGHT"] * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
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
        generator.eval()
        with torch.no_grad():
            for val_lr_imgs, val_hr_imgs in val_dataloader:
                val_lr_imgs = val_lr_imgs.to(DEFINE["DEVICE"])
                val_hr_imgs = val_hr_imgs.to(DEFINE["DEVICE"])
                
                val_gen_imgs = generator(val_lr_imgs)
                ssim_metric.update(val_gen_imgs, val_hr_imgs)

        avg_val_ssim: float = float(ssim_metric.compute().item())
        ssim_metric.reset()

        avg_g_loss: float = gen_loss_epoch / len(train_dataloader)
        avg_d_loss: float = disc_loss_epoch / len(train_dataloader)

        # 履歴の追加
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["val_ssim"].append(avg_val_ssim)

        print(f"[Epoch {epoch:03d}/{DEFINE['EPOCHS']}] [G loss: {avg_g_loss:.4f}] [D loss: {avg_d_loss:.4f}] [Val SSIM: {avg_val_ssim:.4f}]")

        # ---------------------------------------------------------
        # モデルと画像の保存処理 (元の機能を完全復元)
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

        early_stopping(avg_val_ssim)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in Validation SSIM.")
            break

    # 元のコード通り、テストデータセットを用いた最終評価
    print("\nTraining completed. Evaluating on Test Dataset using the best model...")
    
    best_weight_path = f"weights_{FILENAME}/generator_best.pth"
    if os.path.exists(best_weight_path):
        generator.load_state_dict(torch.load(best_weight_path, map_location=DEFINE["DEVICE"]))
        print(f"Loaded best weights from {best_weight_path}")
    
    generator.eval()
    ssim_metric.reset()
    
    with torch.no_grad():
        for test_lr_imgs, test_hr_imgs in test_dataloader:
            test_lr_imgs = test_lr_imgs.to(DEFINE["DEVICE"])
            test_hr_imgs = test_hr_imgs.to(DEFINE["DEVICE"])
            
            test_gen_imgs = generator(test_lr_imgs)
            ssim_metric.update(test_gen_imgs, test_hr_imgs)
            
    final_test_ssim: float = float(ssim_metric.compute().item())
    ssim_metric.reset()
    
    print(f"==> Final Test SSIM: {final_test_ssim:.4f}")
    
    history["final_test_ssim"] = final_test_ssim

    # 学習履歴をJSONとして保存
    with open(f"training_history_{FILENAME}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"History saved to training_history_{FILENAME}.json")

if __name__ == "__main__":
    train()