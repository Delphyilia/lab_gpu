import os
from torchvision.utils import save_image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import random_split
from typing import Tuple, List, Dict
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------------------------------------------------
# Hyperparameters & Configurations
# ---------------------------------------------------------
DEFINE = {
    "INPUT_SIZE": 24,
    "OUTPUT_SIZE": 96,
    "BATCH_SIZE": 16,
    "EPOCHS": 100, # 事前学習は100エポック程度で十分な形になります
    "LEARNING_RATE": 1e-4,
    "PATIENCE": 1000000000,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "RESIDUAL_BLOCKS": 4,
    "OTHERINFO": "PRETRAIN"
}

print(f"Using device: {DEFINE['DEVICE']}")

suffix = DEFINE["OTHERINFO"] if DEFINE["OTHERINFO"] else "BASIC"
FILENAME = f"PRETRAIN_{DEFINE['BATCH_SIZE']}_{DEFINE['EPOCHS']}_{DEFINE['LEARNING_RATE']}_{DEFINE['RESIDUAL_BLOCKS']}_{suffix}"

# ---------------------------------------------------------
# Dataset (SRGANと同じ)
# ---------------------------------------------------------
class STL10SRDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: str, split: str = 'train') -> None:
        super().__init__()
        self.dataset = datasets.STL10(root=root, split=split, download=True)
        self.transform_hr = transforms.Compose([transforms.ToTensor()])
        self.transform_lr = transforms.Compose([
            transforms.Resize((DEFINE["INPUT_SIZE"], DEFINE["INPUT_SIZE"]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[index]
        return self.transform_lr(img), self.transform_hr(img)

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
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class Generator(nn.Module):
    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(DEFINE["RESIDUAL_BLOCKS"])])
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))
        
        upsampling: List[nn.Module] = []
        for _ in range(scale_factor // 2):
            upsampling.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ])
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1
        out = self.upsampling(out)
        out = self.conv3(out)
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
# Main Training Loop (Pretraining)
# ---------------------------------------------------------
def train() -> None:
    os.makedirs(f"weights_{FILENAME}", exist_ok=True)
    os.makedirs(f"saved_images_{FILENAME}", exist_ok=True)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEFINE["DEVICE"])
    
    full_train_dataset = STL10SRDataset(root='./data', split='train')
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=DEFINE["BATCH_SIZE"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=DEFINE["BATCH_SIZE"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    val_fixed_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    fixed_lr_imgs, fixed_hr_imgs = next(iter(val_fixed_dataloader))
    fixed_lr_imgs = fixed_lr_imgs.to(DEFINE["DEVICE"])
    save_image(fixed_hr_imgs, f"saved_images_{FILENAME}/real_hr_samples.png")

    # Generatorのみを定義
    generator = Generator().to(DEFINE["DEVICE"])
    optimizer_G = optim.Adam(generator.parameters(), lr=DEFINE["LEARNING_RATE"], betas=(0.9, 0.999))
    
    # Pretraining用はMSELossのみ
    criterion_content = nn.MSELoss().to(DEFINE["DEVICE"])
    early_stopping = EarlyStopping(patience=DEFINE["PATIENCE"])

    best_ssim: float = -1.0

    print("Pre-Training Started (MSE Loss Only)...")
    for epoch in range(1, DEFINE["EPOCHS"] + 1):
        gen_loss_epoch: float = 0.0

        # --- Training Phase ---
        generator.train()
        for lr_imgs, hr_imgs in train_dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(DEFINE["DEVICE"]), hr_imgs.to(DEFINE["DEVICE"])

            optimizer_G.zero_grad()
            gen_imgs = generator(lr_imgs)
            
            # ピクセルごとのMSEを計算
            loss_G = criterion_content(gen_imgs, hr_imgs)
            loss_G.backward()
            optimizer_G.step()

            gen_loss_epoch += float(loss_G.item())

        # --- Validation Phase ---
        generator.eval()
        with torch.no_grad():
            for val_lr_imgs, val_hr_imgs in val_dataloader:
                val_lr_imgs, val_hr_imgs = val_lr_imgs.to(DEFINE["DEVICE"]), val_hr_imgs.to(DEFINE["DEVICE"])
                val_gen_imgs = generator(val_lr_imgs)
                ssim_metric.update(val_gen_imgs, val_hr_imgs)

        avg_val_ssim: float = float(ssim_metric.compute().item())
        ssim_metric.reset()
        avg_g_loss: float = gen_loss_epoch / len(train_dataloader)

        print(f"[Epoch {epoch:03d}/{DEFINE['EPOCHS']}] [MSE loss: {avg_g_loss:.4f}] [Val SSIM: {avg_val_ssim:.4f}]")

        # モデル保存
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save(generator.state_dict(), f"weights_{FILENAME}/generator_best.pth")
            print(f"  --> Best SSIM updated! Saved to weights_{FILENAME}/generator_best.pth")

        # 画像保存
        with torch.no_grad():
            gen_imgs_fixed = generator(fixed_lr_imgs)
            save_image(gen_imgs_fixed, f"saved_images_{FILENAME}/epoch_{epoch:03d}.png")

        early_stopping(avg_val_ssim)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()