import torch
from unet import UNet
from diffusion import DenoiseDiffusion
from torch.optim import Adam
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import paths
import os
import time
from torch import nn
from typing import Optional

import matplotlib.pyplot as plt
import wandb


class Trainer:
    def __init__(self, parallel: Optional[bool] = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 96
        self.image_channels = 3
        self.n_channels = 64
        self.channel_multipliers = [1, 2, 2, 4]
        self.is_attention = [False, False, True, True]

        self.n_steps = 1000
        self.n_samples = 16

        self.batch_size = 256
        self.learning_rate = 2e-5
        self.epochs = 100

        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        )
        if parallel:
            self.eps_model = nn.DataParallel(self.eps_model, device_ids=[0, 1, 2])
        self.eps_model.to(self.device)

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        self.optimizer = Adam(self.eps_model.parameters(), lr=self.learning_rate)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size),
                torchvision.transforms.ToTensor(),
            ]
        )
        # TODO

        self.manual_seed = 22446688
        self.time_measurement = None
        # self.generator1 = torch.Generator().manual_seed(22446688)

        # self.train_data, self.val_data = torch.utils.data.random_split(ImageFolder('data', transform=self.transform),[0.8,0.2], generator = self.generator1)
        # self.train_dataloader = DataLoader(self.train_data, self.batch_size, shuffle=True, pin_memory=True)
        # self.val_dataloader = DataLoader(self.val_data, self.batch_size, shuffle=True, pin_memory=True)
        # TODO: test data

    def set_seeds(self, seed: Optional[int] = 22446688) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # def add_data(self, train_path: str, val_path: str) -> None:
    #     self.train_data = ImageFolder(train_path, transform=self.transform)
    #     self.val_data = ImageFolder(val_path, transform=self.transform)

    #     self.set_seeds(self.manual_seed)

    #     self.train_dataloader = DataLoader(
    #         self.train_data,
    #         self.batch_size,
    #         shuffle=True,
    #         pin_memory=True,
    #         num_workers=8,
    #     )
    #     self.val_dataloader = DataLoader(
    #         self.val_data, self.batch_size, shuffle=True, pin_memory=True, num_workers=8
    #     )
    #     # TODO: test data

    def add_dataloaders(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def add_fig_path(self, fig_path: str):
        self.fig_path = fig_path

    def add_model_path(self, model_path: str):
        self.model_path = model_path

    def add_logs_path(self, logs_path: str):
        self.logs_path = logs_path

    def add_sample_path(self, sample_path: str):
        self.sample_path = sample_path

    def add_paths(self, _paths: dict):
        # self.add_data(_paths["data"]["train"], _paths["data"]["valid"])
        self.add_fig_path(_paths["figs"])
        self.add_sample_path(_paths["samples"])
        self.add_model_path(_paths["models"])
        self.add_logs_path(_paths["logs"])

    def training_step(self):
        running_loss = 0.0
        self.set_seeds(self.manual_seed)
        for batch in self.train_dataloader:
            data, labels = batch[0], batch[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_dataloader)
        return avg_loss

    def train(self):
        wandb.watch(self.eps_model, log="all")
        wandb.log(
            {
                "image_size": self.image_size,
                "image_channels": self.image_channels,
                "n_channels": self.n_channels,
                "channel_multipliers": self.channel_multipliers,
                "is_attention": self.is_attention,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            }
        )
        self.time_measurement = time.time()
        print("Training started...")
        for ep in range(self.epochs):
            self.time_measurement = time.time()
            epoch = ep + 1
            print(f"Epoch {epoch}")

            self.eps_model.train()
            avg_loss = self.training_step()

            running_vloss = 0.0

            self.eps_model.eval()
            print("\tValidating...")
            running_vloss = 0.0
            with torch.no_grad():
                self.set_seeds(self.manual_seed)
                for batch in self.val_dataloader:
                    val_data, val_labels = batch[0], batch[1]
                    val_data = val_data.to(self.device)
                    vloss = self.diffusion.loss(val_data)
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / len(self.val_dataloader)

            wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_vloss})

            epoch_finished = time.time()
            print(
                f"\tEpoch finished in {epoch_finished - self.time_measurement} seconds"
            )
            print(f"\tTrain Loss: {avg_loss} | Val Loss: {avg_vloss}")

            if epoch % 5 == 0:
                # TODO: model path
                torch.save(
                    self.eps_model.state_dict(),
                    os.path.join(self.model_path, f"sample_{epoch}.pth"),
                )
                self.sample(n_samples=self.n_samples, filename=f"sample_{epoch}.png")

    def show_image(self, img: torch.Tensor, title: Optional[str] = ""):
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        # plt.title(title)
        img = img.transpose(1, 2, 0)
        plt.imsave(os.path.join(self.sample_path, title), img)
        plt.imshow(img.transpose(1, 2, 0))

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        n_samples = xt.shape[0]
        for t in range(self.n_steps)[::-1]:
            xt = self.diffusion.p_sample(
                xt, xt.new_full((n_samples,), t, dtype=torch.long)
            )
        return xt

    def sample(
        self,
        n_samples: Optional[int] = 16,
        filename: Optional[str] = "sample.png",
        model_path: Optional[nn.Module] = None,
    ):
        with torch.no_grad():
            if model_path is not None:
                self.eps_model.load_state_dict(torch.load(model_path))
                self.eps_model.eval()
            self.set_seeds(self.manual_seed)
            xt = torch.randn(
                [n_samples, self.image_channels, self.image_size, self.image_size],
                device=self.device,
            )
            x0 = self._sample_x0(xt, self.n_steps)
            for i in range(n_samples):
                plt.subplot(n_samples // 4, n_samples // 4, i + 1)
                self.show_image(x0[i])
            plt.tight_layout()
            # TODO: fig path
            plt.savefig(os.path.join(self.fig_path, filename))
