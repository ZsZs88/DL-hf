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
from typing import Optional, Tuple
from pytorch_fid import fid_score

import matplotlib.pyplot as plt
import numpy as np
import random
import wandb


class Trainer:
    def __init__(self, parallel: Optional[bool] = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: image_size, batch_size
        # DEFAULT VALUES IF DATALOADERS HAVE TO BE CREATED
        self.image_size = (64, 64)
        self.batch_size = 256

        self.image_channels = 3
        self.n_channels = 64
        self.channel_multipliers = [1, 2, 2, 4]
        self.is_attention = [False, False, True, True]

        self.n_steps = 1000
        self.n_samples = 16

        self.learning_rate = 2e-5
        self.epochs = 100

        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        )
        # TODO: WATCH OUT FOR PARALLEL
        self.parallel = parallel
        if self.parallel:
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

    def add_model(self, model_path: str):
        state_dict = torch.load(model_path)
        # print(state_dict.keys())
        if not self.parallel:
            state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
        # print(state_dict.keys())
        self.eps_model.load_state_dict(state_dict)

    def modify_imagesize(self, image_size: Tuple[int, int]):
        self.image_size = image_size

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

    def _sample_x0(self, xt: torch.Tensor, n_steps: int):
        n_samples = xt.shape[0]
        for t in range(self.n_steps)[::-1]:
            xt = self.diffusion.p_sample(
                xt, xt.new_full((n_samples,), t, dtype=torch.long)
            )
        return xt

    def get_img(self, img: torch.Tensor) -> np.ndarray:
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        img = img.transpose(1, 2, 0)
        return img

    def show_image(
        self, img: np.ndarray, title: Optional[str] = None, save: Optional[bool] = False
    ):
        if title is None:
            title = f"{random.randint(0, 100000)}.png"
        if save:
            plt.imsave(os.path.join(self.sample_path, title), img)
        plt.imshow(img)

    # Sampling for GUI
    def sample_one_for_GUI(
        self,
        model_path: Optional[nn.Module] = None,
    ) -> np.ndarray:
        with torch.no_grad():
            # self.set_seeds(self.manual_seed)
            if model_path is not None:
                self.eps_model.load_state_dict(torch.load(model_path), strict=False)
            self.eps_model.eval()
            xt = torch.randn(
                [
                    1,
                    self.image_channels,
                    self.image_size[0],
                    self.image_size[1],
                ],
                device=self.device,
            )
            x0 = self._sample_x0(xt, self.n_steps)
            img = self.get_img(x0[0])
            return img

    def sample_without_figs(
        self,
        n_samples: Optional[int] = 16,
        filename: Optional[str] = "sample.png",
        model_path: Optional[nn.Module] = None,
    ):
        with torch.no_grad():
            if model_path is not None:
                self.eps_model.load_state_dict(torch.load(model_path))
            self.eps_model.eval()
            # self.set_seeds(self.manual_seed)
            xt = torch.randn(
                [
                    n_samples,
                    self.image_channels,
                    self.image_size[0],
                    self.image_size[1],
                ],
                device=self.device,
            )
            x0 = self._sample_x0(xt, self.n_steps)
            for i in range(n_samples):
                img = self.get_img(x0[i])
                self.show_image(img, title=f"{i}.png", save=True)

    def sample(
        self,
        n_samples: Optional[int] = 16,
        filename: Optional[str] = "sample.png",
        model_path: Optional[nn.Module] = None,
    ):
        self.sample_without_figs(
            n_samples=n_samples, filename=filename, model_path=model_path
        )
        plt.savefig(os.path.join(self.fig_path, filename))

    def test_FID(self, path1, path2) -> float:
        fid_value = fid_score.calculate_fid_given_paths(
            # TODO
            [path1, path2],
            batch_size=64,
            device=self.device,
            dims=2048,
        )
        return fid_value


# if __name__ == "__main__":
#     wandb.init(project="dl-hf_celeba")
#     trainer = Trainer()
#     trainer.add_paths(paths.other)

#     RANDOM_SEED = 43

#     generator = torch.Generator().manual_seed(RANDOM_SEED)
#     torch.manual_seed(RANDOM_SEED)
#     torch.cuda.manual_seed_all(RANDOM_SEED)
#     torch.backends.cudnn.deterministic = True

#     data = ImageFolder(root=paths.other["root"], transform=trainer.transform)
#     train_data, val_data, test_data = random_split(
#         data, [0.8, 0.1, 0.1], generator=generator
#     )

#     train_dataloader = DataLoader(
#         train_data,
#         batch_size=trainer.batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         drop_last=True,
#     )
#     val_dataloader = DataLoader(
#         val_data,
#         batch_size=trainer.batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         drop_last=True,
#     )
#     test_dataloader = DataLoader(
#         test_data,
#         batch_size=trainer.batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#         drop_last=True,
#     )

#     trainer.add_dataloaders(train_dataloader, val_dataloader, test_dataloader)
#     trainer.train()
