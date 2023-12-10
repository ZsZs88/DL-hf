"""
Trainer class implementation for interacting with the model.
This class is responsible for giving an interface to the model for easier training, sampling, testing, etc.
"""

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
    """Trainer class for interacting with the model."""

    def __init__(self, parallel: Optional[bool] = True) -> None:
        """
        Initialize the trainer class.
        parallel: whether to use parallelization or not
        """

        # device - cuda or cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: image_size, batch_size
        # DEFAULT VALUES IF DATALOADERS HAVE TO BE CREATED
        self.image_size = (64, 64)
        self.batch_size = 256

        # Major model parameters
        self.image_channels = 3
        self.n_channels = 64
        self.channel_multipliers = [1, 2, 2, 4]
        self.is_attention = [False, False, True, True]

        # Iteration steps for forward diffusion
        self.n_steps = 1000

        # Number of samples to generate
        self.n_samples = 16

        # Training parameters
        self.learning_rate = 2e-5
        self.epochs = 100

        # U-Net model for backward diffusion
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        )

        # TODO: WATCH OUT FOR PARALLEL
        # Parallelize the model if possible
        self.parallel = parallel
        if self.parallel:
            self.eps_model = nn.DataParallel(self.eps_model, device_ids=[0, 1, 2])
        self.eps_model.to(self.device)

        # Whole diffusion model
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Optimizer
        self.optimizer = Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image transformation
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
        """
        Set seeds for reproducibility.
        seed: seed to use
        """

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
        # test_dataloader: DataLoader,
    ) -> None:
        """
        Add dataloaders to the trainer.
        train_dataloader: dataloader for training
        val_dataloader: dataloader for validation
        #test_dataloader: dataloader for testing
        """

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # self.test_dataloader = test_dataloader

    def add_fig_path(self, fig_path: str) -> None:
        """
        Add path for saving figures.
        fig_path: path to save figures
        """

        self.fig_path = fig_path

    def add_model_path(self, model_path: str) -> None:
        """
        Add path for saving models.
        model_path: path to save models
        """

        self.model_path = model_path

    def add_logs_path(self, logs_path: str) -> None:
        """
        Add path for saving logs.
        logs_path: path to save logs
        """

        self.logs_path = logs_path

    def add_sample_path(self, sample_path: str) -> None:
        """
        Add path for saving samples.
        sample_path: path to save samples
        """

        self.sample_path = sample_path

    def add_paths(self, _paths: dict) -> None:
        """
        Add paths to the trainer.
        _paths: dictionary containing paths
        """

        # self.add_data(_paths["data"]["train"], _paths["data"]["valid"])
        self.add_fig_path(_paths["figs"])
        self.add_sample_path(_paths["samples"])
        self.add_model_path(_paths["models"])
        self.add_logs_path(_paths["logs"])

    def add_model(self, model_path: str) -> None:
        """
        Add model to the trainer.
        model_path: path to the model
        """

        state_dict = torch.load(model_path)
        if not self.parallel:
            state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
        self.eps_model.load_state_dict(state_dict)

    def modify_imagesize(self, image_size: Tuple[int, int]) -> None:
        """
        Modify image size.
        image_size: new image size
        """

        self.image_size = image_size

    def training_step(self) -> float:
        """Training step for one epoch."""

        running_loss = 0.0

        # Train one epoch in batches
        for batch in self.train_dataloader:
            # We only need the data (labels are not used)
            data, labels = batch[0], batch[1]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        # Calculate average loss
        avg_loss = running_loss / len(self.train_dataloader)
        return avg_loss

    def train(self) -> None:
        """Train the model."""

        # Initialize wandb logging
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

        # Timestamp for logging
        self.time_measurement = time.time()
        print("Training started...")

        # Train for epochs
        for ep in range(self.epochs):
            # Set seeds for reproducibility
            self.set_seeds(self.manual_seed)

            self.time_measurement = time.time()
            epoch = ep + 1
            print(f"Epoch {epoch}")

            # Set model to train mode
            self.eps_model.train()

            # Train one epoch
            avg_loss = self.training_step()

            running_vloss = 0.0

            # Set model to eval mode
            self.eps_model.eval()
            print("\tValidating...")
            running_vloss = 0.0

            # Validate one epoch
            with torch.no_grad():
                for batch in self.val_dataloader:
                    val_data, val_labels = batch[0], batch[1]
                    val_data = val_data.to(self.device)
                    vloss = self.diffusion.loss(val_data)
                    running_vloss += vloss.item()

            # Calculate average validation loss
            avg_vloss = running_vloss / len(self.val_dataloader)

            wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_vloss})

            # Timestamp for logging
            epoch_finished = time.time()
            print(
                f"\tEpoch finished in {epoch_finished - self.time_measurement} seconds"
            )
            print(f"\tTrain Loss: {avg_loss} | Val Loss: {avg_vloss}")

            # Save model and sample images every 5 epochs
            if epoch % 5 == 0:
                torch.save(
                    self.eps_model.state_dict(),
                    os.path.join(self.model_path, f"sample_{epoch}.pth"),
                )
                self.sample(n_samples=self.n_samples, filename=f"sample_{epoch}.png")

    def _sample_x0(self, xt: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Sample x_0"""
        n_samples = xt.shape[0]
        for t in range(n_steps)[::-1]:
            xt = self.diffusion.p_sample(
                xt, xt.new_full((n_samples,), t, dtype=torch.long)
            )
        return xt

    def get_img(self, img: torch.Tensor) -> np.ndarray:
        """
        Get image from tensor
        img: image tensor
        """

        img = img.clip(0, 1)
        img = img.cpu().numpy()
        img = img.transpose(1, 2, 0)
        return img

    def show_image(
        self, img: np.ndarray, title: Optional[str] = None, save: Optional[bool] = False
    ) -> None:
        """
        Show image.
        img: image to show
        title: title of the image
        save: whether to save the image or not
        """

        if title is None:
            title = f"{random.randint(0, 100000)}.png"
        if save:
            plt.imsave(os.path.join(self.sample_path, title), img)
        # TODO: ez kell
        # plt.imshow(img)

    def sample_one_for_GUI(
        self,
        model_path: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """
        Sample one image from the model for GUI.
        model_path: path to the model
        """

        with torch.no_grad():
            # TODO: better solution for sampling only once
            self.set_seeds(np.random.randint(0, 100000))

            # Add model to sample from
            if model_path is not None:
                self.add_model(model_path)
            self.eps_model.eval()

            # Random noise to generate image from
            xt = torch.randn(
                [
                    1,
                    self.image_channels,
                    self.image_size[0],
                    self.image_size[1],
                ],
                device=self.device,
            )

            # Sample x_0
            x0 = self._sample_x0(xt, self.n_steps)

            # Get image
            img = self.get_img(x0[0])
            return img

    def sample_without_figs(
        self,
        n_samples: Optional[int] = 16,
        model_path: Optional[nn.Module] = None,
        batching: Optional[bool] = False,
    ) -> None:
        """
        Sample images from the model without saving figures.
        n_samples: number of samples to generate
        model_path: path to the model
        batching: whether to sample in batches or not
        """

        with torch.no_grad():
            # Add model to sample from
            if model_path is not None:
                self.add_model(model_path)
            self.eps_model.eval()

            # TODO: refactor this
            if batching:
                batches = n_samples // 16
                for j in range(batches):
                    # Random noise to generate image from
                    xt = torch.randn(
                        [
                            16,
                            self.image_channels,
                            self.image_size[0],
                            self.image_size[1],
                        ],
                        device=self.device,
                    )

                    # Sample x_0
                    x0 = self._sample_x0(xt, self.n_steps)

                    # Get images
                    for i in range(16):
                        img = self.get_img(x0[i])
                        self.show_image(img, title=f"{16*j + i}.png", save=True)
            else:
                # Random noise to generate image from
                xt = torch.randn(
                    [
                        n_samples,
                        self.image_channels,
                        self.image_size[0],
                        self.image_size[1],
                    ],
                    device=self.device,
                )

                # Sample x_0
                x0 = self._sample_x0(xt, self.n_steps)

                # Get images
                for i in range(n_samples):
                    img = self.get_img(x0[i])
                    self.show_image(img, title=f"{i}.png", save=True)

    def sample(
        self,
        n_samples: Optional[int] = 16,
        filename: Optional[str] = "sample.png",
        model_path: Optional[nn.Module] = None,
    ) -> None:
        """
        Sample images from the model with saving figures.
        n_samples: number of samples to generate
        filename: filename to save the figure
        model_path: path to the model
        """

        self.sample_without_figs(
            n_samples=n_samples, filename=filename, model_path=model_path
        )
        plt.savefig(os.path.join(self.fig_path, filename))

    def test_FID(self, path1: str, path2: str, batch_size: Optional[int] = 64) -> float:
        """
        Calculate FID score.
        path1: path to the first dataset directory
        path2: path to the second dataset directory
        batch_size: batch size for calculating FID
        """

        fid_value = fid_score.calculate_fid_given_paths(
            # TODO
            [path1, path2],
            batch_size=batch_size,
            device=self.device,
            dims=192,
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
