""" Diffusion Model """

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather


class DenoiseDiffusion:
    """
    Denoise Diffusion
    """

    def __init__(
        self, eps_model: nn.Module, n_steps: int, device: torch.device
    ) -> None:
        """
        Initialize Denoise Diffusion
        eps_model: U-Net for backward diffusion
        n_steps: iteration steps for forward diffusion
        device: device to use
        """

        super().__init__()
        self.eps_model = eps_model

        # Create beta linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # Calculate alpha
        self.alpha = 1.0 - self.beta

        # Calculate alpha bar
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Number of steps
        self.n_steps = n_steps

        # Calculate sigma^2
        self.sigma2 = self.beta

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(x_t|x_0) distribution
        """

        # gather alpha bar and calculate mean and variance
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from q(x_t|x_0)
        """

        # Calculate epsilon
        if eps is None:
            eps = torch.randn_like(x0)

        # get q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)

        # Sample from q(x_t|x_0)
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from p(x_{t-1}|x_t)
        """

        # epsilon_theta
        eps_theta = self.eps_model(xt, t)

        # gather alpha_bar
        alpha_bar = gather(self.alpha_bar, t)

        # gather alpha
        alpha = gather(self.alpha, t)

        # Calculate eps_coef
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5

        # Calcualte mean and variance
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)

        # calculate epsilon
        eps = torch.randn(xt.shape, device=xt.device)

        # Sample
        return mean + (var**0.5) * eps

    def loss(
        self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simplified Loss function
        """

        # Get batch size
        batch_size = x0.shape[0]

        # Get random step for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        # get noise
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample x_t for q(x_t|x_0)
        xt = self.q_sample(x0, t, eps=noise)

        # Get epsilon_theta
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
