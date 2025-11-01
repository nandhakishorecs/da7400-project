import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEmbedding(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(StateEmbedding, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, state):
        return self.phi(state)


class EMA_VectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates.
    Normalizes both inputs and codebook entries.
    Returns quantized vector (projection), indices, and commitment loss.
    """

    def __init__(self, num_codes, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Parameter(torch.randn(num_codes, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", torch.randn(num_codes, embedding_dim))

    def forward(self, z):
        # Normalize input and codebook
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, D)

        z = F.normalize(z, dim=-1)
        codebook = F.normalize(self.embedding, dim=-1)

        # Compute squared distances
        d = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul( z, codebook)
            + torch.sum(codebook ** 2, dim=1)
        )

        # Find nearest codebook vector
        indices = torch.argmin(d, dim=1)
        z_q = codebook[indices]

        # EMA codebook update (during training)
        if self.training:
            one_hot = F.one_hot(indices, self.num_codes).type_as(z)
            cluster_size = one_hot.sum(0)
            embed_sum = torch.matmul(one_hot.t(), z)

            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_codes * self.eps)
                * n
            )
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Projection of z into code vector (quantized projection)
        z_proj = torch.sum(z * z_q, dim=1, keepdim=True) * z_q

      #   # Commitment loss (optional term)
      #   commitment_loss = F.mse_loss(z.detach(), z_q)

        return z_proj, indices
