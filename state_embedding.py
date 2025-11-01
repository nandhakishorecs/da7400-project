import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F


class StateEmbedding(nn.Module):
      def __init__(self, state_dim, output_dim):
            super(StateEmbedding,self).__init__()

            """
            Takes a raw state and makes a representation

            """

            self.input_dim = state_dim
            self.output_dim = output_dim

            self.phi = nn.Sequential(
                  nn.Linear(in_features= self.input_dim, out_features = self.output_dim),
                  nn.ReLU(),
                  nn.Linear(self.output_dim, out_features = self.output_dim),
                  nn.ReLU()
            )

            # self.mask_logits = nn.Parameter(torch.zeros(output_dim))

            self.apply(self._weights_init)

      def forward(self, state):
            x = self.phi(state)

            return x 

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)
                  # nn.init.constant_(m.bias, 0)

class VectorQuantizer(nn.Module):
      def __init__(self,output_dim, num_codes):
            super(VectorQuantizer, self).__init__()

            self.num_codes = num_codes

            self.codebook = nn.Parameter(torch.randn(self.num_codes, output_dim))



      def forward(self, embedding):

            """
            embedding: (B, D) or (D,)
            Returns:
                  z_out: projection of embedding onto the best code vector (B, D)
                  indices: index of chosen code per batch element (B,)
            """
            if embedding.dim() == 1:
                  embedding = embedding.unsqueeze(0)  # (1, D)

            # Compute norms for each code vector
            codebook = self.codebook  # (K, D)
            code_norms = torch.sum(codebook ** 2, dim=1, keepdim=True)  # (K, 1)

            # Compute dot products (B, K)
            dots = torch.matmul(embedding, codebook.t())

            # Compute projection coefficients α_i = (z·e_i) / ||e_i||^2
            alpha = dots / code_norms.t()  # (B, K)

            # Compute projected vectors for each code: α_i * e_i
            projections = alpha.unsqueeze(-1) * codebook.unsqueeze(0)  # (B, K, D)

            # Reconstruction errors for each code
            diff = embedding.unsqueeze(1) - projections  # (B, K, D)
            errors = torch.sum(diff ** 2, dim=-1)        # (B, K)

            # Choose the code with minimal reconstruction error
            indices = torch.argmin(errors, dim=1)  # (B,)

            # Select the projection corresponding to that code
            z_out = projections[torch.arange(embedding.size(0)), indices]  # (B, D)

            return z_out, indices


      def _weights_init(self, m):
            
            if isinstance(m, nn.Parameter):
                  nn.init.xavier_uniform_(m.weight)


class SoftHashEmbedding(nn.Module):
      def __init__(self, state_dim, embed_dim, num_bins):
            super().__init__()
            self.num_bins = num_bins
            self.embed_dim = embed_dim

            # prototypes / bins
            self.codebook = nn.Parameter(torch.randn(num_bins, embed_dim))
            
            # small MLP that outputs softmax over bins
            self.bin_predictor = nn.Sequential(
                  nn.Linear(state_dim, 128),
                  nn.ReLU(),
                  nn.Linear(128, num_bins)
            )

      def forward(self, state):
            probs = F.softmax(self.bin_predictor(state), dim=-1)  # (batch, k)
            embedding = probs @ self.codebook                     # (batch, embed_dim)
            return embedding
      


            


