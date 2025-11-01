import torch 
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
      def __init__(self, input_dim, n_actions, embedding_dim=256):
            super(Policy, self).__init__()

            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.output_dim = n_actions 

            self.policy_net = nn.Sequential(
                  nn.Linear(self.input_dim, self.output_dim)
            )

            self.apply(self._weights_init)

      def forward(self, state_embedding):
            action_logits = self.policy_net(state_embedding)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs

      def _weights_init(self, m):
            
            if isinstance(m, nn.Linear):
                  nn.init.xavier_uniform_(m.weight)

# Remove SacActor class if you're not using it