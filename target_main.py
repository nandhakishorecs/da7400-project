import gymnasium as gym
from state_embedding import StateEmbedding, VectorQuantizer
from critic import ValueNetwork
from actor import Policy
from target_base_agent import BaseAgent
import torch
# from embedding_state2 import StateEmbedding, EMA_VectorQuantizer



EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 3e-4
EPISODES = 2000
CODEBOOK_DIM = 64

if __name__ == "__main__":

      env = gym.make('LunarLander-v2')
      ACTION_DIM = env.action_space.n
      STATE_DIM = env.observation_space.shape[0]

      embedding_net = StateEmbedding(STATE_DIM, EMBEDDING_DIM)
      embedding_net.to(DEVICE)

      target_embedding = StateEmbedding(STATE_DIM, EMBEDDING_DIM)
      target_embedding.to(DEVICE)
      target_embedding.load_state_dict(embedding_net.state_dict())

      # codebook = VectorQuantizer(EMBEDDING_DIM, 512)
      # codebook.to(DEVICE)

      critic = ValueNetwork(EMBEDDING_DIM)
      critic.to(DEVICE)

      target_critic = ValueNetwork(EMBEDDING_DIM)
      target_critic.to(DEVICE)
      target_critic.load_state_dict(critic.state_dict()) 

      actor = Policy(EMBEDDING_DIM, ACTION_DIM, HIDDEN_DIM)
      actor.to(DEVICE)

      # Remove SAC-specific parameters (alpha, target_entropy)
      agent = BaseAgent(embedding_net, target_embedding, critic, target_critic, actor, env, DEVICE, BATCH_SIZE, GAMMA, LEARNING_RATE)

      agent.train(EPISODES)