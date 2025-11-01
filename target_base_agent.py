from critic import ValueNetwork
from actor import Policy
from buffer import Buffer
from state_embedding import StateEmbedding, VectorQuantizer
# from embedding_state2 import StateEmbedding, EMA_VectorQuantizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque 
import numpy as np
from tqdm import tqdm

import gymnasium as gym


class BaseAgent:
      def __init__(self, embedding:StateEmbedding, target_embedding:StateEmbedding, critic:ValueNetwork, target_critic:ValueNetwork, actor:Policy, env:gym.Env, device:str, batch_size:int, gamma:float, lr:float=3e-4):

            self.env = env
            
            self.critic = critic
            self.target_critic = target_critic

            self.actor = actor

            self.embedding = embedding
            self.target_embedding = target_embedding

            self.device = device

            self.buffer = Buffer(1000000)

            self.batch_size = batch_size

            self.gamma = gamma

            # self.codebook = codebook

            # Remove SAC-specific parameters
            self.actor_params = list(self.actor.parameters()) + list(self.embedding.parameters()) 
            self.critic_params = list(self.critic.parameters()) + list(self.embedding.parameters()) 
            
            self.optimizer_actor = torch.optim.Adam(self.actor_params, lr=lr)
            self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=lr)
            

      def compute_critic_loss(self, states, next_states, rewards, dones):
            with torch.no_grad():
                  next_state_embeddings = self.target_embedding(next_states)
                  # next_state_embeddings, _ = self.codebook(next_state_embeddings)

                  target_values = self.target_critic(next_state_embeddings).squeeze()
                  y = rewards + self.gamma * (1 - dones) * target_values

            state_embeddings = self.embedding(states)
            # state_embeddings, _ = self.codebook(state_embeddings)

            values = self.critic(state_embeddings).squeeze()
            return F.mse_loss(values, y)

      def compute_actor_loss(self, states, actions, rewards, next_states, dones):
            # Basic Actor-Critic: policy gradient with value function as baseline
            state_embeddings = self.embedding(states)

            # state_embeddings, _ = self.codebook(state_embeddings)

            action_probs = self.actor(state_embeddings)
            
            # Get log probabilities of taken actions
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Calculate advantages (TD error)
            with torch.no_grad():

                  current_values = self.critic(state_embeddings).squeeze()

                  next_state_embeddings = self.target_embedding(next_states)
                  # next_state_embeddings, _ = self.codebook(next_state_embeddings)

                  next_values = self.target_critic(next_state_embeddings).squeeze()
                  target_values = rewards + self.gamma * (1 - dones) * next_values
                  advantages = target_values - current_values
                  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy gradient loss
            actor_loss = -(log_probs * advantages).mean()
            
            return actor_loss

      def soft_update(self, tau=0.005):
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.embedding.parameters(), self.target_embedding.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                     

      def train(self, n_episodes):

            reward_history = deque(maxlen=100)

            pbar = tqdm(range(n_episodes), desc="Initializing...", unit="episode", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

            for ep in pbar:
                  state, _ = self.env.reset()
                  ep_reward = 0
                  done = False

                  step_count = 0

                  while not done:
                        state_tensor = torch.FloatTensor(state).to(self.device)
                        state_embedding = self.embedding(state_tensor)

                        # state_embedding, _ = self.codebook(state_embedding)

                        # Get action from policy
                        probs = self.actor(state_embedding)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()

                        next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                        done = terminated or truncated

                        next_state_tensor = torch.FloatTensor(next_state)
                        
                        # Store in buffer
                        self.buffer.add(state_tensor.detach().cpu(), action.cpu(), reward, next_state_tensor, done)

                        state = next_state
                        ep_reward += reward

                        step_count += 1

                        pbar.set_description(
                              f"Ep {ep+1}/{n_episodes} | "
                              f"Step {step_count} | "
                              f"Avg R: {np.round(np.mean(reward_history), 3) if len(reward_history)>0 else 0} | "
                        )

                        # Train when we have enough samples
                        if len(self.buffer) > self.batch_size:

                              states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                        
                              states = states.to(self.device)
                              actions = actions.to(self.device)
                              rewards = rewards.to(self.device)
                              next_states = next_states.to(self.device)
                              dones = dones.to(self.device)

                              # Update critic
                              critic_loss = self.compute_critic_loss(states, next_states, rewards, dones)

                              self.optimizer_critic.zero_grad()
                              critic_loss.backward()
                              self.optimizer_critic.step()

                              # Update actor
                              if step_count % 2 == 0:
                                    actor_loss = self.compute_actor_loss(states, actions, rewards, next_states, dones)

                                    self.optimizer_actor.zero_grad()
                                    actor_loss.backward()
                                    self.optimizer_actor.step()

                              self.soft_update()
                  
                  # Update reward history
                  reward_history.append(ep_reward)
                  avg_reward = np.mean(reward_history)

                  pbar.set_description(
                        f"Ep {ep+1}/{n_episodes} | "
                        f"Avg R: {avg_reward:.1f} | "
                        f"Steps: {step_count} | "
                        )