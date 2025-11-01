import gymnasium as gym
from state_embedding import StateEmbedding, VectorQuantizer, SoftHashEmbedding
from critic import ValueNetwork
from actor import Policy
from base_agent import BaseAgent
import torch
import wandb
from argument_parser import parser
import sweep_configuration
# from embedding_state2 import StateEmbedding, EMA_VectorQuantizer

torch.manual_seed(0)

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 3e-4
EPISODES = 2000
CODEBOOK_DIM = 64

def create_name(configuration:dict):
      l = [f'{k}-{v}' for k,v in configuration.items() if k not in ['output_size', 'wandb_entity', 'wandb_project', 'wandb_sweep', 'sweep_id', 'wandb']]
      return '_'.join(l)

def train():

      run =  wandb.init(entity = config['wandb_entity'], project = config['wandb_project'], config = config)

      sweep_config = wandb.config
      config.update(sweep_config)

      run.name = create_name(wandb.config)

      BATCH_SIZE = config['batch_size']
      embedding_coeff = config['embedding_loss_coeff']

      env = gym.make('LunarLander-v2')
      ACTION_DIM = env.action_space.n
      STATE_DIM = env.observation_space.shape[0]

      embedding_net = StateEmbedding(STATE_DIM, EMBEDDING_DIM)
      embedding_net.to(DEVICE)

      # target_embedding = StateEmbedding(STATE_DIM, EMBEDDING_DIM)
      # target_embedding.to(DEVICE)

      # codebook = VectorQuantizer(EMBEDDING_DIM, 10)
      # codebook.to(DEVICE)

      critic = ValueNetwork(EMBEDDING_DIM)
      critic.to(DEVICE)

      target_critic = ValueNetwork(EMBEDDING_DIM)
      target_critic.to(DEVICE)
      target_critic.load_state_dict(critic.state_dict()) 

      actor = Policy(EMBEDDING_DIM, ACTION_DIM, HIDDEN_DIM)
      actor.to(DEVICE)

      # Remove SAC-specific parameters (alpha, target_entropy)
      agent = BaseAgent(embedding_net, critic, target_critic, actor, env, DEVICE, BATCH_SIZE, GAMMA, LEARNING_RATE, embedding_loss_coeff=embedding_coeff)

      agent.train(EPISODES)

if __name__ == "__main__":

      args = parser.parse_args()

      config = args.__dict__

      if args.wandb_sweep:
            sweep_config = sweep_configuration.sweep_config
            if not args.sweep_id:
                  sweep_id = wandb.sweep(sweep_config, project=config['wandb_project'], entity=config['wandb_entity'])
            else:
                  sweep_id = args.sweep_id

            wandb.agent(sweep_id, function=train, count=20)
            wandb.finish()

            


      