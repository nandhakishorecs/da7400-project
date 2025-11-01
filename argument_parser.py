import argparse

parser = argparse.ArgumentParser(description = "Train an RL agent for LunarLander-v2")

parser.add_argument('-b', '--batch_size', 
                  type = int, default = 256, 
                  help = 'Batch size')

parser.add_argument('-ec', '--embedding_loss_coeff', 
                  type = float, default = 0.4,
                  help = 'Embedding loss coefficient')

parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

parser.add_argument('-we', '--wandb_entity', 
                  type = str, default = 'da24d008-iit-madras' ,
                  help = 'Wandb Entity used to track experiments in the Weights & Biases dashboard')

parser.add_argument('-wp', '--wandb_project', 
                  type = str, default = 'da7400-test' ,
                  help = 'Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument('--wandb_sweep', action='store_true', help='Enable W&B sweep')

parser.add_argument('--sweep_id', type = str, help = "Sweep ID", default = None)