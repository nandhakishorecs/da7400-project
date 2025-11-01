sweep_config = {
            "method": "bayes",  # Use Bayesian optimization for hyperparameter tuning
            "metric": {"name": "avg_reward_100ep", "goal": "maximize"},
            "parameters": {
                "batch_size":{"values": [128, 256]},
                "embedding_loss_coeff":{"values":[0, 0.2, 0.4, 0.8]}
            }
        }