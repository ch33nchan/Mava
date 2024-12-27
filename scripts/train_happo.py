import jax
from mava.algorithms.happo import HAPPO
from mava.configs.happo_config import HAPPOConfig
from mava.environments import create_environment
from mava.trainers import Trainer

def main():
    config = HAPPOConfig()
    environment = create_environment(config)
    algorithm = HAPPO(config)
    trainer = Trainer(config, environment, algorithm)
    trainer.train()

if __name__ == "__main__":
    main()
