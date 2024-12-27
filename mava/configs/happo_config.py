from mava.configs.system.ppo.ff_ippo import FFIPPOConfig as BaseConfig

class HAPPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.algorithm = 'HAPPO'
        self.clip_param = 0.2
        self.num_agents = 4
        self.lr = 3e-4
        self.network = {
            'pre_torso': {
                '_target_': 'mava.networks.torsos.MLPTorso',
                'layer_sizes': [128, 128],
                'use_layer_norm': False,
                'activation': 'relu'
            },
            'post_torso': {
                '_target_': 'mava.networks.torsos.MLPTorso',
                'layer_sizes': [128, 128],
                'use_layer_norm': False,
                'activation': 'relu'
            }
        }
