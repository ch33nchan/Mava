import jax
import jax.numpy as jnp
from mava.algorithms.base import Algorithm
from mava.utils import update_policy
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardQNet as QNetwork
from mava.networks.distributions import TanhTransformedDistribution
from mava.utils.network_utils import get_action_head

class HAPPO(Algorithm):
    def __init__(self, config):
        super().__init__(config)
        self.clip_param = config.clip_param
        self.num_agents = config.num_agents
        self.actor_networks = [Actor(config.network) for _ in range(self.num_agents)]
        self.critic_network = QNetwork(config.network)
        self.optimizer = optax.adam(config.lr)
        self.opt_state = self.optimizer.init(self.critic_network.params)

    def update(self, data):
        advantages = data['advantages']
        old_log_probs = data['old_log_probs']
        returns = data['returns']
        observations = data['observations']
        actions = data['actions']

        for agent_id in range(self.num_agents):
            agent_advantages = advantages[:, agent_id]
            agent_old_log_probs = old_log_probs[:, agent_id]
            agent_observations = observations[:, agent_id]
            agent_actions = actions[:, agent_id]

            def loss_fn(params):
                log_probs = self.actor_networks[agent_id](params, agent_observations, agent_actions)
                ratio = jnp.exp(log_probs - agent_old_log_probs)
                surr1 = ratio * agent_advantages
                surr2 = jnp.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_advantages
                return -jnp.mean(jnp.minimum(surr1, surr2))

            new_params, opt_state = update_policy(self.actor_networks[agent_id].params, loss_fn, self.optimizer, self.opt_state)
            self.actor_networks[agent_id].params = new_params
            self.opt_state = opt_state

        return self.actor_networks, self.critic_network
