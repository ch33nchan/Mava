# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, Union

import dm_env
import numpy as np

from mava.types import OLT

try:
    import pettingzoo  # noqa: F401

    _has_petting_zoo = True
except ModuleNotFoundError:
    _has_petting_zoo = False

if _has_petting_zoo:

    # Prevent circular import issue.
    if TYPE_CHECKING:
        from mava.wrappers.pettingzoo import (
            PettingZooAECEnvWrapper,
            PettingZooParallelEnvWrapper,
        )

PettingZooEnv = Union["PettingZooAECEnvWrapper", "PettingZooParallelEnvWrapper"]


class ConcatAgentIdToObservation:
    """Concat one-hot vector of agent ID to obs.

    We assume the environment has an ordered list
    self.possible_agents.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self._num_agents = len(environment.possible_agents)

    def reset(self) -> dm_env.TimeStep:
        """Reset environment and concat agent ID."""
        timestep, extras = self._environment.reset()
        old_observations = timestep.observation

        new_observations = {}

        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat agent ID"""
        timestep, extras = self._environment.step(actions)

        old_observations = timestep.observation
        new_observations = {}
        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        timestep, extras = self.reset()
        observations = timestep.observation
        return observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class ConcatPrevActionToObservation:
    """Concat one-hot vector of agent prev_action to obs.

    We assume the environment has discreet actions.

    TODO (Claude) support continuous actions.
    """

    def __init__(self, environment: Any):
        self._environment = environment

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment and add zero action."""
        timestep, extras = self._environment.reset()
        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        # TODO double check this, because possible agents could shrink
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat prev actions."""
        timestep, extras = self._environment.step(actions)
        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )
            agent_one_hot_action[actions[agent]] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                terminal=agent_olt.terminal,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            extras,
        )

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        timestep, extras = self.reset()
        observations = timestep.observation
        return observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
