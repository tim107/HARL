import copy

import mate
import numpy as np
from mate.agents import (RandomCameraAgent, GreedyCameraAgent, HeuristicCameraAgent, RandomTargetAgent,
                         GreedyTargetAgent, HeuristicTargetAgent)
import warnings

# Bad practice
warnings.filterwarnings("ignore")


class CameraVGreedyEnv:
    def __init__(self, env_args):
        base_env = mate.make('MultiAgentTracking-v0')
        base_env = mate.RenderCommunication(base_env)
        if env_args["stationary_type"] == "random":
            stationary_agent = RandomTargetAgent
        elif env_args["stationary_type"] == "greedy":
            stationary_agent = GreedyTargetAgent
        elif env_args["stationary_type"] == "heuristic":
            stationary_agent = HeuristicTargetAgent

        env = mate.MultiCamera(base_env, target_agent=stationary_agent())
        self.env = env
        self.n_agents = env.num_teammates  # 4
        self.share_observation_space = self.env.teammate_joint_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.camera_agents = GreedyCameraAgent().spawn(env.num_cameras)

    def step(self, actions):
        camera_joint_action = actions
        results = self.env.step(camera_joint_action)
        obs, rewards, done, info = results
        for item in info:
            item.update({"coverage_rate": self.env.coverage_rate})
        state = copy.deepcopy(obs)
        dones = [done] * self.n_agents
        available_actions = self._get_avail_actions()
        return obs, state, [[rewards]], dones, info, None

    def reset(self):
        obs = self.env.reset()
        state = copy.deepcopy(obs)

        available_actions = self._get_avail_actions()
        return obs, state, None

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        self.env.close()

    def _get_avail_actions(self):
        # I think this should work, because avail actions only seems to be relevant for discrete action spaces
        avail_actions = [None] * self.n_agents
        return avail_actions


class TargetVGreedyEnv:
    def __init__(self, env_args):
        base_env = mate.make('MultiAgentTracking-v0')
        base_env = mate.RenderCommunication(base_env)
        if env_args["stationary_type"] == "random":
            stationary_agent = RandomCameraAgent
        elif env_args["stationary_type"] == "greedy":
            stationary_agent = GreedyCameraAgent
        elif env_args["stationary_type"] == "heuristic":
            stationary_agent = HeuristicCameraAgent
        env = mate.MultiTarget(base_env, camera_agent=stationary_agent())
        self.env = env
        self.n_agents = env.num_teammates  # 8
        self.share_observation_space = self.env.teammate_joint_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.target_agents = GreedyTargetAgent().spawn(env.num_targets)

    def step(self, actions):
        target_joint_action = actions
        results = self.env.step(target_joint_action)
        obs, rewards, done, info = results
        state = copy.deepcopy(obs)
        dones = [done] * self.n_agents
        available_actions = self._get_avail_actions()
        return obs, state, [[rewards]], dones, info, None

    def reset(self):
        obs = self.env.reset()
        state = copy.deepcopy(obs)

        available_actions = self._get_avail_actions()
        return obs, state, None

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        self.env.close()

    def _get_avail_actions(self):
        # I think this should work, because avail actions only seems to be relevant for discrete action spaces
        avail_actions = [None] * self.n_agents
        return avail_actions
