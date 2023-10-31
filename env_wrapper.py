# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
An example of integrating new tasks into MARLLib
About ma-gym: https://github.com/koulanurag/ma-gym
doc: https://github.com/koulanurag/ma-gym/wiki

Learn how to transform the environment to be compatible with MARLlib:
please refer to the paper: https://arxiv.org/abs/2210.13708

Install ma-gym before use
"""

# you have to put first the file machines.yaml in venv/lib64/../../marllib/envs/base_env/config

import numpy as np
from gym.spaces import Dict as GymDict
from gym.spaces import Box
# import supersuit as ss
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from env import MachineMultiAgentEnv
from multiagent_env import MachineMultiAgentEnv
from dataset_generator import dataset_generator 


# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    "machines": {
        "description": "machines cooperate for scheudling jobs",
        "team_prefix": ("M"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    }
}

# must inherited from MultiAgentEnv class
class RLlibMAGym(MultiAgentEnv):

    def __init__(self, env_config): #come fare per obs space and action space di dimensione diversa ?
    
        super().__init__()
        g = env_config["dataset"] #
        #g = dataset_generator() # I need the dataset generator for instanziate the environment. # MAYBE THIS SOULDBE PUT INTO env_config in input
        env : MachineMultiAgentEnv = MachineMultiAgentEnv(g) # inizialize env

        self.env_config = env_config
        self.agents = env.agents
        self.num_agents = len(self.agents)

        # env = ss.multiagent_wrappers.pad_action_space_v0(env)
        # env.reset()
        # env = ss.multiagent_wrappers.pad_observations_v0(env)

        env.reset()
        self.action_space = env.action_spaces['M0'] # I can putone action space
        # print("global_state", global_state.shape[0])

        self.observation_space = GymDict({
            "obs": Box(
            low=0.,
            high=1000.0,
            shape=(env.observation_spaces['M0'].shape[0],),
            dtype=np.dtype("float64")), 
            
            "state":  Box(
            low=0.,
            high=1000.0,
            shape=(env.observation_spaces['M0'].shape[0],),
            dtype=np.dtype("float64"))})
        
        self.env = env

    def reset(self): #ok
        original_obs = self.env.reset()
        obs = {}
        global_state = self.env.state()
        for name in self.agents:
            # Convert the observation to a Box space
            obs[name] ={"obs": np.array(original_obs[name]), "state": global_state}
        #obs["state"] = self.env.state()
        return obs

    def step(self, action_dict): #ok
        o, rewards, terminated, _ = self.env.step(action_dict)
        obs = {}
        global_state = self.env.state()
        for key in action_dict.keys():
            if key in o.keys():
                obs[key] = {
                    "obs": np.array(o[key]),
                    "state": global_state
                }

        done = sum(terminated.values()) == self.num_agents
        terminated_all = terminated["__all__"]

        dones = {"__all__": True if (done or terminated_all) else False}
        return obs, rewards, dones, {}


    def render(self, mode=None):
        return self.env.render()
        

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
