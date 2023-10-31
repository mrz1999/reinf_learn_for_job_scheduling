
# %% Imports
from multiagent_env import MachineMultiAgentEnv 
from dataset_generator import dataset_generator
import warnings
warnings.filterwarnings('ignore')

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from env_wrapper  import RLlibMAGym

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' to suppress SettingWithCopyWarning

# Test environment
g = dataset_generator()
env = MachineMultiAgentEnv(g)
env.reset()

# %% [markdown]
# ## ALGORITMO E REGISTRAZIONE DELL'ENVIRONMENT CON MARLLIB

# %%
ENV_REGISTRY["machines"] = RLlibMAGym

env = marl.make_env(environment_name="machines", 
                    map_name="machines"
                    )


# %%

mappo = marl.algos.mappo(hyperparam_source="common") 

# {"core_arch": "mlp", "encode_layer": "128-256"}
# {"core_arch": "lstm"}
# {"core_arch": "gru"}

model = marl.build_model(env, mappo, {"core_arch": "lstm"})


# %%
mappo.fit(env, model, 
         stop={'episode_reward_mean': 1000, 'timesteps_total': 500000}, # termination criteria
         local_mode=True,  # if debug (if local_mode = True it didn't consider the gpus and cpus configurations)
         num_gpus=1, # gpu accelerator
         num_workers=10, # cpu accelerator
         share_policy='all', # 'group' or 'all' or 'individual'
         checkpoint_freq=10, checkpoint_end=True, # checkpoint for saving the model
         )
