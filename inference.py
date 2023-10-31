# %%

from marllib import marl
from dataset_generator import dataset_generator
from marllib.envs.base_env import ENV_REGISTRY
from env_wrapper  import RLlibMAGym
# %%
ENV_REGISTRY["machines"] = RLlibMAGym
PARAMS_PATH = '/home/VICOMTECH/uotamendi/Projects/multiagent_rl/experiments/multi_agents/mappo_mlp_machines/MAPPOTrainer_machines_machines_38569_00000_0_2023-10-20_00-24-08/params.json'
MODEL_PATH = '/home/VICOMTECH/uotamendi/Projects/multiagent_rl/experiments/multi_agents/mappo_mlp_machines/MAPPOTrainer_machines_machines_38569_00000_0_2023-10-20_00-24-08/checkpoint_000050/checkpoint-50'

dataset = dataset_generator()
# %%
# prepare the environment
env = marl.make_env(environment_name="machines",
                    map_name="machines",
                    dataset=dataset
                    )

# %%

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")

# %%

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo,{"core_arch": "mlp", "encode_layer": "128-256"})

# %%

result = mappo.render(env, model,
             restore_path={'params_path': PARAMS_PATH,  # experiment configuration
                        'model_path': MODEL_PATH, # checkpoint path,
                        'soft_horizon': True,  # soft horizon avoid env reset
                        'horizon': 0,  # max steps per episode,
                        'num_episodes': 0,  # number of episodes to run,
                        'timesteps': 0,  # number of timesteps to run,
                        'render': False},  # render
            stop={'timesteps_total': 0},
            local_mode=True,
            num_workers=10,
            share_policy="all",
            checkpoint_end=False)

# %%
env[0].env.render()
# %%
#    job  machine  start_time  end_time  setup_time operators
# 0    4        2           0      63.0         0.0       [3]
# 1    5        3           0      58.0         0.0       [4]
# 2    6        0           1      62.0         0.0       [6]
# 3    2        3          61     134.0         5.0      [11]
# 4    7        0          63     132.0         6.0       [6]
# 5    1        2          65     116.0         0.0       [2]
# 6    8        2         125     169.0         4.0       [2]
# 7    3        0         145     211.0         5.0       [7]
