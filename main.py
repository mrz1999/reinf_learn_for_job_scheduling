# %%
from utils import *
from dataset_generator import dataset_generator
pa = Parameters()
g = dataset_generator()

pa.num_jobs = g.J
pa.num_machine = g.M
pa.num_operators = g.O
pa.backlog_size = pa.num_jobs

pa.job_slots = 5 

pa.timesteps = 2000000

# %%
g = dataset_generator()
day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup = from_generate_to_dataset(g)
mask_env = initialize_env(pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup)

# %%
# TRAINING

checkpoint = CheckpointCallback(
        save_freq=1000000, 
        save_path=f"./{mask_env.pa.output_dir}/logs/", 
        name_prefix="mask_ppo", 
        save_replay_buffer=True,
        save_vecnormalize=True )

start_time_mask_ppo = time.time()

Maskable_PPO_alg(mask_env, checkpoint)
mask_ppo_time = time.time() - start_time_mask_ppo
print('time: ', mask_ppo_time)

# %%
# INFERENCE

model_mask_ppo = MaskablePPO.load(f'{mask_env.pa.output_dir}/models/PPO_mask_model')
Maskable_PPO_test(mask_env, model_mask_ppo)
print(f'mask ppo train time: {mask_ppo_time}')

# Find the best scheduling between the 10 test
t = float('inf')
for i in range(10):
    dataset = pd.read_csv(f'{mask_env.pa.output_dir}/results/mask_ppo{i+1}.csv')
    my_max = max(dataset['END TIME']) 
    if my_max < t:
        t = my_max
        best_ds = i+1
        
print('the best ds is the {}, with a time of {}'.format(best_ds, t))



