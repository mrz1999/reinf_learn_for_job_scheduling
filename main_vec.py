# %% IMPORT
import time

# Choose your Environment. In the repository are implemented three type with little difference between one and another
# from Environment1_stop import Environment
# from Environment2_drastic import Environment
from Environment3_default_max import Environment

import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from numpy.random import default_rng
import itertools

# %% DEFINE ALL THE PARAMETERS
class Parameters():

    def __init__(self):

        # DATASET  PARAMETERS
        self.num_jobs = 20
        self.num_machine = 6
        self.num_operators = 30
        self.job_slots = 5 # maximum allowed number of job in the queue
        self.backlog_size = self.num_jobs # size of the backlog that is the maximum number of job I can put in wait when the queue is full
        self.max_time = 20 # max execution time of a job
        self.max_op = 4 # maximum number of operator that a job can require
        self.max_setup_time = 10 # maximum setup time between two jobs on the same machine

        # ALG PARAMETERS
        self.lr = 0.00045
        self.gamma = 0.95
        self.seed = 3
        self.verb = 1 # verbose in the rl algortihm

        # LEARN FUNCTION PARAMETERS
        self.timesteps = 5000000 #  total number of samples (env steps) to train on
        self.log = 100 # number of timestep after which the alg print the new info

        self.output_dir = 'output_vec'
        self.num_env = 4

# %% INIZIALIZE DATASET AND ENVIRONMENT FUNCTION
def inizialize_env():

    pa = Parameters()
    num_jobs = pa.num_jobs
    num_machine = pa.num_machine
    num_operators = pa.num_operators
    max_time = pa.max_time
    max_num_op = pa.max_op
    max_setup_time = pa.max_setup_time

    np.random.seed(seed=pa.seed)

    # day_job_code dataset

    day = np.ones(num_jobs, dtype = int)
    job = np.arange(num_jobs)+1 #NON VOGLIO CHE NESSUN JOB ABBIA CODICE 0 PERCHÈ HO MESSO 0 COME DEFAULT NELLA LISTA DEI RUNNING JOB E INDICA CHE NON STA RUNNANOD NESSUN JOB 
    
    rng = default_rng()
    job_code = rng.choice(300, size=num_jobs, replace=False)

    day_job_jobcode = pd.DataFrame(list(zip(day, job, job_code)), columns = ['Day', 'Job', 'Code'])
    # print(day_job_jobcode)

    code = []
    machine = []

    # code_machine_op_t dataset

    for one_code in job_code:
        code.append(one_code)
        code.append(one_code) #I put two machine for each job randomly, so I will need two times each jobcode
        machines = np.random.choice(np.arange(num_machine), 2, replace = False)
        machine.extend(list(machines))

    operator = np.random.randint(1,max_num_op, len(machine))
    time = np.random.randint(1,max_time,len(machine))
    jobcode_machine_op_t = pd.DataFrame(list(zip(code, machine, operator, time)), columns = ['Code', 'Machine', 'Operator','Time'])
    # print(jobcode_machine_op_t)

    # turn_abilities dataset

    operator = np.arange(num_operators)
    turn = np.random.choice(['TA','TB','TC'], size = num_operators )
    machine = np.random.choice(machine, size = num_operators)
    turn_abilities = pd.DataFrame(list(zip(operator, turn, machine)), columns = ['Operator', 'Turn', 'Machine'])
    # print(turn_abilities)

    combinations = {}
    for machine in np.arange(num_machine):
        job_on_mach = list(jobcode_machine_op_t[jobcode_machine_op_t['Machine'] == machine]['Code'])
        combinations[machine] = list(itertools.permutations(job_on_mach, 2))
        combinations[machine].extend([(job,job) for job in job_on_mach])
    
    job_from = []
    job_to = []
    mach_col = []
    for machine in np.arange(num_machine):
        just_one_mach = combinations[machine]

        job_from.append(list(zip(*just_one_mach))[0])
        job_to.append(list(zip(*just_one_mach))[1])
        mach_col.append([machine]*len(combinations[machine]))

    # I have a list of tuple but I want a single list of numbers for columns
    job_from = [item for t in job_from for item in t]   
    job_to = [item for t in job_to for item in t]
    mach_col = [item for t in mach_col for item in t]

    setuptime = np.random.randint(1,max_setup_time,len(job_from))
    j1_j2_m_setup = pd.DataFrame(list(zip(job_from,job_to,mach_col,setuptime)), columns = ['Job_from', 'Job_to', 'Machine', 'SetupTime'])

    for row in j1_j2_m_setup.itertuples():
        if row.Job_from == row.Job_to:
            j1_j2_m_setup['SetupTime'].loc[row.Index] = 0
    # print(j1_j2_m_setup)

    # SAVE THE DATASET
    day_job_jobcode.to_csv(f'{pa.output_dir}/datasets/day_job_jobcode')
    jobcode_machine_op_t.to_csv(f'{pa.output_dir}/datasets/jobcode_machine_op_t')
    turn_abilities.to_csv(f'{pa.output_dir}/datasets/turn_abilities')
    j1_j2_m_setup.to_csv(f'{pa.output_dir}/datasets/j1_j2_m_setup')

    env = Environment(pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup)

    return env
# %% 
def make_env():
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = inizialize_env()
        env.reset()
        return env
    return _init

def linear_schedule(initial_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            # progress_remaining = 1.0 - (num_timesteps / total_timesteps)
            return progress_remaining * initial_value

        return func

# %% DQN
def DQN_alg(vecenv, env, checkpoint):
    
    checkpoint_callback = checkpoint

    # CREATE AND TRAIN THE MODEL
    model_dqn = DQN("MlpPolicy", 
                    vecenv, 
                    learning_rate=linear_schedule(0.001),
                    gamma = env.pa.gamma, 
                    exploration_fraction = 0.15, 
                    verbose=env.pa.verb, 
                    seed = env.pa.seed, 
                    tensorboard_log=f'./{env.pa.output_dir}/tensorboard/')
    
    model_dqn.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log, 
                    callback= checkpoint_callback)
    
    # SAVE THE MODEL
    model_dqn.save(f"{env.pa.output_dir}/models/DQN_model")

# TEST THE TRAINED MODEL
def DQN_test(env, model_dqn):
    i = 0
    obs, _ = env.reset()
    while i<10:
        action, _states = model_dqn.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)
        
        obs, reward, terminated, _, info = env.step(int(action))
        # print(obs.reshape(20,5))
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = f'{env.pa.output_dir}/results/dqn{i}.csv'
            dataset.to_csv(name)
            obs, _ = env.reset()     

# %% A2C
def A2C_alg(vecenv, env, checkpoint):

    checkpoint_callback = checkpoint
    
    # CREATE AND TRAIN THE MODEL
    model_a2c = A2C("MlpPolicy", 
                    vecenv, 
                    learning_rate=linear_schedule(0.001),
                    gamma = env.pa.gamma, 
                    verbose=env.pa.verb, 
                    seed=env.pa.seed, 
                    tensorboard_log=f'./{env.pa.output_dir}/tensorboard/')
    
    model_a2c.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log, 
                    callback = checkpoint_callback)
    
    # SAVE THE MODEL
    model_a2c.save(f"{env.pa.output_dir}/models/A2C_model")

# TEST THE TRAINED MODEL
def A2C_test(env, model_a2c):

    i = 0
    obs, _ = env.reset()
    while i<10:
        action, _states = model_a2c.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)
        
        obs, reward, terminated, _, info = env.step(int(action))
        # print(obs.reshape(20,5))
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = f'{env.pa.output_dir}/results/a2c'+str(i)+'.csv'
            dataset.to_csv(name)
            obs, _ = env.reset()
    
# %% PPO
def PPO_alg(vecenv, env, checkpoint):

    checkpoint_callback = checkpoint
    
    # CREATE AND TRAIN THE MODEL
    model_ppo = PPO("MlpPolicy", 
                    vecenv, 
                    learning_rate=linear_schedule(0.001), 
                    n_steps=1024,
                    gamma = 0.9,
                    gae_lambda=0.98,
                    n_epochs=4,
                    ent_coef=0.1,
                    verbose=env.pa.verb,
                    seed = env.pa.seed, 
                    tensorboard_log=f'./{env.pa.output_dir}/tensorboard/')    
    
    model_ppo.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log, 
                    callback = checkpoint_callback)
    
    # SAVE THE MODEL
    model_ppo.save(f"{env.pa.output_dir}/models/PPO_model")

# TEST THE TRAINED MODEL
def PPO_test(env, model_ppo):

    i = 0
    obs, _ = env.reset()

    while i<10:
        action, _states = model_ppo.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)

        obs, reward, terminated, _, info = env.step(int(action))
        # print(obs.reshape(20,5))
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = f'{env.pa.output_dir}/results/ppo{i}.csv'
            dataset.to_csv(name)
            obs, _ = env.reset()

# %% MAIN

def main():

    pa = Parameters()
    output_dir = pa.output_dir

    if not os.path.exists(f'{output_dir}/datasets'):
        os.makedirs(f'{output_dir}/datasets')  

    if not os.path.exists(f'{output_dir}/models'):
        os.makedirs(f'{output_dir}/models')  

    if not os.path.exists(f'{output_dir}/logs'):
        os.makedirs(f'{output_dir}/logs')   

    if not os.path.exists(f'{output_dir}/results'):
        os.makedirs(f'{output_dir}/results')  

    if not os.path.exists(f'{output_dir}/tensorboard'):
        os.makedirs(f'{output_dir}/tensorboard')

    vec_env = SubprocVecEnv([make_env() for i in range(pa.num_env)])
    env = inizialize_env()

    checkpoint = CheckpointCallback(
    save_freq=100000,
    save_path=f'./{output_dir}/logs',
    name_prefix="dqn_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )

    # DQN
    initial_time = time.time()

    DQN_alg(vec_env, env, checkpoint)
    dqn_time = time.time() - initial_time

    model_dqn = DQN.load(f'{output_dir}/models/DQN_model')
    DQN_test(env, model_dqn)
    print(f'dqn train time: {dqn_time}')

    checkpoint = CheckpointCallback(
    save_freq=100000,
    save_path=f'./{output_dir}/logs',
    name_prefix="a2c_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )

    # A2C
    start_time_a2c = time.time()

    A2C_alg(vec_env, env, checkpoint)
    a2c_time = time.time() - start_time_a2c
    
    model_a2c = A2C.load(f'{output_dir}/models/A2C_model')
    A2C_test(env, model_a2c)
    print(f'a2c train time: {a2c_time}')


    checkpoint = CheckpointCallback(
    save_freq=100000,
    save_path=f'./{output_dir}/logs',
    name_prefix="ppo_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )

    # PPO
    start_time_ppo = time.time()
    
    PPO_alg(vec_env, env, checkpoint)
    ppo_time = time.time() - start_time_ppo
    
    model_ppo = PPO.load(f'{output_dir}/models/PPO_model')
    PPO_test(env, model_ppo)
    print(f'a2c train time: {ppo_time}')

    print(f'dqn time: {dqn_time}, a2c time: {a2c_time}, ppo time: {ppo_time}')

import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
# %%
