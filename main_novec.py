# %% IMPORT
import time
from Environment2_drastic import Environment
import pandas as pd
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import warnings
warnings.filterwarnings("ignore")

import os
from numpy.random import default_rng
import numpy as np
import itertools

# %% DEFINE ALL THE PARAMETERS
class Parameters():

    def __init__(self):

        # DATASET  PARAMETERS
        self.num_jobs = 8
        self.num_machine = 4
        self.num_operators = 50
        self.job_slots = 5 # maximum allowed number of job in the queue
        self.backlog_size = 50 # size of the backlog that is the maximum number of job I can put in wait when the queue is full
        self.max_time = 10 # max execution time of a job
        self.max_op = 4 # maximum number of operator that a job can require
        self.max_setup_time = 10 # maximum setup time between two jobs on the same machine

        # ALG PARAMETERS
        self.lr = 0.00045
        self.gamma = 0.95
        self.seed = 3
        self.verb = 1 # verbose in the rl algortihm

        # LEARN FUNCTION PARAMETERS
        self.timesteps = 1000000 #  total number of samples (env steps) to train on
        self.log = 1000 # number of timestep after which the alg print the new info

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

    # j1_j2_m_setup dataset
    
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

    day_job_jobcode.to_csv('datasets_novec/day_job_jobcode')
    jobcode_machine_op_t.to_csv('datasets_novec/jobcode_machine_op_t')
    turn_abilities.to_csv('datasets_novec/turn_abilities')
    j1_j2_m_setup.to_csv('datasets_novec/j1_j2_m_setup')

    env = Environment(pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup)

    return env

# %% LR SCHEDULE
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
def DQN_alg(env, checkpoint):
    
    checkpoint_callback = checkpoint

    # CREATE AND TRAIN THE MODEL
    model_dqn = DQN("MlpPolicy", 
                    env, 
                    learning_rate=linear_schedule(0.001), 
                    gamma = env.pa.gamma, 
                    exploration_fraction = 0.15, 
                    verbose=env.pa.verb, 
                    seed = env.pa.seed, 
                    tensorboard_log='./tensorboard')
    
    model_dqn.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log, 
                    callback = checkpoint_callback)
    
    # SAVE THE MODEL
    model_dqn.save("models_novec/DQN_model")

# TEST THE TRAINED MODEL
def DQN_test(env, model_dqn):
    i = 0
    obs = env.reset()

    while i<10:
        action, _ = model_dqn.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)
        
        obs, reward, terminated, info = env.step(int(action))
        # print(obs.reshape(20,5))
        
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = 'results_novec/dqn'+str(i)+'.csv'
            dataset.to_csv(name)
            obs = env.reset()     

# %% A2C
def A2C_alg(env, checkpoint):

    checkpoint_callback = checkpoint
    
    # CREATE AND TRAIN THE MODEL
    model_a2c = A2C("MlpPolicy", 
                    env, 
                    learning_rate=linear_schedule(0.001), 
                    gamma = env.pa.gamma, 
                    verbose=env.pa.verb, 
                    seed=env.pa.seed, 
                    tensorboard_log='./tensorboard')
    
    model_a2c.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log, 
                    callback = checkpoint_callback)
    
    # SAVE THE MODEL
    model_a2c.save("models_novec/A2C_model")

# TEST THE TRAINED MODEL
def A2C_test(env, model_a2c):
    i = 0
    obs = env.reset()
    
    while i<10:
        action, _ = model_a2c.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)
        
        obs, reward, terminated, info = env.step(int(action))
        # print(obs.reshape(20,5))
        
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = 'results_novec/a2c'+str(i)+'.csv'
            dataset.to_csv(name)
            obs = env.reset()
    
# %% PPO
def PPO_alg(env, checkpoint):

    checkpoint_callback = checkpoint

    # CREATE AND TRAIN THE MODEL
    model_ppo = PPO("MlpPolicy", 
                    env, 
                    learning_rate=linear_schedule(0.001),
                    n_steps = 1024, 
                    gamma = env.pa.gamma,
                    gae_lambda=0.98,
                    n_epochs = 4,
                    ent_coef=0.1, 
                    verbose=env.pa.verb, 
                    seed = env.pa.seed, 
                    tensorboard_log='./tensorboard/')
    
    model_ppo.learn(total_timesteps=env.pa.timesteps, 
                    log_interval=env.pa.log//100, 
                    callback = checkpoint_callback)
    
    # SAVE THE MODEL
    model_ppo.save("models_novec/PPO_model")

# TEST THE TRAINED MODEL
def PPO_test(env, model_ppo):
    i = 0
    obs = env.reset()

    while i<10:
        action, _states = model_ppo.predict(obs, deterministic=True)
        
        print('episode: ', i)
        print(action)
        
        obs, reward, terminated, info = env.step(int(action))
        # print(obs.reshape(20,5))

        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = 'results_novec/ppo'+str(i)+'.csv'
            dataset.to_csv(name)
            obs = env.reset()

# %% MAIN

def main():
    if not os.path.exists('datasets_novec'):
        os.makedirs('datasets_novec')  

    if not os.path.exists('models_novec'):
        os.makedirs('models_novec')  

    if not os.path.exists('logs_novec'):
        os.makedirs('logs_novec')   

    if not os.path.exists('results_novec'):
        os.makedirs('results_novec')  

    if not os.path.exists('tensorboard'):
        os.makedirs('tensorboard')
    
    env = inizialize_env()

    checkpoint = CheckpointCallback(
    save_freq=1000000,
    save_path="./logs_novec",
    name_prefix="dqn_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    
    # DQN
    initial_time = time.time()

    DQN_alg(env, checkpoint)
    dqn_time = time.time() - initial_time

    model_dqn = DQN.load('models_novec/DQN_model')
    DQN_test(env, model_dqn)
    print(f'dqn train time: {dqn_time}')

    # A2C
    start_time_a2c = time.time()

    A2C_alg(env, checkpoint)
    a2c_time = time.time() - start_time_a2c
    
    model_a2c = A2C.load('models_novec/A2C_model')
    A2C_test(env, model_a2c)
    print(f'a2c train time: {a2c_time}')

    # PPO
    start_time_ppo = time.time()
    
    PPO_alg(env, checkpoint)
    ppo_time = time.time() - start_time_ppo
    
    model_ppo = PPO.load('models_novec/PPO_model')
    PPO_test(env, model_ppo)
    print(f'a2c train time: {ppo_time}')

    print(f'dqn time: {dqn_time}, a2c time: {a2c_time}, ppo time: {ppo_time}')

main()
# %%
