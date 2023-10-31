# UTILS FOR THE ACTION MASKING ENVIRONMENT

import pandas as pd
import os
import numpy as np
from numpy.random import default_rng
import itertools # for generating all the possible combinations of two jobs on the same machine for generating the setup time
import time

import gymnasium as gym

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy #(MlpPolicy)
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from Environment4_action_mask import Environment as MaskEnv

import warnings
warnings.filterwarnings("ignore")

class Parameters():

    def __init__(self):

        # DATASET  PARAMETERS
        self.num_jobs = 15
        self.num_machine = 5
        self.num_operators = 30
        self.job_slots = 5 # maximum allowed number of job in the queue
        self.backlog_size = self.num_jobs # size of the backlog that is the maximum number of job I can put in wait when the queue is full

        #this three parameter are not needed if we are generating the dataset with the generator

        # self.max_time = 10 # max execution time of a job
        # self.max_op = 4 # maximum number of operator that a job can require
        # self.max_setup_time = 5 # maximum setup time between two jobs on the same machine

        # ALG PARAMETERS
        self.lr = 0.00045
        self.gamma = 0.95
        self.batch_size = 64
        self.seed = 3
        self.verb = 1 # verbose in the rl algortihm

        # LEARN FUNCTION PARAMETERS
        self.timesteps = 1000000 #  total number of samples (env steps) to train on
        self.log = 20 # number of timestep after which the alg print the new info

        # ENVIRONMENT PARAMETERS
        self.output_dir = 'output'
        self.num_env = 4 # number of parallel environments (with action mask I cannot use parellel env so it is useless)

def get_dataset(pa, create_dataset = True, dataset_path = None):
    if create_dataset == False:
        assert(dataset_path != None)

        # upload the dataset from the path given as input
        day_job_jobcode = pd.read_csv(f'{dataset_path}/day_job_jobcode')
        jobcode_machine_op_t = pd.read_csv(f'{dataset_path}/jobcode_machine_op_t')
        turn_abilities = pd.read_csv(f'{dataset_path}/turn_abilities')
        j1_j2_m_setup = pd.read_csv(f'{dataset_path}/j1_j2_m_setup')

    else:

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
            code.append(one_code) 
            code.append(one_code)  #I put three machine for each job randomly, so I will need two times each jobcode
            machines = np.random.choice(np.arange(num_machine), 3, replace = False)
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

        if not os.path.exists(f'{pa.output_dir}/datasets'):
            os.makedirs(f'{pa.output_dir}/datasets')  

        day_job_jobcode.to_csv(f'{pa.output_dir}/datasets/day_job_jobcode')
        jobcode_machine_op_t.to_csv(f'{pa.output_dir}/datasets/jobcode_machine_op_t')
        turn_abilities.to_csv(f'{pa.output_dir}/datasets/turn_abilities')
        j1_j2_m_setup.to_csv(f'{pa.output_dir}/datasets/j1_j2_m_setup')

    return day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup

def from_generate_to_dataset(g):
    '''Generate the dataset needed for initialize the single agent environment, starting from the output of the generator.'''

    # day_job_jobcode 
    day_job_jobcode = pd.DataFrame()
    day_job_jobcode['Day'] = [1]*len(day_job_jobcode)
    day_job_jobcode['Job'] = g._u['job']
    day_job_jobcode['Code'] = day_job_jobcode['Job']

    # jobcode_machine_op_t
    jobcode_machine_op_t = pd.merge(g._r, g._pt, on=['job','machine'])
    jobcode_machine_op_t.rename(columns={'job':'Code', 'machine':'Machine', 'resources':'Operator', 'processing_time':'Time'}, inplace=True)

    # turn_abilities
    turn_abilities = pd.merge(g._tm, g._sk, on=['operator'])
    turn_abilities.rename({'operator':'Operator', 'shift':'Turn', 'machine':'Machine'}, axis=1, inplace=True)
    shift = {1:'TA',2:'TB',3:'TC'}
    turn_abilities=turn_abilities.replace({'Turn':shift})

    # j1_j2_m_setup
    j1_j2_m_setup = g._st.rename(columns={'job_from':'Job_from', 'job_to':'Job_to', 'machine':'Machine', 'setup_time':'SetupTime'})

    return day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup


def initialize_env(pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup):
    mask_env = MaskEnv(pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup)

    mask_env = ActionMasker(mask_env, mask_fn)

    return mask_env

def mask_fn(env: gym.Env) -> np.ndarray:
    '''Function which calls the action_mask method of the environment to get the action mask'''
    return env.action_masks()

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

def Maskable_PPO_alg(env, checkpoint=None):
        
    # CREATE AND TRAIN THE MODEL
    model_mask_ppo = MaskablePPO(MaskableActorCriticPolicy, 
                    env, 
                    # learning_rate = float(3e-5),
                    learning_rate=linear_schedule(0.0005), 
                    n_steps=1024,
                    batch_size = env.pa.batch_size,
                    gamma = 0.99,
                    gae_lambda=0.98,
                    n_epochs=10,
                    ent_coef=0.01,
                    target_kl = 0.5, 
                    verbose=env.pa.verb,
                    seed = env.pa.seed, 
                    tensorboard_log=f'./{env.pa.output_dir}/tensorboard/',
                    device = 'cuda'
                    )    
    
    if checkpoint != None:
        checkpoint_callback = checkpoint
        model_mask_ppo.learn(total_timesteps=env.pa.timesteps, 
                        log_interval=env.pa.log, 
                        callback = checkpoint_callback)
    else:
  
        model_mask_ppo.learn(total_timesteps=env.pa.timesteps, 
                        log_interval=env.pa.log)
    
    # SAVE THE MODEL
    model_mask_ppo.save(f"{env.pa.output_dir}/models/PPO_mask_model")

def Maskable_PPO_test(env, model_mask_ppo):
    i = 0
    obs, _ = env.reset()

    while i<10:
        action, _states = model_mask_ppo.predict(obs, action_masks=env.action_masks(), deterministic = True)
        
        print('episode: ', i)
        print(action)

        obs, reward, terminated, _, info = env.step(int(action))
        # print(obs.reshape(20,5))
        if terminated:
            # print(f"episode: {i} \n observation: {obs}")
            i +=1
            dataset = env.render()
            name = f'{env.pa.output_dir}/results/mask_ppo{i}.csv'
            dataset.to_csv(name)
            obs,_ = env.reset()

def create_output_directory(output_dir):
    # if not os.path.exists(f'{output_dir}/datasets'):
    #     os.makedirs(f'{output_dir}/datasets')  

    if not os.path.exists(f'{output_dir}/models'):
        os.makedirs(f'{output_dir}/models')  

    if not os.path.exists(f'{output_dir}/logs'):
        os.makedirs(f'{output_dir}/logs')   

    if not os.path.exists(f'{output_dir}/results'):
        os.makedirs(f'{output_dir}/results')  

    if not os.path.exists(f'{output_dir}/tensorboard'):
        os.makedirs(f'{output_dir}/tensorboard')




