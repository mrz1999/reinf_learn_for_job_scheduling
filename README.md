# reinf_learn_for_job_scheduling
In this repository the problem of Unrelated Parallel Machine with Setup Time and Resources will be solved by using Reinforcement Learning. Two different approach will be used: SIngle Agent and Multi Agent

In the main it is developed the Single Agent system. 
For using this system you simply should create a new virtual environment and install the requirements.txt .
The last version is the Masked Environment.
The Environment is in the file 'Environment4_action_mask.py'. For using it you can simply import it: 
from Environment4_action_mask import Environment

The file dataset_generator.py can be used for generating a new dataset, you can change the number of jobs, operators, machines, and other parameter for generating your dataset. You can also fix a seed (it uses the function that are in the folder src).

In the file main.py you can change the information about your training: e.g. the number of episodes (pa.timesteps), the settings about the checkpoints.
The algorithm used in MaskablePPO from stable-baselines3 library.


In the branch single_agent it will be presented the same environment of the main but adapted for using one of the algorithm of stabl-baseliens3 without the Mask (the environment in this branch differs from the one of the main for the reward function, it the maskable env I have changed some rewards for adapting the bonus because the maskable env never choose bad action).
In the branch there are 3 Environments almost egual. The only difference is in the termination criteria. The best Environmnet should be the 3.
There are 2 different main depending on if using vectorization (for fasting the computation) or not. The main are developped in such a way to testing the Environment eith three algorithms from stable-baseliens3 library: DQN, A2C and PPO.

Finally the other branch is about the multi agent.
Using this branch is a little bit more difficult, is not sufficient installing a requirements file. You can read the instruction in the file Instruction.md . We try to create a file (setup.sh) that authomatically install all the tools you need, but it is not completed. 

The environment is developed in the file multiagent_env.py. For using he library MARLLIB we need a wrapper of the Environment that in the file env_wrapper.py . 
For running the algorithm you can use the file train.py.  You can cange the algorithm among 'mappo', 'coma' and 'matrpo' that are the one we tried and secure work. For the ohter algorithms of the library, you should first check if they work. 