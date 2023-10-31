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
