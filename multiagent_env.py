# %% import
import warnings
import gym
import numpy as np
import random
import pandas as pd
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

from ray.rllib.env import MultiAgentEnv

from src.generator import Generator

from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.utils.typing import (
    AgentID,
    MultiAgentDict,
)


import pandas as pd
# default='warn' to suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

warnings.filterwarnings('ignore')

# %% supplementary classes


class ActionBuffer:

    def __init__(self, _agent_ids: Set[str]):
        self._actions = {}
        for agent_id in _agent_ids:
            self._actions[agent_id] = []

    def add_action(self, agent_id: str, action):
        if agent_id not in self._actions:
            self._actions[agent_id] = []
        self._actions[agent_id].append(action)

    @property
    def get_agent(self, agent_id: str):
        return self._actions[agent_id]

    def add_action(self, agent_id: str, action: int):
        if agent_id not in self._actions:
            self._actions[agent_id] = []
        self._actions[agent_id].append(action)

    def get_last_action(self, agent_id: str):
        if agent_id not in self._actions:
            return None
        return self._actions[agent_id][-1]


# %% multi agent env

class MachineMultiAgentEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "machines"}

    """An environment that hosts multiple independent agents.

        Agents are identified by (string) agent ids. Note that these "agents" here
        are not to be confused with RLlib Algorithms, which are also sometimes
        referred to as "agents" or "RL agents".

        The preferred format for action- and observation space is a mapping from agent
        ids to their individual spaces.
        """

    def __init__(self, g: Generator, render_mode=None):
        """Initializes a multi-agent env.

        """

        super().__init__()

        self.render_mode = render_mode
        self._initial_state = g
        self._job_slot = 20

        self.agents = set()  # will be the set of alive agents
        self.agents.update(["M{}".format(i)
                           for i in range(self._initial_state.M)])
        self.agents = list(self.agents)
        # will be the set of all the agents (alive and dead)
        self._agent_ids = self.agents.copy()
        # will be the set of all the agents (alive and dead)
        self.possible_agents = self.agents.copy()
        # Create a mapping from agent_id to agent_index (from string to integer agent_name_mappping['M1]=1)
        self.agent_name_mapping = {}
        for i in range(self._initial_state.M):
            self.agent_name_mapping["M{}".format(i)] = (i)

        list(range(self._initial_state.M))

        self.preprocess_data()  # create all the dataset needed from the environment

        self._current_time = 0

        self._current_observations = None
        self._current_states = None

        self.flag_global_state = True
        if self.flag_global_state:
            self.global_state = None

        self._actions: ActionBuffer = ActionBuffer(self.possible_agents)

        self.available_operators = {}
        self.working_operators = {agent_id: []
                                  for agent_id in self.possible_agents}

        self.terminateds = set()
        self.truncateds = set()

        self.output_df = pd.DataFrame(
            columns=['job', 'machine', 'start_time', 'end_time', 'setup_time', 'operators'])

        self.machine_job = {machine:
                            list(
                                self.jobs_df['job'][self.jobs_df['machine'] == self.agent_name_mapping[machine]])
                            for machine in self.possible_agents}

        self._setup_space()

        # create the dict which map each integer output of the NN to the correspective job_id
        self.nn_to_action()

        self.no_action_count = {agent: 0 for agent in self.agents}
        self.null_action_count = {agent: 0 for agent in self.agents}
        self.max_no_action = 30
        self.max_null_action = 30

    def _setup_space(self):
        '''Define the action_spaces and the observation_spaces that are two dicts, which contains the action and the observation for each agent.'''
        # I will have one action space for each agent. Each agent chan choose one action among its possible jobs.
        # At the same way I will have one observation space for each agent. The observation of one agent will be formed by: 2 integer (availability and number of available resource) and 5 information for each job that can be executed on that machine.

        self.real_action_spaces = {}  # this are the real action and obs space
        self.real_observation_spaces = {}

        self.action_spaces = {}  # this are the padded action and obs space
        self.observation_spaces = {}
        global_shape = 0

        for machine, machine_code in self.agent_name_mapping.items():
            # let's calculate the number of possible jobs for a specific machine
            machine_poss_jobs = len(
                self.jobs_df[self.jobs_df['machine'] == machine_code])

            self.real_action_spaces[machine] = gym.spaces.Discrete(
                machine_poss_jobs+1)
            # the high parameter can be fixed in a better way than 999.
            self.real_observation_spaces[machine] = gym.spaces.Box(
                low=0, high=999, shape=(2 + 5 * machine_poss_jobs, ), dtype=np.float64)

            global_shape += 2 + 5 * machine_poss_jobs

        if self.flag_global_state:
            # the observation space idenfity by the key 'state' will be the global state that in this environment is the union of all the single observations of the different agents.
            self.real_observation_spaces['state'] = gym.spaces.Box(
                low=0, high=999, shape=(global_shape, ), dtype=np.float64)

        # UNIFORME THE SPACE FORMAT (IN THIS WAY YOU DON'T HAVE TO USE SUPERSUIT)
        max_obs_space = max([self.real_observation_spaces[agent].shape
                             for agent in self.real_observation_spaces.keys()])
        max_action_space = max([self.real_action_spaces[agent].n
                                for agent in self.real_action_spaces.keys()])
        self.action_spaces = {agent: gym.spaces.Discrete(max_action_space)
                              for agent in self.real_action_spaces.keys()}
        self.observation_spaces = {agent: gym.spaces.Box(low=0, high=999, shape=max_obs_space, dtype=np.float64)
                                   for agent in self.real_observation_spaces.keys()}

    def action_space(self, agent: AgentID):
        return self.action_spaces[agent]

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def preprocess_data(self):
        '''
        This function preprocess the data to generate the required dataframes
        For this purpose, the following dataframes are generated:
        - Job information (job_id, machine_id, remaining_units, processing_time, resource_requirement, remaining_time)
        - Setup-time information job_from, job_to, machine, setup_time
        - Operator information (operator_id, day, start_time, end_time)
        - Operator-machine information (operator_id, machine_id, skill)

        The dataframes are generated using the initial state of the environment.
        '''
        # Generate the required data
        # Create dataframes with the next information:
        # - Job information (job_id, machine_id, remaining_units, processing_time, resource_requirement, remaining_time)

        self.jobs_df = self._initial_state._u.merge(
            self._initial_state._pt, on='job')
        self.jobs_df["remaining_time"] = 0
        self.jobs_df.rename(columns={"unit": "remaining_units"}, inplace=True)
        self.jobs_df = self.jobs_df.merge(
            self._initial_state._r, on=['job', 'machine'])

        # Setup-time information job_from, job_to, machine, setup_time
        self.setup_df = self._initial_state._st

        # Operator information (operator_id, day, start_time, end_time, machine_operability, assigned)
        self.operators_df = []
        days = 1
        for _, row in self._initial_state._tm.iterrows():
            shift = row['shift']
            for day in range(days):
                start_time = day * 24 + (shift-1) * (8 * 60)
                end_time = start_time + (8 * 60)
                self.operators_df.append({"operator": row['operator'],
                                          "day": day,
                                          "start_time": start_time,
                                          "end_time": end_time,
                                          "machine_operability": len(self._initial_state.operator_machines(row['operator'])),
                                          "assigned": 0})
        self.operators_df = pd.DataFrame(self.operators_df)
        # Operator-machine information (operator_id, machine_id, skill)
        self.operators_machine_df = self._initial_state._sk

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> MultiAgentDict:
        """Resets the env and returns observations from ready agents.

        Args:
            seed: An optional seed to use for the new episode.

        Returns:
            A dict with new observations for each ready agent.
        """

        # Call super's `reset()` method to (maybe) set the given `seed`.
        # ssuper().reset()

        # Reset the env and get the initial observations.

        # if seed is not None:
        #     np.random.seed(seed)

        self.preprocess_data()

        self.agents = self.possible_agents.copy()
        # self.machine_job = {machine:
        #                     list(self.jobs_df['job'][self.jobs_df['machine'] == self.agent_name_mapping[machine]])
        #                     for machine in self.possible_agents}

        self.available_operators = {}
        self.working_operators = {agent_id: []
                                  for agent_id in self.possible_agents}

        # create the dict which map each integer output of the NN to the correspective job_id
        self.nn_to_action()

        # Return the initial observations.
        self._actions: ActionBuffer = ActionBuffer(self.agents)

        if self.flag_global_state:
            self.global_state = None

        self._current_states = self._get_states()
        self._current_observations = self._get_observations()

        self._current_time = 0
        self.output_df = pd.DataFrame(
            columns=['job', 'machine', 'start_time', 'end_time', 'setup_time', 'operators'])

        self.no_action_count = {agent: 0 for agent in self.agents}
        self.null_action_count = {agent: 0 for agent in self.agents}

        # in info I put the info if the agent is available or not
        info = {agent_id: self._current_observations[agent_id][0]
                for agent_id in self.possible_agents}

        # return self._current_states, info # if use gymnasium, use this return
        return self._current_observations

    def state(self):
        '''Return a global view of the environment appropriate for centralized training decentralized execution methods. I have implemented it as the union of all the single observations of the different agents.'''

        return self.global_state

    def _get_states(self) -> MultiAgentDict:
        """Returns the state for each possible agent.

        Returns:
            The states for each ready agent.
        """
        return {
            agent: {'actual_state': self._get_state(agent)['actual_state'],
                    'available_jobs': self._get_state(agent)['available_jobs']}
            for agent in self.possible_agents
        }

    def _get_state(self, agent_id: str) -> Dict[str, np.ndarray]:
        """Returns the states for the given agent.

        Args:
            agent: The  agent ID (like 'M0', 'M4', .. etc.).

        Returns:
            A dictionary containing the state for the given agent.
            The output contains the following information for machine agents:
            - actual state of the machine:
                [job_id, remaining_units, remaining_time, required_resources, available_resources]
            - available jobs (a dictionary of dictionaries). Each job is a dictionary with the id as key and the following information as values:
                [remaining_units, processing_time,  resource_requirements, setup_time, num_machine(e.g. in how many machines can the job be executed)]
        """

        agent = self.agent_name_mapping[agent_id]

        # I have two possiblities: the agent is alive, or the agent is dead (if the agent is alive it will be in the list of the agents, otherwise it will be only in the list of the possible_agents)

        if agent_id in self.agents:
            # THE AGENT IS ALIVE

            # Calculate the operators that are able to work on the selected machine
            machine_operators = list(
                self.operators_machine_df[self.operators_machine_df['machine'] == agent]['operator'])

            # Calculate the operators that are availale NOW (not all the operators that can work on that machine)
            available_operators = self.operators_df[
                (self.operators_df['operator'].isin(machine_operators)) &
                (self.operators_df['assigned'] == 0) &
                (self.operators_df['start_time'] <= self._current_time) &
                (self.operators_df['end_time'] > self._current_time)
            ]

            n_operators = len(available_operators)

            # Store the operator_id of the available operators now.
            # (machine_operability is on how many machine that operator can work)
            self.available_operators[agent_id] = {'operators': available_operators['operator'].values.tolist(),
                                                  'machine_operability': available_operators['machine_operability'].values.tolist()}

            # Get the CURRENT STATE of the machine (So the information about the job that is running on the machine at the current time)
            # Check if there is a job that is running on that machine
            current_job = self.jobs_df[
                (self.jobs_df['machine'] == agent) &
                (self.jobs_df['remaining_time'] > 0)
            ]

            if len(current_job) > 0:
                current_job = current_job.iloc[0]
                current_job_state = {'job_id': current_job['job'],
                                     'remaining_units': current_job['remaining_units'],
                                     'remaining_time': current_job['remaining_time'],
                                     'required_resources': current_job['resources'],
                                     'available_resources': n_operators}
                current_job = current_job['job']
            else:
                current_job = None
                current_job_state = {'job_id': None,
                                     'available_resources': n_operators}

            # Get the AVAILABLE JOBS for the machine (5 info for each job: remaining_units, processing_time, resources requirements, setup time, num_machine).

            # I select just the job that are available to be executed on that machine
            available_jobs = self.jobs_df[(self.jobs_df['machine'] == agent) &
                                          (self.jobs_df['remaining_units'] > 0) &
                                          (self.jobs_df['remaining_time'] == 0) &
                                          (self.jobs_df['resources']
                                           <= n_operators)
                                          ]
            # Remove jobs that are been executing by other machines. Bcs if the job can be done on more machines and on one of them is not executing, it will have a row with remaining_time 0 and so it will be selected in the previous line. But we don't want it

            # array of jobs to remove
            job_to_remove = self.jobs_df[(
                self.jobs_df['remaining_time'] != 0)]['job'].values
            if len(job_to_remove) > 0:
                available_jobs = available_jobs[~available_jobs['job'].isin(
                    job_to_remove)]
            # Calculate the SETUP TIME to the available jobs

            # Consider the case in which I don't have any previous job
            if len(self._actions._actions[agent_id]) == 0:
                available_jobs.loc[:, 'setup_time'] = np.zeros(
                    len(available_jobs))
            else:
                # If on that machine was executed before another job, I need to calculate the setup time from the last job executed
                previous_job = self._actions.get_last_action(agent_id)

                machine_setups = self.setup_df[
                    (self.setup_df['machine'] == agent) &
                    (self.setup_df['job_from'] == previous_job)
                ]
                available_jobs = available_jobs.merge(
                    machine_setups, left_on='job', right_on='job_to')

            # I calculate the number of machines on which can be execute each available job
            num_machines = [self.jobs_df['job'].value_counts()[job_id]
                            for job_id in available_jobs['job']]
            # Convert available jobs to a list of certain columns
            available_jobs_state = {job_id: {
                'remaining_units': available_jobs['remaining_units'].loc[available_jobs['job'] == job_id].values[0],
                'processing_time': available_jobs['processing_time'].loc[available_jobs['job'] == job_id].values[0],
                'required_resources': available_jobs['resources'].loc[available_jobs['job'] == job_id].values[0],
                'setup_time': available_jobs['setup_time'].loc[available_jobs['job'] == job_id].values[0],
                'num_machines': num_machines[idx]
            } for idx, job_id in enumerate(available_jobs['job'])}
        else:
            # THE AGENT IS DEAD

            current_job = None
            current_job_state = {'job_id': 999,  # I will put a job_id that is not possible, so I can recognize that the agent is dead and I cannot assign job to it because if I put None in the observation I will interpret it as the machine is available.
                                 'available_resources': 0}

            available_jobs_state = {}

        return {"actual_state": current_job_state, "available_jobs": available_jobs_state}

    def _get_observations(self, states=None) -> MultiAgentDict:
        """Returns the observations for each ready agent.

        Args:
            states: A dictionary containing the states at which I want to calculate the observations for all the agents

        Returns:
            A dict containing the observations for each ready agent.
        """
        # If the state is defined, I calculate the observation at that particular state, otherwise I use the current state of the environment.
        obs = {}
        state = []

        if states is None:
            states = self._current_states

        for agent in self.possible_agents:  # tutti gli agenti

            if agent in self.agents:  # agenti vivi

                real_obs = self._get_observation(agent, states[agent])
                padding_obs = self.add_padding_obs(real_obs)
                obs[agent] = padding_obs

            if self.flag_global_state:

                if agent in self.agents:

                    # this is the size dedicated to this agent in the global state
                    real_len_obs = 2 + 5 * len(self.machine_job[agent])
                    # la lunghezze dell'osservazione per il singolo agente dorebbe sempre essere questa perchè io metto 0 quando qualche job non si può fare ma mantenedo la size fissata
                    assert (len(real_obs) == real_len_obs)
                    # calculate the global state
                    state = np.append(state, real_obs)

                else:
                    # if the agent is dead, I put 0 in the global state
                    state = np.append(state, np.zeros(
                        2 + 5 * len(self.machine_job[agent])))

        if self.flag_global_state:
            self.global_state = state

        return obs

    def _get_observation(self, agent, state=None) -> List:
        """Returns the observation for the given agent.

        Args:
            agent: The agent ID (like 'M0', 'M1', .. etc.).
            state: The state of the agent. If the state is not specified, the current state of the agent will be used.


        Returns:
           Return a list that is the observation of the agent and it contains the following information:
           [machine_availability, available_resources, (remaining_units, processing time, resources requirements, setup time, num_machine) for each possible job on that machine]
        """
        if state is None:
            state = self._current_states[agent]

        # Get the observation of the machine to give as input to the policy

        # The first element of the observation will be a binary variable (1 if the machine is available (is not executing any job), 0 if it is occupied).
        if state['actual_state']['job_id'] == None:
            obs = [1]
        else:
            obs = [0]

        # The second element of the observation will be the number of available operators for the machine
        obs.append(state['actual_state']['available_resources'])

        # The remaining elements of the observation will be the information about all the possible jobs for the machine.
        # If is not possible execute a specific job now, the vaue corresponding to this job in the observation will be 0.
        for job in self.machine_job[agent]:

            # This are the job that can execute now from the machine.
            available_jobs = state['available_jobs']

            if job in available_jobs.keys():

                obs.extend([available_jobs[job]['remaining_units'],
                            available_jobs[job]['processing_time'],
                            available_jobs[job]['required_resources'],
                            available_jobs[job]['setup_time'],
                            available_jobs[job]['num_machines']])
            else:
                # If the job is not one of the available for that machine, it will have all the values equal to 0.
                obs.extend([0, 0, 0, 0, 0])
        return np.array(obs)

    def add_padding_obs(self, obs):
        # I have to add the padding for having all the observation of the same size
        # I have one unique shape for each agent
        size = self.observation_spaces['M0'].shape[0]
        padding = [0]*(size-len(obs))
        obs = np.append(obs, np.array(padding))
        return obs

    def nn_to_action(self):
        '''The output of the NN will be an integer. Each integer will correspond to a specific job. We build a dictionary that will map each integer to the job_id for each agent (machine).'''

        self.nn_to_action_dict = {agent_id: {nn:
                                             job_id for nn, job_id in enumerate(self.machine_job[agent_id])}
                                  for agent_id in self.agents}
        # print('real action dictionary: ', self.nn_to_action_dict)

        for agent_id in self.agents:
            # action that I have added for the padding:
            for key in range(self.real_action_spaces[agent_id].n, self.action_spaces[agent_id].n):
                self.nn_to_action_dict[agent_id][key-1] = 'null'

            self.nn_to_action_dict[agent_id][self.action_spaces[agent_id].n-1] = 'no action'

    def get_job_from_nn(self, agent_id: str, nn: int) -> int:
        '''Given the agent_id and the integer output of the NN, it returns the job_id'''
        return (self.nn_to_action_dict[agent_id][nn])

    def there_are_poss_act(self, agent_id: str) -> bool:
        '''Returns True if there are action that can be executed by the agent at the current timesteps, False otherwise.
        It will take as input the id of the agent'''
        if len(self._current_states[agent_id]['available_jobs']) > 0 & \
                (self._current_states[agent_id]['actual_state']['job_id'] == None):
            return False
        else:
            return True

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        Takes as input a dictionary with as key the agent ('M0','M1', ... etc) and as value the output of the NN. 
        ATTENTION!! The output of the NN is not the code of the action that the agent wants to perform, but is an index which can be mapped to the axction to perform throug the nn_to_action_dict.

        Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        FLOW:
            1) Check if the action selected are possible
                - Have two machines selected the same action ?
                - Is the action one of the remaining ones or it was already executed ?
                - Are enough operators available for that action ? 
                - Is available the machine on which I want to execute the action or it was already doing something else ? 
            2) Execute the possible actions
            3) Add the executed action to the output dataframe
            4) Update the state of the environment
            5) Calculate the reward
            6) Calculate the new observation

        Returns:
            Tuple containing:
            1) new observations for each ready agent, 
            2) reward values for each ready agent. If the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).

        """
        obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}

        # Convert the output of the NN in the action_dict with the code of the operation to perform.
        action_dict = {agent_id: self.get_job_from_nn(
            agent_id, nn_output) for agent_id, nn_output in action_dict.items()}
        if action_dict == {}:
            print("!!! FORCED TERMINATION !!!")
            terminated['__all__'] = True
        # print('action dictionary: ', action_dict)
        # print('jobs dataset: ', self.jobs_df)
        for ag in action_dict.keys():  # count the number of consecutive no action when something else was possible
            if action_dict[ag] == 'no action':
                # check that no other action where possible
                if self.there_are_poss_act(ag):
                    self.no_action_count[ag] += 1
                else:
                    # reset the counter because choosing the no action was the best option
                    self.no_action_count[ag] = 0
            else:
                self.no_action_count[ag] = 0  # reset the counter

        # Check if the action selected are possible
        # First check if two or more machine have selected the same action.
        truncated = self._actions_not_overlapping(action_dict)

        # Then check if all the actions are valid for the respective machine
        for agent_id, action in action_dict.items():
            agent = self.agent_name_mapping[agent_id]

            if action != 'no action':  # THE NO ACTION IS ALWAYS POSSIBLE, SO I DON'T NEED TO CHECK

                # First check that the machine is not already executing something else
                if self._current_states[agent_id]['actual_state']['job_id'] != None:
                    # assert(self._current_observations[agent_id][0] == 0) # when I update the current state, I update also the current observation, so if in the current state I have ajob oon that machine, I will found it also in the observation.

                    # In this case I want a bad reward, because he has to learn to wait when the machine is already executing something else.
                    truncated[agent_id] = True

                # Check if the action is valid for the machine
                if action not in self._current_states[agent_id]['available_jobs'].keys():
                    truncated[agent_id] = True

                # ____________________ SOSTITUITO DALL'IF DI SOPRA (POSSO USARE QUELLO DI SOTTO SE VOGLIO FARE QUALCHE DIFFERENZA PER I DIFFERENTI CASI IN CUI L'AZIONE NON È VALIDA) ______________________
                # # Per ogni agente le azioni tra cui sceglie sono solo i job che possono essere eseguiti su quella macchina, quindi il check sulle azioni possibili non serve.

                # res_not_available = self._not_enough_res(agent_id, action)   # Check if enough operators are available for executing that action
                # act_not_valid = self._not_available_act(agent_id, action) # Check if the machine was already executing something when it was assigned another job
                # is_executed = self._job_terminated(agent_id, action) # Check if the action was already executed (I have not other units to do for that job) or is in execution by another machine.

                # if truncated[agent_id] == True:
                    # print("The action {} is not valid for agent {}.".format(action, agent_id))
            if action == 'null':  # I'm choosing one of the action that I have added for making all the action space equal
                truncated[agent_id] = True

        # I want to count the number of consecutive no action (the one that have truncated = TRUE)
        for ag in truncated.keys():
            if truncated[ag] == True:
                self.null_action_count[ag] += 1
            else:
                self.null_action_count[ag] = 0  # reset the counter

        # EXECUTE THE NOT TRUNCATED ACTIONS

        # Select the set of operator to assign to each job
        op_to_assign = self.choose_ops_to_assign(
            action_dict, truncated, mode='greedy')

        for agent_id, op_list in op_to_assign.items():  # Store new info about the operator working now
            self.working_operators[agent_id] = op_list

        for agent_id, action in action_dict.items():
            # execute the action
            if (action != 'no action') and (truncated[agent_id] == False):

                self._make_action_machine(
                    agent_id, action, op_to_assign[agent_id])
                self.add_action_to_output(
                    agent_id, action, op_to_assign[agent_id])

        reward = self.get_total_reward(action_dict)

        # print('output dataset: ', self.render())
        # After all the actions are computed, a time step will pass and all the remaining time and units of the jobs will be updated.
        self.make_time_pass()

        # Check if the episode is terminated
        # NOTE: an agent could have jobs that are still executing, but if it has not more jobs to schedule it will be considered as terminated
        for agent_id in action_dict.keys():
            terminated[agent_id] = self._is_done(agent_id)
            info[agent_id] = {}

        # Force the termination of the agents which have select more than 10 consecutive no action
        for agent_id, no_action_count in self.no_action_count.items():
            if no_action_count >= self.max_no_action:
                terminated[agent_id] = True

        # Force the termination of the agents which have select more than 5 consecutive null action
        for agent_id, null_action_count in self.null_action_count.items():
            if null_action_count >= self.max_null_action:
                terminated[agent_id] = True

        if '__all__' not in terminated.keys():
            terminated['__all__'] = all(terminated.values())

        # We calculate the new state and observation of the envitonment.
        self._current_states = self._get_states()

        obs = self._get_observations()

        # If some agent is terminated (no more job to execute) I have to remove it from the list of the agents
        agent_to_remove = []
        for agent_id in self.agents:
            try:
                if terminated[agent_id]:
                    agent_to_remove.append(agent_id)
            except:
                # print('I cannot remove the agent {} bcs the terminated was {}'.format(agent_id, terminated))
                pass
        for element in agent_to_remove:
            self.agents.remove(element)

        # In this code I have use truncated when the action was not possible. But in this case I don't want to stop the agent so the truncated that I will giVe as return will be False for every agent because the only case in which I want that it stops is when there is the terminated
        # truncated = {agent_id: False for agent_id in self.possible_agents}
        # if self.agents == []:
        #     assert(terminated['__all__'] == True)
        # # print(self.action_mask())
        # if terminated['__all__']:
        #     # print('END OF THE EPISODE !!!')
        #     print('output dataset: ', self.render())
        # print('rewards: ', reward)
        return obs, reward, terminated, info

    def _not_enough_res(self,  agent: str, action: int):
        """
        ''' Check if there are enough operators available for the action selected by the agent

        Args:
            agent: The agent name ('M0, 'M1', ...) .
            action: The action to be taken by the agent.

        Returns:
            False if the action is valid for the given agent_id (machine type), True otherwise (the action can not be computed).

        """
        # The no action will be the last output of the NN
        if action == self.action_spaces[agent].n:
            return True  # The null action is always possible
        else:
            machine_state = self._current_states[agent]
            available_res = machine_state['actual_state']['available_resources']
            action_job = machine_state['available_jobs'][action]
            required_res = action_job['required_resources']
            if available_res >= required_res:
                return False
            else:
                return True

    def _job_terminated(self, agent_id: str, action: int, state: Dict = None):
        '''Check if the action was already executed (I have not other units to do for that job).
        It returns False if the action is not terminated and can be executed now, True otherwise .
        '''

        if state is None:
            state = self._current_states[agent_id]

        completed = state['available_jobs'][action]['remaining_units'] == 0
        # NON HO BISOGNO DI CONTROLLARE SE IL JOB È IN ESECUZIONE PERCHÈ IN QUESTO CASO NON LO TROVEREI NELLO STATO
        # in_execution = state['available_jobs'][action]['remaining_time'] != 0

        # in_execution = (self.jobs_df['remaining_time'][(self.jobs_df['job'] == action) &
        #                                       (self.jobs_df['machine'] == agent)] != 0).values[0]
        # if completed or in_execution:
        if completed:
            # either the action is executing now by another machine or I have already terminated all the units I needed for finishing this job. (I can not execute the job)
            return True
        else:
            # the action is not already finished and no machine is executing it, so it can be executed now. (I can execute the job)
            return False

    def _not_available_act(self, agent: str, action: int):
        '''Check if the selected action is a valid one for the machine. If the machine is already executing a job, the only valid action is the 'no action'.

        Args:
            agent: The agent id ('M0, 'M1', ...) 
            action: The action to be taken by the agent

        Returns:
            False if the action is valid for the given agent_id (machine type), True otherwise (the action can not be executed).
        '''

        # The no action will be the last output of the NN
        if action == self.action_spaces[agent].n:
            return True  # The null action is always possible
        else:

            machine_state = self._current_states[agent]
            current_job = machine_state['actual_state']['job_id']

            if current_job is None:
                return False  # there is not a job in the machine, so the action is valid
            else:
                return True  # the machine is already executing a job, so the action is not valid

    def _actions_not_overlapping(self, action_dict: MultiAgentDict) -> MultiAgentDict:
        """
        Check if in the set of actions to be performed there are two or more jobs selected by the same agent (same machine).

        Args:
            action_dict: dictionary with as key the agent ('M0','M1', ... etc) and as value the action that the agent wants to perform.

        Returns:
          dictionary with agent as key and True if its action is not overlapping with anyone else, False otherwise as value.
        """

        # This is the list of all the actions to perform
        actions = list(action_dict.values())
        # The 'no action' are not actions that are overlapping also if there is more the one, so I do not consider them. The same for the 'null' action.
        if 'no action' in actions:
            while actions.count('no action'):
                actions.remove('no action')
        if 'null' in actions:
            while actions.count('null'):
                actions.remove('null')
        # If all the actions are the no actions I don't have to check if they are overlapping.
        if actions != []:
            # This is a dictionary with as key the action and as value the number of times that the action is performed
            counted_act = Counter(actions)
            if max(counted_act.values()) > 1:  # There are at least two agents wit the same action
                # print('There is at least one conflict')

                agent_saved = []  # list of agent which will execute the action in case there is a conflict

                # Choose one of the agent to execute the action and bloch the other agents with the same action
                for action, act_count in counted_act.items():
                    if act_count > 1:

                        # list of agent with same action
                        agents = [agent for agent in action_dict.keys(
                        ) if action_dict[agent] == action]

                        # choose the agent which implies less time for executing the action
                        times = {agent:
                                 self._current_states[agent]['available_jobs'][action]['processing_time']
                                 for agent in agents
                                 if action in self._current_states[agent]['available_jobs'].keys()}
                        if times != {}:
                            agent_saved.append(min(times, key=times.get))

                return {(agent): (True if (counted_act[action_dict[agent]] > 1) & (agent not in agent_saved) else False) for agent in action_dict.keys()}

        # Either in case all the actions are the no actions, or in case each action is choosen once so there is no conflict
        return {agent: False for agent in action_dict.keys()}

    def choose_ops_to_assign(self, action_dict: MultiAgentDict, truncated: MultiAgentDict, mode: str = 'random') -> MultiAgentDict:
        op_to_assign = {}
        already_assigned_op = []

        for agent, action in action_dict.items():
            if (action != 'no action') and (truncated[agent] == False):
                op_to_assign[agent] = self.choose_op_to_assign_manualcheck(
                    agent, action, already_assigned_op, mode)
                already_assigned_op.extend(op_to_assign[agent])

        return op_to_assign

    def choose_op_to_assign_manualcheck(self, agent_id: str, action: int, already_assigned_op: list, mode: str = 'random') -> list:
        '''Choose the operator to assign to the machine for this job.
        Args:
            agent_id: The agent id ('M0','M1','M2', ...)
            action: The job to be executed by the agent.
            mode: The mode to choose the operators to assign. It can be 'random' or 'greedy'.
        Output:
            oper_to_assign: The list of operators to assign to the job.
        '''
        # I give another input to this function that is the already_assigned_op. This is a list of operators that are already assigned to other jobs. So could not be assigned to this job.

        disponible_op = self.available_operators[agent_id]
        n_op_needed = math.ceil(
            self._current_states[agent_id]['available_jobs'][action]['required_resources'])

        # Check that the number of operators available is greater or equal to the number of operators needed for the job, after removing the operator already selected for other jobs.
        available_op = [op_id for op_id in disponible_op['operators']
                        if op_id not in already_assigned_op]
        assert (len(available_op) >= n_op_needed)

        if mode == 'random':
            oper_to_assign = random.sample(
                available_op, math.ceil(n_op_needed))

        elif mode == 'greedy':

            oper_to_assign = []  # list of operator that I will assign to the job
            ordered_op = [{op_id: order}
                          for order, op_id in sorted(zip(disponible_op['machine_operability'], disponible_op['operators']))
                          if op_id in available_op
                          ]  # it's a list of dict. Each dict has aas key the operator_id and as value the respective operability.

            # I start from the operator with the lowest operability. With this line I select the lowest operability.
            mach_op = list(ordered_op[0].values())[0]

            while len(oper_to_assign) < n_op_needed:
                same_operatibility = [list(op_dict.keys())[0]
                                      for op_dict in ordered_op
                                      if list(op_dict.values())[0] == mach_op]
                # I remove the operators that are already assigned to other jobs
                same_operatibility = [
                    op_id for op_id in same_operatibility if op_id not in already_assigned_op]
                if len(same_operatibility) > 0:
                    if n_op_needed - len(oper_to_assign) >= len(same_operatibility):
                        oper_to_assign.extend(same_operatibility)
                    else:
                        # this random choice can be changed with a more intelligent choice. Taking into account the operator needed by the other agent.
                        adding_op = random.sample(same_operatibility, int(
                            n_op_needed - len(oper_to_assign)))
                        oper_to_assign.extend(adding_op)

                mach_op += 1

                # I check that I don't go over the maximum operability of the machine. Because it's a while and I'm afraid of going in loop. It should not happen. I will delete this line as I'm sure that it will not go in infinite loop. I put max+1 for the checking bcs I update mach_op before checking the condition of the while.
                assert (mach_op <= max(
                    disponible_op['machine_operability']) + 1)

        else:
            raise ValueError(
                "The mode {} is not valid. The mode can be 'random' or 'greedy'.".format(
                    mode)
            )

        return oper_to_assign

    def choose_op_to_assign(self, agent_id: str, action: int, mode: str = 'random'):
        '''Choose the operator to assign to the machine for this job.
        Args:
            agent_id: The agent id ('M0','M1','M2', ...)
            action: The job to be executed by the agent.
            mode: The mode to choose the operators to assign. It can be 'random' or 'greedy'.
        Output:
            oper_to_assign: The list of operators to assign to the job.
        '''

        available_op = self.available_operators[agent_id]
        n_op_needed = self._current_states[agent_id]['available_jobs'][action]['required_resources']

        if mode == 'random':
            oper_to_assign = random.sample(
                available_op['operators'], math.ceil(n_op_needed))

        elif mode == 'greedy':
            ordered_op = [x for _, x in sorted(
                zip(available_op['machine_operability'], available_op['operators']))]
            oper_to_assign = ordered_op[:math.ceil(n_op_needed)]

        else:
            raise ValueError(
                "The mode {} is not valid. The mode can be 'random' or 'greedy'.".format(
                    mode)
            )

        return oper_to_assign

    def _make_action_machine(self, agent_id: str, action: int, oper_to_assign: list = []):
        """ 
        Make the action for the given agent_id of machine type

        Args:
            agent_id: The agent id ('M0','M1','M2', ...).
            action: The job to be executed by the agent.
            oper_to_assign: The list of operators to assign to the job (empty list if the action is the noaction).
        """

        agent = self.agent_name_mapping[agent_id]

        # Check if the list of operators to assign is not empty in case I have a job to schedule
        assert (len(oper_to_assign) > 0)

        # I have to put the info of the new job to be executed in the datasets

        # Change remaining time in the jobs dataset (It will the time for completng one unit plus the setp time needed from the machine for starting this job).
        self.jobs_df['remaining_time'][(self.jobs_df['job'] == action) &
                                       (self.jobs_df['machine'] == agent)] = \
            int(self.jobs_df[(self.jobs_df['job'] == action) &
                             (self.jobs_df['machine'] == agent)]
                ['processing_time']) + \
            self._current_states[agent_id]['available_jobs'][action]['setup_time']

        # update the assigned columm in the operators dataset
        self.operators_df['assigned'][self.operators_df['operator'].isin(
            oper_to_assign)] = 1

        self._actions.add_action(agent_id, action)

    def add_action_to_output(self, agent_id: str, action: int, op_to_assign: list = []):

        agent = self.agent_name_mapping[agent_id]

        job_duration = self._current_states[agent_id]['available_jobs'][action]['processing_time']

        setup_time = self._current_states[agent_id]['available_jobs'][action]['setup_time']

        new_action = {'job': action,
                      'machine': agent,
                      'start_time': self._current_time,
                      'end_time': self._current_time + setup_time + job_duration,
                      'setup_time': setup_time,
                      'operators': op_to_assign}  # HERE I'M PUTTING THE SCHEDULE JUST FOR ONE ITEM OF PRODUCTION. (the setup time is referred to the time needed before starting executing the effective job: start_time + setup_time = effective start time)

        self.output_df.loc[len(self.output_df)] = new_action

    def _is_done(self, agent_id: str) -> bool:
        """Returns whether the given agent is done (e.g. I don't have any job that could be executed on that machine).

        Args:
            agent_id: The agent ID (like 'M0', 'M4', .. etc.).

        Returns:
            Whether the given agent is terminated.
        """

        all_jobs_of_the_mach = self.machine_job[agent_id]

        job_terminated = list(set(self.jobs_df['job'][
            (self.jobs_df['job'].isin(all_jobs_of_the_mach)) &
            (self.jobs_df['remaining_units'] == 0)
        ].values))  # I use set because I want just one time the job. Without set if a job can be done on two machines, it will be counted two times because it will appear two times in the dataset.

        # In the job terminated for that machine I have to consider also if there is some job that can be done on that machine with one unit remaining that is already in execution by another machine.
        job_almost_terminated = list(self.jobs_df['job'][
            (self.jobs_df['job'].isin(all_jobs_of_the_mach)) &
            (self.jobs_df['remaining_units'] == 1) &
            (self.jobs_df['remaining_time'] != 0)
        ].values)

        job_terminated.extend(job_almost_terminated)

        if len(job_terminated) == len(all_jobs_of_the_mach):
            done = True
        else:
            done = False
        return done

    def get_total_reward(self, action_dict: MultiAgentDict):
        '''Calculate the reward of taking an action in a given state
            Args:
            action_dict: dictionary with as key the agent ('M0','M1', ... etc) and as value the action that the agent wants to perform.

            Output:
            Reward: the reward obtained by computing all the set of actions in the given state.
            '''

        # I HAVE IMPLEMENTED TWO REWARDS. reward_all can be used if you want just on e reward for all the agent. Instead, rewards is a dictionary with as key the agent_id and as value the reward for that agent.

        rewards = {}
        overlapping = self._actions_not_overlapping(action_dict)

        for agent_id, action in action_dict.items():
            rewards[agent_id] = self.get_single_reward(agent_id, action)

        # REWARD ALL
        # I compute the mean of the rewards of the single agents.
        reward_all = np.mean(list(rewards.values()))

        n_overlapped = list(overlapping.values()).count(True)
        # I subtract the number of overlapped actions to the reward as malus because it's a bad choice.
        reward_all -= n_overlapped

        # SINGLE REWARD FOR AGENT
        # I will put a malus for the agents which have choosen the same machines.
        for agent_id in rewards.keys():
            if overlapping[agent_id] == True:
                rewards[agent_id] -= 1/4

        # Bad reward if the agents have choosen the same operators (I HAVE CHANGED THE FUNCTION FOR THE OPERATORS, NOW I DON'T NEED TO DEVELOP THIS PART OF THE REWARD)

        return rewards

    def get_single_reward(self, agent_id: str, action: int, state=None):
        '''Calculate the reward of the single agent taking an action in a given state'''

        if state is None:
            # this is the state before performing the action. Because after calling the reward function I update the state
            state = self._current_states[agent_id]

        agent = self.agent_name_mapping[agent_id]
        reward = 0  # Inizialize the reward to 0

        # If the action is the no action, check if there was at least one real action available for that agent

        # NO ACTION
        if action == 'no action':
            # If there was at least one job available for that agent, the reward is negative
            # If there were not any job available for that agent, the reward is positive because WAIT was the best action to choose.

            # First check if the machine is executing something else. In case affermative, no action is the best choice.
            if self._current_states[agent_id]['actual_state']['job_id'] != None:
                reward += 1/2
                return reward

            # jobs that can be executed on that machine
            possible_job = list(state['available_jobs'].keys())
            if len(possible_job) == 0:
                # I have no possible action, so the no action was the best choice.
                reward += 1/2
            else:
                # penalty for choosing the no action when it was not the best choice.
                reward -= 1/2

        # NULL ACTION
        elif action == 'null action':
            reward -= 1  # penalty for choosing a null action

        # ACTION
        else:

            # ACTION NOT POSSIBLE
            # In case the machine is already executing another job or job not in the possible ones for that machine NOW the env should learn to assign always the no_action.
            if (self._current_states[agent_id]['actual_state']['job_id'] != None) or (action not in state['available_jobs'].keys()):
                # print('machine not available')
                # assert(self._current_observations[agent_id][0] == 0) # Check that the machine is not available (the first element of the observation is 0 if the machine is not available)
                reward -= 1/2

            # THE FOLLOWING ROWS SOULD BE COMPLETELY SUBSTITUED BY THE IF ABOVE
            # if self._not_enough_res(agent_id, action):
            #     print('not enough res')
            #     reward -= 3
            # elif self._not_available_act(agent_id, action):
            #     print('not available act')
            #     reward -= 3
            # elif self._job_terminated(agent_id, action):
            #     print('job terminated')
            #     reward -= 3

            else:
                # POSSIBLE ACTION
                # print('possible action')
                reward += 1/4  # bonus for choosing an AVAILABLE ACTION

                # If there were others agents which could compute that action the reward will be the difference between the time needed to compute the job on the selected agent and the time needed by the fastest agent to compute the job.

                # TIME NEEDED TO COMPUTE THE JOB on the selected machine.
                processing_time = self._current_states[agent_id]['available_jobs'][action]['processing_time']
                setup_time = self._current_states[agent_id]['available_jobs'][action]['setup_time']
                time_needed = processing_time + setup_time

                # Now we want to know how much time we could have saved if we had chosen the best action. So we calculated the time that others machine culd have implied
                # this is an array of machine integer code on which the job can be executed.
                possible_mach = self.jobs_df[self.jobs_df['job']
                                             == action]['machine'].values
                # list of the time needed for computing the job on each of the possible machines
                total_time = []
                for machine in possible_mach:

                    machine_id = 'M' + str(machine)

                    processing_time = self.jobs_df[self.jobs_df['job'] ==
                                                   action][self.jobs_df['machine'] == machine]['processing_time'].values[0]

                    if self._actions._actions[machine_id] == []:
                        setup_time = 0
                    else:

                        setup_time = self.setup_df[self.setup_df['job_from'] == self._actions.get_last_action(
                            machine_id)][self.setup_df['job_to'] == action][self.setup_df['machine'] == machine]['setup_time'].values[0]

                    total_time.append(processing_time + setup_time)

                if setup_time == 0:
                    reward += 1/4
                else:
                    reward -= 1/4

                # is 0 in the best case, otherwise is negative if I have another machine on which the job could have be done faster.
                time_advantage = (
                    min([other_time - time_needed for other_time in total_time]))*(1/2)
                if time_advantage == 0:
                    reward += 1/2  # bonus for choosing the best action
                else:
                    reward += time_advantage/5

                # We do the same with the NUMBER OF OPERATORS REQUIRED.
                required_res = self._current_states[agent_id]['available_jobs'][action]['required_resources']

                other_required_res = list(
                    self.jobs_df[self.jobs_df['job'] == action]['resources'])

                n_op_advantage = (
                    min([res - required_res for res in other_required_res]))
                if n_op_advantage == 0:
                    reward += 1/2  # bonus for choosing the best action
                else:
                    reward += n_op_advantage/2

        return reward

    
    
    def render(self):
        '''Return a dataset that can be used for visualizing the jobs scheduling.'''
        print("The fuking render")
        print(self.output_df)
        self.output_df.to_csv('output.csv')
        return True

    def make_time_pass(self):
        '''When the action is the no action and there is no job in the machine, the time pass of one step and the state of the machine is updated.'''

        # If the remainint time is 1, after passing the time, the job will be terminated.
        job_terminated = set(
            self.jobs_df['job'][self.jobs_df['remaining_time'] == 1].values)
        machine_terminated = self.jobs_df['machine'][self.jobs_df['remaining_time'] == 1].values

        try:
            # Check that the number of jobs terminated is equal to the number of machines terminated. For being sure of select good element from the datatset
            assert (len(job_terminated) == len(machine_terminated))
        except:
            raise AssertionError(
                f'job terminated: {job_terminated} machine terminated: {machine_terminated} dataset: {self.jobs_df}')

        self._current_time += 1
        self.jobs_df['remaining_time'] = self.jobs_df['remaining_time'].apply(
            lambda x: x-1 if x > 0 else 0)

        # For the job terminated in this time step, I have to update the remaining_units, because I've just finished one unit. (Here I don't care about the machine bcs I want to update the units for the jobin general).
        self.jobs_df['remaining_units'][self.jobs_df['job'].isin(
            job_terminated)] -= 1

        # I have to free the operators which are working on the machine on which were executing the jobs that are now terminated.
        op_to_free = []

        for machine in machine_terminated:
            # we need the machine_id
            for machine_id, mach in self.agent_name_mapping.items():
                if mach == machine:
                    break
            op_to_free.append(self.working_operators[machine_id][0])

        self.operators_df['assigned'][self.operators_df['operator'].isin(
            op_to_free)] = 0

    def one_action_mask(self, agent):
        '''Return a list of 0 and 1 that indicates which actions are possible for the agent.'''

        mask = []
        if self._current_states[agent]['actual_state']['job_id'] == None:
            # this represents the output of the NN (excluded the last one: no action)
            for i in range(self.action_space(agent).n - 1):
                job = self.get_job_from_nn(agent, i)
                if job in self._current_states[agent]['available_jobs'].keys():
                    mask.append(1)
                else:
                    mask.append(0)
        else:  # current job id is None
            # If there is not a job in the machine, all the other actions are not possible.
            mask.extend([0]*self.action_space(agent).n)
        # The no action will be the last output of the NN and it is always possible
        mask.append(1)
        return mask

    def action_mask(self):
        '''Return a dictionary with the mask for each agent. The mask is a list of 0 and 1 that indicates which actions are possible for the agent.'''

        # Note that the action mask works with the attribute self._current_state so be careful that every time you used the _current_state is updated with the new state of the environment.
        # In the env it should always be like this because the state is updated after each step
        assert (self._current_states == self._get_states()
                ), 'Your state is not updated with the current one!'
        action_mask = {ag: self.one_action_mask(
            ag) for ag in self.possible_agents}
        return action_mask

    # FOR USING QMIX I NEED TO SPECIFY AN AGENT GROUPING

    def with_agent_groups(
        self,
        groups: Dict[str, List],
        obs_space: gym.Space = None,
            act_space: gym.Space = None) -> "MultiAgentEnv":
        ''' An agent group is a list of agent IDs that are mapped to a single
        logical agent. All agents of the group must act at the same time in the
        environment. Agent grouping is required to leverage algorithms such as Q-Mix.
        The rewards of all the agents in a group are summed. The individual
        agent rewards are available under the "individual_rewards" key of the
        group info return.
        Args:
            groups: Mapping from group id to a list of the agent ids
                of group members. If an agent id is not present in any group
                value, it will be left ungrouped. The group id becomes a new agent ID
                in the final environment.
            obs_space: Optional observation space for the grouped
                env. Must be a tuple space. If not provided, will infer this to be a
                Tuple of n individual agents spaces (n=num agents in a group).
            act_space: Optional action space for the grouped env. (as for the obs_space)

        Output: 
            The grouped environment.

        EXAMPLES: grouped_env = env.with_agent_groups(env, { "group1": ["agent1", "agent2", "agent3"], 
                                                            "group2": ["agent4", "agent5"], 
                                                            })
        '''
        return MultiAgentEnv

    # FOR THE MULTIAGENTENV class from RLLIB
    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        """Checks if the observation space contains the given key.

        Args:
            x: Observations to check.

        Returns:
            True if the observation space contains the given all observations
                in x.
        """
        for agent, job in x.items():
            if job not in self.observation_spaces[agent]:
                return False
        return True

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        """Checks if the action space contains the given action.

        Args:
            x: Actions to check.

        Returns:
            True if the action space contains all actions in x.
        """
        for agent, action in x.items():
            if action not in self.action_spaces[agent]:
                return False
        return True

# %%
