# %%

# machine number 80
# operator number 171
# maximum operator number 9
# job number 352

import numpy as np
import random
import copy
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
# %% [markdown]
# ## RL environment

# %%
class Environment(gym.Env):
    """
    Description:
        The environment consists of a set of M machines, J jobs (which can be performed on some machines) and O operators (which have the ability to perform some jobs). 
        The agent starts the episode with no job allocated to any machine. (The environment can also be implemented for instantiated from a random assignation).
        The goal is to schedule all the jobs in the requestd date.
    Observation:
        Type: Box( num_machine * job_slots * 5)
        The idea of the observation is a bi-dimensional vector (num_machine*job_slots, 5) that is than flatten:
        Each row will represents a combination job, machine, let's immagine that row i is the combination of job in the job slot 'j' and machine 'm'
        Num                 Observation     
        (i,0)               Remaining time on machine m   
        (i,1)               Number of available operator on machine m
        (i,2)               Total time of job already scheduled on machine m         
        (i,3)               Time required for completed the job in the the job slot j on machine m
                            (-9 if that job can not be done on machine m)
        (i,4)               Num of operators required from job in the the job slot j on machine m        
                            (-9 if that job can not be done on machine m)
    Actions:
        Type: Discrete(num_machine * job_slots + 1)
        Each action is an integer which indicates a combination job machine.
        Let's imagine that action j corresponds to combination job in the jobslot 'k', machine 'n'
        Num                         Action
        j                           Assign the job stored in the k-th slot to machine n.
        num_machine*job_slot+1      Don't do anything. Let just time passes.
    
    Reward:
        At each time step the agent will receive a reward signal of the following 
        magnitude:

            r = -5 if the action is not possible
            r = -1 if it decide to not doing anything when there were possible jobs to schedule
            r = +5 if it devide to not doing anything and it was the best choice
            r = +3 - 
                max(time scheduled on machine on which I can do that job - time scheduled on the choosen machine) - 
                max(time required from the job on the others machine - time required from the job on the choosen machine) +
                5 (only if it was the best way to schedule that job)
            r = -15 if it choose a job slot empty
            r = +15 if it assignes all the jobs
            r = -15 if the episode ends without assigning all the jobs  

    Episode termination:
        The episode terminates when the agent allocates all the jobs
        There is early criteria of termination when the agent choose an empty job slot or when choose the no action (wait and not doing anything) when there were some other action possible.
    """

    def __init__(self,pa, day_job_jobcode, jobcode_machine_op_t, turn_abilities, j1_j2_m_setup):
        super(Environment, self).__init__()
        '''
        This function can be used for configuring the environment
        INPUT:
        pa: is a class which contains all the information about the parameters
        day_job_jobcode: is a pandas dataframe of 3 columns: ['Day', 'Job', 'Code']
        jobcode_machine_op_t: pandas dataframe of 4 columns: ['Code', 'Machine', 'Operator','Time']
        turn_abilities: pandas dataframe of 3 columns: ['Operator', 'Turn', 'Machine']
        j1_j2_m_setup: pandas dataframe of 4 columns: ['Job_from', 'Job_to', 'Machine', 'Setuptime']
        '''

        #datasets
        self.day_job_jobcode = day_job_jobcode
        self.jobcode_machine_op_t = jobcode_machine_op_t
        self.turn_abilities = turn_abilities
        self.j1_j2_m_setup = j1_j2_m_setup

        self.jobcode_to_integer() # I add the JobCode column to the jobcode_machine_op_t dataset

        tot_days = self.day_job_jobcode['Day'].unique() #this variable is a list of all the day for which I have to schedule jobs

        self.pa = pa # pa is a class contains all the parameters

        self.curr_time = 0 # I will update this variable each time I have terminated the action for one turn
        self.days = tot_days # all the days I have to analyze
        
        self.jobs_code = list(jobcode_machine_op_t['JobCode'].unique())
        self.machines_code = list(jobcode_machine_op_t['Machine'].unique())
        self.operators_code = list(turn_abilities['Operator'].unique())

        # I don't need this assert because it can happen that I have a machine but no job can be done on it, expecially if I use random dataset.
        # assert(self.pa.num_machine == len(self.machines_code))

        # the following three attributes are dictionaries which have as key the identifier of the class and as the value the related class.
        self.jobs = dict(zip(self.jobs_code, [Job(job_code, jobcode_machine_op_t) for job_code in self.jobs_code]))
        self.machines = dict(zip(self.machines_code, [Machine(machine_code, jobcode_machine_op_t, turn_abilities) for machine_code in self.machines_code]))
        self.operators = dict(zip(self.operators_code, [Operator(op_code, turn_abilities) for op_code in self.operators_code]))

        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)

        self.inizialize_queue()

        # Now let's define the state: the state will be modified only when an action is actually performed (in the step function)
        self.state = State(self.jobs, self.machines, self.operators, self.job_slot, self.job_backlog)
        
        self.finish = False # It will be True when the agents terminates the episode

        self._setup_space()
        self.scheduled_action = {machine:[] for machine in self.machines_code} # it will be a dictionary which has as key the machine and as value the list of the action scheduled on that machine

    def jobcode_to_integer(self):
        '''It transforms the job code (that are string of number and letters) into integer number'''
        # I will add a column to the dataset jobcode_machine_op_t with analogus information of the JobCode information but with integer.
        code = list(self.jobcode_machine_op_t['Code'].unique())
        new_code = []
        code_dict = {}
        for idx,element in enumerate(code):
            code_dict[element] = idx
        for element in self.jobcode_machine_op_t['Code']:
            new_code.append(code_dict[element])

        # add the new column to the dataframe
        self.jobcode_machine_op_t['JobCode'] = new_code

        # then I have also to change the setup dataset (I will substitute the column Job_to and Job_from with the respective jobcode (integer) for each job).
        self.j1_j2_m_setup['Job_from'] = [code_dict[job] for job in self.j1_j2_m_setup['Job_from']]
        self.j1_j2_m_setup['Job_to'] = [code_dict[job] for job in self.j1_j2_m_setup['Job_to']]

    def _setup_space(self):
        
        # The observation space will have two dimensions. For each machine-job I will record: 
        # - the remaining time on that machine 
        # - the number of available operator 
        # - the time occupied by the job scheduled on that machine
        # - the time required by the job 
        # - the number of operator required by the job. 
        # - So the shape will be (num_machines*num_job_slots) * 5
        self.observation_space = spaces.Box(
                                    low = np.array([0]*(self.pa.num_machine*self.pa.job_slots*5)), 
                                    high = np.array([99]*(self.pa.num_machine*self.pa.job_slots*5)), 
                                    shape = (self.pa.num_machine*self.pa.job_slots*5,), 
                                    dtype = np.float32 )

        # The action is assign a job to a machine or to not assign anything, 
        # so the number of action is the number of machine * number of jobs I'm considering + 1
        self.action_space = spaces.Discrete(self.pa.num_machine*self.pa.job_slots + 1)

    def observe(self, state):
        # INPUT: state must be an instance of the state class.

        # this is a compact representation (non a graphic one)
        # For each machine and each job in a jobslot I store:
        # - remaining time on that machine (0 if the machine is available)
        # - number of free operator that can work on that machine
        # - total time of all the jobs scheduled on that machine until now (I want omogeneous distribution of jobs so I don't want that a machine has a very long queue while another one is empty. I use this idea also for the reward)
        # - time of execution of the job (if the job can not be executed on that machine: -9)
        # - operator needed for execute that job (if the job can not be executed on that machine: -9)

        compact_repr = np.zeros((self.pa.num_machine*self.pa.job_slots,5))
        cr_mach = 0 # counts the index of the couple machine-job I'm considering

        for machine_code, machine in state.machines.items():
            machine_info = []
            operator_av = sum(state.operators[op_code].available for op_code in machine.operators)
            machine_info.extend([machine.remaining_time, operator_av, machine.time_scheduled])

            for job_code in state.job_slot.slot:

                machine_job = machine_info.copy()

                if job_code == None: # I have no job in that position of the job_slot
                    compact_repr[cr_mach,:] = machine_job.extend([-9,-9])
                else:
                    job = state.jobs[job_code]

                    if machine_code in job.suitable_machine: # If I can execute that job on that machine, i put the info of the job
                        index = np.where(job.suitable_machine == machine_code)[0] # I want the information about that job on that machine (because in al the info of the job I have info about all the mahcien on which I can perform that job)

                        # Note that the time of the job will be the actual time + the setup time needed
                        machine_job.extend([job.job_time[index][0] + self.get_setup_time(job_code, machine_code), job.operators[index][0]]) 

                    else: # if I can not execute that job on that machine, I put default value for the job
                        machine_job.extend([-9, -9]) 

                compact_repr[cr_mach,:] = machine_job
                cr_mach +=1
        compact_repr = np.reshape(compact_repr, self.pa.num_machine*self.pa.job_slots*5).astype(np.float32)
            
        return compact_repr
    
    def dict_indexNN_JobMach(self, state):
        '''It will create a dictionary which has as key the index of the output of the NN and as value the correpondent list [job_code, action_code]'''

        # The output of the NN depends on the observation. So the index job-machine of the output it's in the same order in which compare in the observation
        # -9 is the default job code when the job slot is empty

        index_to_couple = {}
        index = 0
        for machine_code in self.machines.keys():
            for job_code in state.job_slot.slot:
                if job_code == None:
                    index_to_couple[index] = [-9, machine_code]
                else:
                    index_to_couple[index] = [job_code, machine_code]
                index += 1
        return index_to_couple

    def nn_to_code(self, action_nn):
        '''It will take as input the output of the NN and will return a list [job_code, machine_code]'''
        # It's before taking the action, when I saw the output of the NN. So the actual state is self.state. 
        #NOTA CHE IL CASO DI NO ACTION VIENE CONSIDERATO E TRATTATO A PARTE
        my_dict = self.dict_indexNN_JobMach(self.state)
        return my_dict[action_nn]

    def inizialize_queue(self):
        '''At first I fill the queue with random jobs and I put all the other in the backlog.'''

        if len(self.jobs_code) < self.pa.job_slots: # I can put directly all the jobs in the queue
            for idx, code in enumerate(self.jobs_code):
                self.job_slot.slot[idx] = code 

        else:
            queue = random.sample(self.jobs_code, self.pa.job_slots)
            assert(len(queue) == len(self.job_slot.slot))
            self.job_slot.slot = queue
            backlog = [job_code for job_code in self.jobs_code if job_code not in queue]

            for idx, code in enumerate(backlog):
                self.job_backlog.backlog[idx] = code
            self.job_backlog.curr_size = len(backlog)

    def update_queue(self, state, job_action):
        '''Every time an action is selected, I have a new empty space in the queue of the jobs so I can 
        put a new job and remove it from the Backlog'''

        index = state.job_slot.slot.index(job_action)
        state.job_slot.slot[index] = None

        if not state.end_backlog: # In case I have still element in the backlog
            backlog_element = state.job_backlog.backlog # I have a list of backlog_size element, which contains The job in the backlog and None for the empty slot. I want only the index of the code in the backlog
            backlog_job = [job for job in backlog_element if job != None ]
            new_job = random.choice(backlog_job)
            state.job_slot.slot[index] = new_job

            backlog_index = state.job_backlog.backlog.index(new_job)
            state.job_backlog.backlog[backlog_index] = None
            if state.job_backlog.curr_size != 0:
                state.job_backlog.curr_size -= 1
            if state.job_backlog.curr_size == 0:
                state.end_backlog = True

        return state
            
    def is_possible_action(self, state, action):
        '''It will return True if the action is possible, False otherwise'''

        job_code, machine_code = action

        job = state.jobs[job_code]
        machine = state.machines[machine_code]
        op_code = machine.operators #operators code which can work on that machine (it's  a numpy array)
        index = np.where(job.suitable_machine==machine_code)[0]
        num_op = job.operators[index]

        # I want to stop the algorithm and giving a very bad reward every time it tries to schedule a job on a machine which is not suitable for that job, or which is occupied or when the necessary number of worker is not availabe 
        if machine_code not in job.suitable_machine or \
           machine.available == 0 or \
           (sum([state.operators[operator].available for operator in op_code]) < num_op)[0]: # if I don't put [0] I obtain a numpy array [False] or [True]
            return(False)
        else:
            return(True)

    def get_setup_time(self, job_to, machine_code) -> int:
        '''It will return the setup time on the input machine when passing from the last scheduled job on it to the one given as input.

        Input:
            job_to: job code of the job that I want to schedule
            machine: machine code of the machine on which I want to schedule the job
        
        Output:
            setup time: time needed to pass from the last scheduled job on that machine to the job_to in input
        '''
        if self.scheduled_action[machine_code] == []:
            return 0 # I have never used the machine for a job so I don't to wait a setup time
        else:
            
            last_job = self.scheduled_action[machine_code][-1]
            # print('machine code', machine_code, 'job to', job_to, 'last job', last_job)
            setup_time = self.j1_j2_m_setup[self.j1_j2_m_setup['Machine']==machine_code] \
                                            [self.j1_j2_m_setup['Job_from']==last_job] \
                                            [self.j1_j2_m_setup['Job_to']==job_to]['SetupTime'].values[0]
        return setup_time

    def step(self, action):
        """
        Take an action in the environment and observe the next transition.

        Args:
            action: An indicator of the action to be taken.

        Returns:
            The next transition.
        """
        
        if action == self.pa.num_machine*self.pa.job_slots: #no action
            return self.no_action_step()
            
        else:
            action = self.nn_to_code(action)
            job_code, machine_code = action
            # print('code of the job to execute',job_code)
            # print('job slot: ', self.state.job_slot.slot)

            if job_code == -9: # I have selected no job so it's like don't do any action
                return self.no_job_action()
            else:
                is_possible = self.is_possible_action(self.state, action) #this will be true if the action is possible

                info = {}
                reward = self.get_reward(self.state, action)

                if is_possible: 

                    self.state = self._get_next_state(self.state, action)
                    self.update_time(pass_time = False)
                    done = self.finish
                    # Update the list of the action scheduled on that machine
                    self.scheduled_action[machine_code].append(job_code)
                else:
                    done = True
                
                obs = self.observe(self.state)
            
            if done == True: # If I'm teminating the episode
                if self.state.job_slot.slot != [None] * self.pa.job_slots: # not all job are assigned
                    reward -= 15
                else:
                    reward += 15
        return obs, reward, done, False, info
    
    def reset(self, seed=0, options=None):
        """
        Reset the environment to execute a new episode.

        Returns: State representing the initial position of the agent.
        """

        self.curr_time = 0

        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)

        self.inizialize_queue()

        self.state = State(self.jobs, self.machines, self.operators, self.job_slot, self.job_backlog)

        self.finish = False

        obs = self.observe(self.state)
        self.scheduled_action = {machine:[] for machine in self.machines_code}
        return obs, {}
    
    def render(self):
        """
        Render a state of the environment.

        Args:
            mode: one of 'human' or 'rgb_array'. Human added only for compatibility.
            All rendering will be done in 'rgb arrays' via NumPy.

        Returns:
            A numpy.ndarray or None.

        """
        # assert mode in ['human', 'rgb_array']
        
        job_code = []
        job_start_time = []
        job_end_time = []
        macchina = [] 
        workers = []
        
        for job in self.state.jobs.values():
            job_code.append(job.ds_code)
            job_start_time.append(job.start_time)
            job_end_time.append(job.finish_time)
            macchina.append(job.machine)
            workers.append(job.op)

        # time_len = max(job_end_time) #is the time needed for completing all the jobs
        df = pd.DataFrame(list(zip(job_code, job_start_time, job_end_time, macchina, workers)), columns=['JOB CODE','START TIME','END TIME', 'MACHINE','OPERATORS'])

        return(df)

    def get_reward(self, state, action):
        """
        Compute the reward obtained by taking action 'a' at state 's'.

        Args:
            state: the state of the agent prior to taking the action.
            action: the action taken by the agent.

        Returns:
            A float representing the reward signal received by the agent.

        """
        if action == 'no action':
            # decide which is the reward if i decide to not compute action (something negative bcs I want to terminate as soon as possible. But if there no available action, wait is the best option so implement different case)

            # Check if the job in the job slots are all not possible, because machine are occupied or operator are not free. In this case give a positive reward. Other wise if an action was possible, give a negative one but not so exagerated, like -5.

            for job_code in state.job_slot.slot:
                if job_code != None: #  I don't have to do anything in case the job slot is empty
                    job = state.jobs[job_code]
                    machine_codes = job.suitable_machine
                    for machine_code in machine_codes:
                        machine = state.machines[machine_code]
                        if machine.available == 1:
                            op_code = machine.operators
                        
                            # calculate the number of operator needed:
                            index = np.where(job.suitable_machine==machine_code)[0]
                            num_op = job.operators[index]

                            if sum([state.operators[operator].available for operator in op_code]) >= num_op: 
                                return -3 # There is a possible action to execute but the system has preferred to wait (no good)
                            else:
                                reward = 5 # No available action so was good waiting
                        else: reward = 5


        # I'M TAKING AN ACTION
        else: 

            job_code, machine_code = action

            job = state.jobs[job_code]
            machine = state.machines[machine_code]
            op_code = machine.operators #operators code which can work on that machine (it's  a numpy array)

            # I check on which machine I could execute that job
            possible_machine = job.suitable_machine


            index = np.where(job.suitable_machine==machine_code)[0]
            num_op = job.operators[index]
            # GIVE NEGATIVE REWARD IF THE ACTION IS NOT POSSIBLE
            # Is the machine one of the possible for that job ? 
            if machine_code not in possible_machine:
                # print('MACHINE NOT POSSIBLE')
                reward = -5

            # Is the machine available ? 
            elif machine.available == 0:
                # print('machine not available')
                reward = -5
            
            # Is the required number of operator available? 

            elif sum([state.operators[operator].available for operator in op_code]) < num_op: 
                # print('not enough operator')
                reward = -5

            # CALCULATE THE REWARD WHEN I EXECUTE A POSSIBLE ACTION
            else:
                reward = 3 #bonus for assigning a correct job


                time = job.job_time[index]
                setup_time = self.get_setup_time(job_code, machine_code)
                time = time + setup_time # the real time I have to take in account is the time of job + the setup time because this is the time in which I have to occupy the machine

                # I calculate the maximum difference between the time scheduled on the chosen machine and the time that I could have had with another one (I want an omogeneous distribution among all the machines so I want that this difference is the minimum possible).
                #(In this way I'm considering also the machine that can not be available, but the idea is that I want to scheudle in the best way and if next time he try to schedule in the best way but the machine is not available, the alg will obtain a negative reward))
                difference = max([(machine.time_scheduled + time) - (state.machines[mach].time_scheduled + job_mach_time + self.get_setup_time(job_code,mach)) for mach, job_mach_time in zip(possible_machine, job.job_time)])

                # GIVE A POSITIVE REWARD IF IT WAS THE BEST POSSIBLE ACTION (in terms of omogenous scheduling on all the machine)
                if difference == 0:
                    reward += 5
                else:
                    reward -= difference
                
                # give a reward depending on the time that the job could have used on other machine instead of the one choosen
                diff_time = max([time-(other_time + self.get_setup_time(job_code,mach)) for other_time, mach in zip(job.job_time, possible_machine)]) 
                reward -= diff_time
                reward = float(reward)

                # a part of reward will be done taking in account the number of operators implied vs the number of operator required by the other machines that could execute that job.
                diff_op = max(num_op - other_n_op for other_n_op in job.operators)
                if diff_op == 0:
                    reward += 2
                else:
                    reward -= float(diff_op)/4

        return reward

    def _get_next_state(self, old_state, action):
        """
        Gets the next state after the agent performs action 'a' in state 's'.

        Args:
            state: current state (before taking the action).
            action: move performed by the agent (it will be a list of two values integer which represents the position of the job in the queue and the number of the machine on which perform the job).

        Returns: a State instance representing the new state.
        """
        state = copy.deepcopy(old_state) # I don't want to modify the state of the environment but create a new state instance which represents how the new state could be.

        job_code, machine_code = action 
        job = state.jobs[job_code]
        machine = state.machines[machine_code] 

        # First check if the machine for executing the job is free
        if machine.available == 0:
            return state
        else:
            # Check if the number of operator required is available
            op_code = machine.operators #operators code which can work on that machine (it's  a numpy array)

            index = np.where(job.suitable_machine==machine_code)[0]
            num_op = job.operators[index]

            if sum([state.operators[operator].available for operator in op_code]) >= num_op:

                operator_available = [code for code in op_code if state.operators[code].available == 1]
                operator_code = np.random.choice(operator_available, num_op, replace = False)
                operators = [state.operators[op_code] for op_code in operator_code]
                
                # Now we can execute the action on 'machine' with the element of operators as operators
                
                # Change the attribute of the job
                job.actual_time = float(job.job_time[index]) + self.get_setup_time(job_code, machine_code) # I have to add the setup time
                job.num_op = num_op

                job.status = 0
                job.start_time = self.curr_time
                job.finish_time = job.start_time + job.actual_time
                job.machine = machine_code
                
                job.op = [op.code for op in operators]

                # Assign the job to the machine
                machine.available = 0
                machine.running_job = job_code
                machine.remaining_time = job.actual_time
                machine.time_scheduled += job.actual_time

                # Change the availability of the operators 
                for op in operators:
                    op.available = 0
                    op.job = job_code
                    op.machine = machine

                # Update the job that are currently running (it containts the job class for all the jobs that are running)
                state.running_job = [job for job in state.jobs.values() if job.status == 0] #QUESTO SECONDO ME SI DEVE CAMBIARE
                
                # Update the job that are in the queue and in the backlog. (If there is at least one job in the backlog, I put it into the queue)
                state = self.update_queue(state, job_code)
            else:
                return state
        return state 

    def update_time(self, pass_time = True):
        '''This function is for free the resources. 
        Remember that I occupy machine and operator as soon as in the step function I assign the job to a machine'''
        # When I complete an action I need the time to go forward.

        if pass_time:
            self.curr_time += 1 # I update the time only in case of the 'no action' action 
        
            # JOB
            for job in self.state.jobs.values():
                job.update_status(self.curr_time)

            # MACHINE

            for machine in self.state.machines.values():
                machine.update_time()

            # OPERATOR (free the operator when the job is completly executed)
            for operator in self.state.operators.values():
                job_code = operator.job
                if job_code is not None:
                    if self.state.jobs[job_code].status == 1:
                        operator.available = 1
                        operator.job = None

        # Check if the episode is terminated because I have completed all the jobs, and I have more job to assign in the backlog or in the job slot.

        if self.state.job_backlog.curr_size == 0 and \
           self.state.job_slot.slot == [None] * self.pa.job_slots:
            self.finish = True

    def there_are_possible_act(self):
        for job_code in self.state.job_slot.slot:
                if job_code != None: #  I don't have to do anything in case the job slot is empty
                    job = self.state.jobs[job_code]
                    machine_codes = job.suitable_machine
                    for machine_code in machine_codes:
                        machine = self.state.machines[machine_code]
                        if machine.available == 1:
                            op_code = machine.operators
                        
                            # calculate the number of operator needed:
                            index = np.where(job.suitable_machine==machine_code)[0]
                            num_op = job.operators[index]

                            if sum([self.state.operators[operator].available for operator in op_code]) >= num_op: 
                                return True
                            else:
                                return False
                        else: return False

    def no_action_step(self):
        '''This function will be used  when the NN choose the no action.
        In case the selected action is the No action the only thing to do is to put the time forward of one time step.
        We check if there are possible action that could have be done instead on wait, if so we will stop the episode because the policy choose an action that is not good.
        '''
       
        reward = self.get_reward(self.state, 'no action')

        self.update_time()

        obs = self.observe(self.state)
        info = {}

        #check if there were other possible action, in that case stop the alg
        if self.there_are_possible_act():
            done = True
            reward -= 15 # I'm not terminating the episode correctly
        else: 
            done = self.finish


        return obs, reward, done, False, info

    def no_job_action(self):
        '''This function will be used  two cases: - The NN choose the no action - it is chosen an empty jobslot.
        When it choose an empty slot I don't want to consider it as a time step but I want the he tries again to chose
        another job. So the pass_time input will be False in this case, otherwise it will be True as default. '''


        obs = self.observe(self.state)
        reward = -15
        info = {}
        done = True

        return obs, reward, done, False, info


# %%
class State():

    def __init__(self, jobs, machines, operators, job_slot, backlog):

        self.job_now = None # WHEN I CONSIDER A JOB I HAVE TO PUT IT HERE. Here I want the jobcode I calculate as the added column, not the one in the input dataset

        self.jobs = jobs # dict with job code as key and the class instance as value
        self.machines = machines # dict of machines with key and value analogus to the previous attribute
        self.operators = operators # dict of operators with key and value analogus to the previous attribute
        
        self.running_job = np.zeros(len(self.machines)) # list which containts the job class for all the jobs that are running

        self.job_slot = job_slot # it's an instance of the JobSlot class
        self.job_backlog = backlog

        if self.job_backlog.curr_size != 0:
            self.end_backlog = False
        else:
            self.end_backlog = True

# %%
class Job():

    def __init__(self, job_code, jobcode_machine_op_t):
        
        self.id = job_code        
        self.ds_code = jobcode_machine_op_t.loc \
                                     [jobcode_machine_op_t['JobCode'] == job_code]\
                                     ['Code'].values[0]


        self.suitable_machine = jobcode_machine_op_t.loc \
                                     [jobcode_machine_op_t['JobCode'] == job_code]\
                                     ['Machine'].values #np.array of machine on which that job can be executed
        
        self.operators = jobcode_machine_op_t.loc\
                                     [jobcode_machine_op_t['JobCode'] == job_code]\
                                     ['Operator'].values #np.array. operators[i] is the number of operators needed for executing job on suitable_machine[i]
        
        
        self.job_time = jobcode_machine_op_t.loc\
                                     [jobcode_machine_op_t['JobCode'] == job_code]\
                                     ['Time'].values #np.array. job_time[i] is the time for computing the job on suitable_machine[i]
    
        self.status = -1 #it will be -1 if the job is not executed, 0 if it is being executing, 1 if it is already executed
    
        self.start_time = -1 #this will be the time at which I'm starting evaluating the job
        self.finish_time = float('inf') #Integer: the remaining time of that job. It will be infinite, until the job is executed by some machine

        # When I decide on which machine I want to execute the job, I save also info about the time and the number of operator needed for executing it on a specific machine  
        self.machine = None # this will be the machine on which I will execute the job
        self.actual_time = None
        self.num_op = None
        self.op = [] # this will be the list of the operators that will execute this job


    def update_status(self, curr_time):
        if self.status == 0:
            if curr_time >= self.finish_time:
                self.status = 1

# %%
class Machine():

    def __init__(self, machine_code, jobcode_machine_op_t, turn_abilities):

        self.code = machine_code

        self.poss_job = jobcode_machine_op_t.loc\
                                     [jobcode_machine_op_t['Machine'] == machine_code]\
                                     ['JobCode'].values

        self.operators = turn_abilities.loc\
                              [turn_abilities['Machine'] == self.code]\
                              ['Operator'].values #let's have a list with all the codes of the operators that can work on this machine.

        self.available = 1 #1 if it is available, 0 otherwise.
        self.running_job = None
        self.remaining_time = 0

        self.time_scheduled = 0 # I will add here the time for every job I schedule on this machine


    def update_time(self):
        '''When an operation is performed, time needs to go forward'''
        
        if self.remaining_time == 0:
            return 'No Job is running on this machine.'
        self.remaining_time -= 1

        if self.remaining_time == 0:

            self.running_job = None
            self.available = 1

# %%
class Operator():
    # Non considero il turno degli operai.
    def __init__(self, op_code, turn_abilities):

        self.code = op_code

        self.abilities = turn_abilities.loc\
                                     [turn_abilities['Operator'] == op_code]\
                                     ['Machine'].values # each element per un operator con la list delle macchine su cui possono lavorare   
        self.turn = turn_abilities.loc\
                                 [turn_abilities['Operator'] == op_code]\
                                 ['Turn'].values  # a list con ogni elemento che è un turno per un operatore.            

        self.available = 1 # 1 if it is available, 0 otherwise.
        self.job = None # it will be the code of the job when it will be assigned to a job.
        self.machine = None # it will be the code of the machine when the operator will be assigned to a job.


# %%
class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.job_slots # each element will be just the code of the job that is in the queue

class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0
