
import itertools

import numpy as np
import pandas as pd
from src.job import Job, Job_Split
from src.machine import Machine, MachineTask
from src.operator import Operator, OperatorChain, OperatorTask, Worker
from src.task import Task


def initialize_jobs(jobs_df: pd.DataFrame):
    '''Initialize the jobs from the dataframe

    Args:
        jobs_df (pd.DataFrame): Dataframe with the jobs

    Returns:
        list[Job]: List of jobs
    '''
    jobs = []
    for row in jobs_df.iterrows():
        job = Job(idx = row[1]["job_id"], reference = row[1]["Code"], 
                day = row[1]["Proc_day"], production = row[1]["units"], cost = 10)
        jobs.append(job)
    return jobs


def split_jobs(jobs: 'list[Job]', production_times_df: pd.DataFrame, max_hours: int = 6):
    '''Split the jobs in order to have a maximum duration of 6 hours
    
    Args:
        jobs (list[Job]): List of jobs
        production_times_df (pd.DataFrame): Dataframe with the production times of each machine
        max_hours (int, optional): Maximum duration of the jobs. Defaults to 6.
        
    '''
    max_duration = max_hours  * 60 # in minutes
    for job in jobs:
        max_prod_time = production_times_df[production_times_df['Code'] == job.reference]['unit_production_time'].max()
        total_prod = job.production 
        max_per_split = int(np.floor(max_duration / max_prod_time))
        while total_prod > max_per_split:
            job.add_split(Job_Split(job, max_per_split))
            total_prod -= max_per_split
        job.add_split(Job_Split(job, total_prod))
        

def initialize_machines(skill_matrix_df: pd.DataFrame):
    '''Initialize the machines from the dataframe
    
    Args:
        skill_matrix_df (pd.DataFrame): Dataframe with the machines
        
    Returns:
        list[Machine]: List of machines'''
    machines = []
    for e,label in enumerate(skill_matrix_df["machine"].unique()):
        machine = Machine(idx = e, name = label)
        machines.append(machine)
    return machines

def initialize_workers(skill_matrix_df: pd.DataFrame):
    '''Initialize the workers from the dataframe

    Args:
        skill_matrix_df (pd.DataFrame): Dataframe with the workers skills

    Returns:
        list[Worker]: List of workers'''
    operators = []
    for e,label in enumerate(skill_matrix_df["personal"].unique()):
        operator = Worker(idx = e, name = label, cost = 10)
        operators.append(operator)
    return operators

def initialize_shift(workers: 'list[Worker]', planning_days: 'list[int]', shift_df: pd.DataFrame):
    '''Initialize the shift of the workers, one shift per day. 
    
    
    Args:
        workers (list[Worker]): List of workers
        planning_days (list[int]): List of planning days
        shift_df (pd.DataFrame): Dataframe with the shift of each worker
    '''
    shift_duration = 8 * 60
    for worker in workers:
        aux_= shift_df[worker.name == shift_df['personal']]
        if len(aux_) == 0:
            print("No shift for worker: ", worker.name)
            continue
        shift = aux_['turno'].values[0]
        if shift == 'Default':
            shift = 0
        elif shift == 'TA':
            shift = 0
        elif shift == 'TB':
            shift = 1
        elif shift == 'TC':
            shift = 2
        for i in planning_days:
            start_time = (float(i) * 1440) + 360 + shift * shift_duration
            end_time = (float(i)  * 1440) + 360 + (shift+1) * shift_duration
            worker.add_worker_shift([start_time, end_time])


def initialize_operators(workers: 'list[Worker]', planning_days: 'list[int]'):
    '''Initialize the operators from the workers generating one operator per day. 
    Operator = Shift of a worker in a day
    
    Args:
        workers (list[Worker]): List of workers
        planning_days (list[int]): List of planning days
        
    Returns:
        list[Operator]: List of operators'''
    operators = []
    shift_duration = 8 * 60
    for worker in workers:
        id = int(worker.name.split("FT")[1])
        if id < 100:
            shift = 0
        elif id < 180:
            shift = 1
        else:
            shift = 2
        for i in planning_days:
            start_time = (float(i) * 1440) + 360 + shift * shift_duration
            end_time = (float(i)  * 1440) + 360 + (shift+1) * shift_duration
            operator = Operator(worker=worker, start_time=start_time, end_time=end_time) 
            worker.add_worker_split(operator)
            operators.append(operator)
    return operators




def initialize_tasks(jobs: 'list[Job]', machines_dict: 'dict[Machine]', workers_dict: 'dict[Worker]', prod_df, skill_df, mod_df, MAX_DAY_DELAY=1):
    '''Initialize the tasks from the jobs
    
    Args:
        jobs (list[Job]): List of jobs
        machines_dict (dict[Machine]): Dictionary of machines
        workers_dict (dict[Worker]): Dictionary of workers
        prod_df (pd.DataFrame): Dataframe with the production times of each machine
        skill_df (pd.DataFrame): Dataframe with the skills of each worker
        mod_df (pd.DataFrame): Dataframe with the required working capacity of each worker
        MAX_DAY_DELAY (int, optional): Maximum day delay of the operators. Defaults to 1.
        
    Returns:
        list[Task]: List of tasks'''
    
    tasks = []
    for job in jobs:
        for split in job.splits:
            machine_list = prod_df[prod_df['Code']==job.reference]
            mod_list =  mod_df[mod_df['Code']==job.reference] 
            for row in machine_list.iterrows():
                machine_name = row[1]['machine']
                process_time = row[1]['unit_production_time']
                if machine_name not in machines_dict:
                    continue
                machine = machines_dict[machine_name]
                mod = mod_list[mod_list['machine']==machine_name]['operators'].values[0]
                task = Task(kind="normal", job_split = split, machine = machine, process_time= process_time, mod = mod)
                split.add_task(task)
                machine.add_task(task)
                tasks.append(task)
                worker_list = skill_df[skill_df['machine']==machine_name]["personal"]
                for worker_name in worker_list:
                    if worker_name not in workers_dict:
                        continue
                    worker = workers_dict[worker_name]
                    for operator in worker.splits:
                        day_diff = (operator.start_time // 1440) - job.day
                        if (0 <= day_diff) and (day_diff <= MAX_DAY_DELAY): # operator can only be asigned to 
                            operator.add_task(task)
    return tasks



def initialize_doplex_tasks(jobs: 'list[Job]', machines_dict: 'dict[Machine]', workers_dict: 'dict[Worker]', prod_df, skill_df, mod_df):
    '''Initialize the tasks from the jobs for the doplex problem
    
    Args:
        jobs (list[Job]): List of jobs
        machines_dict (dict[Machine]): Dictionary of machines
        workers_dict (dict[Worker]): Dictionary of workers
        prod_df (pd.DataFrame): Dataframe with the production times of each machine
        skill_df (pd.DataFrame): Dataframe with the skills of each worker
        mod_df (pd.DataFrame): Dataframe with the required working capacity of each worker
        MAX_DAY_DELAY (int, optional): Maximum day delay of the operators. Defaults to 1.
        
    Returns:
        list[Task]: List of tasks'''
    
    tasks = []
    for job in jobs:
        for split in job.splits:
            machine_list = prod_df[prod_df['Code']==job.reference]
            mod_list =  mod_df[mod_df['Code']==job.reference] 
            for row in machine_list.iterrows():
                machine_name = row[1]['machine']
                process_time = row[1]['unit_production_time']
                if machine_name not in machines_dict:
                    print("Machine not found in the skill matrix: ", machine_name)
                    continue
                machine = machines_dict[machine_name]
                mod = mod_list[mod_list['machine']==machine_name]['operators'].values[0]
                task = Task(kind="normal", job_split = split, machine = machine, process_time= process_time, mod = mod)
                
                worker_list = skill_df[skill_df['machine']==machine_name]["personal"]
                worker_assignments = 0
                for worker_name in worker_list:
                    if worker_name not in workers_dict:
                        continue
                    worker = workers_dict[worker_name]
                    worker.add_task(task)
                    worker_assignments += 1
                if worker_assignments > 0:
                    split.add_task(task)
                    machine.add_task(task)
                    tasks.append(task)
                else:
                    print("No workers can be assigned to task: ", task)
                    

    return tasks


def initialize_machine_tasks(machines: 'list[Machine]', setup_times_df):
    '''Initialize the machine tasks asignations in graph format
    
    Args:
        machines (list[Machine]): List of machines
        setup_times_df (pd.DataFrame): Dataframe with the setup times of each machine
    
    Returns:
        int: Number of machine task edges'''
    no_task_machines = []
    machine_task_edges = 0
    no_setup_times = 0
    for machine in machines:
        if len(machine.tasks) == 0:
            no_task_machines.append(machine)
            continue
        machine_setups = setup_times_df[setup_times_df['machine']==machine.name]
        for task_i in [machine.start_task]+machine.tasks: # add start task in i (n + m)
            if task_i.normal:
                from_setups = machine_setups[machine_setups['Code_from']==task_i.job_split.job.reference]
            for task_j in [machine.end_task]+machine.tasks: # add end task in j (n + m)
                if task_i.idx != task_j.idx:
                    day_diff = 0
                    cost = 0
                    if task_i.normal and task_j.normal: # only connect tasks with difference of 3 days, except machine start and machine end. Example: job of day 1 can be after job of day 2 but not after job of day 5
                        day_diff = abs(task_j.job_split.job.day - task_i.job_split.job.day)
                    if (day_diff <= 3): # TO-DO 
                        if task_i.normal and task_j.normal:
                            if task_i.job_split.job.reference == task_j.job_split.job.reference:
                                cost = 0
                            else:
                                cost = from_setups[from_setups['Code_to']==task_j.job_split.job.reference]['setup_time']
                                if len(cost) == 0:
                                    no_setup_times += 1
                                    continue
                                else:
                                    cost = cost.values[0] 
                        machine_task = MachineTask(task_i, task_j, machine=machine, setup_cost=cost)
                        task_i.add_ouput_x_edge(machine_task)
                        task_j.add_input_x_edge(machine_task)
                        machine.add_task_edge(machine_task)
                        machine_task_edges += 1
    for machine in no_task_machines:
        machines.remove(machine)
    print(f"Machines with no tasks: {len(no_task_machines)}")
    print(f"No setup times found: {no_setup_times}")
    return machine_task_edges


def initialize_operator_taks(operators: 'list[Operator]', ONE_MACHINE_PER_SHIFT : bool = False ):
    '''Initialize the operator tasks asignations in graph format
    
    Args:
        operators (list[Operator]): List of operators
        ONE_MACHINE_PER_SHIFT (bool, optional): If True, the operator can only work in one machine per shift. Defaults to False.
        
    Returns:
        int: Number of operator task edges'''
    operator_task_edges = 0
    no_task_operators = []
    shift_duration = 8 * 60
    for operator in operators:
        if len(operator.tasks) == 0:
            no_task_operators.append(operator)
            continue
        for task_i in [operator.start_task]+operator.tasks: # add start task in i (n + w)
            # from_setups = transition_matrix_df[transition_matrix_df['Machine_from']==task_i.machine.name]
            for task_j in [operator.end_task]+operator.tasks: # add end task in j (n + w)
                if (task_i.idx != task_j.idx) and not (task_i.dummy and task_j.dummy): # cant go from task i to i, or from start to end
                    cost = 0
                    time = 0
                    if task_i.normal and task_j.normal:
                        if ONE_MACHINE_PER_SHIFT:
                            if task_i.machine.name != task_j.machine.name:
                                continue
                        #cost = from_setups[from_setups['Machine_to']==task_j.machine.name]['transition_time'].values[0]
                        time = (task_i.job_split.production * task_i.process_time) + (task_j.job_split.production * task_j.process_time)
                    if time <= shift_duration:
                        operator_task_edge = OperatorTask(task_i, task_j, operator, cost)
                        task_i.add_ouput_y_edge(operator_task_edge)
                        task_j.add_input_y_edge(operator_task_edge) 
                        operator.add_task_edge(operator_task_edge)
                        operator_task_edges += 1
                    
    for operator in no_task_operators:
        operators.remove(operator)
    print(f"Operators with no tasks: {len(no_task_operators)}")

    return operator_task_edges


def initialize_operator_chains(operators: 'list[Operator]', max_tasks: int = 3, max_hours: int = 8, max_machines: int = 2):
    '''Initialize the operator task asignations in chain format
    
    Args:
        operators (list[Operator]): List of operators
        max_tasks (int, optional): Maximum number of tasks in a chain. Defaults to 3.
        max_hours (int, optional): Maximum number of hours in a chain. Defaults to 8.
        max_machines (int, optional): Maximum number of machines in a chain. Defaults to 2.
        
    Returns:
        int: Number of operator chains'''
    operator_chains = 0
    no_task_operators = []
    for operator in operators:
        if len(operator.tasks) == 0:
            no_task_operators.append(operator)
            continue
        chains = get_possible_chains(operator.tasks, max_tasks, max_hours, max_machines)
        for chain in chains:
            operator_chain = OperatorChain(operator, chain)
            for task in chain:
                task.add_chain(operator_chain)
            operator.add_chain(operator_chain)
            operator_chains += 1
    for operator in no_task_operators:
        operators.remove(operator)
    print(f"Operators with no tasks: {len(no_task_operators)}")
    return operator_chains

def get_possible_chains(tasks: 'list[Task]', max_tasks: int = 3, max_hours: int = 8, max_machines: int = 2):
    '''
        Function that takes a list of tasks and returns all the possible chains of tasks based on a condition

        Example:
            tasks = [task1, task2, task3]
            max_machines = 2
            max_tasks = 3
            max_hours = 10

            possible_chains = [[task1], [task2], [task3], [task1, task2], [task1, task3], [task2, task1], [task2, task3], [task3, task1], [task3, task2], [task1, task2, task3]]

        Args:
            tasks (list): list of tasks
            max_machines (int): maximum number of machines that can be used in a chain
            max_hours (int): maximum number of hours that can be used in a chain

        Returns:
            combinations (list[list[task]]): list of all the possible chains of tasks
    '''
    combinations = []
    max_time = max_hours * 60
    for i in range(1, max_tasks+1):
        combinations += list(itertools.combinations(tasks, i))
    possible_chains = []
    for combination in combinations:
        time = 0
        machine_changes = 0
        aux_machine = None
        possible_chain = True
        for task in combination:
            if aux_machine is None:
                aux_machine = task.machine.name
                machine_changes += 1
            else:
                if aux_machine != task.machine.name:
                    machine_changes += 1
                    aux_machine = task.machine.name
            time += task.process_time * task.job_split.production
            if time > max_time:
                possible_chain = False
                break
            if machine_changes > max_machines:
                possible_chain = False
                break
        if possible_chain:
            possible_chains.append(combination)
    return possible_chains