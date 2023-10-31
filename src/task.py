import itertools

from src.job import Job
from src.time import Time


class Task:
    '''Task class that represents a task in the production plan
    
    Attributes:
        idx (int): index in the task , matrix or list
        kind (str): kind of the task, starting, normal or ending
        machine (Machine): machine to which the task belongs [Optional]
        job (Job): job to which the task belongs [Optional]
        time (Time): start time of the task
        asignation_var (str): cplex asignation variable of the task
        var (str): cplex variable of the task
        process_time (int): unit process time of the task (in minutes)
        mod (float): Required working capacity of the task
        pulp_var (Pulp Variable): pulp variable of the task
        doplex_var (Doplex Interval Variable): doplex variable of the task
        input_x_edges (list[MachineTask]): list of input x edges of the task [Optional]
        output_x_edges (list[MachineTask]): list of output x edges of the task [Optional]
        input_y_edges (list[OperatorTask]): list of input y edges of the task [Optional]
        output_y_edges (list[OperatorTask]): list of output y edges of the task [Optional]
        worker_tasks (list[WorkerTaskSecuence]): list of worker-tasks doplex asignations [Optional]
        chains (list[OperatorChain]): list of chains of the task [Optional]
        '''
    count = itertools.count()

    def __init__(self, kind: str, machine: object = None, job: Job = None, process_time: int = 0, mod: float = None):
        '''Task constructor
        
        Args:
            kind (str): kind of the task, starting, normal or ending
            machine (Machine): machine to which the task belongs [Optional]
            job (job): job split to which the task belongs [Optional]
            process_time (int): unit process time of the task (in minutes)
            mod (float): Required working capacity of the task

        '''
        self.idx = next(Task.count)
        self.job = job
        self.machine = machine
        self.kind = kind # starting, normal, ending
        self.time = None
        self.asignation_var = None
        self.var = kind        
        self.process_time = process_time
        self.mod = mod
        self.pulp_var = None
        self.doplex_var = None
        
        if self.normal: # we don't need asignation and production for starting and ending tasks
            self.var = f"m{self.machine.name}_js{self.job.idx}_t{self.idx}"
            self.asignation_var = f"a_{self.idx}" # cplex variable
            self.time = Time(self.idx, self.job.start_lb, self.job.start_ub) 
            self.production = Production(self.idx, self.job.production)

        self.input_x_edges = []
        self.output_x_edges = []
        self.input_y_edges = []
        self.output_y_edges = []
        self.worker_tasks = []
        self.chains = []
    
    @property
    def starting(self):
        '''Returns True if the task is a starting task, False otherwise
        
        Returns:
            bool: True if the task is a starting task, False otherwise'''
        return self.kind == "starting"

    @property
    def normal(self):
        '''Returns True if the task is a normal task, False otherwise
        
        Returns:
            bool: True if the task is a normal task, False otherwise'''
        return self.kind == "normal"

    @property
    def ending(self):
        '''Returns True if the task is a ending task, False otherwise
        
        Returns:
            bool: True if the task is a ending task, False otherwise'''
        return self.kind == "ending"

    @property
    def dummy(self):
        '''Returns True if the task is a starting or ending task, False otherwise
        
        Returns:
            bool: True if the task is a starting or ending task, False otherwise'''
        return self.kind in ["starting", "ending"]

    def add_input_x_edge(self, edge):
        '''Adds an input x edge to the task
        
        Args:
            edge (MachineTask): input x edge to add'''
        self.input_x_edges.append(edge)
    
    def add_ouput_x_edge(self, edge):
        '''Adds an output x edge to the task
        
        Args:
            edge (MachineTask): output x edge to add'''
        self.output_x_edges.append(edge)

    def add_input_y_edge(self, edge):
        '''Adds an input y edge to the task
        
        Args:
            edge (OperatorTask): input y edge to add'''
        self.input_y_edges.append(edge)
    
    def add_ouput_y_edge(self, edge):
        '''Adds an output y edge to the task
        
        Args:
            edge (OperatorTask): output y edge to add'''
        self.output_y_edges.append(edge)
    
    def add_chain(self, chain):
        '''Adds a chain to the task
        
        Args:
            chain (OperatorChain): chain to add
        '''
        self.chains.append(chain)
        
    def add_worker_task(self, worker_task):
        '''Adds a worker-task doplex asignation to the task
        
        Args:
            worker_task (WorkerTaskSecuence): worker-task doplex asignation to add
        '''
        self.worker_tasks.append(worker_task)
    def set_time(self, time):
        '''Sets the time of the task

        Args:
            time (Time): time to set
        '''
        self.time = time
        
    def __repr__(self):
        if self.dummy:
            return f"Task {self.idx} {self.kind} with {self.time}"
        else:
            return f"Task {self.idx} {self.kind} of machine {self.machine.idx} of job {self.job.idx} with {self.time}"


class Production:

    def __init__(self, task_idx : Task, max_production : int):
        self.var = f"up_{task_idx}"
        self.lb = [0]
        self.ub = [max_production]
