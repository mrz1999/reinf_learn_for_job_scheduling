
from src.task import Task
from src.time import Time


class Machine:
    '''Machine class that represents a machine in the production plan

    Attributes:
        idx (int): index in the machine , matrix or list
        name (str): name of the machine
        start_task (Task): starting task of the machine
        end_task (Task): ending task of the machine
        tasks (list[Task]): list of tasks of the machine
        task_edges (list[MachineTask]): list of machine tasks of the machine
        secuence_var (Doplex Secuence Variable): doplex secuence variable of the machine
        doplex_var (Doplex Interval Variable): doplex variable
    '''
    def __init__(self, idx: int, name: str, start_time: int = 0):
        '''Machine constructor

        Args:
            idx (int): index in the machine , matrix or list
            name (str): name of the machine
            start_time (int): start time of the machine
        '''
        self.idx = idx
        self.name = name
        self.start_task = Task("starting")
        self.start_task.set_time(Time(f"m{self.idx}_start", start_time, None)) # start time of the machine
        self.end_task = Task("ending")
        self.end_task.set_time(Time(f"m{self.idx}_end", start_time, None)) # end time of the machine
        self.tasks : 'list[Task]' = []
        self.task_edges :'list[MachineTask]' = []
        self.doplex_var = None
        self.secuence_var = None

    def add_task(self, task: Task):
        '''Add a task to the machine

        Args:
            task (Task): task to add
        '''
        self.tasks.append(task)
        
    def add_task_edge(self, edge):
        '''Add a task edge to the machine
        
        Args:
            edge (MachineTask): task edge to add
        '''
        self.task_edges.append(edge)

    def __repr__(self):
        return f"Machine {self.idx} {self.name}, tasks: {len(self.tasks)}, task edges: {len(self.task_edges)}"

class MachineTask:
    '''Machine task class that represents a machine task asignation, in a graph format.
    
    Attributes:
        tail (Task): from task
        head (Task): to task
        machine (Machine): machine of the tasks
        cost (float): setup cost from task tail to task head
        var (str): cplex variable name
        pulp_var (pulp variable): pulp variable
        doplex_var (Doplex Interval Variable): doplex variable

    '''
    def __init__(self, task_i: Task, task_j: Task, machine: Machine, setup_cost: float):
        '''Machine task constructor
        
        Args:
            task_i (Task): from task
            task_j (Task): to task
            machine (Machine): machine of the tasks
            setup_cost (float): setup cost from task tail to task head
            
        '''
        self.tail = task_i # From task i
        self.head = task_j # To task j
        self.machine = machine
        self.cost = setup_cost
        self.var = f"x_{self.tail.idx}_{self.head.idx}"
        if self.tail.dummy and self.head.dummy:
            self.var += f"_{self.machine.name}"
        self.pulp_var = None
        self.doplex_var = None

    def __repr__(self):
        return f"From {self.tail} to {self.head} with cost {self.cost}"

