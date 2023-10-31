from src.task import Task
from src.time import Time

class Operator:
    ''' Operator class that represents a shift of worker in the production plan
    
    Attributes:
        idx (int): index in the operator , matrix or list
        name (str): name of the operator
        start_time (float): start time of the shift (in minutes)
        end_time (float): end time of the shift (in minutes)
        start_task (Task): starting task of the operator
        end_task (Task): ending task of the operator
        tasks (list[Task]): list of tasks of the operator
        task_edges (list[OperatorTask]): list of operator tasks asignations of the operator
        chains (list[Chain]): list of chains of the operator
        activation_var (str): activation Cplex variable of the operator
        '''
    
    def __init__(self,  idx: int, name: str, start_time: int, end_time: int):
        '''Operator constructor
        
        Args:
            worker (Worker): worker of the operator
            start_time (int): start time of the shift (in minutes)
            end_time (int): end time of the shift (in minutes)
        '''
        self.idx = idx
        self.name = name
        self.start_time = start_time
        self.end_time = end_time

        self.start_task : Task = Task("starting")
        self.start_task.set_time(Time(f"o{self.idx}_start", self.start_time, self.start_time))
        self.end_task : Task = Task("ending")
        self.end_task.set_time(Time(f"o{self.idx}_end", self.end_time, self.end_time)) 
        
        self.activation_var = f"c_{self.idx}"
        self.tasks : 'list[Task]' = []
        self.task_edges : 'list[OperatorTask]' = []

    def add_task(self, task: Task):
        '''Add a task to the operator

        Args:
            task (Task): task to add
        '''
        self.tasks.append(task)
        
    def add_task_edge(self, edge):
        '''Add a task edge to the operator

        Args:
            edge (OperatorTask): task edge to add
        '''
        self.task_edges.append(edge)


    def __repr__(self):
        return f"Operator {self.idx} {self.name} [{self.start_time, self.end_time}] with tasks: {len(self.tasks)}, task edges: {len(self.task_edges)}"
    
class OperatorTask:
    '''Operator task class that represents a task asignation of worker in a shift. It is a edge of the asignation graph.
    

    Attributes:
        tail (Task): from task
        head (Task): to task
        operator (Operator): operator of the task
        cost (float): cost of moving from the machine of the tail task to the machine of the head task
        var (str): Cplex variable of the task
        pulp_var (pulp variable): pulp variable of the task
    '''

    def __init__(self, task_i: Task, task_j: Task, operator : Operator,  travel_cost: float):
        self.tail = task_i # From task i
        self.head = task_j # To task j
        self.operator = operator
        self.cost = travel_cost
        self.var = f"y_{self.tail.idx}_{self.head.idx}_{self.operator.idx}"
        self.binary_var = f"b_{self.tail.idx}_{self.head.idx}_{self.operator.idx}"
        self.pulp_var = None

    def __repr__(self):
        return f"Operator task of {self.operator} from {self.tail} to {self.head} with cost {self.cost}"
