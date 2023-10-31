import itertools

START_HOUR = 0 
class Job:
    ''' Documentation in pydoc format

    Job class that represents a job in the production plan

    Attributes:
        idx (int): index in the job , matrix or list
        reference (str): reference of the job
        day (int): production day
        production (int): production quantity
        cost (float): unit cost
        start_lb (int): lower bound of the start time (in minutes)
        start_ub (int): upper bound of the start time
        tasks (list[Tasks]): list of tasks of the job

    '''
    def __init__(self, idx: int, reference: str, day: int, production: int, cost: float):
        ''' Job constructor

        Args:
            idx (int): index in the job , matrix or list
            reference (str): reference of the job
            day (int): production day
            production (int): production quantity
            cost (float): unit cost

        '''
        self.idx = idx # index in the job , matrix or list
        self.reference = reference
        self.day = day
        self.production = production
        self.cost = cost
        self.start_lb =  START_HOUR + self.day * 1440
        self.start_ub = None
        self.tasks : 'list[Task]' = []
        

    def add_task(self, task):
        '''Add a task to the job split'''
        self.tasks.append(task)

    def lock_start(self, day: int, hour: int):
        '''Lock the start time of the job

        Args:
            day (int): day of the start time
            hour (int): hour of the start time
        '''

        self.start_lb = day * 1440 + hour * 60
        self.start_ub = day * 1440 + hour * 60
    

    def __repr__(self):
        '''Job representation'''
        return f"Job {self.idx}: ref {self.reference}, day {self.day}, production {self.production}, cost {self.cost}"

class Job_Split:
    '''Job split class that represents a job split in the production plan
    
    Attributes:
        idx (int): index in the job , matrix or list
        job (Job): job to which the split belongs
        production (int): production quantity
        tasks (list[Task]): list of tasks of the split
        doplex_var (Doplex Interval Variable): doplex variable
    
    '''
    count = itertools.count()

    def __init__(self, job: Job, production: int):
        '''Job split constructor

        Args:
            job (Job): job to which the split belongs
            production (int): production quantity
        '''
        self.idx = next(Job_Split.count)
        self.job = job
        self.production = production
        self.doplex_var = None
        self.tasks = []

    def add_task(self, task):
        '''Add a task to the job split'''
        self.tasks.append(task)
    
    def __repr__(self):
        return f"Job-split {self.idx}: {self.production}, {self.job}"


