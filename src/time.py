import itertools


class Time:
    '''Time class that represents a time in the production plan
    
    Attributes:
        idx (int): index in the time , matrix or list
        lb (int): lower bound of the time (in minutes)
        ub (int): upper bound of the time (in minutes)
        var (str): cplex variable
        pulp_var (Pulp Variable): pulp variable'''
    def __init__(self, idx: int, lb: int = 0, ub: int = None, var: int = None):
        '''Time constructor

        Args:
            idx (int): index in the time , matrix or list
            lb (int): lower bound of the time (in minutes)
            ub (int): upper bound of the time (in minutes)
        '''
        self.idx = idx 
        self.lb = lb if lb is None else [lb]
        self.ub = ub if ub is None else [ub]
        self.pulp_var = None
        if var is None:
            self.var = f"t_{self.idx}" # cplex variable 
        else:
            self.var = var
    
    def __repr__(self):
        return f"time {self.var}: {self.lb} - {self.ub}"
