from src.operator import Operator, OperatorTask
from src.machine import Machine, MachineTask
from src.job import Job
from src.task import Task
import numpy as np
import pandas as pd
import math


class Generator:

    def __init__(self, J: int, M: int, O: int, seed: int = None):

        if seed is not None:
            np.random.seed(seed)
            
        self.J = J
        self.M = M
        self.O = O

        self._u: pd.DataFrame = pd.DataFrame(columns=['job', 'unit'])
        self._st: pd.DataFrame = pd.DataFrame(
            columns=['job_from', 'job_to', 'machine', 'setup_time'])
        self._pt: pd.DataFrame = pd.DataFrame(
            columns=['job', 'machine', 'processing_time'])
        self._r: pd.DataFrame = pd.DataFrame(
            columns=['job', 'machine', 'resources'])
        self._tr: pd.DataFrame = pd.DataFrame(
            columns=['machine_1', 'machine_2', 'transition_time'])
        self._sk: pd.DataFrame = pd.DataFrame(columns=['operator', 'machine'])
        self._tm: pd.DataFrame = pd.DataFrame(columns=['operator', 'shift'])

    def units(self, min: int, max: int):
        for j in range(1, self.J+1):
            unit = np.random.randint(min, max)
            row = {'job': j, 'unit': unit}
            self._u = pd.concat([self._u, pd.DataFrame([row])], ignore_index=True)

    def setup_times(self, job_machine_min: float, job_machine_max: float, min: int, max: int):
        machine_job = pd.DataFrame(columns=['machine', 'job'])
        rows = []
        for j in range(1, self.J+1):
            in_machines = math.ceil(np.random.uniform(
                job_machine_min, job_machine_max)*self.M)

            machines = np.random.choice(self.M, in_machines, replace=False)
            
            for m in machines:
                rows.append({'machine': m, 'job': j})
        
        machine_job = pd.concat([machine_job, pd.DataFrame(rows)], ignore_index=True)

        rows = []
        for m in range(self.M):
            jobs = machine_job[machine_job['machine'] == m]['job'].values
            for j in range(len(jobs)):
                for i in range(len(jobs)):
                    if i != j:
                        rows.append({'job_from': jobs[i], 'job_to': jobs[j], 'machine': m, 'setup_time': np.random.randint(min, max)})
                        
                    else:
                        rows.append({'job_from': jobs[i], 'job_to': jobs[j], 'machine': m, 'setup_time': 0})
        self._st = pd.concat([self._st, pd.DataFrame(rows)], ignore_index=True)
        # Drop duplicates and keep first
        self._st = self._st.drop_duplicates(
            subset=['job_from', 'job_to', 'machine'], keep='first')

    def processing_times(self, min: int, max: int):
        rows = []
        for m in range(self.M):
            m_j = self._st[self._st['machine'] == m]['job_from'].values
            for j in m_j:
                rows.append({'job': j, 'machine': m, 'processing_time': np.random.randint(min, max)})
        
        self._pt = pd.concat([self._pt, pd.DataFrame(rows)], ignore_index=True)
        # Drop duplicates and keep first
        self._pt = self._pt.drop_duplicates(
            subset=['job', 'machine'], keep='first')

    def resources(self, min: float, max: float):
        rows = []
        for m in range(self.M):
            m_j = self._st[self._st['machine'] == m]['job_from'].values
            for j in m_j:
                resource = np.random.uniform(min, max)
                if resource > 1:
                    resource = int(resource)
                else:
                    resource = 0.5
                rows.append({'job': j, 'machine': m, 'resources': resource})
        
        self._r = pd.concat([self._r, pd.DataFrame(rows)], ignore_index=True)
        # Drop duplicates and keep first
        self._r = self._r.drop_duplicates(
            subset=['job', 'machine'], keep='first')

    def transition_time(self, max: int, prob_near: float):
        rows = []
        for m in range(self.M):
            for z in range(m, self.M):  # Without repeating the same machine
                if np.random.uniform(0, 1) < prob_near:
                    rows.append({'machine_1': m, 'machine_2': z, 'transition_time': 0})
                    rows.append({'machine_1': z, 'machine_2': m, 'transition_time': 0})
                elif z != m:
                    tr =  np.random.randint(0, max)
                    rows.append({'machine_1': m, 'machine_2': z, 'transition_time': tr})
                    rows.append({'machine_1': z, 'machine_2': m, 'transition_time': tr})
                else:
                    rows.append({'machine_1': m, 'machine_2': z, 'transition_time': 0})
                    rows.append({'machine_1': z, 'machine_2': m, 'transition_time': 0})
        
        self._tr = pd.concat([self._tr, pd.DataFrame(rows)], ignore_index=True)
        # Drop duplicates and keep first
        self._tr = self._tr.drop_duplicates(
            subset=['machine_1', 'machine_2'], keep='first')

    def skill_matrix(self, operator_machine_min: float, operator_machine_max: float,
                        min_operators_per_machine: int):
        rows = []
        for o in range(self.O):
            skills = math.ceil(np.random.uniform(
                operator_machine_min, operator_machine_max)*self.M)

            machines = np.random.choice(self.M, skills, replace=False)
            for m in machines:
                rows.append({'operator': o, 'machine': m})
        self._sk = pd.concat([self._sk, pd.DataFrame(rows)], ignore_index=True)

        rows = []
        for m in range(self.M):
            operators = self._sk[self._sk['machine'] == m]['operator'].values
            if len(operators) < min_operators_per_machine:
                for i in range(min_operators_per_machine - len(operators)):
                    rows.append({'operator': np.random.randint(0, self.O), 'machine': m})
        self._sk = pd.concat([self._sk, pd.DataFrame(rows)], ignore_index=True)
        #Drop duplicates and keep first
        self._sk = self._sk.drop_duplicates(
            subset=['operator', 'machine'], keep='first')

    def operator_timetable(self, t1_number: float, t2_number: float, t3_number: float):
        t1 = math.ceil(self.O * t1_number)
        t2 = math.ceil(self.O * t2_number)
        rows = []
        for o in range(self.O):
            if o < t1:
                rows.append({'operator': o, 'shift': 1})
            elif o < t1 + t2:
                rows.append({'operator': o, 'shift': 2})
            else:
                rows.append({'operator': o, 'shift': 3})
        self._tm = pd.concat([self._tm, pd.DataFrame(rows)], ignore_index=True)

    def reindex(self):
        self._u = self._u.set_index('job')
        self._st = self._st.set_index(['job_from', 'job_to', 'machine'])
        self._pt = self._pt.set_index(['job', 'machine'])
        self._r = self._r.set_index(['job', 'machine'])
        self._tr = self._tr.set_index(['machine_1', 'machine_2'])
        self._sk = self._sk.set_index(['operator', 'machine'])
        self._tm = self._tm.set_index(['operator'])

    def u(self, j: int):
        try:
            return self._u.loc[j][0]
        except KeyError:
            return None

    def st(self, j: int, i: int, m: int):
        try:
            return self._st.loc[j, i, m][0]
        except KeyError:
            return None

    def pt(self, j: int, m: int):
        try:
            return self._pt.loc[j, m][0]
        except KeyError:
            return None

    def r(self, j: int, m: int):
        try:
            return self._r.loc[j, m][0]
        except KeyError:
            return None

    def tr(self, m1: int, m2: int):
        try:
            return self._tr.loc[m1, m2][0]
        except KeyError:
            return None

    def sk(self, o: int, m: int):
        try:
            return self._sk.loc[o, m][0]
        except KeyError:
            return None

    def tm(self, o: int):
        try:
            return self._tm.loc[o][0]
        except KeyError:
            return None

    def job_machines(self, j: int):
        try:
            st = self._st.reset_index()
            machines = st[st['job_from'] == j]['machine'].unique()
            return machines
        except KeyError:
            return None

    def operator_machines(self, o: int):
        try:
            sk = self._sk.reset_index()
            machines = sk[sk['operator'] == o]['machine'].unique()
            return machines
        except KeyError:
            return None

    def machine_operators(self, m: int):
        try:
            sk = self._sk.reset_index()
            operators = sk[sk['machine'] == m]['operator'].unique()
            return operators
        except KeyError:
            return None

    def machine_jobs(self, m: int):
        try:
            pt = self._pt.reset_index()
            jobs = pt[pt['machine'] == m]['job'].unique()
            return jobs
        except KeyError:
            return None


def initializer(g: Generator):
    jobs: 'list[Job]' = []
    machines: 'list[Machine]' = []
    operators: 'list[Operator]' = []
    tasks: 'list[Task]' = []
    print('Generating jobs...')
    for j in range(g.J):
        job = Job(idx=j, reference='J'+str(j),
                  day=0, production=g.u(j), cost=1)
        jobs.append(job)

    print('Generating machines...')
    for m in range(g.M):
        machine = Machine(idx=m, name='M'+str(m))
        machines.append(machine)

    print('Generating operators...')
    for o in range(g.O):
        shift = g.tm(o)
        start_time = (shift-1) * (8 * 60)
        end_time = shift * (8 * 60)
        operator = Operator(idx=o, name='O'+str(o), start_time=start_time, end_time=end_time)
        operators.append(operator)
        
    print('Generating tasks...')
    for job in jobs:
        job_machines = g.job_machines(job.idx)
        for m in job_machines:
            machine: Machine = machines[m]
            resource = g.r(job.idx, machine.idx)
            process_time = g.pt(job.idx, machine.idx)
            task = Task(kind="normal",
                        job=job,
                        machine=machine,
                        process_time=process_time,
                        mod=resource)
            job.add_task(task)
            machine.add_task(task)
            tasks.append(task)
            
            machine_operators = g.machine_operators(machine.idx)
            for o in machine_operators:
                operators[o].tasks.append(task)

    N = len(tasks)
    print(f'N = {N}')
    print('Generating x tasks...')
    no_task_machines = []
    x_edges = 0
    for machine in machines:
        if len(machine.tasks) == 0:
            no_task_machines.append(machine)
        # add start task in i (n + m)
        for task_i in [machine.start_task]+machine.tasks:
            for task_j in [machine.end_task]+machine.tasks:
                if task_i.idx != task_j.idx:
                    cost = 0
                    if task_i.normal and task_j.normal:
                        cost = g.st(task_i.job.idx,
                                    task_j.job.idx, machine.idx)
                        if cost is None:
                            raise Exception(
                                f"No setup cost for job {task_i.job.idx} and job {task_j.job.idx} on machine {machine.idx}")
                    
                    machine_task = MachineTask(
                        task_i, task_j, machine=machine, setup_cost=cost)
                    task_i.add_ouput_x_edge(machine_task)
                    task_j.add_input_x_edge(machine_task)
                    machine.add_task_edge(machine_task)
                    x_edges += 1

    for machine in no_task_machines:
        machines.remove(machine)
    print('Generating y tasks...')
    no_task_operator = []
    y_edges = 0
    for operator in operators:
        if len(operator.tasks) == 0:
            no_task_operator.append(operator)
        for task_i in [operator.start_task] + operator.tasks:
            for task_j in [operator.end_task] + operator.tasks:
                if task_i.idx != task_j.idx:
                    cost = 0
                    if task_i.normal and task_j.normal:
                        cost = g.tr(task_i.machine.idx, task_j.machine.idx)
                        if cost is None:
                            raise Exception(
                                f"No transition cost for machine {task_i.machine.idx} and machine {task_j.machine.idx}")
                            
                    operator_day_task = OperatorTask(
                        task_i, task_j, operator, cost)
                    task_i.add_ouput_y_edge(operator_day_task)
                    task_j.add_input_y_edge(operator_day_task)
                    operator.add_task_edge(operator_day_task)
                    y_edges += 1

    return jobs, machines, operators, operator, tasks, N, x_edges, y_edges