from src.generator import Generator

def dataset_generator():
    '''Return a generator with the data defined in this function. This generator will be given as input to the Environment for generating all the datasets needed. '''
    # GENERATE THE DATASET
    J = 8
    M = 4
    O = 40

    # Units
    u_min = 1
    u_max = 2

    # Setup times
    job_machine_min = 0.1 
    job_machine_max = 0.4
    st_min = 0
    st_max = 10

    # Processing times
    pt_min = 40
    pt_max = 70

    # Resources
    r_min = 0.5
    r_max = 2

    # Transition times
    tr_max = 5
    prob_near = 0.2

    # Skill matrix
    operator_machine_min = 0.1
    operator_machine_max = 0.2
    min_operators_in_machine = 3

    # Operator timetable
    t1_number = 0.3
    t2_number = 0.3
    t3_number = 0.3

    g = Generator(J, M, O, seed = 3)
    g.units(u_min, u_max)
    g.setup_times(job_machine_min, job_machine_max, st_min, st_max)
    g.processing_times(pt_min, pt_max)
    g.resources(r_min, r_max)
    g.transition_time(tr_max, prob_near)
    g.skill_matrix(operator_machine_min, operator_machine_max,
                min_operators_in_machine)
    g.operator_timetable(t1_number, t2_number, t3_number)
    return g