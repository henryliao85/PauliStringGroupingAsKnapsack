# Multithreading
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

# Basic Packages
import time
import pandas as pd

# SOO Algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from miscellaneous import load_operators


# Hyperparam for Multithreading
N_THREADS = 48
POOL = ThreadPool(N_THREADS)
RUNNER = StarmapParallelization(POOL.starmap)

POP_SIZE = 1000
N_OFFSPRINGS = 500
N_GEN = 10_000

FILENAME = "test.csv"


Operator = str
Operators = list[Operator]
OperatorIndex = int
OperatorIndices = list[OperatorIndex]
GroupIndex = int
GroupIndices = list[GroupIndex]

DIR = "./OperatorSet.txt"
OPS = [e.upper() for e in load_operators(DIR)]

NUM_OPERATORS = len(OPS)
NUM_MAX_GROUP = NUM_OPERATORS

XL = np.zeros(NUM_OPERATORS)
XU = NUM_MAX_GROUP * np.ones(NUM_OPERATORS)


def lookup_table():
    pass


def is_string_not_commuting(g: Operator, ge: Operator) -> bool:
    if g == ge:
        return False
    v = False
    for s1, s2 in zip(g, ge):
        if s1 == "I" or s2 == "I" or s1 == s2:
            continue
        v = not v
    return v


def num_of_noncommuting_in_same_group(g: Operator, gg: Operators) -> int:
    return sum([1 if is_string_not_commuting(g, ge) else 0 for ge in g])


def nums_of_noncommuting(x: GroupIndices) -> list[int]:
    # [sum([1 if gid1 == gid2 and is_string_not_commuting(op1,op2) else 0 for op2, gid2 in (OPS, x) ]) for op1,gid1 in (OPS, x)]
    rec = []
    t = [list(i) for i in zip(*[OPS, x])]
    for op1, gid1 in t:
        count = 0
        for op2, gid2 in t:
            if gid1 == gid2 and is_string_not_commuting(op1, op2):
                count += 1
        rec += [count]
    return rec


class MyProblem(ElementwiseProblem):
    def __init__(self, elementwise=True, **kwargs):
        super().__init__(
            n_var=NUM_OPERATORS,
            n_obj=1,
            n_ieq_constr=NUM_OPERATORS,
            xl=XL,
            xu=XU,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = len(set(x))
        out["G"] = nums_of_noncommuting(list(x))


def main():
    # define the problem by passing the starmap interface of the thread pool
    problem = MyProblem(elementwise_runner=RUNNER)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        n_offsprings=N_OFFSPRINGS,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", N_GEN)

    start = time.time()

    res = minimize(
        problem, algorithm, termination, seed=10, save_history=True, verbose=True
    )

    end = time.time()

    print(f"=================================")
    print(f"Runtime: {end-start} seconds!!!")
    print(f"=================================")

    X = res.X
    F = res.F
    print(X)
    print(F)

    columns = OPS
    df = pd.DataFrame(X, columns=columns)
    df.to_csv(FILENAME, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
