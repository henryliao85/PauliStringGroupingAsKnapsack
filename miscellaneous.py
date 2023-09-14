import math
import numpy as np
from numpy.random import randint


Dir = str
Operator = str
Operators = list[Operator]


def load_operators(dir: Dir) -> Operators:
    with open(dir, "r+") as f:
        return eval(f.read())


# 1: do not commute
# 0: commute
def compare_operators(x: Operator, y: Operator) -> int:
    return (
        1
        # commute = 1, noncommute = -1
        - math.prod(
            [1 if xx == yy or xx == "I" or yy == "I" else -1 for xx, yy in zip(x, y)]
        )
    ) // 2


def num_of_noncommuting_in_same_group(e: Operator, g: Operators) -> int:
    return sum([compare_operators(e, gg) for gg in g])


def random_pauli_string(len: int, num: int, dir: Dir = "OperatorSet.txt") -> None:
    CHARS = ["i", "x", "y", "z"]
    with open(dir, "w+") as f:
        f.write(
            str(["".join([CHARS[rr] for rr in r]) for r in randint(4, size=(num, len))])
        )
