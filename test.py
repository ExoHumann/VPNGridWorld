
from GridWorld import GridWorld
from math import prod
from itertools import count
import numpy as np
# env = GridWorld()
# env.reset()
# print(env.observation_space.shape)

A = np.random.binomial(1, wall_pct, size=(2, 2))

print(A)