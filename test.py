
from GridWorld import GridWorld
from math import prod
from itertools import count
import numpy as np
# env = GridWorld()
# env.reset()
# print(env.observation_space.shape)
import math
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical





class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        hidden_units = 32
        hidden_units2 = 64
        self.affine1 = nn.Linear(3, hidden_units)
        self.affine2 = nn.Linear(hidden_units, hidden_units2)
        self.affine15 = nn.Linear(hidden_units, hidden_units)
        # critic's layer
        self.value_head = nn.Linear(hidden_units2, 1)
        self.v_current = torch.zeros((2,2), requires_grad=False)
        self.v_next = torch.zeros((2,2), requires_grad=False)
    def forward(self, x):
        """
        forward of both actor and critic
        """



        x = F.relu(self.affine1(x))
        """i = 0
        while i < 20:
            x = F.relu(self.affine15(x))
            i += 1"""

        for i in range(3):
            x = F.relu(self.affine15(x))
            #x[0] = 10 #illegal
            #self.v_next[0, 0] = torch.max(x)  #this CANT be gradient descendet on
            #K = torch.max(x)  #this CAN be gradient descendet on
            self.v_next[0, 0] = x[0]






        """
        if torch.max(x) < 10:
            x = x+1 #this is legal
            #x[:] = 50   #this shit is illegal
        """
        x = F.relu(self.affine2(x* self.g[0]))

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return state_values
model = Policy()
decay_rate = 0.001 / 100
optimizer = optim.Adam(model.parameters(), lr=0.001)

# reset gradients
optimizer.zero_grad()
k = 0
# sum up all the values of policy_losses and value_losses
for i in range(10000):
    output = model(torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32))
    loss = (output - 10)**2
    if i%1000==0 and i > 0:
        k += 1
        print(loss)
        if k>4:
            k = 4
        optimizer = optim.Adam(model.parameters(), lr=0.001/math.exp(k))
        print("learning rate is", 0.001/math.exp(k*0.3))

        print("output is ", output)
# perform backprop

    loss.backward(retain_graph=True)   #added retain_graph=True because of regularization
    optimizer.step()
