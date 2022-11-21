# import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from GridWorld import GridWorld
from math import prod
import time

TUHE = np.array([[1,0,0,0,0,2,0,1,0,1],
                [1,0,0,0,0,0,1,0,1,0],
                [1,1,0,0,1,0,0,1,1,1],
                [0,0,0,0,1,0,0,1,0,1],
                [1,1,0,1,1,0,1,1,1,0],
                [1,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [1,0,1,0,0,1,0,0,0,1],
                [0,1,1,0,0,0,0,0,1,1],
                [1,0,1,0,1,3,1,1,0,1]])

n_steps_givup = 40  # Number of steps before giving up
N_EPISODES = 1000  # Total number of training episodes 

learning_rate = 3e-2
gamma = 0.99
seed = 0#543
fps = 0
render = True
if fps:
    render = True
log_interval = 40

wall_pct = 0.0
map_size = 4
map = [map_size]*4
non_diag = True


if seed:
    env = GridWorld(wall_pct=0.6, seed=seed, map=map, non_diag=True, resetter=0, space_fun=GridWorld.test)
    torch.manual_seed(seed)
else:
    env = GridWorld(wall_pct=0.6, map=map, non_diag=True, resetter=0, space_fun=GridWorld.test)
env.reset()

# env.reset_to(TUHE)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        hidden_units = 32
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)

        # actor's layer
        self.action_head = nn.Linear(hidden_units, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(hidden_units, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = state.flatten()
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns) #laver returns om til torch tensors
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()  #calculating advantage, value.item() = the state value we got

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)      #

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))    #l1 smoothed absolute error

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # show updated action weights
    # print(model.get_parameter("action_head.weight"))
    # grads = []
    # for name, param in model.named_parameters():
    #     print(name, param)
        # grads.append(param.view(-1))
    # grads = torch.cat(grads)
    # print()


    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 0

    # run infinitely many episodes
    for i_episode in range(N_EPISODES): #count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, n_steps_givup):

            # select action from policy
            # print(f"{i_episode}, {t} - selecting action")
            action = select_action(state)

            # take the action
            if render:
                time.sleep(fps)
            state, reward, done = env.step(action)
            

            if render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        # print(f"{i_episode} - finishing episode")
        finish_episode()

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

            if abs(running_reward - 1.00) < eps and is_solved(100):
            # if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break
    
def play():
    # env = GridWorld(map_size=(4,4,5,5), render=True, rewards=(0.0, 100.0))
    model.eval()
    env.render = True
    state = env.reset()
    wins = 0
    total = 0

    # for i in range(100):
    i = 0
    while True:
        # pick best action
        state = state.flatten()
        state = torch.from_numpy(state).float()
        probs, _ = model(state)
        action = probs.argmax().item()

        # take action
        time.sleep(0.1)
        state, reward, done = env.step(action)

        i += 1
        if done or i > 50:  # Complete or give up, max 50 steps
            state = env.reset()
            if i <= 50: 
                wins += 1
            total += 1
            i = 0
            print(f'wins: {wins}/{total}')
            if total == 100:
                break

def is_solved(eps=100):
    """Convergence test over arg 'eps' episodes

       returns true If it can get 100 wins in a rough without using 50 or more, steps
    """

    model.eval()
    state = env.reset()
    wins = 0
    total = 0

    i = 0
    while True:
        # pick best action
        state = state.flatten()
        state = torch.from_numpy(state).float()
        probs, _ = model(state)
        action = probs.argmax().item()

        # take action
        state, reward, done = env.step(action)

        i += 1
        if done:  # Complete
            state = env.reset()            
            wins += 1
            i = 0
            if wins == eps:
                model.train()
                return True
        elif i > 50:
            model.train()
            print(f'Failed evaluation: {wins}/{eps}')
            return False




if __name__ == '__main__':
    main()       #training the model until convergence
    play()        #evaluation/testing the final model, renders the output