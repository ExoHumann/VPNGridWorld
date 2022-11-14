
import argparse
import gym

import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from GridWorld import GridWorld
from math import prod
import time

Map = np.array([[1, 0, 0, 0, 0, 2, 0, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                 [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                 [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 0, 1, 0, 1, 3, 1, 1, 0, 1]])

n_steps_givup = 40  # Number of steps before giving up  #max steps allowed in train2
N_EPISODES = 20000  # Total number of training episodes
K = 10
test_size = 100 #number of test attempts
learning_rate = 3e-2
gamma = 0.99
seed = 0  # 543
max_allowed_steps = n_steps_givup #max steps allowed in test
allowed_action_dict = {-1: -1, 0: -1, 1: -1, -1: 0, 0: 0, 1: 0, -1: 1, 0: 1,
                       1: 1}  # i left, j right #staying at the same place is allowed

fps = 0
render = False
renderTest = True
if fps:
    render = True
log_interval = 40

wall_pct = 0.0
map_size = 5
map_size = [map_size] * 4
non_diag = False


# env = gym.make('CartPole-v1', render_mode="rgb_array")
if seed:
    env = GridWorld(map_size=map_size, seed=seed, render=render, non_diag=non_diag, rewards=(0.0, 1.0),
                    wall_pct=wall_pct)
    torch.manual_seed(seed)
else:
    env = GridWorld(map_size=map_size, render=render, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)
env.reset()

env.reset_to(Map)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Embedding(nn.Module):
    """Maps input to latent space?"""
    def __init__(self):
        super(Embedding, self).__init__()
        hidden_units = 32
        hidden_units2 = 64
        output_dims = env.observation_space.shape[1]*env.observation_space.shape[2]

        #input should contain
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)
        self.affine2 = nn.Linear(hidden_units, hidden_units2)


        # r_outs's head
        self.r_out = nn.Linear(hidden_units2, output_dims)

        # r_in's head
        self.r_in = nn.Linear(hidden_units2, output_dims)

        # transition probability head
        #self.p = nn.Linear(hidden_units2, output_dims)


        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.v_current = torch.zeros(size=output_dims)
        self.v_next = torch.zeros(size=output_dims)

        #self.values = np.zeros(())
    def forward(self, x):
        """

        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))


        r_out = self.r_out(x)

        r_in = self.r_out(x)

        #p = self.p(x)

        #value iteration
        for k in range(K):
            for i, j in allowed_action_dict.items():
                self.v_next = torch.max(self.v_current,torch.max([]))

        return r_in, r_out, p

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        hidden_units = 32
        hidden_units2 = 64
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)
        self.affine2 = nn.Linear(hidden_units, hidden_units2)


        # actor's layer
        self.action_head = nn.Linear(hidden_units2, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(hidden_units2, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

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
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)  # laver returns om til torch tensors
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()  # calculating advantage, value.item() = the state value we got

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)  #

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))  # l1 smoothed absolute error

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
    list_of_running_reward = []
    list_of_i_episode = []
    # run infinitely many episodes
    for i_episode in range(N_EPISODES):  # count(1):

        # reset environment and episode reward
        state = env.reset(new_grid=False)
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

            # if render:
            #     env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward #helps with tracking trainprogress
        running_reward = 0.02 * ep_reward + (1 - 0.02) * running_reward

        # perform backprop
        # print(f"{i_episode} - finishing episode")
        finish_episode()

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            list_of_i_episode.append(i_episode)
            list_of_running_reward.append(running_reward)


            if abs(running_reward - 1.00) < eps and is_solved(100):
                # if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))

                break
    plt.figure(figsize=(10, 5))
    plt.plot(list_of_i_episode, list_of_running_reward, 'r.-', label='Running average')
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.grid(linestyle=':')
    plt.legend()
    plt.show()


def play():
    # env = GridWorld(map_size=(4,4,5,5), render=True, rewards=(0.0, 100.0))
    model.eval()
    env.render = True
    state = env.reset(new_grid=False)

    wins_baseline = 0
    total_baseline = 0
    # for i in range(100):
    i = 0
    while True:
        # baseline
        baselineProps = torch.tensor([1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])
        baseline_m = Categorical(baselineProps)
        baseline_action = baseline_m.sample().item()
        state, reward, done = env.step(baseline_action)

        i += 1
        if done or i > max_allowed_steps:  # Complete or give up, max 50 steps
            state = env.reset(new_grid=False)
            env.render = False
            if i <= max_allowed_steps:
                wins_baseline += 1
            total_baseline += 1

            if total_baseline == test_size:
                break

    print(f'wins baseline: {wins_baseline} attempts baseline: {total_baseline}')


    i = 0
    env.render = renderTest
    state = env.reset(new_grid=False)
    wins = 0
    total = 0
    while True:
        # pick best action
        state = state.flatten()
        state = torch.from_numpy(state).float()
        probs, _ = model(state)
        action = probs.argmax().item()
        m = Categorical(probs)
        action = m.sample().item()

        #vi bliver stuck i den samme position, derfor performer den bedre uden argmax
        # take action
        time.sleep(0.1)
        state, reward, done = env.step(action)

        i += 1
        if done or i > max_allowed_steps:  # Complete or give up, max 50 steps
            state = env.reset(new_grid=False)
            if i <= max_allowed_steps:
                wins += 1
            total += 1
            i = 0
            print(f'wins: {wins} attempts: {total}')
        if total == test_size:
            break


def is_solved(eps=100):
    """Convergence test over arg 'eps' episodes

       returns true If it can get 100 wins in a rough without using 50 or more, steps
    """

    model.eval()
    state = env.reset(new_grid=False)
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
            state = env.reset(new_grid=False)
            wins += 1
            i = 0
            if wins == eps:
                model.train()
                return True
        elif i > max_allowed_steps:
            model.train()
            print(f'Failed evaluation: {wins}/{eps}')
            return False


if __name__ == '__main__':
    main()  # training the model until convergence
    play()  # evaluation/testing the final model, renders the output