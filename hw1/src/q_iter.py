import gym
import torch
import numpy as np
from itertools import count
from torch.optim import Optimizer
import random

from src.utils import device
from src.networks import ValueFunctionQ


DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START

# simple MSE loss
def loss(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
    mean_square_error = (value - target)**2
    return mean_square_error


def greedy_sample(Q: ValueFunctionQ, state: np.array):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(env: gym.Env, Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        action = greedy_sample(Q, state)
    return action

def train_one_epoch(
        seed: int,
        env: gym.Env,
        Q: ValueFunctionQ,
        optimizer: Optimizer,
        gamma: float = 0.99
    ) -> float:
    Q.train()

    # Reset the environment and get a fresh observation
    state, info = env.reset(seed = seed)

    episode_reward: float = 0.0

    for t in count():
        # TODO: Generate action using epsilon-greedy policy
        action = eps_greedy_sample(env, Q, state)

        # TODO: Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated:
            next_state = None

        # Calculate the target
        with torch.no_grad():
            # TODO: Compute the target Q-value
            if terminated:
                target_val = reward
            else:
                V_next_state = Q(next_state).max() 
                #print('V_next_state', V_next_state)
                target_val = reward + gamma * V_next_state
            #pass

        # TODO: Compute the loss
        old_val = Q(state, action)
        mse_loss = loss(old_val, target_val)

        # TODO: Perform backpropagation and update the network
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        # TODO: Update the state
        state = next_state

        # TODO: Handle episode termination
        done = truncated or terminated
        if done:
            break

    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
