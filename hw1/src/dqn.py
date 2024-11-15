import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer
import random

from src.utils import device
from src.networks import ValueFunctionQ
from src.buffer import ReplayBuffer, Transition

DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# simple MSE loss
# Hint: used for optimize Q function
def loss(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


def greedy_sample(Q: ValueFunctionQ, state):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(env: gym.Env ,Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # You can copy from your previous implementation
    # With probability eps, select a random action
    # With probability (1 - eps), select the best action using greedy_sample
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        action = greedy_sample(Q, state)
    return action

def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        gamma: float,
        memory: ReplayBuffer,
        optimizer: Optimizer
):
    if len(memory) < memory.batch_size:
        return

    batch_transitions = memory.sample()
    batch = Transition(*zip(*batch_transitions))

    states = np.stack(batch.state)
    actions = np.stack(batch.action)
    rewards = np.stack(batch.reward)
    next_states = np.stack(batch.next_state)
    dones = torch.Tensor(batch.done)
    #valid_next_states = np.stack(tuple(
    #    filter(lambda s: s is not None, batch.next_state)
    #))
    #next_states_with_none = np.stack(batch.next_state)

    #nonterminal_mask = tensor(
    #    tuple(map(lambda s: s is not None, batch.next_state)),
    #    type=torch.bool
    #)
    # print("number of transitions", len(memory))
    # print("states shape", states.shape)
    # print("actions shape", actions.shape)
    # print("rewards shape", rewards.shape)
    # print("next_states shape", next_states.shape)
    

    rewards = tensor(rewards)

    # TODO: Update the Q-network
    # Hint: Calculate the target Q-values
    # Initialize targets with zeros
    target_val_batch = torch.zeros(size=(memory.batch_size, 1), device=DEVICE)
    with torch.no_grad():
        V_next_state, _ = target_Q(next_states).max(dim = 1)
        #V_next_state, _ = nonterminal_mask * target_Q(next_states_with_none).max(dim = 1)
        #print("V_next_state shape", V_next_state.shape)
        #print("target_val_batch shape", target_val_batch.shape)
        target_val_batch = rewards + gamma * V_next_state * (1 - dones) #* nonterminal_mask
        #print("target_val_batch shape after broadcasting", target_val_batch.shape)
        #pass  # Students are expected to compute the target Q-values here
    
    old_val_batch = Q(states).gather(1, tensor(actions, type = torch.int64).view(-1, 1))
    #print("old_val_batch shape", old_val_batch.shape)
    mse_loss = loss(old_val_batch, target_val_batch)

    optimizer.zero_grad()
    mse_loss.backward()
    optimizer.step()



def train_one_epoch(
        seed: int,
        env: gym.Env,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        RB: ReplayBuffer,
        optimizer: Optimizer,
        gamma: float = 0.99,
        train_interval: int = 10,
) -> float:
    # Make sure target isn't being trained
    Q.train()
    target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset(seed = seed)

    episode_reward: float = 0.0

    for t in count():
        # TODO: Complete the train_one_epoch for dqn algorithm
        action = eps_greedy_sample(env, Q, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        #if terminated:
        #    print("next_state on termination", next_state)
            #next_state = None

        #if truncated:
        #    print("next_state on truncation", next_state)

        # transition = Transition(state, action, next_state, reward)
        RB.push(state, action, next_state, reward, terminated)
        if t % train_interval == 0:
            #print("training when t =", t)
            optimize_Q(Q, target_Q, gamma, RB, optimizer)

        done = truncated or terminated
        if done:
            break
        #pass


    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
