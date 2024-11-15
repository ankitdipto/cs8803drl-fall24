import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ, Policy
from src.buffer import ReplayBuffer, Transition

DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# Hint: loss you can use to update Q function
def loss_Q(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


# Hint: loss you can use to update policy
def loss_pi(
        log_probabilities: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    return -1.0 * (log_probabilities * advantages).mean()

# Hint: you can use similar implementation from dqn algorithm
def optimize_Q(
        V_values_buffer: torch.Tensor,
        Q_values_of_traj: torch.Tensor, 
        # Q: ValueFunctionQ,
        # target_Q: ValueFunctionQ,
        # policy: Policy,
        # gamma: float,
        # batch: Transition,
        optimizer: Optimizer
):
    # states = np.stack(batch.state)
    # actions = np.stack(batch.action)
    # rewards = np.stack(batch.reward)
    # valid_next_states = np.stack(tuple(
    #     filter(lambda s: s is not None, batch.next_state)
    # ))

    # nonterminal_mask = tensor(
    #     tuple(map(lambda s: s is not None, batch.next_state)),
    #     type=torch.bool
    # )

    # actions_, log_probabilities = policy.sample_multiple(states)
    # actions_ = actions_.unsqueeze(-1)[nonterminal_mask]

    # rewards = tensor(rewards)
    # batch_size = len(rewards)
    # TODO: Update the Q-network

    # calculate the target
    # targets = torch.zeros(size=(batch_size, 1), device=DEVICE)
    # with torch.no_grad():
        # Hint: Compute the target Q-values
        #pass

    LOSS = loss_Q(V_values_buffer, Q_values_of_traj)
    print("In optimize_Q", V_values_buffer.requires_grad, Q_values_of_traj.requires_grad)
    optimizer.zero_grad()
    LOSS.backward()
    optimizer.step()





# Hint: you can use similar implementation from reinforce algorithm
def optimize_policy(
        V_values_buffer: torch.Tensor,
        Q_values_of_traj: torch.Tensor,
        log_probs_buffer: torch.Tensor,
        # policy: Policy,
        # Q: ValueFunctionQ,
        # batch: Transition,
        optimizer: Optimizer
):
    # states = np.stack(batch.state)

    #actions, log_probabilities = policy.sample_multiple(states)

    #actions = actions.unsqueeze(-1)
    #log_probabilities = log_probabilities.unsqueeze(-1)

    # TODO: Update the policy network

    with torch.no_grad():
        # Hint: Advantages
        # A(s,a) = Q(s,a) - V(s)
        advantages = Q_values_of_traj - V_values_buffer
    LOSS = loss_pi(log_probs_buffer, advantages)
    optimizer.zero_grad()
    LOSS.backward()
    optimizer.step()


def train_one_epoch(
        seed: int,
        env: gym.Env,
        policy: Policy,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        #memory: ReplayBuffer,
        optimizer_Q: Optimizer,
        optimizer_pi: Optimizer,
        gamma: float = 0.99,
        n_step_return: bool = False
) -> float:
    # make sure target isn't fitted
    policy.train(), Q.train(), target_Q.eval()

    V_values_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    Q_values_next_buffer = []

    # Reset the environment and get a fresh observation
    state, info = env.reset(seed = seed)
    episode_reward = 0
    for t in count():

        # TODO: Complete the train_one_epoch for actor-critic algorithm
        action, log_prob_of_action = policy.sample(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        V_value = Q.V(state)

        rewards_buffer.append(reward)
        V_values_buffer.append(V_value)
        log_probs_buffer.append(log_prob_of_action)

        assert next_state is not None
        Q_value_next = target_Q.V(next_state)
        Q_values_next_buffer.append(Q_value_next)

        if terminated:
            #print("terminated at step", t)
            # If terminated we will not do bootstrapping
            Q_values_next_buffer[-1] = 0
            break
            
        if truncated:
            # If truncated we will do bootstrapping
            # Leaving Q_value_next of this step as it was
            break
        # TODO: Store the transition in memory

        # Hint: Use replay buffer!
        # Hint: Check if replay buffer has enough samples

        state = next_state # Very crucial! If you miss this step it will go for an infinite loop without you knowing.

    #print("V_values_buffer", V_values_buffer[:5])
    #print("Q_values_next_buffer", Q_values_next_buffer[:5])
    #Q_values_of_traj = np.zeros(len(V_values_buffer))
    Q_values_of_traj = torch.zeros(len(V_values_buffer), dtype = torch.float32, device = DEVICE)
    if n_step_return: # for TD(n) target
        for timestep in reversed(range(len(rewards_buffer))):
            Q_value_next = rewards_buffer[timestep] + gamma * Q_value_next
            Q_values_of_traj[timestep] = Q_value_next

    else: # for TD(0) target
        for timestep in range(len(rewards_buffer)):
            temp = rewards_buffer[timestep] + gamma * Q_values_next_buffer[timestep]
            #if timestep % 50 == 0:
            #    print("temp, Q_value_next_buffer[timestep]", temp, Q_values_next_buffer[timestep])
                # print("temp grad", temp.grad_fn)
            Q_values_of_traj[timestep] = temp
    
    #print("Q_values_of_traj", Q_values_of_traj[:5])
    
    V_values_buffer = torch.stack(V_values_buffer)
    log_probs_buffer = torch.stack(log_probs_buffer)
    Q_values_of_traj = torch.FloatTensor(Q_values_of_traj)

    #print("V_values_buffer", V_values_buffer[:5])
    #print("log_probs_buffer", log_probs_buffer[:5])
    
    optimize_Q(V_values_buffer, Q_values_of_traj, optimizer_Q)
    optimize_policy(V_values_buffer, Q_values_of_traj, log_probs_buffer, optimizer_pi)
    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
