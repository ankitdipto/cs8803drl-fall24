import gym
import torch
import numpy as np
from typing import Tuple
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import Policy


DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

# Hint loss you can use
def loss(
        epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor
    ) -> torch.Tensor:
    return -1.0 * (epoch_log_probability_actions * epoch_action_rewards).mean()


def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        optimizer: Optimizer,
        max_timesteps=5_000
    ) -> Tuple[float, float]:

    policy.train()

    epoch_total_timesteps = 0

    # Action log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    # Loop through episodes
    while True:
        # Stop if we've done over the total number of timesteps
        if epoch_total_timesteps > max_timesteps:
            break

        # Running total of this episode's rewards
        episode_reward: float = 0

        # Reset the environment and get a fresh observation
        state, info = env.reset()

        # Loop through timesteps until the episode is done (or the max is hit)
        for t in count():
            epoch_total_timesteps += 1

            # TODO: Sample an action from the policy
            action, log_prob_of_action = policy.sample(tensor(state))
            # print('action', action)
            # TODO: Take the action in the environment
            state, reward, terminated, truncated, info = env.step(action)

            # TODO: Accumulate the reward
            episode_reward += reward
            epoch_action_rewards.append(reward)

            # TODO: Store the log probability of the action
            epoch_log_probability_actions.append(log_prob_of_action)
            
            # Finish the action loop if this episode is done
            done = terminated or truncated
            if done:
                # TODO: Assign the episode reward to each timestep in the episode

                break

    weights = [episode_reward] * epoch_total_timesteps
    epoch_log_probability_actions_tensor = torch.stack(epoch_log_probability_actions)
    epoch_action_rewards_tensor = torch.tensor(weights)
    # TODO: Calculate the policy gradient loss
    Loss = loss(epoch_log_probability_actions_tensor, epoch_action_rewards_tensor)

    # TODO: Perform backpropagation and update the policy parameters
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    # Placeholder return values (to be replaced with actual calculations)
    return 0.0, 0.0
