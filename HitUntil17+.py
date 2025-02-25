# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:44:04 2025

@author: Guita
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create the Blackjack environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Track rewards for each episode
total_rewards_fixed = []

# Number of episodes to run the fixed strategy
num_episodes = 100000

# Function to implement the fixed strategy
def fixed_strategy(state):
    return 1 if state[0] < 16 else 0  # Hit if sum < 16, otherwise Stand

# Run the fixed strategy
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = fixed_strategy(state)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

    total_rewards_fixed.append(total_reward)

    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Convert rewards to numpy array for easier analysis
rewards_array = np.array(total_rewards_fixed)

# Moving average function for smoothing
def moving_average(data, window_size=1000):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

# Plot Moving Average of Rewards
plt.figure(figsize=(10, 5))
plt.plot(moving_average(rewards_array, window_size=5000), label="Moving Average Reward", color="b")
plt.axhline(y=0, color='r', linestyle='--', label="Break-even Line")
plt.xlabel('Episode (x5000)')
plt.ylabel('Average Reward')
plt.title('Moving Average of Rewards over Episodes')
plt.legend()
plt.show()

# Win/Loss Distribution
win_count = (rewards_array == 1).sum()
loss_count = (rewards_array == -1).sum()
draw_count = (rewards_array == 0).sum()

plt.figure(figsize=(6, 6))
plt.pie(
    [win_count, loss_count, draw_count],
    labels=["Wins", "Losses", "Draws"],
    autopct="%1.1f%%",
    colors=["green", "red", "gray"],
    startangle=140,
)
plt.title("Win/Loss/Draw Distribution")
plt.show()

