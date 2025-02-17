# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:44:04 2025

@author: Guita
"""

import gymnasium as gym
import matplotlib.pyplot as plt

# Create the Blackjack environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Track rewards for each episode
total_rewards_fixed = []

# Number of episodes to run the fixed strategy
num_episodes = 100000

# Function to implement the fixed strategy
def fixed_strategy(state):
    # If player's sum is less than 17, hit; otherwise, stand
    if state[0] < 17:
        return 1  # Hit
    else:
        return 0  # Stand

# Run the fixed strategy for the given number of episodes
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = fixed_strategy(state)  # Use fixed strategy
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

    total_rewards_fixed.append(total_reward)

    # Debugging: Print total reward for every 1000 episodes
    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Plot the rewards for the fixed strategy
plt.plot(total_rewards_fixed)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance of Fixed Strategy')
plt.show()

# Calculate the win rate for the fixed strategy
win_count_fixed = total_rewards_fixed.count(1)
win_rate_fixed = win_count_fixed / num_episodes  # Total win percentage
print(f"\nFixed Strategy Final Win Rate: {win_rate_fixed * 100}%")
