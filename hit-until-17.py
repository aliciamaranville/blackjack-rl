import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the Blackjack environment
env = gym.make('Blackjack-v1', sab=True)

# Track rewards, busts, and outcomes (win/loss/draw))
total_rewards_basic = []
wins_basic = 0
losses_basic = 0
draws_basic = 0

# Number of episodes to run
num_episodes = 100_000

def fixed_strategy(state):
    return 1 if state[0] < 17 else 0  # Hit if sum < 17, otherwise Stand

# Run the strategy for the given number of episodes
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    player_hand = 0

    while not done:
        action = fixed_strategy(state)  # Use the fixed strategy
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Track the player's hand and dealer's hand
        player_sum, dealer_card, usable_ace = state
        player_hand = player_sum
        
        # Track wins/losses/draws
        if done:
            if reward == 1:
                wins_basic += 1
            elif reward == -1:
                losses_basic += 1
            else:
                draws_basic += 1

        state = next_state

    total_rewards_basic.append(total_reward)

# Convert rewards to numpy array
rewards_array = np.array(total_rewards_basic)

# Moving average plot for smoother trends
window_size = 5000
moving_avg_rewards = np.convolve(rewards_array, np.ones(window_size) / window_size, mode="valid")

# Create the figure for visualizations
fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# Plot Moving Average of Rewards
axs[0].plot(moving_avg_rewards, label="Moving Average Reward (Hit Until 17)", color="b")
axs[0].axhline(y=0, color='r', linestyle='--', label="Break-even Line")
axs[0].set_xlabel('Episode (x5000)')
axs[0].set_ylabel('Average Reward')
axs[0].set_title('Moving Average of Rewards')
axs[0].legend()

# Plot Pie Chart for Win/Loss/Draw Percentages
labels = ['Wins', 'Losses', 'Draws']
sizes = [wins_basic, losses_basic, draws_basic]
colors = ['lightgreen', 'red', 'gray']
axs[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
axs[1].set_title("Win/Loss/Draw Distribution")

plt.tight_layout()
plt.show()


