import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from collections import defaultdict


# ----------- Setup -----------------

# Parameters
epsilon = 0.1  
num_episodes = 500000

# Create Gymnasium blackjack environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Initialize Q-table and returns table
Q = np.zeros((32, 11, 2, 2))  
returns = np.zeros((32, 11, 2, 2))  
num_visits = np.zeros((32, 11, 2, 2))  


# ----------- Monte Carlo Implimentation -----------------

# Function to estimate the estimating state-action values Using Monte Carlo first visit.
def monte_carlo_fv():
    num_wins = 0 
    total_rewards = []  # For every episode, store the total reward
    win_rates = []  # For every 1000 episodes, store win rate
    

    for episode in range(num_episodes):
        s, _ = env.reset()  # reset environment, get inital state
        episode_history = []  # Store (state, action, reward) for this episode
        done = False
        total_reward = 0
        visited_pairs = set()  # Tracks first occurrences of (state, action) pairs

        while not done:
            a = e_greedy(s)  # use epsilon greedy policy to select action, a
            next_s, reward, done, truncated, _ = env.step(a)  # Take action and observe outcome
            episode_history.append((s, a, reward))  # Store episode history
            s = next_s  # Move to next state
            total_reward += reward  

        # Process episode history and update Q-values
        for i, (s, a, reward) in enumerate(episode_history):
            state_index = (s[0] - 1, s[1] - 1, s[2], a)  # Convert state to index
            
            if state_index not in visited_pairs:  # First-visit check
                visited_pairs.add(state_index)  # Mark this (state, action) as visited
                returns[state_index] += reward 
                num_visits[state_index] += 1  
                Q[state_index] = returns[state_index] / num_visits[state_index]  # Compute new Q-value

        total_rewards.append(total_reward)  # Store total reward for this episode
        if reward == 1:
            num_wins += 1  # Increment win count

        if (episode + 1) % 1000 == 0:  # Compute win rate every 1000 episodes
            win_rates.append(num_wins / (episode + 1))

    return total_rewards, win_rates, num_wins / num_episodes  



# declare epsilon greedy policy with state s
def e_greedy(s): 
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  
    else:
        return np.argmax(Q[s[0] - 1, s[1] - 1, s[2], :]) 



# -------- Evaluation and Plotting -----------------


# Evaluates the final policy using the learned Q-values over the given number of episodes
def policy_evaluation(Q, num_episodes=100000):

    num_wins = 0  
    num_draws = 0  
    num_loss = 0  

    for _ in range(num_episodes):
        s, _ = env.reset()  # Initialize environment, get initial state
        done = False  
        
        while not done:
            a = np.argmax(Q[s[0] - 1, s[1] - 1, s[2], :])  # Select best action using Q-values
            next_s, reward, done, truncated, _ = env.step(a)  # Take action and observe next state and reward
            s = next_s  # go to next state

        # Update counters
        if reward == 1:
            num_wins += 1  # Count win
        elif reward == 0:
            num_draws += 1  # Count draw
        else:
            num_loss += 1  # Count loss

    return num_wins, num_draws, num_loss  




# Run monte_carlo_first_visit to get values
total_rewards, win_rates, total_training_win_rate = monte_carlo_fv()



# Evaluate our final policy
num_wins, num_draws, num_loss = policy_evaluation(Q, num_episodes=100000)


# Compute state values and policy
state_value = defaultdict(float)
policy = defaultdict(int)

for player_sum in range(12, 22):
    for dealer_card in range(1, 11):
        for usable_ace in [0, 1]:
            best_action = np.argmax(Q[player_sum - 1, dealer_card - 1, usable_ace, :])
            policy[(player_sum, dealer_card, usable_ace)] = best_action
            state_value[(player_sum, dealer_card, usable_ace)] = np.max(Q[player_sum - 1, dealer_card - 1, usable_ace, :])

# Create grids
player_vals, dealer_vals = np.meshgrid(np.arange(12, 22), np.arange(1, 11))



# Function to get the state values
def get_state_values(usable_ace):
    return np.array([
        [state_value[(player, dealer, usable_ace)] for dealer in range(1, 11)]
        for player in range(12, 22)
    ])



# 3D Surface Plot of State Values (No Usable Ace)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(1, 11), np.arange(12, 22))  # Swap dealer and player
Z = get_state_values(0)

ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.set_xlabel("Dealer Showing")  # Now correct
ax.set_ylabel("Player Sum")  # Now correct
ax.set_zlabel("State Value")
ax.set_title("Monte Carlo First Visit - State Value Function (No Usable Ace)")
plt.show()

# 3D Surface Plot of State Values (With Usable Ace)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

Z = get_state_values(1)

ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.set_xlabel("Dealer Showing")  # Now correct
ax.set_ylabel("Player Sum")  # Now correct
ax.set_zlabel("State Value")
ax.set_title("Monte Carlo First Visit - State Value Function (With Usable Ace)")
plt.show()





# Policy Heatmap (No Usable Ace)
policy_no_ace = np.array([
    [policy[(player, dealer, 0)] for dealer in range(1, 11)]
    for player in range(12, 22)
])

plt.figure(figsize=(8, 6))
sns.heatmap(policy_no_ace, annot=True, cmap="coolwarm", linewidths=0.5, cbar=False, xticklabels=range(1, 11), yticklabels=range(12, 22))
plt.xlabel("Dealer Showing")
plt.ylabel("Player Sum")
plt.title("Monte Carlo First Visit - Policy Heatmap (No Usable Ace)")
plt.show()

# Policy Heatmap (With Usable Ace)
policy_with_ace = np.array([
    [policy[(player, dealer, 1)] for dealer in range(1, 11)]
    for player in range(12, 22)
])

plt.figure(figsize=(8, 6))
sns.heatmap(policy_with_ace, annot=True, cmap="coolwarm", linewidths=0.5, cbar=False, xticklabels=range(1, 11), yticklabels=range(12, 22))
plt.xlabel("Dealer Showing")
plt.ylabel("Player Sum")
plt.title("Monte Carlo First Visit - Policy Heatmap (With Usable Ace)")
plt.show()




# Pie Chart of Win/Draw/Loss Rates
total_games = num_wins + num_draws + num_loss
win_percent = (num_wins / total_games) * 100
draw_percent = (num_draws / total_games) * 100
loss_percent = (num_loss / total_games) * 100

labels = ['Win', 'Draw', 'Loss']
sizes = [win_percent, draw_percent, loss_percent]
colors = ['green', 'gray', 'red']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title("Monte Carlo First Visit - Final Policy Win/Draw/Loss Rates")
plt.show()




# Line Plot: Total Rewards over Episodes
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Monte Carlo First Visit - Training Performance')
plt.show()



# Line Plot: Win Rate over Episodes
plt.plot(range(1000, num_episodes + 1, 1000), win_rates)
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Monte Carlo First Visit - Win Rate Over Time')
plt.show()



# Plot Rolling Average of Win Rate
rolling_win_rate = np.convolve(win_rates, np.ones(10)/10, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(range(10000, num_episodes + 1, 1000)[:len(rolling_win_rate)], rolling_win_rate)
plt.xlabel("Episode")
plt.ylabel("Win Rate (Smoothed)")
plt.title("Monte Carlo First Visit - Smoothed Win Rate Over Time")
plt.show()



# Print final results
print(f"Total Win Rate during Training: {total_training_win_rate * 100:.2f}%")
print(f"Final Win Rate with Learned Policy: {win_percent:.2f}%")
print(f"Final Draw Rate: {draw_percent:.2f}%")
print(f"Final Loss Rate: {loss_percent:.2f}%")

