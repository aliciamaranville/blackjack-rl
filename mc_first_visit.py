import gymnasium as gym
import numpy as np
import random
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# ----------- Setup -----------------
NUM_EPISODES = 100_000
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9

# ----------- Monte Carlo Agent Code -----------
class BlackjackAgent:
    def __init__(self, env, epsilon, discount_factor):
        """Initialize the agent with Q-values, epsilon, and discount factor."""
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))  # State-action Q-values
        self.returns = defaultdict(list) # Stores returns
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.training_error = []

    def get_action(self, obs):
        """Choose an action based on epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            return int(np.argmax(self.q_values[obs]))  # Greedy action

    def update(self, episode_history):
        """Update the Q-values using Monte Carlo Every-Visit method."""
        G = 0  # Initialize return
        visited_pairs = set()

        for s, a, reward in reversed(episode_history):
            G = reward + self.discount_factor * G
            if (s, a) not in visited_pairs:
                visited_pairs.add((s,a))
                self.returns[(s, a)].append(G)
                self.q_values[s][a] = np.mean(self.returns[(s,a)])


def monte_carlo_fv(env, agent, num_episodes):
    """Monte Carlo First Visit algorithm for Blackjack."""
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    episode_rewards = []
    num_wins = 0  # Track wins

    for episode in tqdm(range(num_episodes)):
        s, _ = env.reset()  # Reset environment, get initial state s
        done = False
        episode_history = []  # Store (state, action, reward) for this episode
        episode_reward = 0

        while not done:
            a = agent.get_action(s)  # Use epsilon-greedy policy to select action
            next_s, reward, done, truncated, _ = env.step(a)  # Take action a
            episode_history.append((s, a, reward))
            s = next_s
            episode_reward += reward

        # Update Q-values using Monte Carlo method
        agent.update(episode_history)
        episode_rewards.append(episode_reward)

        if episode_reward > 0:  # Winning episode
            num_wins += 1

    return env, episode_rewards

def create_grids(agent, usable_ace=False):
    """Create value and policy grids for plotting."""
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(np.arange(12, 22), np.arange(1, 11))

    # Create value grid
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # Create policy grid
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def plot_value_and_policy(value_grid, policy_grid, title):
    """Plot value and policy grids."""
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # Plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    ax1.set_title(f"State Values: {title}")
    ax1.set_xlabel("Player Sum")
    ax1.set_ylabel("Dealer Showing")
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # Plot the policy
    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False, ax=ax2)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player Sum")
    ax2.set_ylabel("Dealer Showing")
    ax2.set_xticks(np.arange(0.5, len(policy_grid[0])))
    ax2.set_xticklabels(np.arange(12, 22))
    ax2.set_yticks(np.arange(0.5, len(policy_grid)))
    ax2.set_yticklabels(np.arange(1, 11))

    # Add legend for actions
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    # Initialize agent with learning parameters
    agent = BlackjackAgent(
        env=env,
        epsilon=EPSILON,
        discount_factor=DISCOUNT_FACTOR
    )

    # Run Monte Carlo training
    env = monte_carlo_fv(env, agent, NUM_EPISODES)

    # Create and plot value and policy grids
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    plot_value_and_policy(value_grid, policy_grid, title="With Usable Ace")

    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    plot_value_and_policy(value_grid, policy_grid, title="Without Usable Ace")

