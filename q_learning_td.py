import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from matplotlib.patches import Patch

# Constants
NUM_EPISODES = 100_000
WINDOW_SIZE = 5000
LEARNING_RATE = 0.01
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
TRACE_DECAY = 0.6  # Lambda for TD(0.6)

# ----------- Q-learning with TD(0.6) -----------
class BlackjackAgent:
    def __init__(self, env, learning_rate, epsilon, discount_factor, trace_decay):
        """Initialize the agent with Q-values, learning rate, and eligibility traces."""
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.e_values = defaultdict(lambda: np.zeros(env.action_space.n))  # Eligibility traces
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.trace_decay = trace_decay
        self.training_error = []

    def get_action(self, obs):
        """Choose an action using an epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            return int(np.argmax(self.q_values[obs]))  # Greedy action

    def update(self, obs, action, reward, terminated, next_obs):
        """Update Q-values using TD(0.6) with eligibility traces."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        # Update eligibility trace for the visited state-action pair
        self.e_values[obs][action] += 1  # Accumulating traces

        # Update all Q-values using eligibility traces
        for state, actions in self.e_values.items():
            for act in range(len(actions)):
                self.q_values[state][act] += self.lr * td_error * self.e_values[state][act]
                self.e_values[state][act] *= self.discount_factor * self.trace_decay  # Decay traces

        self.training_error.append(td_error)

    def reset_traces(self):
        """Reset eligibility traces at the start of each episode."""
        self.e_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

def train_qlearning_agent(env, agent, num_episodes):
    """Train the Q-learning agent with eligibility traces."""
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    rewards = []

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        agent.reset_traces()  # Reset eligibility traces at episode start

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        rewards.append(ep_reward)

    return env, rewards

# ----------- Policy Grid Creation -----------
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

# ----------- Plotting Functions -----------
def plot_training_metrics(env, agent, rolling_length=500):
    """Plot training metrics: rewards, lengths, and training error."""
    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))

    # Plot rolling average of episode rewards
    reward_moving_average = np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    ) / rolling_length
    axs[0].plot(reward_moving_average)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Average Reward")

    # Plot rolling average of episode lengths
    length_moving_average = np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    ) / rolling_length
    axs[1].plot(length_moving_average)
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Length")

    # Plot rolling average of training error
    training_error_moving_average = np.convolve(
        np.array(agent.training_error), np.ones(rolling_length), mode="same"
    ) / rolling_length
    axs[2].plot(training_error_moving_average)
    axs[2].set_title("Training Error")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Average Error")

    plt.tight_layout()
    plt.show()

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

# ----------- Main Execution -----------
if __name__ == "__main__":
    # Initialize environment and agent
    env = gym.make("Blackjack-v1", sab=True)
    agent = BlackjackAgent(
        env=env,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        discount_factor=DISCOUNT_FACTOR,
        trace_decay=TRACE_DECAY,
    )

    # Train the agent
    env = train_qlearning_agent(env, agent, NUM_EPISODES)

    # Plot training metrics
    #plot_training_metrics(env, agent)

    # Create and plot value and policy grids
    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    plot_value_and_policy(value_grid, policy_grid, title="With Usable Ace")

    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    plot_value_and_policy(value_grid, policy_grid, title="Without Usable Ace")
