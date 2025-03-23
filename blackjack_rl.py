import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from matplotlib.patches import Patch
import q_learning as qlearning
import q_learning_td as tdlearning
import mc_every_visit as mcev
import mc_first_visit as mcfv

# Constants
NUM_EPISODES = 100_000
WINDOW_SIZE = 5000
LEARNING_RATE = 0.01
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
TRACE_DECAY = 0.6
HIT_UNTIL = 17
GAMES = 100_000

def moving_average(data, window_size):
    """Computes the moving average for smoothing reward trends."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def draw_card():
    """Simulates drawing a card from a standard deck."""
    return min(random.randint(1, 13), 10)

def play_basic_strategy():
    """Plays using basic strategy (hit until reaching at least 17)."""
    total = 0
    while total < HIT_UNTIL:
        total += draw_card()
    return total if total <= 21 else 0

def play_strategy(env, policy_usable, policy_no_ace):
    """Plays using a learned policy from one of the algorithms."""
    obs, _ = env.reset()
    done = False
    while not done:
        player_sum, dealer_card, usable_ace = obs
        policy = policy_usable if usable_ace else policy_no_ace
        action = policy.get(obs, 1)  # Default to hitting if state unseen
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return 0 if reward == 0 else (21 if reward == 1 else 0)  # Convert to score

def dealer_play():
    """Simulates the dealer's strategy (hit until reaching at least 17)."""
    total = 0
    while total < 17:
        total += draw_card()
    return total if total <= 21 else 0

def simulate_game(strategy):
    """Simulates a game and returns the outcome."""
    player_score = strategy()
    dealer_score = dealer_play()
    return 'win' if player_score > dealer_score else 'loss' if player_score < dealer_score else 'tie'

def train_agent(env, agent, train_function):
    """Trains a Blackjack agent and extracts its policy."""
    env, rewards = train_function(env, agent, NUM_EPISODES)
    q_values = agent.q_values
    policy_usable_ace = {state: np.argmax(q_values[state]) for state in q_values if state[2]}
    policy_no_ace = {state: np.argmax(q_values[state]) for state in q_values if not state[2]}

    # Track win rates every 1000 episodes
    win_rates = []
    for i in range(0, NUM_EPISODES, 1000):
        wins_in_range = sum([1 for reward in rewards[i:i+1000] if reward > 0])  # Count wins in this range
        win_rate = wins_in_range / 1000 * 100  # Calculate percentage of wins
        win_rates.append(win_rate)

    return env, policy_usable_ace, policy_no_ace, rewards, win_rates

def setup_and_train_agents():
    """Initializes, trains, and extracts policies for both Q-learning and TD-learning agents."""
    mcfv_env = gym.make("Blackjack-v1", sab=True)
    mcfv_agent = mcfv.BlackjackAgent(mcfv_env, EPSILON, DISCOUNT_FACTOR)
    mcfv_env, mcfv_policy_usable, mcfv_policy_no_ace, mcfv_rewards, mcfv_wins = train_agent(mcfv_env, mcfv_agent, mcfv.monte_carlo_fv)

    mcev_env = gym.make("Blackjack-v1", sab=True)
    mcev_agent = mcev.BlackjackAgent(mcev_env, EPSILON, DISCOUNT_FACTOR)
    mcev_env, mcev_policy_usable, mcev_policy_no_ace, mcev_rewards, mcev_wins = train_agent(mcev_env, mcev_agent, mcev.monte_carlo_ev)

    q_env = gym.make("Blackjack-v1", sab=True)
    q_agent = qlearning.BlackjackAgent(q_env, LEARNING_RATE, EPSILON, DISCOUNT_FACTOR)
    q_env, q_policy_usable, q_policy_no_ace, q_rewards, q_wins = train_agent(q_env, q_agent, qlearning.train_qlearning_agent)
    
    td_env = gym.make("Blackjack-v1", sab=True)
    td_agent = tdlearning.BlackjackAgent(td_env, LEARNING_RATE, EPSILON, DISCOUNT_FACTOR, TRACE_DECAY)
    td_env, td_policy_usable, td_policy_no_ace, td_rewards, td_wins = train_agent(td_env, td_agent, tdlearning.train_qlearning_agent)
    
    return q_env, q_agent, q_policy_usable, q_policy_no_ace, q_rewards, q_wins, \
        td_env, td_agent, td_policy_usable, td_policy_no_ace, td_rewards, td_wins, \
            mcev_env, mcev_agent, mcev_policy_usable, mcev_policy_no_ace, mcev_rewards, mcev_wins, \
                mcfv_env, mcfv_agent, mcfv_policy_usable, mcfv_policy_no_ace, mcfv_rewards, mcfv_wins

def visualize_policies(agent, learning_module):
    """Generates policy and value function visualizations."""
    for usable_ace in [True, False]:
        value_grid, policy_grid = learning_module.create_grids(agent, usable_ace=usable_ace)
        title = "With Usable Ace" if usable_ace else "Without Usable Ace"
        learning_module.plot_value_and_policy(value_grid, policy_grid, title)

def plot_average_rewards(q_rewards, td_rewards, mcev_rewards, mcfv_rewards):
    """Plots the smoothed average rewards over the first 1,000 episodes."""
    episodes = np.arange(1000)

    q_rewards = q_rewards[:1000]
    td_rewards = td_rewards[:1000]
    mcev_rewards = mcev_rewards[:1000]
    mcfv_rewards = mcfv_rewards[:1000]

    q_rewards_smoothed = moving_average(q_rewards, WINDOW_SIZE)
    td_rewards_smoothed = moving_average(td_rewards, WINDOW_SIZE)
    mcev_rewards_smoothed = moving_average(mcev_rewards, WINDOW_SIZE)
    mcfv_rewards_smoothed = moving_average(mcfv_rewards, WINDOW_SIZE)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, q_rewards_smoothed, label="Q-Learning", color="red")
    plt.plot(episodes, td_rewards_smoothed, label="TD-Learning", color="green")
    plt.plot(episodes, mcev_rewards_smoothed, label="MC Every Visit", color="blue")
    plt.plot(episodes, mcfv_rewards_smoothed, label="MC First Visit", color="orange")

    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Average Reward")
    plt.title("Smoothed Average Rewards Over First 1,000 Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_win_rates(q_win_rates, td_win_rates, mcev_win_rates, mcfv_win_rates):
    """Plots smoothed win rates every 1000 episodes for each algorithm."""
    plt.figure(figsize=(10, 5))
    episodes = np.arrange(len(q_win_rates))
    
    plt.plot(episodes, q_win_rates, label="Q-Learning", color="red")
    plt.plot(episodes, td_win_rates, label="TD-Learning", color="green")
    plt.plot(episodes, mcev_win_rates, label="MC Every Visit", color="blue")
    plt.plot(episodes, mcfv_win_rates, label="MC First Visit", color="orange")

    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.title("Smoothed Win Rates Every 1,000 Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()




def run_simulations(q_env, q_policy_usable, q_policy_no_ace, td_env, td_policy_usable, td_policy_no_ace, mcev_env, mcev_policy_usable, mcev_policy_no_ace, mcfv_env, mcfv_policy_usable, mcfv_policy_no_ace):
    """Runs multiple blackjack simulations for different strategies."""
    results = {"basic": {'win': 0, 'loss': 0, 'tie': 0},
               "q_learning": {'win': 0, 'loss': 0, 'tie': 0},
               "td_learning": {'win': 0, 'loss': 0, 'tie': 0},
               "mcev": {'win': 0, 'loss': 0, 'tie': 0},
               "mcfv": {'win': 0, 'loss': 0, 'tie': 0}}
    
    for _ in range(GAMES):
        results["basic"][simulate_game(play_basic_strategy)] += 1
        results["q_learning"][simulate_game(lambda: play_strategy(q_env, q_policy_usable, q_policy_no_ace))] += 1
        results["td_learning"][simulate_game(lambda: play_strategy(td_env, td_policy_usable, td_policy_no_ace))] += 1
        results["mcev"][simulate_game(lambda: play_strategy(mcev_env, mcev_policy_usable, mcev_policy_no_ace))] += 1
        results["mcfv"][simulate_game(lambda: play_strategy(mcfv_env, mcfv_policy_usable, mcfv_policy_no_ace))] += 1

    return results

def print_results(results):
    """Prints win percentages for each strategy."""
    for strategy, outcome in results.items():
        win_rate = outcome['win'] / GAMES * 100
        print(f"{strategy.capitalize()} Strategy Win %: {win_rate:.2f}%")

if __name__ == "__main__":
    # Train agents and extract policies
    q_env, q_agent, q_policy_usable, q_policy_no_ace, q_rewards, q_win_rates, \
    td_env, td_agent, td_policy_usable, td_policy_no_ace, td_rewards, td_win_rates, \
    mcev_env, mcev_agent, mcev_policy_usable, mcev_policy_no_ace, mcev_rewards, mcev_win_rates, \
    mcfv_env, mcfv_agent, mcfv_policy_usable, mcfv_policy_no_ace, mcfv_rewards, mcfv_win_rates = setup_and_train_agents()

    # Visualize learned policies
    #visualize_policies(q_agent, qlearning)
    #visualize_policies(td_agent, tdlearning)
    #visualize_policies(mcev_agent, mcev)
    #visualize_policies(mcfv_agent, mcfv)

    plot_average_rewards(q_rewards, td_rewards, mcev_rewards, mcfv_rewards)

    # Plot win rates every 1000 episodes
    plot_win_rates(q_win_rates, td_win_rates, mcev_win_rates, mcfv_win_rates)
    
    # Run and evaluate simulations
    results = run_simulations(q_env, q_policy_usable, q_policy_no_ace, td_env, td_policy_usable, td_policy_no_ace, mcev_env, mcev_policy_usable, mcev_policy_no_ace, mcfv_env, mcfv_policy_usable, mcfv_policy_no_ace)
    print_results(results)
