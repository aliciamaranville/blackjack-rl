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

# Constants
NUM_EPISODES = 100_000
WINDOW_SIZE = 5000
LEARNING_RATE = 0.01
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (NUM_EPISODES / 2)
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
TRACE_DECAY = 0.6
HIT_UNTIL = 17
GAMES = 100_000

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
    """Plays using a learned policy from Q-learning or TD-learning."""
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
    env = train_function(env, agent, NUM_EPISODES)
    q_values = agent.q_values
    policy_usable_ace = {state: np.argmax(q_values[state]) for state in q_values if state[2]}
    policy_no_ace = {state: np.argmax(q_values[state]) for state in q_values if not state[2]}
    return env, policy_usable_ace, policy_no_ace

def setup_and_train_agents():
    """Initializes, trains, and extracts policies for both Q-learning and TD-learning agents."""
    q_env = gym.make("Blackjack-v1", sab=True)
    q_agent = qlearning.BlackjackAgent(q_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)
    q_env, q_policy_usable, q_policy_no_ace = train_agent(q_env, q_agent, qlearning.train_qlearning_agent)
    
    td_env = gym.make("Blackjack-v1", sab=True)
    td_agent = tdlearning.BlackjackAgent(td_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR, TRACE_DECAY)
    td_env, td_policy_usable, td_policy_no_ace = train_agent(td_env, td_agent, tdlearning.train_qlearning_agent)
    
    return q_env, q_agent, q_policy_usable, q_policy_no_ace, td_env, td_agent, td_policy_usable, td_policy_no_ace

def visualize_policies(agent, learning_module):
    """Generates policy and value function visualizations."""
    for usable_ace in [True, False]:
        value_grid, policy_grid = learning_module.create_grids(agent, usable_ace=usable_ace)
        title = "With Usable Ace" if usable_ace else "Without Usable Ace"
        learning_module.plot_value_and_policy(value_grid, policy_grid, title)

def run_simulations(q_env, q_policy_usable, q_policy_no_ace, td_env, td_policy_usable, td_policy_no_ace):
    """Runs multiple blackjack simulations for different strategies."""
    results = {"basic": {'win': 0, 'loss': 0, 'tie': 0},
               "q_learning": {'win': 0, 'loss': 0, 'tie': 0},
               "td_learning": {'win': 0, 'loss': 0, 'tie': 0}}
    
    for _ in range(GAMES):
        results["basic"][simulate_game(play_basic_strategy)] += 1
        results["q_learning"][simulate_game(lambda: play_strategy(q_env, q_policy_usable, q_policy_no_ace))] += 1
        results["td_learning"][simulate_game(lambda: play_strategy(td_env, td_policy_usable, td_policy_no_ace))] += 1

    
    return results

def print_results(results):
    """Prints win percentages for each strategy."""
    for strategy, outcome in results.items():
        win_rate = outcome['win'] / GAMES * 100
        print(f"{strategy.capitalize()} Strategy Win %: {win_rate:.2f}%")

if __name__ == "__main__":
    # Train agents and extract policies
    q_env, q_agent, q_policy_usable, q_policy_no_ace, td_env, td_agent, td_policy_usable, td_policy_no_ace = setup_and_train_agents()
    
    # Visualize learned policies
    visualize_policies(q_agent, qlearning)
    visualize_policies(td_agent, tdlearning)
    
    # Run and evaluate simulations
    results = run_simulations(q_env, q_policy_usable, q_policy_no_ace, td_env, td_policy_usable, td_policy_no_ace)
    print_results(results)