import random
import numpy as np
from env import StaticGridEnv

# Initialize the environment
env = StaticGridEnv(42)

# Hyperparameters
epsilon = 0.2  # Exploration rate
alpha = 0.1    # Learning rate
gamma = 0.99   # Discount factor
num_episodes = 1000
max_steps_per_episode = 100

# Initialize Q-table
state_space_size = 100  # 10x10 grid
action_space_size = env.action_space
Q_table = np.zeros((state_space_size, action_space_size))

# Metrics
total_rewards_per_episode = []
steps_per_episode = []
successful_episodes = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_rewards = 0
    steps = 0

    for step in range(max_steps_per_episode):
        # Choose an action (epsilon-greedy policy)
        if random.uniform(0, 1) < epsilon:
            action =  random.randint(0,3)# Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit

        # Take the action
        next_state, reward, done, _ = env.step(action)

        # Update Q-table
        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action]
        )

        # Update metrics
        total_rewards += reward
        steps += 1

        # Render the environment (optional)
        env.render(episode=episode, learning_type="Q-learning")

        # Transition to the next state
        state = next_state

        if done:
            if reward > 0:  # Assuming positive reward indicates success
                successful_episodes += 1
            break

    # Track metrics
    total_rewards_per_episode.append(total_rewards)
    steps_per_episode.append(steps)

    # Decay epsilon
    epsilon = max(0.01, epsilon * 0.995)

# Close the environment
env.close()

# Print metrics
print(f"Total Rewards per Episode: {total_rewards_per_episode}")
print(f"Steps per Episode: {steps_per_episode}")
print(f"Number of Successful Episodes: {successful_episodes}")
