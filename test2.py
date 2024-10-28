import random
import numpy as np
from env import StaticGridEnv

# Initialise the environment
env = StaticGridEnv ()

print(np.random.uniform(0.0, 0.01, (5, 3)))

# Start a new episode
state = env.reset()
done = False

while not done:
    # Take a random action for this demonstration
    action = random.choice ([0, 1, 2, 3])

    # Execute the action
    next_state , reward , done , _ = env.step(action)

    # Render the environment
    env.render(episode=0, learning_type="Random Policy")

    # Update the state
    state = next_state

# Close the environment
env.close()

print("Done!")