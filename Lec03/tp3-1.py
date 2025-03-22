# %% [markdown]
# # TP3 INFO8003
# The idea behind this notebook is to get familiar with RL algorithms related to continuous domain. In this notebook we focus on the fitted-Q algorithm and the Q-learning algorithm.

# %% [markdown]
# We describe the domain below:
#
# - **State space**: $S = \{(p,v) \in \mathbb{R}^2 | |p| \leq 1, |v| \leq 3 \}$ and a *terminal state*. A terminal state can be seen as a regular state in which the system is stuck and for which all the future rewards obtained in the aftermath are zero.
#     - A terminal state is reached if $|p_{t+1}| > 1$ or $|v_{t+1}| > 3$.
#
# - **Action space**: $ A = \{4,-4\}$.
# - **Dynamics**: $\dot{p} = v$, $\dot{v} =  \frac{a}{m (1+Hill^\prime(p)^2)} - \frac{g Hill^\prime(p)}{1+Hill^\prime(p)^2} - \frac{s^2 Hill^{\prime}(p) Hill^{\prime \prime}(p) }{1+Hill^\prime(p)^2}$,
#     where $m = 1$, $g = 9.81$ and
#
#   $$
#   Hill(p) =
#   \begin{cases}
#     p^2 + p & \text{if} \quad p < 0 \\
#     \frac{p}{\sqrt{1+5p^2}} & \text{otherwise}.
#   \end{cases}
#   $$
#
#     - The discrete-time dynamics is obtained by discretizing the time with the time between $t$ and $t+1$ chosen equal to $0.1s$.
# - **Integration time step**: $0.001$.
# - **Reward signal**:
#   $$
#   r(p_t,v_t,a_t) =
#   \begin{cases}
#     -1 & \text{if} \quad p_{t+1} < -1 \quad \text{or} \quad |v_{t+1}| > 3 \\
#     1 & \text{if} \quad p_{t+1} > 1 \quad \text{and} \quad |v_{t+1}| \le 3 \\
#     0 & \text{otherwise}.
#   \end{cases}
#   $$
#
# - **Discount factor**: $\gamma = 0.95$.
# - **Time horizon**: $T \rightarrow +\infty$.
# - **Initial state**: $p_0 \sim \mathcal{U}(\left[-0.1, 0.1 \right])$, $v_0 = 0$.
#
# This domain is a *car on the hill* problem, and will be referred to by this name from now on. The figure here below shows an illustration of the domain.
#
# <p align="center">
#     <img src="caronthehill_display.jpeg" alt="Display of the position $p=0$ and the speed $s=1$ of the car.">
#     </p>
#

# %% [markdown]
# The implementation of this domain has already been implemented for you to answer the following questions

# %%
import gymnasium as gym
import pygame
import imageio
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Union
from display_caronthehill import save_caronthehill_image


class CarOnHillEnv(gym.Env):
    """
    Car on Hill environment following the Gymnasium interface.

    State space: position [-1, 1], velocity [-3, 3]
    Action space: {-4, 4}
    """


    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.action_space = spaces.Discrete(2)  # 0: -4, 1: 4
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -3.0]),
            high=np.array([1.0, 3.0]),
            dtype=np.float64
        )

        # Physics parameters
        self.dt = 0.001
        self.m = 1.0
        self.g = 9.81

        # Initial state bounds
        self.initial_position_range = (-0.1, 0.1)
        self.initial_velocity = 0.0

        # Discount factor
        self.gamma = 0.95

        # Initialize state
        self.state = None
        self.steps = 0

        self.render_mode = render_mode
        self.frames = []



    def _hill_function(self, p: float) -> float:
        if p < 0:
            return p**2 + p
        return p / np.sqrt(1 + 5 * p**2)

    def _hill_derivative(self, p: float) -> float:
        if p < 0:
            return 2 * p + 1
        return 1 / (1 + 5 * p**2)**(3/2)

    def _hill_second_derivative(self, p: float) -> float:
        if p < 0:
            return 2
        return -15 * p / (1 + 5 * p**2)**(5/2)

    def _dynamics(self, p: float, v: float, a: float) -> Tuple[float, float]:
        """Simulate dynamics for one time step (0.1s) using Euler integration."""
        steps = int(0.1 / self.dt)

        for _ in range(steps):
            hill_deriv = self._hill_derivative(p)
            hill_second = self._hill_second_derivative(p)

            v_dot = (a / (self.m * (1 + hill_deriv**2)) -
                    (self.g * hill_deriv) / (1 + hill_deriv**2) -
                    (v**2 * hill_deriv * hill_second) / (1 + hill_deriv**2))

            p += v * self.dt
            v += v_dot * self.dt

        return p, v

    def _get_reward(self, next_p: float, next_v: float) -> float:
        if next_p < -1 or abs(next_v) > 3:
            return -1
        elif next_p > 1 and abs(next_v) <= 3:
            return 1
        return 0

    def _is_terminal(self, p: float, v: float) -> bool:
        return abs(p) > 1 or abs(v) > 3

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.frames = []
        p = self.np_random.uniform(*self.initial_position_range)
        v = self.initial_velocity

        self.state = np.array([p, v], dtype=np.float32)
        self.steps = 0

        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None, "Call reset before using step method."

        force = 4 if action == 1 else -4
        p, v = self.state
        next_p, next_v = self._dynamics(p, v, force)
        next_state = np.array([next_p, next_v], dtype=np.float32)

        reward = self._get_reward(next_p, next_v)
        terminated = self._is_terminal(next_p, next_v)
        truncated = False  # Infinite time horizon

        self.state = next_state
        self.steps += 1
        if self.render_mode == "gif":
            self.render(next_p, next_v)
        return next_state, reward, terminated, truncated, {}

    def render(self, position: float, velocity: float):
        """Render the current state of the environment."""
        if self.render_mode == "gif":
            frame = save_caronthehill_image(position, max(min(velocity, 3), -3))
            self.frames.append(frame)


    def save_gif(self, filename="car_on_hill.gif"):
        """Save the collected frames as a GIF."""
        if self.render_mode == "gif" and self.frames:
            imageio.mimsave(filename, self.frames, fps=10)
            print(f"GIF saved as {filename}")

# %% [markdown]
# You can render a trajectory using the following code

# %%
env = CarOnHillEnv(render_mode="gif")


num_steps = 100
state, _ = env.reset()
for _ in range(num_steps):
    action = env.action_space.sample() # We implement a random policy here
    next_state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state

env.save_gif("car_on_hill.gif")

# %% [markdown]
# ## Part 1: Fitted Q iteration

# %% [markdown]
# ### Question 1: Fitted Q Iteration Algorithm
#
# Implement the Fitted-Q-Iteration algorithm for the car on the hill environment. It should use a sklearn model for the regression algorithm. Propose two stopping rules for the computation of the $\widehat{Q}_N$-functions sequence and motivate them.
#
# ### Algorithm
#
# #### Inputs:
# - A set of four-tuples $\mathcal{F}$ (experience replay buffer)
# - A regression algorithm
#
# #### Initialization:
# - Set $N$ to 0.
# - Let $\hat{Q}_N$ be a function equal to zero everywhere on $\mathcal{S} \times \mathcal{A}$.
#
# #### Iterations:
# Repeat until stopping conditions are reached
#
# 1. **Increment Iteration Counter:**
#    - $N \leftarrow N + 1$.
#
# 2. **Build the Training Set:**
#    - Construct the training set $\mathcal{TS} = \{(i^l, o^l)\}, l = 1, \ldots, \#\mathcal{F}$ based on the function $\hat{Q}_{N-1}$ and on the full set of four-tuples $\mathcal{F}$:
#      \[
#      \begin{aligned}
#      i^l &= (s^l, a^l), \\
#      o^l &= r^l + \gamma \max_{a' \in \mathcal{A}} \hat{Q}_{N-1}(s^l_{+1}, a')
#      \end{aligned}
#      \]
#
# 3. **Induce the Function:**
#    - Use the regression algorithm to induce from $\mathcal{TS}$ the function $\hat{Q}_N(s, a)$.
#
#

# %%
# You can use the implementation here bellow as a starting point
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

class FittedQIteration:
    def __init__(self, model, gamma: float, action_space: List[int]):
        """
        Initialize the Fitted Q-Iteration algorithm.

        Parameters:
        - model: A regression model from scikit-learn used to approximate the Q-function.
        - gamma: The discount factor for future rewards.
        - action_space: A list of possible actions in the environment.
        """
        self.model = model
        self.gamma = gamma
        self.action_space = action_space
        self.q_function = None
        self.iteration_history = []

    def train(self, experience_replay: List[Tuple[np.ndarray, int, float, np.ndarray]],
              stopping_criteria: str, max_iterations: int = 100,
              convergence_threshold: float = 0.01):
        """
        Train the Q-function using the Fitted Q-Iteration algorithm.

        Parameters:
        - experience_replay: A list of experience tuples (state, action, reward, next_state).
        - stopping_criteria: The criteria to stop training ('iterations', 'convergence').
        - max_iterations: Maximum number of iterations for the 'iterations' stopping criteria.
        - convergence_threshold: Threshold for the 'convergence' stopping criteria.
        """
        # Initialize Q function to zero
        N = 0
        prev_q_values = np.zeros(len(experience_replay))

        pbar = tqdm(total=max_iterations)

        while True:
            N += 1
            pbar.update(1)

            # Build training set
            inputs = []
            targets = []

            for s, a, r, s_next in experience_replay:
                # Prepare input (state-action pair)
                inputs.append(np.concatenate([s, [self.action_space.index(a)]]))

                # Calculate target using the Bellman equation
                if s_next is None or self._is_terminal(s_next):  # Terminal state
                    target = r
                else:
                    # Get max Q-value for next state across all actions
                    next_q_values = np.array([self._predict_single(s_next, action)
                                             for action in self.action_space])
                    target = r + self.gamma * np.max(next_q_values)

                targets.append(target)

            # Convert to numpy arrays
            X = np.array(inputs)
            y = np.array(targets)

            # Fit regression model
            self.model.fit(X, y)

            # Save the model as the Q-function
            self.q_function = self.model

            # Check stopping criteria
            if stopping_criteria == 'iterations' and N >= max_iterations:
                break

            if stopping_criteria == 'convergence':
                # Calculate current Q-values for the same state-action pairs
                current_q_values = np.array([self._predict_single(s, a)
                                           for s, a, _, _ in experience_replay])

                # Calculate change in Q-values
                q_change = np.mean(np.abs(current_q_values - prev_q_values))
                self.iteration_history.append(q_change)

                # Store current Q-values for next iteration comparison
                prev_q_values = current_q_values.copy()

                # Check if converged
                if q_change < convergence_threshold:
                    break

        pbar.close()
        print(f"Training completed after {N} iterations")
        return self.iteration_history

    def _is_terminal(self, state):
        """Check if a state is terminal."""
        p, v = state
        return abs(p) > 1 or abs(v) > 3

    def _predict_single(self, state, action):
        """Predict Q-value for a single state-action pair."""
        if self.q_function is None:
            return 0

        # Prepare input for prediction
        X = np.concatenate([state, [self.action_space.index(action)]])
        X = X.reshape(1, -1)

        return self.q_function.predict(X)[0]

    def predict_Q(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the Q-values for all actions given a state.

        Parameters:
        - state: The current state for which to predict Q-values.

        Returns:
        - An array of Q-values for each action in the action space.
        """
        if self.q_function is None:
            return np.zeros(len(self.action_space))

        return np.array([self._predict_single(state, a) for a in self.action_space])

    def predict_action(self, state: np.ndarray) -> int:
        """
        Predict the best action for a given state based on the Q-function.

        Parameters:
        - state: The current state for which to predict the best action.

        Returns:
        - The action with the highest Q-value.
        """
        q_values = self.predict_Q(state)
        return self.action_space[np.argmax(q_values)]

    def plot_convergence(self):
        """Plot the convergence history of Q-values."""
        if len(self.iteration_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.iteration_history)
            plt.title('Convergence of Q-values')
            plt.xlabel('Iteration')
            plt.ylabel('Mean absolute change in Q-values')
            plt.grid(True)
            plt.show()

# %% [markdown]
# ### Question 2: Generating Sets of One-Step System Transitions
#
# Propose two strategies for generating sets of one-step system transitions and motivate them.

# %%
def generate_uniform_random_samples(env, n_samples=10000):
    """
    Generate one-step system transitions using uniform random sampling.

    Parameters:
    - env: The environment
    - n_samples: Number of samples to generate

    Returns:
    - A list of transitions (state, action, reward, next_state)
    """
    samples = []

    for _ in range(n_samples):
        # Generate random state within bounds
        p = np.random.uniform(-1, 1)
        v = np.random.uniform(-3, 3)
        state = np.array([p, v])

        # Choose random action
        action = np.random.choice([0, 1])  # 0: -4, 1: 4
        force = 4 if action == 1 else -4

        # Compute next state and reward
        next_p, next_v = env._dynamics(p, v, force)
        next_state = np.array([next_p, next_v])

        reward = env._get_reward(next_p, next_v)

        # Check if terminal
        if env._is_terminal(next_p, next_v):
            next_state = None

        samples.append((state, force, reward, next_state))

    return samples

def generate_episode_based_samples(env, n_episodes=100, max_steps=100):
    """
    Generate one-step system transitions by simulating episodes.

    Parameters:
    - env: The environment
    - n_episodes: Number of episodes to simulate
    - max_steps: Maximum steps per episode

    Returns:
    - A list of transitions (state, action, reward, next_state)
    """
    samples = []

    for _ in range(n_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            # Choose random action
            action = np.random.choice([0, 1])  # 0: -4, 1: 4
            force = 4 if action == 1 else -4

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store the transition
            samples.append((state.copy(), force, reward, next_state.copy() if not terminated else None))

            # Update state
            state = next_state.copy()

            # Check termination
            if terminated or truncated:
                break

    return samples

# %% [markdown]
# ## Question 3
#
# Use the following supervised learning techniques:
# - Linear Regression
# - Extremely Randomized Trees
# - Neural Networks
#
# Build and motivate your neural network structure.
# These techniques are implemented in the `scikit-learn` libraries.
# Derive the policy $\widehat{\mu}_*$ from $\widehat{Q}$ and display the Q-values and the policy in a colored 2D grid. Use red for action a = -4 and blue for action a = 4, with a resolution of 0.01 for the state space display.
#

# %%
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

models = {
    "LinearRegression": LinearRegression(),
    "MLPRegressor": MLPRegressor(
        hidden_layer_sizes=(64, 64),  # Two hidden layers with 64 neurons each
        activation='relu',            # ReLU activation function
        solver='adam',                # Adam optimizer
        alpha=0.0001,                 # L2 regularization
        batch_size='auto',            # Automatic batch size
        learning_rate='adaptive',     # Adaptive learning rate
        max_iter=1000,                # Maximum iterations
        early_stopping=True,          # Early stopping
        validation_fraction=0.1,      # Validation set size
        n_iter_no_change=10,          # Iterations with no improvement for early stopping
        random_state=42               # Random seed for reproducibility
    ),
    "ExtraTreesRegressor": ExtraTreesRegressor( # Hint: make some quick research in the litterature to find interesting parameters
        n_estimators=100,            # Number of trees
        max_depth=None,              # No maximum depth limit
        min_samples_split=2,         # Minimum samples to split a node
        min_samples_leaf=1,          # Minimum samples at leaf node
        max_features='auto',         # Auto feature selection
        bootstrap=False,             # Don't use bootstrap samples
        n_jobs=-1,                   # Use all available cores
        random_state=42              # Random seed for reproducibility
    )
}

# Define stopping conditions
stopping_conditions = ['iterations', 'convergence']

# Define tuple generation techniques
tuple_generation_techniques = [
    ('uniform', lambda env: generate_uniform_random_samples(env, n_samples=10000)),
    ('episode', lambda env: generate_episode_based_samples(env, n_episodes=100, max_steps=100))
]

def visualize_policy_and_qvalues(fqi, model_name, generation_technique, stopping_condition):
    """
    Visualize the Q-values and policy derived from the trained model.

    Parameters:
    - fqi: Trained FittedQIteration instance
    - model_name: Name of the model used
    - generation_technique: Name of the data generation technique
    - stopping_condition: Name of the stopping condition used
    """
    # Create a grid of states
    p_range = np.arange(-1, 1.01, 0.01)
    v_range = np.arange(-3, 3.01, 0.01)

    # Initialize arrays to store Q-values and policy
    q_values = np.zeros((len(p_range), len(v_range)))
    policy = np.zeros((len(p_range), len(v_range)))

    # Calculate Q-values and policy for each state
    for i, p in enumerate(p_range):
        for j, v in enumerate(v_range):
            state = np.array([p, v])

            # Skip states outside the valid region
            if abs(p) > 1 or abs(v) > 3:
                q_values[i, j] = np.nan
                policy[i, j] = np.nan
                continue

            # Get Q-values for each action
            q_vals = fqi.predict_Q(state)

            # Store max Q-value and best action
            q_values[i, j] = np.max(q_vals)
            policy[i, j] = np.argmax(q_vals)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Q-values
    im1 = ax1.imshow(q_values.T, origin='lower', aspect='auto',
                    extent=[-1, 1, -3, 3], cmap='viridis')
    ax1.set_title(f'Q-values - {model_name} - {generation_technique} - {stopping_condition}')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Velocity')
    plt.colorbar(im1, ax=ax1, label='Q-value')

    # Create custom colormap for policy (red for -4, blue for 4)
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'blue'], N=2)

    # Plot policy
    im2 = ax2.imshow(policy.T, origin='lower', aspect='auto',
                   extent=[-1, 1, -3, 3], cmap=cmap, vmin=0, vmax=1)
    ax2.set_title(f'Policy - {model_name} - {generation_technique} - {stopping_condition}')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Velocity')
    cbar = plt.colorbar(im2, ax=ax2, label='Action')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['-4', '4'])

    plt.tight_layout()
    plt.savefig(f'policy_{model_name}_{generation_technique}_{stopping_condition}.png')
    plt.show()

results = {}
trained_models = {}

env = CarOnHillEnv()

# Train FQI for each combination
for model_name, model in models.items():
    for gen_name, gen_func in tuple_generation_techniques:
        # Generate experience replay buffer
        experience_replay = gen_func(env)

        for stopping_condition in stopping_conditions:
            print(f"Training {model_name} with {gen_name} data using {stopping_condition} stopping condition")

            # Create FQI instance
            fqi = FittedQIteration(model=model, gamma=env.gamma, action_space=[-4, 4])

            # Train the model
            history = fqi.train(
                experience_replay=experience_replay,
                stopping_criteria=stopping_condition,
                max_iterations=50 if stopping_condition == 'iterations' else 100,
                convergence_threshold=0.01
            )

            # Store the trained model
            key = (model_name, gen_name, stopping_condition)
            trained_models[key] = fqi

            # Visualize policy and Q-values
            visualize_policy_and_qvalues(fqi, model_name, gen_name, stopping_condition)

# Choose a best model for evaluation
best_model_key = ('ExtraTreesRegressor', 'episode', 'convergence')  # Change this based on your evaluation
your_model = trained_models[best_model_key]

# %%
#play the policy

def estimate_expected_return(env, fqi, n_episodes=100, max_steps=200):
    """
    Estimate the expected return of a policy derived from FQI.

    Parameters:
    - env: The environment
    - fqi: Trained FittedQIteration instance
    - n_episodes: Number of episodes to simulate
    - max_steps: Maximum steps per episode

    Returns:
    - Average return across all episodes
    """
    total_return = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        step = 0
        gamma_power = 1  # γ^t

        for _ in range(max_steps):
            # Choose action according to the learned policy
            action_idx = np.argmax(fqi.predict_Q(state))
            action = 1 if fqi.action_space[action_idx] == 4 else 0  # Convert to env action

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update return
            episode_return += gamma_power * reward
            gamma_power *= env.gamma

            # Update state
            state = next_state
            step += 1

            # Check termination
            if terminated or truncated:
                break

        total_return += episode_return

    return total_return / n_episodes

# Create a table of expected returns
results_table = np.zeros((len(models), len(tuple_generation_techniques), len(stopping_conditions)))

for i, model_name in enumerate(models.keys()):
    for j, (gen_name, _) in enumerate(tuple_generation_techniques):
        for k, stopping_condition in enumerate(stopping_conditions):
            key = (model_name, gen_name, stopping_condition)
            if key in trained_models:
                fqi = trained_models[key]
                expected_return = estimate_expected_return(env, fqi)
                results_table[i, j, k] = expected_return
                print(f"Expected return for {model_name}, {gen_name}, {stopping_condition}: {expected_return:.4f}")

# Display results in a formatted table
print("\nExpected Returns:")
print("-" * 80)
print(f"{'Model':<20} | {'Generation':<10} | {'Stopping':<10} | {'Return':<10}")
print("-" * 80)

for i, model_name in enumerate(models.keys()):
    for j, (gen_name, _) in enumerate(tuple_generation_techniques):
        for k, stopping_condition in enumerate(stopping_conditions):
            print(f"{model_name:<20} | {gen_name:<10} | {stopping_condition:<10} | {results_table[i, j, k]:<10.4f}")

# Play the policy of the best model
env = CarOnHillEnv(render_mode="gif")
best_model = trained_models[best_model_key]

num_steps = 100
state, _ = env.reset()
for _ in range(num_steps):
    q_values = best_model.predict_Q(state)
    action_idx = np.argmax(q_values)
    action = 1 if best_model.action_space[action_idx] == 4 else 0  # Convert to env action
    next_state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state

env.save_gif("car_on_hill.gif")

# %% [markdown]
# ### Question 4: Estimate and Display Expected Return
#
# Estimate and display the expected return of $\widehat{\mu}_N^*$ in a table for each:
#   - Supervised learning algorithm.
#   - One-step system transitions generation strategy.
#   - Stopping rule.

# %%
def estimate_expected_return(env, fqi, n_episodes=100, max_steps=200):
    """
    Estimate the expected return of a policy derived from FQI.

    Parameters:
    - env: The environment
    - fqi: Trained FittedQIteration instance
    - n_episodes: Number of episodes to simulate
    - max_steps: Maximum steps per episode

    Returns:
    - Average return across all episodes
    """
    total_return = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        step = 0
        gamma_power = 1  # γ^t

        for _ in range(max_steps):
            # Choose action according to the learned policy
            action_idx = np.argmax(fqi.predict_Q(state))
            action = 1 if fqi.action_space[action_idx] == 4 else 0  # Convert to env action

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update return
            episode_return += gamma_power * reward
            gamma_power *= env.gamma

            # Update state
            state = next_state
            step += 1

            # Check termination
            if terminated or truncated:
                break

        total_return += episode_return

    return total_return / n_episodes

# Create a table of expected returns
results_table = np.zeros((len(models), len(tuple_generation_techniques), len(stopping_conditions)))

for i, model_name in enumerate(models.keys()):
    for j, (gen_name, _) in enumerate(tuple_generation_techniques):
        for k, stopping_condition in enumerate(stopping_conditions):
            key = (model_name, gen_name, stopping_condition)
            if key in trained_models:
                fqi = trained_models[key]
                expected_return = estimate_expected_return(env, fqi)
                results_table[i, j, k] = expected_return
                print(f"Expected return for {model_name}, {gen_name}, {stopping_condition}: {expected_return:.4f}")

# Display results in a formatted table
print("\nExpected Returns:")
print("-" * 80)
print(f"{'Model':<20} | {'Generation':<10} | {'Stopping':<10} | {'Return':<10}")
print("-" * 80)

for i, model_name in enumerate(models.keys()):
    for j, (gen_name, _) in enumerate(tuple_generation_techniques):
        for k, stopping_condition in enumerate(stopping_conditions):
            print(f"{model_name:<20} | {gen_name:<10} | {stopping_condition:<10} | {results_table[i, j, k]:<10.4f}")

# Play the policy of the best model
env = CarOnHillEnv(render_mode="gif")
best_model = trained_models[best_model_key]

num_steps = 100
state, _ = env.reset()
for _ in range(num_steps):
    q_values = best_model.predict_Q(state)
    action_idx = np.argmax(q_values)
    action = 1 if best_model.action_space[action_idx] == 4 else 0  # Convert to env action
    next_state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state

env.save_gif("car_on_hill.gif")

# %% [markdown]
# ### Question 5: Results Discussion
#
# Discuss the impact on the results for each:
# - Supervised learning algorithm.
# - One-step system transitions generation strategies.
# - Stopping rules.
#

# %% [markdown]
# #### Answer:
