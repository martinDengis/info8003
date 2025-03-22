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

    def train(self, experience_replay: List[Tuple[np.ndarray, int, float, np.ndarray]], stopping_criteria: str):
        """
        Train the Q-function using the Fitted Q-Iteration algorithm.

        Parameters:
        - experience_replay: A list of experience tuples (state, action, reward, next_state).
        - stopping_criteria: The criteria to stop training.
        """
        # Initialize Q function to zero
        if self.q_function is None:
            self.q_function = lambda s, a: 0

        inputs = []
        targets = []


    def predict_Q(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the Q-values for all actions given a state.

        Parameters:
        - state: The current state for which to predict Q-values.

        Returns:
        - An array of Q-values for each action in the action space.
        """
        return np.array([self.q_function(state, a) for a in self.action_space])

    def predict_action(self, state: np.ndarray) -> int:
        """
        Predict the best action for a given state based on the Q-function.

        Parameters:
        - state: The current state for which to predict the best action.

        Returns:
        - The action with the highest Q-value.
        """
        return np.argmax(np.array([self.q_function(state, a) for a in self.action_space]))

# %% [markdown]
# ### Question 2: Generating Sets of One-Step System Transitions
#
# Propose two strategies for generating sets of one-step system transitions and motivate them.

# %%
# your code

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

models = {
    "LinearRegression": LinearRegression(),
    "MLPRegressor": MLPRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor() # Hint: make some quick research in the litterature to find interesting parameters
}

# Define stopping conditions
stopping_conditions = []

# Define tuple generation techniques
tuple_generation_techniques = []

trained_models = {}

# Train FQI for each combination
for model_name, model in models.items():
    for stopping_condition in stopping_conditions:
        for technique in tuple_generation_techniques:
            pass

# %%
#play the policy

env = CarOnHillEnv(render_mode="gif")

num_steps = 100
state, _ = env.reset()
for _ in range(num_steps):
    action = your_model.predict(state)
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
# your code

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

# %% [markdown]
# ## Part 2: Parametric Q-Learning
#

# %% [markdown]
#
# ### Question 1: Parametric Q-Learning Algorithm
#
# Implement a routine which computes a parametrized approximation of the Q-function via the Parametric Q-Learning algorithm. Use a neural network as the approximation architecture, and motivate its structure.
#

# %%
import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List, Tuple

class ParametricQLearning:
    def __init__(self, model, gamma: float, action_space: List[int], learning_rate: float = 0.01):
        """
        Initialize the Parametric Q-Learning algorithm.

        Parameters:
        - model: A scikit-learn model used for approximating the Q-function.
        - gamma: Discount factor for future rewards.
        - action_space: List of possible actions.
        - learning_rate: Learning rate for updating the Q-function.
        """
        self.model = model
        self.gamma = gamma
        self.action_space = action_space
        self.learning_rate = learning_rate
        # Initialize Q-function as a lambda function
        self.q_function = lambda s, a: self.model.predict([np.append(s, a)])[0]

    def train(self, env, num_episodes: int, max_steps: int):
        """
        Train the Q-learning model over a number of episodes.

        Parameters:
        - env: The environment to train on.
        - num_episodes: Number of episodes to train for.
        - max_steps: Maximum number of steps per episode.
        """

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - An action from the action space.
        """
        pass

    def update_q_function(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Update the Q-function using the Bellman equation.

        Parameters:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state after taking the action.
        """
        pass

    def predict_action(self, state: np.ndarray) -> int:
        """
        Predict the best action for a given state.

        Parameters:
        - state: The current state.

        Returns:
        - The action with the highest Q-value.
        """
        pass

# %% [markdown]
# ### Question 2: Policy Derivation and Visualization
#
# Derive the policy $\widehat{\mu}_*$ from $\widehat{Q}$ and display it in a colored 2D grid. Use red for action a = -4 and blue for action a = 4, with a resolution of 0.01 for the state space display.
#

# %%
# your code

# %% [markdown]
# ### Question 3: Expected Return Estimation
#
# Estimate and show the expected return of $\widehat{\mu}^*$.
#

# %%
# your code

# %% [markdown]
# ### Question 4: Experimental Protocol Design
#
# Design an experimental protocol to compare Fitted Q Iteration (FQI) and Parametric Q-Learning. Use a curve plot where the x-axis represents the number of one-step system transitions and the y-axis represents the expected return.

# %%
# your code

# %% [markdown]
# ### Question 5: Results Discussion
#
# Discuss the results obtained by running the experimental protocol. Consider the differences in performance between FQI and Parametric Q-Learning, and any insights gained from the comparison.

# %% [markdown]
# #### Answer:


