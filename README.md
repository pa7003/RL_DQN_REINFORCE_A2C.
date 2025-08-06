# RL_DQN_REINFORCE_A2C.
This repository contains the implementation of three fundamental Reinforcement Learning algorithms: Deep Q-Network (DQN), Policy Gradient (REINFORCE), and Actor-Critic (A2C). All algorithms are implemented using PyTorch and trained on the `CartPole-v1` environment from Gymnasium.

## Table of Contents

1.  [Assignment Overview](#assignment-overview)
2.  [Environment](#environment)
3.  [Implemented Algorithms](#implemented-algorithms)
4.  [Setup and Installation](#setup-and-installation)
5.  [How to Run the Code](#how-to-run-the-code)
6.  [Evaluation and Analysis](#evaluation-and-analysis)
7.  [Expected Deliverables](#expected-deliverables)

## Assignment Overview

The goal of this assignment is to implement and analyze the learning behavior of DQN, REINFORCE, and Actor-Critic agents. Each agent is trained for a specified number of episodes, and their performance is evaluated using plots, key metrics, and a comparative discussion.

## Environment

The chosen environment for all implementations is `CartPole-v1` from OpenAI Gymnasium.

* **Observation Space:** A 4-dimensional continuous space representing the cart's position, velocity, pole's angle, and angular velocity.
* **Action Space:** A 2-dimensional discrete space representing actions: 0 (push cart left) and 1 (push cart right).
* **Reward:** +1 for every step the pole remains upright.
* **Episode Termination:** The pole falls over (angle > $\pm$12 degrees), cart moves off screen (position > $\pm$2.4), or episode length exceeds 500 steps. The environment is considered "solved" if the average reward over 100 consecutive episodes is $\geq$ 195.0.

## Implemented Algorithms

1.  **Deep Q-Network (DQN):**
    * Neural Network for Q-value estimation.
    * Experience Replay Buffer.
    * Target Network mechanism.
    * $\epsilon$-greedy policy for exploration.
2.  **Policy Gradient (REINFORCE):**
    * Stochastic Policy Network with logits output.
    * Monte Carlo Return calculation.
    * Loss computation using log-probability of actions.
3.  **Actor-Critic (A2C):**
    * Separate Actor (policy) and Critic (value) networks.
    * Temporal Difference (TD) error computation.
    * Advantage estimation.
    * Entropy regularization (optional but included).

## Setup and Installation

To run the provided Jupyter Notebook (`RL_Assignment_DQN_REINFORCE_A2C.ipynb`), you will need Python 3.8+ and the following libraries. It is highly recommended to use Google Colab for execution, as it provides a ready-to-use environment with GPU access.

1.  **Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com/).
2.  **Upload the Notebook:** Upload `RL_Assignment_DQN_REINFORCE_A2C.ipynb`.
3.  **Change Runtime Type:** Go to `Runtime` -> `Change runtime type` and select `GPU` under `Hardware accelerator` for faster training, although `CartPole-v1` can run on CPU.

The necessary libraries are installed within the notebook's first cell:
```bash
!pip install gymnasium[classic_control]
!pip install pygame
```

## How to Run the Code

1.  **Open the Notebook:** Load `RL_Assignment_DQN_REINFORCE_A2C.ipynb` in Google Colab.
2.  **Run All Cells:** Execute all cells sequentially. You can do this by going to `Runtime` -> `Run all`.
3.  **Observe Training Progress:** The notebook will print real-time updates on episode rewards and moving averages for each algorithm.
4.  **View Plots:** After training, performance plots will be displayed, including a consolidated comparison.
5.  **Check Metrics:** Evaluation metrics (final average reward, convergence episode) will be printed.
6.  **Inference Demonstration:** The notebook will load the saved models and run a few episodes with rendering for each algorithm, demonstrating their learned policies.

## Evaluation and Analysis

The notebook includes dedicated sections for:

* **Consolidated Plots:** A single plot comparing the moving average rewards of all three algorithms.
* **Hyperparameter Table:** A table summarizing the hyperparameters used for each agent.
* **Evaluation Metrics:** Code to calculate and report:
    * Final average reward over the last 100 episodes.
    * Number of episodes to converge (defined as achieving an average reward of $\geq$ 195.0 over 100 consecutive episodes for CartPole-v1).
* **Discussion and Analysis:** A detailed qualitative discussion (provided within the notebook) on the learning curves, stability, variance, and comparative performance of DQN, REINFORCE, and Actor-Critic.


