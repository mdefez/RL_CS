import numpy as np
from collections import defaultdict
from tqdm import tqdm


class MonteCarloControlAgent:
    def __init__(self, action_space_shape, alpha = 0.01, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_shape = action_space_shape
        self.alpha = alpha
        self.name = "Monte-Carlo"

        self.Q = defaultdict(lambda: np.zeros(self.action_space_shape))


    def reset(self):
        self.Q = defaultdict(lambda: np.zeros(self.action_space_shape))

    def policy(self, state):        # Epsilon greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_shape)
        return np.argmax(self.Q[state])

    def generate_episode(self, env):
        episode = []
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = self.policy(state)
            next_state, reward, done, _, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            score += reward
            if score == 3_000:      # Finish and add big reward
                episode.pop(0)
                episode.append((state, action, 50))
                done = True

        return episode, score

    def update(self, episode):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                self.Q[state][action] += self.alpha * (G - self.Q[state][action])
                visited.add((state, action))

    def train(self, env, num_episodes=1000):
        self.reset()
        for _ in tqdm(range(num_episodes), desc = "Training MC"):
            episode, score = self.generate_episode(env)
            self.update(episode)

    def test(self, env, n_runs):
        scores = []
        
        old_epsilon = self.epsilon
        self.epsilon = 0

        for _ in tqdm(range(n_runs), desc = "Testing"):
            scores.append(self.generate_episode(env)[1])
        
        self.epsilon = old_epsilon

        return scores
    




class SarsaLambdaAgent:
    def __init__(self, action_space_shape, gamma=0.99, alpha=0.1, epsilon=0.1, lam=0.9):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lam = lam
        self.action_space_shape = action_space_shape
        self.name = "SARSA"

        self.Q = defaultdict(lambda: np.zeros(self.action_space_shape))
        self.E = defaultdict(lambda: np.zeros(self.action_space_shape)) 

    def reset(self):
        self.Q = defaultdict(lambda: np.zeros(self.action_space_shape))
        self.E = defaultdict(lambda: np.zeros(self.action_space_shape))  

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_shape)
        return np.argmax(self.Q[state])

    def reset_traces(self):
        self.E = defaultdict(lambda: np.zeros(self.action_space_shape))

    def generate_episode(self, env):
        episode = []
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = self.policy(state)
            next_state, reward, done, _, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            score += reward
            if score == 3_000:
                done = True

        return episode, score

    def train_episode(self, env):
        state, _ = env.reset()
        action = self.policy(state)

        self.reset_traces()
        done = False

        score = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated

            next_action = self.policy(next_state)

            td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

            self.E[state][action] += 1

            for s in self.E:
                self.Q[s] += self.alpha * td_error * self.E[s]
                self.E[s] *= self.gamma * self.lam

            state = next_state
            action = next_action

        return score

    def train(self, env, num_episodes=1000):
        self.reset()
        for _ in tqdm(range(num_episodes), desc = "Training SARSA"):
            self.train_episode(env)


    def test(self, env, n_runs):
        scores = []

        old_epsilon = self.epsilon
        self.epsilon = 0
        for _ in tqdm(range(n_runs), desc = "Testing"):
            scores.append(self.generate_episode(env)[1])
        
        self.epsilon = old_epsilon

        return scores

