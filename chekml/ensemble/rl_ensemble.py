import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class RLEnsemble:
    """
    RL-based ensemble where an agent learns to select models.
    Special function: Q-learning for adaptive model selection.
    """
    def __init__(self, models, num_episodes=100):
        self.models = list(models.values())  # List of trained models
        self.model_names = list(models.keys())
        self.num_episodes = num_episodes
        self.q_table = np.zeros(len(self.models))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        for model in self.models:
            model.fit(X_train, y_train)

        for episode in range(self.num_episodes):
            action = self._select_action()
            y_pred = self.models[action].predict(X_val)
            reward = -mean_squared_error(y_val, y_pred)
            self._update_q_table(action, reward)

    def _select_action(self):
        if np.random.rand() < self.epsilon:
            return random.randint(0, len(self.models) - 1)
        return np.argmax(self.q_table)

    def _update_q_table(self, action, reward):
        self.q_table[action] += self.alpha * (reward + self.gamma * np.max(self.q_table) - self.q_table[action])
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        best_action = np.argmax(self.q_table)
        return self.models[best_action].predict(X_test)
