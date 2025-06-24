import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class TradingEnvironment:
    def __init__(self, data, initial_balance=100000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        return self.get_state()

    def get_state(self):
        if self.current_step >= len(self.data):
            return None

        row = self.data.iloc[self.current_step]
        state = np.array([
            row['Close'] / 10000,  # Normalized price
            row['Volume'] / 1000000,  # Normalized volume
            row['RSI'] / 100,  # RSI
            row['MACD'],  # MACD
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held / 1000,  # Normalized shares
        ])
        return state

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}

        current_price = self.data.iloc[self.current_step]['Close']

        # Actions: 0=Hold, 1=Buy, 2=Sell
        reward = 0

        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price

        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance += self.shares_held * current_price
            reward = self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        next_state = self.get_state()

        # Calculate reward based on portfolio value change
        portfolio_value = self.balance + self.shares_held * current_price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance

        done = self.current_step >= len(self.data) - 1

        return next_state, reward, done, {}

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        target = rewards + 0.95 * np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target

        self.model.fit(states, target_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
