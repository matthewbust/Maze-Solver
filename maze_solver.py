import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt 

class Maze:
    """5x5 Grid environment with walls to form a maze"""
    
    def __init__(self, maze_array=None):
        self.size = 10
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        if maze_array is None:
            self.maze = self.generate_solvable_maze()
        else:
            self.maze = maze_array
        self.reset()

    def generate_solvable_maze(self):
        maze = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.2:
                    maze[i, j] = 1
        
        maze[self.start_pos] = 0
        maze[self.goal_pos] = 0
        
        for i in range(self.size):
            maze[0, i] = 0
        for i in range(self.size):
            maze[i, self.size-1] = 0
        
        return maze
    
    def reset(self):
        self.agent_pos = list(self.start_pos)
        return self.get_state()
    
    def get_state(self):
        state = np.copy(self.maze).astype(float)
        state[self.agent_pos[0], self.agent_pos[1]] = 0.5
        return state.flatten()
    
    def step(self, action):
        old_pos = self.agent_pos.copy()
        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
        
        if action == 0:
            self.agent_pos[1] -= 1
        elif action == 1:
            self.agent_pos[0] -= 1
        elif action == 2:
            self.agent_pos[1] += 1
        elif action == 3:
            self.agent_pos[0] += 1
        
        if (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.size or
            self.agent_pos[1] < 0 or self.agent_pos[1] >= self.size):
            self.agent_pos = old_pos
            return self.get_state(), -0.1, False, {'hit': 'boundary'}
        
        if self.maze[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos
            return self.get_state(), -0.2, False, {'hit': 'wall'}
        
        if tuple(self.agent_pos) == self.goal_pos:
            return self.get_state(), 10, True, {'result': 'win'}
        
        new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        distance_reward = (old_dist - new_dist)
        
        return self.get_state(), distance_reward - 0.1, False, {'result': 'move'}


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.batch_size = 128
        self.update_target_freq = 10
        
        self.memory = deque(maxlen=50000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        targets = rewards + self.gamma * np.amax(
            self.target_model.predict(next_states, verbose=0), axis=1
        ) * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes=300, max_steps=100):
    state_size = 100
    action_size = 4
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    win_history = []
    
    print("Begin Training")
    print(f"Maze Size: 10x10 | Episodes: {episodes} | Max Steps: {max_steps}\n")
    
    for episode in range(episodes):
        maze = Maze()
        state = maze.reset()
        total_reward = 0
        won = False
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = maze.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                won = True
                break
        
        agent.replay()
        
        if (episode + 1) % agent.update_target_freq == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        win_history.append(1 if won else 0)
        
        if (episode + 1) % 20 == 0:
            avg_score = np.mean(scores[-20:])
            win_rate = np.mean(win_history[-20:])
            print(f"Episode {episode+1}/{episodes} | Avg Score: {avg_score:.2f} | Win Rate: {win_rate:.2%} | Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, win_history


def test_agent(agent, num_episodes=5):
    wins = 0
    test_results = []
    
    for episode in range(num_episodes):
        maze = Maze()
        state = maze.reset()
        done = False
        steps = 0
        reward_sum = 0
        path = [(0, 0)]
        
        while not done and steps < 100:
            action = np.argmax(agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            state, reward, done, _ = maze.step(action)
            path.append(tuple(maze.agent_pos))
            reward_sum += reward
            steps += 1
        
        test_results.append((maze, path, done, steps, reward_sum))
        
        if done:
            wins += 1
            print(f"Test {episode+1}: WIN in {steps} steps")
        else:
            print(f"Test {episode+1}: FAILED after {steps} steps")
    
    return test_results



#  VISUALIZATION OF PATH

def plot_path(maze, path, index):
    grid = np.copy(maze.maze).astype(float)

    for (x, y) in path:
        grid[x, y] = 0.5

    grid[0, 0] = 0.8
    grid[maze.size-1, maze.size-1] = 0.9

    plt.subplot(1, 5, index+1)
    plt.imshow(grid, cmap="viridis")
    plt.title(f"Test {index+1}")
    plt.axis("off")


# RUN


if __name__ == "__main__":
    agent, scores, win_history = train_agent(episodes=1000, max_steps=200)
    
    results = test_agent(agent, num_episodes=5)

    plt.figure(figsize=(20, 4))
    for i, (maze, path, done, steps, reward) in enumerate(results):
        plot_path(maze, path, i)

    plt.tight_layout()
    plt.show()
