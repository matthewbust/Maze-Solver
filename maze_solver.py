import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import deque
import random

class Maze:
    """5x5 Maze environment with walls and navigable spaces"""
    
    def __init__(self, maze_array=None):
        """
        Initialize maze
        maze_array: 5x5 numpy array where 1=wall, 0=navigable
        """
        if maze_array is None:
            self.maze = self.generate_solvable_maze()
        else:
            self.maze = maze_array
        
        self.size = 5
        self.start_pos = (0, 0)  # Top-left
        self.goal_pos = (4, 4)   # Bottom-right
        self.reset()
    
    def generate_solvable_maze(self):
        """Generate a random solvable 5x5 maze"""
        maze = np.zeros((5, 5))
        
        # Add random walls (20% density - less for smaller maze)
        for i in range(5):
            for j in range(5):
                if random.random() < 0.1:
                    maze[i, j] = 1
        
        # Ensure start and goal are navigable
        maze[0, 0] = 0
        maze[4, 4] = 0
        
        # Simple path guarantee: clear a path from start to goal
        # Clear right path
        for i in range(5):
            maze[0, i] = 0
        # Clear down path
        for i in range(5):
            maze[i, 4] = 0
        
        return maze
    
    def reset(self):
        """Reset agent to start position."""
        self.agent_pos = list(self.start_pos)
        return self.get_state()
    
    def get_state(self):
        """Return flattened state representation"""
        state = np.copy(self.maze).astype(float)
        # Mark agent position
        state[self.agent_pos[0], self.agent_pos[1]] = 0.5
        return state.flatten()
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        Actions are 0=left, 1=up, 2=right, 3=down
        """
        old_pos = self.agent_pos.copy()
        
        # Map actions to movements
        if action == 0:   # left
            self.agent_pos[1] -= 1
        elif action == 1: # up
            self.agent_pos[0] -= 1
        elif action == 2: # right
            self.agent_pos[1] += 1
        elif action == 3: # down
            self.agent_pos[0] += 1
        
        # Check boundaries
        if (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.size or
            self.agent_pos[1] < 0 or self.agent_pos[1] >= self.size):
            self.agent_pos = old_pos
            return self.get_state(), -0.75, False, {'hit': 'boundary'}
        
        # Check walls
        if self.maze[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos
            return self.get_state(), -0.75, False, {'hit': 'wall'}
        
        # Check if goal reached
        if tuple(self.agent_pos) == self.goal_pos:
            return self.get_state(), 20.0, True, {'result': 'win'}
        
        # Movement cost
        return self.get_state(), -0.1, False, {'result': 'move'}
    
    def render(self, ax=None):
        """Visualize the maze"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        # Create visual representation
        visual = np.copy(self.maze)
        visual[self.agent_pos[0], self.agent_pos[1]] = 0.5  # Agent
        visual[self.goal_pos[0], self.goal_pos[1]] = 0.7    # Goal
        
        ax.clear()
        ax.imshow(visual, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('Maze Environment (5x5)')
        ax.grid(True, which='both', color='black', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, 5, 1))
        ax.set_yticks(np.arange(-0.5, 5, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        return ax


class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # 4 actions
        
        # Hyperparameters
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.003
        self.batch_size = 32
        
        # Experience replay memory
        self.memory = deque(maxlen=2000)
        
        # Build neural network
        self.model = self.build_model()
    
    def build_model(self):
        """Build Deep Q-Network using Keras."""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Q-learning update
        targets = rewards + self.gamma * np.amax(
            self.model.predict(next_states, verbose=0), axis=1
        ) * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load_model(self, filepath='maze_dqn_model.keras'):
        """Load trained model from disk"""
        self.model = keras.models.load_model(filepath)
        self.epsilon = self.epsilon_min  # Use trained model (no exploration)
        print(f"Model loaded from {filepath}")


def train_agent(episodes=100, max_steps=50, render_freq=50):
    """
    Train the DQN agent on maze solving
    Game ends at threshold or completion
    Game starts from position and runs until completion/loss
    """
    maze = Maze()
    state_size = 25  # 5x5 flattened
    action_size = 4   # 4 movement options
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    win_history = []
    
    print("Training Deep Q-Learning Agent...")
    print(f"Maze Size: 5x5 | Episodes: {episodes} | Max Steps: {max_steps}\n")
    
    for episode in range(episodes):
        state = maze.reset()  # Start from position
        total_reward = 0
        won = False
        
        for step in range(max_steps):  # Maximum steps threshold
            action = agent.act(state)
            next_state, reward, done, info = maze.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:  # Game ends on completion
                won = True
                break
        
        agent.replay()  # Replay experiences
        
        scores.append(total_reward)
        win_history.append(1 if won else 0)
        
        # Calculate win rate over last 20 episodes
        win_rate = np.mean(win_history[-20:]) if len(win_history) >= 20 else 0
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {total_reward:.2f} | "
                  f"Win: {'Yes' if won else 'No'} | "
                  f"Win Rate (20): {win_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, maze, scores, win_history

def test_agent(agent, maze, num_episodes=5):
    """Test trained agent"""
    print("\n=== Testing Trained Agent ===")
    
    wins = 0
    for episode in range(num_episodes):
        state = maze.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 50:
            action = np.argmax(agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            state, reward, done, info = maze.step(action)
            total_reward += reward
            steps += 1
        
        if done:
            wins += 1
            print(f"Test {episode + 1}: WIN in {steps} steps (Reward: {total_reward:.2f})")
        else:
            print(f"Test {episode + 1}: FAILED after {steps} steps (Reward: {total_reward:.2f})")
    
    print(f"\nTest Win Rate: {wins}/{num_episodes} = {wins/num_episodes:.1%}")


if __name__ == "__main__":
    # Train the agent
    agent, maze, scores, win_history = train_agent(episodes=100, max_steps=50)
    
    # visualize_training(scores, win_history)
    test_agent(agent, maze, num_episodes=5)