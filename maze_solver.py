import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Maze:
    """5x5 Grid environment with walls to form a maze"""
    
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
        
        # Add random walls (.10 = 10% density)
        for i in range(5):
            for j in range(5):
                if random.random() < 0.1:
                    maze[i, j] = 1
        
        # Ensure start and goal are open
        maze[0, 0] = 0
        maze[4, 4] = 0
        
        # Clear a guaranteed path (the top row and right column will be open)
        for i in range(5):
            maze[0, i] = 0  # Top row
        for i in range(5):
            maze[i, 4] = 0  # Right column
        
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
        
        # Calculate distance to goal before move
        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
        
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
            return self.get_state(), -1.0, False, {'hit': 'boundary'}
        
        # Check walls
        if self.maze[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos
            return self.get_state(), -1.0, False, {'hit': 'wall'}
        
        # Check if goal reached
        if tuple(self.agent_pos) == self.goal_pos:
            return self.get_state(), 10.0, True, {'result': 'win'}
        
        # Distance-based reward
        new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        distance_reward = (old_dist - new_dist) * 0.1  # Reward for getting closer
        
        # Small step penalty
        return self.get_state(), distance_reward - 0.04, False, {'result': 'move'}

class DQNAgent:
    """Deep Q-Network Agent with Target Network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # 4 actions
        
        # Hyperparameters
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_target_freq = 10  # Update target network every N episodes
        
        # Experience replay memory
        self.memory = deque(maxlen=5000)
        
        # Build main and target networks
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        """Build Deep Q-Network using Keras."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
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
        """Train on batch of experiences using target network"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        targets = rewards + self.gamma * np.amax(
            self.target_model.predict(next_states, verbose=0), axis=1
        ) * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def animate_test(maze, path, delay=350):
    """Animate the agent solving a maze."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Maze grid: walls=1 -> black, free=0 -> white
    ax.imshow(maze.maze, cmap="gray_r")

    # Draw start + goal
    ax.text(0, 0, "S", color="green", ha="center", va="center", fontsize=14)
    ax.text(4, 4, "G", color="blue", ha="center", va="center", fontsize=14)

    # Agent point & # Path marker (red) (cyan)
    agent_dot, = ax.plot([], [], "ro", markersize=10)
    path_line, = ax.plot([], [], "c-", linewidth=2)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linewidth=1)

    def update(frame):
        # Build path so far
        xs = [p[1] for p in path[:frame+1]]
        ys = [p[0] for p in path[:frame+1]]

        path_line.set_data(xs, ys)
        agent_dot.set_data(xs[-1], ys[-1])

        return agent_dot, path_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), interval=delay, blit=True, repeat=False
    )

    plt.show()

def train_agent(episodes=300, max_steps=100):
    """
    Train the DQN agent on maze solving with multiple random mazes
    """
    state_size = 25  # 5x5 flattened
    action_size = 4   # 4 movement options
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    win_history = []
    
    print("Begin Training")
    print(f"Maze Size: 5x5 | Episodes: {episodes} | Max Steps: {max_steps}\n")
    
    for episode in range(episodes):
        maze = Maze()
        state = maze.reset()
        total_reward = 0
        won = False
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = maze.step(action)
            
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
        
        win_rate = np.mean(win_history[-20:]) if len(win_history) >= 20 else 0
        
        if (episode + 1) % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Win Rate (20): {win_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, win_history

def test_agent(agent, num_episodes=5):
    """Test trained agent on new random mazes"""
    print("\n=== Testing Trained Agent on New Mazes ===")
    
    wins = 0
    total_steps = []
    test_results = []
    
    for episode in range(num_episodes):
        maze = Maze()
        state = maze.reset()
        done = False
        steps = 0
        total_reward = 0
        path = [(0, 0)]
        
        while not done and steps < 100:
            action = np.argmax(agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            state, reward, done, info = maze.step(action)
            path.append(tuple(maze.agent_pos))
            total_reward += reward
            steps += 1
        
        test_results.append((maze, path, done, steps, total_reward))
        
        if done:
            wins += 1
            total_steps.append(steps)
            print(f"Test {episode + 1}: WIN in {steps} steps (Reward: {total_reward:.2f})")
        else:
            print(f"Test {episode + 1}: FAILED after {steps} steps (Reward: {total_reward:.2f})")
    
    print(f"\nTest Win Rate: {wins}/{num_episodes} = {wins/num_episodes:.1%}")
    if total_steps:
        print(f"Avg steps to win: {np.mean(total_steps):.1f}")

    # TEXT + VISUALS
    for i, (maze, path, won, steps, reward) in enumerate(test_results):
        print(f"\nText view of Test {i + 1}: {'WIN' if won else 'FAILED'}, {steps} steps, reward={reward:.2f}")
        print_maze_with_path(maze, path)
        animate_test(maze, path, delay=350)   # <-- animation

def print_maze_with_path(maze, path):
    """Print the maze grid with the agent's path"""
    grid = []
    
    for i in range(5):
        row = []
        for j in range(5):
            if maze.maze[i, j] == 1:
                row.append('#')
            else:
                row.append(' ')
        grid.append(row)
    
    for pos in path[:-1]:
        if grid[pos[0]][pos[1]] == ' ':
            grid[pos[0]][pos[1]] = '.'
    
    grid[0][0] = 'S'
    grid[4][4] = 'G'
    
    final_pos = path[-1]
    if final_pos != (4, 4) and final_pos != (0, 0):
        grid[final_pos[0]][final_pos[1]] = 'X'
    
    print("  " + "-" * 11)
    for i, row in enumerate(grid):
        print(f"{i} | " + " ".join(row) + " |")
    print("  " + "-" * 11)
    print("    0 1 2 3 4")

if __name__ == "__main__":
    agent, scores, win_history = train_agent(episodes=300, max_steps=100)
    test_agent(agent, num_episodes=5)
