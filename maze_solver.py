import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import os
import json
import matplotlib.pyplot as plt

class Maze:
    """10x10 Grid environment with walls to form a maze"""
    
    def __init__(self, maze_array=None):
        """
        Initialize maze
        maze_array: 10x10 numpy array where 1=wall, 0=navigable
        """
        self.size = 10
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        if maze_array is None:
            self.maze = self.generate_solvable_maze()
        else:
            self.maze = maze_array
        self.reset()

    def generate_solvable_maze(self):
        """Generate a random solvable 10x10 maze with substantial walls using BFS to verify path exists"""
        max_attempts = 20
        
        for attempt in range(max_attempts):
            maze = np.zeros((self.size, self.size))
            
            # Add random walls with HIGHER density for more challenge
            for i in range(self.size):
                for j in range(self.size):
                    if random.random() < 0.40:  # INCREASED: 40% wall density (was 25%)
                        maze[i, j] = 1
            
            # Ensure start and goal are open
            maze[self.start_pos] = 0
            maze[self.goal_pos] = 0
            
            # Verify path exists using BFS
            if self._has_path(maze):
                return maze
        
        # Fallback: create maze with guaranteed path
        return self._create_guaranteed_path_maze()
    
    def _has_path(self, maze):
        """Check if a path exists from start to goal using BFS"""
        from collections import deque
        
        queue = deque([self.start_pos])
        visited = set([self.start_pos])
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        while queue:
            current = queue.popleft()
            
            if current == self.goal_pos:
                return True
            
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                # Check boundaries
                if (0 <= next_pos[0] < self.size and 
                    0 <= next_pos[1] < self.size and
                    next_pos not in visited and
                    maze[next_pos] == 0):
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def _create_guaranteed_path_maze(self):
        """Create maze with a guaranteed winding path and substantial walls (fallback method)"""
        maze = np.ones((self.size, self.size))  # Start with all walls
        
        # Create a random winding path from start to goal
        current = list(self.start_pos)
        path = [tuple(current)]
        maze[tuple(current)] = 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while tuple(current) != self.goal_pos:
            # Prioritize moving toward goal but add randomness
            possible_moves = []
            
            for dx, dy in directions:
                next_pos = [current[0] + dx, current[1] + dy]
                
                if (0 <= next_pos[0] < self.size and 
                    0 <= next_pos[1] < self.size):
                    
                    # Calculate distance to goal
                    dist_to_goal = abs(next_pos[0] - self.goal_pos[0]) + abs(next_pos[1] - self.goal_pos[1])
                    current_dist = abs(current[0] - self.goal_pos[0]) + abs(current[1] - self.goal_pos[1])
                    
                    # Prefer moves that get closer to goal, but allow backtracking
                    if dist_to_goal <= current_dist:
                        possible_moves.append((next_pos, 3))  # Higher weight for good moves
                    else:
                        possible_moves.append((next_pos, 1))  # Lower weight for backtracking
            
            if possible_moves:
                # Weighted random choice
                moves, weights = zip(*possible_moves)
                next_move = random.choices(moves, weights=weights, k=1)[0]
                current = next_move
                maze[tuple(current)] = 0
                path.append(tuple(current))
        
        # Add some additional random open spaces (fewer than before to keep walls substantial)
        for _ in range(self.size):  # REDUCED: Only size extra spaces (was size*2)
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            maze[i, j] = 0
        
        # Ensure start and goal are definitely open
        maze[self.start_pos] = 0
        maze[self.goal_pos] = 0
        
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
            return self.get_state(), -0.1, False, {'hit': 'boundary'}
        
        # Check walls
        if self.maze[self.agent_pos[0], self.agent_pos[1]] == 1:
            self.agent_pos = old_pos
            return self.get_state(), -0.2, False, {'hit': 'wall'}
        
        # Check if goal reached
        if tuple(self.agent_pos) == self.goal_pos:
            return self.get_state(), 10, True, {'result': 'win'}
        
        # Distance-based reward
        new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        distance_reward = (old_dist - new_dist) * 1.0  # Reward for getting closer
        
        # Small step penalty
        return self.get_state(), distance_reward - 0.1, False, {'result': 'move'}


class DQNAgent:
    """Deep Q-Network Agent with Target Network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # 4 actions
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_freq = 10  # Update target network every N episodes
        
        # Experience replay memory
        self.memory = deque(maxlen=20000)
        
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
        
        # Q-learning update using target network
        targets = rewards + self.gamma * np.amax(
            self.target_model.predict(next_states, verbose=0), axis=1
        ) * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath='models/maze_agent'):
        """Save the trained model and parameters"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model weights
        self.model.save(f'{filepath}_model.keras')
        self.target_model.save(f'{filepath}_target_model.keras')
        
        # Save agent parameters
        params = {
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        
        with open(f'{filepath}_params.json', 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"\nSuccess: Model saved to '{filepath}_model.keras'")
        print(f"Success: Parameters saved to '{filepath}_params.json'")
    
    @staticmethod
    def load(filepath='models/maze_agent'):
        """Load a trained model and parameters"""
        # Load parameters
        with open(f'{filepath}_params.json', 'r') as f:
            params = json.load(f)
        
        # Create agent with saved parameters
        agent = DQNAgent(params['state_size'], params['action_size'])
        
        # Load model weights with custom objects for compatibility
        custom_objects = {'mse': 'mean_squared_error'}
        
        try:
            agent.model = keras.models.load_model(
                f'{filepath}_model.keras',
                custom_objects=custom_objects,
                compile=False
            )
            agent.target_model = keras.models.load_model(
                f'{filepath}_target_model.keras',
                custom_objects=custom_objects,
                compile=False
            )
            
            # Recompile the models
            agent.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=agent.learning_rate),
                loss='mse'
            )
            agent.target_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=agent.learning_rate),
                loss='mse'
            )
            
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("The saved model may be from an incompatible version.")
            print("Please delete the old model and retrain.")
            raise
        
        # Restore epsilon
        agent.epsilon = params['epsilon']
        
        print(f"\nSuccess: Model loaded from '{filepath}_model.keras'")
        print(f"Success: Epsilon: {agent.epsilon:.3f}")
        
        return agent


def save_maze(maze, filepath='models/maze_agent_maze.npy'):
    """Save the maze array to disk"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    np.save(filepath, maze.maze)
    print(f"Success: Maze saved to '{filepath}'")


def load_maze(filepath='models/maze_agent_maze.npy'):
    """Load a maze array from disk"""
    if os.path.exists(filepath):
        maze_array = np.load(filepath)
        maze = Maze(maze_array=maze_array)
        print(f"Success: Maze loaded from '{filepath}'")
        return maze
    else:
        print(f"Error: Maze file not found: '{filepath}'")
        return None


def train_agent(episodes=500, max_steps=100, maze=None):
    """
    Train the DQN agent on a single fixed maze
    """
    state_size = 100  # 10x10 flattened
    action_size = 4   # 4 movement options
    
    agent = DQNAgent(state_size, action_size)
    
    # Generate one fixed maze for entire training
    if maze is None:
        maze = Maze()
        print("\nGenerated new solvable maze, solution verified with BFS")
    
    scores = []
    win_history = []
    
    print("Begin Training on Fixed Maze")
    print(f"Maze Size: 10x10 | Episodes: {episodes} | Max Steps: {max_steps}\n")
    
    # Show the maze being trained on
    print("Training Maze:")
    print_maze_with_path(maze, [(0, 0)])
    print()
    
    for episode in range(episodes):
        # Reset to same maze every time
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
        
        # Update target network periodically
        if (episode + 1) % agent.update_target_freq == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        win_history.append(1 if won else 0)
        
        # Calculate win rate over last 20 episodes
        win_rate = np.mean(win_history[-20:]) if len(win_history) >= 20 else 0
        
        if (episode + 1) % 50 == 0:  # Report less frequently for more episodes
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Win Rate (20): {win_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save the trained agent AND the maze
    agent.save()
    save_maze(maze)
    
    return agent, maze, scores, win_history


def test_on_trained_maze(agent, maze, num_runs=5):
    """Test trained agent on the SAME maze it was trained on"""
    print("\n=== Testing on Training Maze ===")
    
    wins = 0
    total_steps = []
    all_paths = []
    
    for run in range(num_runs):
        state = maze.reset()
        done = False
        steps = 0
        total_reward = 0
        path = [(0, 0)]
        
        while not done and steps < 100:
            # Use greedy policy (no exploration)
            action = np.argmax(agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            state, reward, done, info = maze.step(action)
            path.append(tuple(maze.agent_pos))
            total_reward += reward
            steps += 1
        
        all_paths.append((path, done, steps, total_reward))
        
        if done:
            wins += 1
            total_steps.append(steps)
            print(f"Run {run + 1}: WIN in {steps} steps (Reward: {total_reward:.2f})")
        else:
            print(f"Run {run + 1}: FAILED after {steps} steps (Reward: {total_reward:.2f})")
    
    print(f"\nSuccess Rate on Training Maze: {wins}/{num_runs} = {wins/num_runs:.1%}")
    if total_steps:
        print(f"Avg steps to win: {np.mean(total_steps):.1f}")
    
    # Show the best path
    if wins > 0:
        successful_paths = [(p, s) for p, d, s, r in all_paths if d]
        best_path, best_steps = min(successful_paths, key=lambda x: x[1])
        
        print(f"\n--- Best Solution ({best_steps} steps) ---")
        print_maze_with_path(maze, best_path)
        
        # Visualize
        visualize_trained_maze_solution(maze, best_path)
    
    return all_paths


def test_agent(agent, num_episodes=5):
    """Test trained agent on new random mazes"""
    print("\n=== Testing Trained Agent on New Random Mazes ===")
    
    wins = 0
    total_steps = []
    test_results = []
    
    for episode in range(num_episodes):
        maze = Maze()  # New random maze
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
    
    return test_results


def print_maze_with_path(maze, path):
    """Print the maze grid with the agent's path"""
    grid = []
    
    # Create visual grid
    for i in range(maze.size):
        row = []
        for j in range(maze.size):
            if maze.maze[i, j] == 1:
                row.append('#')  # Wall
            else:
                row.append(' ')  # Empty space
        grid.append(row)
    
    # Mark the path
    for pos in path[:-1]:  # All positions except final
        if grid[pos[0]][pos[1]] == ' ':
            grid[pos[0]][pos[1]] = '.'
    
    # Mark start, goal, and final position
    grid[0][0] = 'S'
    grid[maze.size-1][maze.size-1] = 'G'
    
    # Mark final position on loss
    final_pos = path[-1]
    if final_pos != (maze.size-1, maze.size-1) and final_pos != (0, 0):
        grid[final_pos[0]][final_pos[1]] = 'X'
    
    # Print the grid
    print("  " + "-" * (maze.size * 2 + 1))
    for i, row in enumerate(grid):
        print(f"{i:2d} | " + " ".join(row) + " |")
    print("  " + "-" * (maze.size * 2 + 1))
    print("    " + " ".join(f"{i:2d}"[1] for i in range(maze.size)))


def visualize_trained_maze_solution(maze, path):
    """Visualize the solution path on the training maze"""
    grid = np.copy(maze.maze).astype(float)
    
    # Mark the path
    for (x, y) in path:
        grid[x, y] = 0.5
    
    # Mark start and goal
    grid[0, 0] = 0.8
    grid[maze.size-1, maze.size-1] = 0.9
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="viridis")
    plt.title(f"Learned Solution ({len(path)-1} steps)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print("Solution visualization displayed")


if __name__ == "__main__":
    print("="*60)
    print("          10x10 MAZE SOLVER WITH DQN")
    print("="*60)
    
    # Check if a trained model exists
    model_path = 'models/maze_agent'
    maze_path = 'models/maze_agent_maze.npy'
    model_exists = os.path.exists(f'{model_path}_model.keras')
    maze_exists = os.path.exists(maze_path)
    old_model_exists = os.path.exists('maze_agent_10x10.h5')
    
    if old_model_exists and not model_exists:
        print("\nError: Found old .h5 model format from previous TensorFlow version")
        print("This format is incompatible with the current version.")
        print("\nOptions:")
        print("  1. Delete old model and train new one")
        print("  2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            try:
                os.remove('maze_agent_10x10.h5')
                print("Old model deleted. Starting training...\n")
                agent, maze, scores, win_history = train_agent(episodes=500, max_steps=100)
                print("\n--- Training Complete ---")
                test_on_trained_maze(agent, maze, num_runs=5)
            except Exception as e:
                print(f"Error: {e}")
                exit()
        else:
            print("\nExiting...")
            exit()
    
    elif model_exists and maze_exists:
        print("\nTrained model and maze found!")
        
        # Load the model and maze
        try:
            agent = DQNAgent.load(model_path)
            maze = load_maze(maze_path)
            
            if not maze:
                print("\nError: Failed to load maze. Please retrain.")
                exit()
            
            # Show menu after loading
            while True:
                print("\nOptions:")
                print("  1. Test on training maze (5 runs)")
                print("  2. Test on new random mazes (5 mazes)")
                print("  3. Retrain new model on new maze")
                print("  4. Exit")
                
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == '1':
                    # Test on the saved training maze
                    test_on_trained_maze(agent, maze, num_runs=5)
                
                elif choice == '2':
                    # Test on new random mazes
                    test_agent(agent, num_episodes=5)
                
                elif choice == '3':
                    # Retrain
                    print("\n--- Starting New Training ---")
                    agent, maze, scores, win_history = train_agent(episodes=500, max_steps=100)
                    print("\n--- Training Complete ---")
                    test_on_trained_maze(agent, maze, num_runs=5)
                    # Continue loop to allow more testing
                    
                elif choice == '4':
                    print("\nExiting...")
                    exit()
                else:
                    print("\nInvalid choice. Please try again.")
                    
        except Exception as e:
            print(f"\nError: Failed to load model: {e}")
            print("Please delete the models folder and retrain.")
            exit()
    
    elif model_exists and not maze_exists:
        print("\nError: Model found but maze file missing!")
        print("You need to retrain to save both model and maze together.")
        print("\nOptions:")
        print("  1. Retrain new model")
        print("  2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            print("\n--- Starting New Training ---")
            agent, maze, scores, win_history = train_agent(episodes=500, max_steps=100)
            print("\n--- Training Complete ---")
            test_on_trained_maze(agent, maze, num_runs=5)
        else:
            print("\nExiting...")
            exit()
    
    else:
        print("\nNo trained model found. Starting training...\n")
        agent, maze, scores, win_history = train_agent(episodes=500, max_steps=100)
        print("\n--- Training Complete ---")
        test_on_trained_maze(agent, maze, num_runs=5)
