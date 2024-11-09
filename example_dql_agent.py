import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # Increased network capacity for better learning
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQLAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, tau=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = epsilon   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.tau = tau  # Soft update parameter
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Define discrete actions: steering and throttle values
        self.steering_actions = [-0.3, -0.1, 0.0, 0.1, 0.3]
        self.throttle_actions = [0.0, 0.1, 0.3]

    def get_state(self, car, center_building):
        """Get relevant state information."""
        distance_to_center = car.distanceTo(center_building)
        velocity = np.sqrt(car.velocity.x**2 + car.velocity.y**2)
        heading_diff = self._calculate_heading_diff(car, center_building)
        angular_position = np.arctan2(car.center.y - center_building.center.y, car.center.x - center_building.center.x)
        
        state = np.array([
            distance_to_center,
            velocity,
            heading_diff,
            car.heading,
            angular_position,  # Added angular position as part of state
        ])
        return state

    def _calculate_heading_diff(self, car, center_building):
        """Calculate the heading difference between the car and the desired path."""
        v = car.center - center_building.center
        desired_heading = np.mod(np.arctan2(v.y, v.x) + np.pi/2, 2 * np.pi)
        return np.sin(desired_heading - car.heading)
    
    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            action = action_values.argmax().item()
        
        # Convert action index to steering and throttle values
        steering_idx = action // len(self.throttle_actions)
        throttle_idx = action % len(self.throttle_actions)
        
        return self.steering_actions[steering_idx], self.throttle_actions[throttle_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    target = reward + self.gamma * self.target_model(next_state_tensor).max(1)[0].item()
            
            # Prepare target values for the current state-action pair
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target
            
            states.append(state_tensor)
            targets.append(target_f)

        # Batch update
        states = torch.cat(states)
        targets = torch.cat(targets).to(self.device)

        # Calculate loss and optimize model
        self.optimizer.zero_grad()
        loss = nn.SmoothL1Loss()(self.model(states), targets)  # Huber loss
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Soft update of the target model parameters."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())
