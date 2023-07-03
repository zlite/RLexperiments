import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay
class Memory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Environment Initialization
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_episodes = 1000
batch_size = 64
gamma = 0.99

# DQN and Memory Initialization
DQN = DQN(state_dim, action_dim)
memory = Memory(10000)
optimizer = optim.Adam(DQN.parameters())

# Training Loop
for episode in range(max_episodes):
    state, _ = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)
    done = False
    steps = 0
    while not done:
        # epsilon-greedy action selection
        if random.random() < max(0.01, 0.08 - 0.01*(episode/200)): 
            action = env.action_space.sample()
        else:
            action = torch.argmax(DQN(state)).item()

        next_state, reward, _, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        
        # store in memory
        memory.add((state, action, reward, next_state, done))
        
        state = next_state
        steps += 1
        
        if done:
            print(f"Episode: {episode + 1}, Steps: {steps}")
            break
    
    if len(memory.buffer) > batch_size:
        batch = memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(state_batch)
        action_batch = torch.Tensor(action_batch)
        reward_batch = torch.Tensor(reward_batch)
        next_state_batch = torch.cat(next_state_batch)
        done_batch = torch.Tensor(done_batch)

        current_q_values = DQN(state_batch).gather(1, action_batch.long().unsqueeze(1)).squeeze()
        max_next_q_values = DQN(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (1 - done_batch) * gamma * max_next_q_values

        # loss is measured from error between current and newly expected Q values
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training completed.")
import time

# After the training loop
env = gym.make('CartPole-v0')
state, _ = env.reset()  # Extract numpy array from the tuple
done = False
while not done:
    env.render()
    time.sleep(0.02)  # Slow down the rendering
    with torch.no_grad():  # Don't need to calculate gradients here
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.argmax(DQN(state)).item()
    result = env.step(action)  # Get the return value of the step method
    state_tuple, _, done, _, _ = env.step(action)  # Extract tuple from the output
    state = state_tuple  # Extract numpy array from the tuple
env.close()



