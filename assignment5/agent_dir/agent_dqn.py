import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F
from collections import deque
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        hidden_dims = [64, 64]
        layers = []
        self.fc_in = nn.Linear(input_size, hidden_dims[0])
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
        self.fc_net = nn.Sequential(*layers)
        self.fc_final = nn.Linear(hidden_dims[-1], output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        x = self.fc_in(inputs)
        x = self.fc_net(x)
        x = self.fc_final(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        return len(self.buffer)

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        ##################
        # transition: (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        ##################
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 确保状态数组形状一致
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), \
            next_states, np.array(dones, dtype=np.uint8)

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        self.env = env
        self.args = args
        
        # Initialize networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_network = QNetwork(self.state_dim, args.hidden_size, self.action_dim)
        self.target_network = QNetwork(self.state_dim, args.hidden_size, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(10000)
        
        # Training parameters
        self.batch_size = 32
        self.gamma = args.gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 100
        self.steps = 0
        
        # Use CUDA if available
        self.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        # For logging
        self.writer = SummaryWriter()
        self.episode_reward = 0
        self.total_rewards = []
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.episode_reward = 0

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = np.array(observation, dtype=np.float32).flatten()
        if test:
            with torch.no_grad():
                state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
                q_values = self.q_network(state)
                action = q_values.max(1)[1].item()
            return action
        
        # Epsilon-greedy policy during training
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
                q_values = self.q_network(state)
                action = q_values.max(1)[1].item()
            return action

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        print("Env max episode steps: ", self.env.spec.max_episode_steps)

        best_reward = -float('inf')
        
        for episode in range(self.args.n_frames):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # 取第一个元素作为状态
            else:
                state = reset_result
            state = np.array(state, dtype=np.float32).flatten()
            self.init_game_setting()
            done = False
            episode_reward = 0
            
            while not done:
                # Select and perform action
                action = self.make_action(state, test=False)
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_state, reward, terminated, truncated = step_result
                else:
                    next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
                # 确保next_state是正确形状的numpy数组
                next_state = np.array(next_state, dtype=np.float32).flatten()
                
                # Store transition in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train the network
                for _ in range(min(4, int(len(self.replay_buffer) / self.batch_size))):
                    loss = self.train()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                self.steps += 1
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                if done:
                    break

            # Update learning rate
            self.scheduler.step()
            
            self.total_rewards.append(episode_reward)
            avg_reward = np.mean(self.total_rewards[-100:])
            
            # Logging
            self.writer.add_scalar('reward', episode_reward, episode)
            self.writer.add_scalar('avg_reward', avg_reward, episode)
            self.writer.add_scalar('epsilon', self.epsilon, episode)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], episode)
            
            if episode % 10 == 0:
                print(f'Episode {episode}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}'
                      f', Steps: {self.steps}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
            
            # Early stopping if solved
            if avg_reward >= 195 and episode >= 100:
                print(f'Solved at episode {episode} with average reward {avg_reward:.2f}!')
                break
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.q_network.state_dict(), 'best_dqn_model.pth')
                
        
        self.env.close()
        self.writer.close()
