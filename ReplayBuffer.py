import numpy as np
import random
from collections import namedtuple, deque
import torch
import operator


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # .float() is casting float
        # .to(device) is move object both of CPU, GPU, other some device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, hyper_alpha=0.7, hyper_beta=0.5):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            hyper_alpha (float) : hyper paremeter alpha in Prioritized Experience Replay Paper
            hyper_beta (float) : hyper paremeter beta in Prioritized Experience Replay Paper
        """
        self.hyper_alpha = hyper_alpha
        self.hyper_beta = hyper_beta
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority"""
        priorities = np.array([e.priority for e in self.memory if e is not None])
        if len(priorities) > 0:
            max_priority = priorities.max()
        else:
            max_priority = 1.0
            
        e = self.experience(state, action, reward, next_state, done, max_priority)
        
        self.memory.append(e)
    
    def sample(self):
        """Prioritized sample a batch of experiences from memory."""
        
        priorities = np.array([e.priority for e in self.memory if e is not None])
        probabilities_a = priorities ** self.hyper_alpha
        sum_probabilties_a = probabilities_a.sum()
        p_i = probabilities_a / sum_probabilties_a
        
        sampled_indices = np.random.choice(len(self.memory), self.batch_size, p=p_i)
        experiences = [self.memory[idx] for idx in sampled_indices]
        
        N = len(self.memory)
        w_sampled = weights = (N * p_i[sampled_indices]) ** (-1 * self.hyper_beta)
        w_sampled = w_sampled / w_sampled.max()
        
        
        # .float() is casting float
        # .to(device) is move object both of CPU, GPU, other some device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        w = torch.from_numpy(np.vstack(w_sampled)).float()
  
        return (states, actions, rewards, next_states, dones, w, sampled_indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.memory[idx]._replace(priority = priority)