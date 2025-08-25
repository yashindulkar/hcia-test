import torch
import torch.optim as optim
import numpy as np
import uuid
from collections import deque
from .mouse import (
    initialize_grid_with_cheese_types, move, get_reward, ACTIONS,
    MOUSE, CHEESE, TRAP, ORGANIC_CHEESE, WALL, EMPTY
)
from .policy_network import PolicyNetwork, RewardNetwork
from .models import GameState, Trajectory, TrainingSession, PolicyModel
import pickle

class ReinforceTrainer:
    """
    Trainer for REINFORCE algorithm with support for RLHF
    """
    def __init__(self, learning_rate=1e-3, gamma=0.99, max_steps=100):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_steps = max_steps
        
        # Initialize networks
        self.policy_net = PolicyNetwork()
        self.reward_net = RewardNetwork()  # For RLHF
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=learning_rate)
        
        # Training history
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # RLHF parameters
        self.use_learned_reward = False
        self.kl_penalty_weight = 0.1  # KL divergence penalty
        self.baseline_policy = None  # Store baseline policy for KL penalty
        
    def run_episode(self, save_trajectory=True):
        """
        Run a single episode and return trajectory data
        """
        # Initialize environment
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        # Track trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        trajectory_id = str(uuid.uuid4())
        step = 0
        episode_reward = 0
        
        # Track game statistics
        cheese_collected = 0
        organic_cheese_collected = 0
        traps_hit = 0
        end_reason = "max_steps"
        
        while step < self.max_steps:
            # Convert grid to tensor for neural network
            state_tensor = torch.FloatTensor(grid).unsqueeze(0)
            states.append(grid.copy())
            
            # Select action using policy
            action_idx, action_probs = self.policy_net.select_action(state_tensor)
            action = ACTIONS[action_idx]
            actions.append(action)
            
            # Get log probability for this action
            log_prob = self.policy_net.get_action_log_prob(state_tensor, action_idx)
            log_probs.append(log_prob)
            
            # Take action
            new_grid = move(action, grid.copy())
            new_mouse_pos = tuple(np.argwhere(new_grid == MOUSE)[0])
            
            # Calculate reward
            if self.use_learned_reward:
                # Use learned reward from human feedback
                reward = self.reward_net.forward(torch.FloatTensor(new_grid).unsqueeze(0)).item()
            else:
                # Use original environment reward
                reward = get_reward(new_mouse_pos, new_grid)
            
            rewards.append(reward)
            episode_reward += reward
            
            # Update grid
            grid = new_grid
            mouse_pos = new_mouse_pos
            
            # Check for episode end conditions
            if grid[mouse_pos] == CHEESE:
                cheese_collected += 1
                end_reason = "cheese"
                break
            elif grid[mouse_pos] == ORGANIC_CHEESE:
                organic_cheese_collected += 1
                end_reason = "organic_cheese"
                break
            elif grid[mouse_pos] == TRAP:
                traps_hit += 1
                end_reason = "trap"
                break
                
            step += 1
            
            # Save game state if requested
            if save_trajectory:
                game_state = GameState.objects.create(
                    trajectory_id=trajectory_id,
                    mouse_position=str(mouse_pos),
                    reward=reward,
                    step_number=step,
                    action_taken=action
                )
                game_state.set_grid(grid)
                game_state.save()
        
        # Save trajectory summary if requested
        if save_trajectory:
            Trajectory.objects.create(
                trajectory_id=trajectory_id,
                total_reward=episode_reward,
                total_steps=step,
                cheese_collected=cheese_collected,
                organic_cheese_collected=organic_cheese_collected,
                traps_hit=traps_hit,
                episode_ended=(step < self.max_steps),
                end_reason=end_reason,
                policy_version="current"
            )
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step)
        
        return {
            'trajectory_id': trajectory_id,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'total_reward': episode_reward,
            'steps': step,
            'cheese_collected': cheese_collected,
            'organic_cheese_collected': organic_cheese_collected,
            'traps_hit': traps_hit,
            'end_reason': end_reason
        }
    
    def compute_returns(self, rewards):
        """
        Compute discounted returns for REINFORCE
        """
        returns = []
        G = 0
        
        # Compute returns backwards
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        return torch.FloatTensor(returns)
    
    def train_policy(self, num_episodes=100, save_trajectories=True):
        """
        Train policy using REINFORCE algorithm
        """
        # Create training session
        session_id = str(uuid.uuid4())
        session = TrainingSession.objects.create(
            session_id=session_id,
            training_type='baseline' if not self.use_learned_reward else 'rlhf',
            target_episodes=num_episodes,
            notes=f"REINFORCE training with learning_rate={self.learning_rate}, gamma={self.gamma}"
        )
        
        all_trajectories = []
        
        for episode in range(num_episodes):
            # Run episode
            trajectory = self.run_episode(save_trajectory=save_trajectories)
            all_trajectories.append(trajectory)
            
            # Compute returns
            returns = self.compute_returns(trajectory['rewards'])
            
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute policy loss
            policy_loss = 0
            for log_prob, G in zip(trajectory['log_probs'], returns):
                policy_loss += -log_prob * G
            
            # Add KL penalty if using RLHF
            if self.use_learned_reward and self.baseline_policy is not None:
                kl_penalty = self.compute_kl_penalty(trajectory['states'])
                policy_loss += self.kl_penalty_weight * kl_penalty
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            
            # Update session progress
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.episode_rewards))
                session.episodes_completed = episode + 1
                session.current_average_reward = avg_reward
                session.save()
                
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Last Reward = {trajectory['total_reward']:.2f}")
        
        # Complete training session
        session.episodes_completed = num_episodes
        session.current_average_reward = np.mean(list(self.episode_rewards))
        session.is_completed = True
        session.save()
        
        # Save trained model
        policy_model = self.save_policy_model(
            version=f"reinforce_{session_id[:8]}",
            training_type='baseline' if not self.use_learned_reward else 'rlhf',
            training_episodes=num_episodes,
            average_reward=np.mean(list(self.episode_rewards))
        )
        
        session.resulting_policy = policy_model
        session.save()
        
        return all_trajectories, session
    
    def compute_kl_penalty(self, states):
        """
        Compute KL divergence penalty between current policy and baseline
        """
        if self.baseline_policy is None:
            return torch.tensor(0.0)
        
        total_kl = torch.tensor(0.0)
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Current policy probabilities
            current_probs = self.policy_net.forward(state_tensor)
            
            # Baseline policy probabilities
            with torch.no_grad():
                baseline_probs = self.baseline_policy.forward(state_tensor)
            
            # KL divergence
            kl_div = torch.nn.functional.kl_div(
                torch.log(current_probs + 1e-8),
                baseline_probs,
                reduction='batchmean'
            )
            total_kl += kl_div
        
        return total_kl / len(states)
    
    def train_reward_model(self, feedback_data, num_epochs=50):
        """
        Train reward model using Bradley-Terry model on human feedback
        """
        print(f"Training reward model on {len(feedback_data)} feedback pairs...")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for feedback in feedback_data:
                traj_a_states = feedback['trajectory_a_states']
                traj_b_states = feedback['trajectory_b_states']
                preference = feedback['preference']  # 1 if A preferred, 0 if B preferred
                
                # Compute rewards for both trajectories
                reward_a_list = []
                reward_b_list = []
                
                for state in traj_a_states:
                    reward_a_list.append(self.reward_net.forward(
                        torch.FloatTensor(state).unsqueeze(0)
                    ).squeeze())
                
                for state in traj_b_states:
                    reward_b_list.append(self.reward_net.forward(
                        torch.FloatTensor(state).unsqueeze(0)
                    ).squeeze())
                
                # Sum rewards (avoid in-place operations)
                reward_a = torch.stack(reward_a_list).sum() if reward_a_list else torch.tensor(0.0)
                reward_b = torch.stack(reward_b_list).sum() if reward_b_list else torch.tensor(0.0)
                
                # Bradley-Terry model loss
                # P(A > B) = exp(reward_A) / (exp(reward_A) + exp(reward_B))
                prob_a_preferred = torch.sigmoid(reward_a - reward_b)
                
                if preference == 1:  # A is preferred
                    loss = -torch.log(prob_a_preferred + 1e-8)
                else:  # B is preferred
                    loss = -torch.log(1 - prob_a_preferred + 1e-8)
                
                # Update reward network
                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()
                
                total_loss += loss.detach().item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(feedback_data)
                print(f"Reward training epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def enable_rlhf(self):
        """
        Enable RLHF mode and store baseline policy
        """
        self.use_learned_reward = True
        # Store current policy as baseline for KL penalty
        self.baseline_policy = PolicyNetwork()
        self.baseline_policy.load_state_dict(self.policy_net.state_dict())
        self.baseline_policy.eval()
        
        print("RLHF mode enabled. Current policy stored as baseline.")
    
    def save_policy_model(self, version, training_type, training_episodes, average_reward):
        """
        Save current policy to database
        """
        # Serialize model state
        model_data = pickle.dumps(self.policy_net.state_dict())
        
        # Store hyperparameters
        hyperparams = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'max_steps': self.max_steps,
            'use_learned_reward': self.use_learned_reward,
            'kl_penalty_weight': self.kl_penalty_weight
        }
        
        # Create model record
        policy_model = PolicyModel.objects.create(
            version=version,
            model_data=model_data,
            training_type=training_type,
            training_episodes=training_episodes,
            average_reward=average_reward,
            is_active=True
        )
        
        policy_model.set_hyperparameters(hyperparams)
        policy_model.save()
        
        return policy_model
    
    def load_policy_model(self, version):
        """
        Load policy from database
        """
        try:
            model_record = PolicyModel.objects.get(version=version)
            state_dict = pickle.loads(model_record.model_data)
            self.policy_net.load_state_dict(state_dict)
            
            # Load hyperparameters
            hyperparams = model_record.get_hyperparameters()
            if hyperparams:
                self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
                self.gamma = hyperparams.get('gamma', self.gamma)
                self.max_steps = hyperparams.get('max_steps', self.max_steps)
                self.use_learned_reward = hyperparams.get('use_learned_reward', False)
                self.kl_penalty_weight = hyperparams.get('kl_penalty_weight', 0.1)
            
            print(f"Loaded policy model: {version}")
            return True
            
        except PolicyModel.DoesNotExist:
            print(f"Policy model {version} not found")
            return False