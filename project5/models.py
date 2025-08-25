from django.db import models
import json

class GameState(models.Model):
    """Store individual game states and steps in trajectories"""
    grid_state = models.TextField()  # JSON representation of the grid
    mouse_position = models.CharField(max_length=10)  # "(row,col)" format
    reward = models.FloatField(default=0.0)
    step_number = models.IntegerField(default=0)
    trajectory_id = models.CharField(max_length=50)
    action_taken = models.CharField(max_length=10, blank=True)  # up, down, left, right
    created_at = models.DateTimeField(auto_now_add=True)
    
    def set_grid(self, grid):
        """Convert numpy array to JSON string"""
        self.grid_state = json.dumps(grid.tolist())
    
    def get_grid(self):
        """Convert JSON string back to list (can be converted to numpy array)"""
        return json.loads(self.grid_state)
    
    class Meta:
        ordering = ['trajectory_id', 'step_number']
        indexes = [
            models.Index(fields=['trajectory_id', 'step_number']),
        ]

class Trajectory(models.Model):
    """Store complete trajectories for comparison"""
    trajectory_id = models.CharField(max_length=50, unique=True)
    total_reward = models.FloatField()
    total_steps = models.IntegerField()
    cheese_collected = models.IntegerField(default=0)
    organic_cheese_collected = models.IntegerField(default=0)
    traps_hit = models.IntegerField(default=0)
    episode_ended = models.BooleanField(default=False)  # Did episode end naturally?
    end_reason = models.CharField(max_length=50, blank=True)  # cheese, trap, max_steps
    policy_version = models.CharField(max_length=20, default="v1")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Trajectory {self.trajectory_id[:8]}... (Reward: {self.total_reward}, Steps: {self.total_steps})"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Trajectories"

class HumanFeedback(models.Model):
    """Store human preferences between trajectory pairs"""
    PREFERENCE_CHOICES = [
        ('A', 'Trajectory A'),
        ('B', 'Trajectory B'),
    ]
    
    trajectory_a_id = models.CharField(max_length=50)
    trajectory_b_id = models.CharField(max_length=50)
    preferred_trajectory = models.CharField(max_length=1, choices=PREFERENCE_CHOICES)
    feedback_reason = models.TextField(blank=True)
    confidence_level = models.IntegerField(default=3, help_text="1-5 scale, how confident are you?")
    submitted_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback: {self.trajectory_a_id[:8]}... vs {self.trajectory_b_id[:8]}... -> {self.preferred_trajectory}"
    
    class Meta:
        ordering = ['-submitted_at']

class PolicyModel(models.Model):
    """Store trained policy models"""
    TRAINING_TYPE_CHOICES = [
        ('baseline', 'Baseline REINFORCE'),
        ('rlhf', 'RLHF Trained'),
    ]
    
    version = models.CharField(max_length=20, unique=True)
    model_data = models.BinaryField()  # Pickled model state_dict
    training_type = models.CharField(max_length=10, choices=TRAINING_TYPE_CHOICES, default='baseline')
    training_episodes = models.IntegerField()
    average_reward = models.FloatField()
    final_loss = models.FloatField(null=True, blank=True)
    hyperparameters = models.TextField(blank=True)  # JSON string of hyperparameters
    is_active = models.BooleanField(default=True)  # Is this the current active model?
    created_at = models.DateTimeField(auto_now_add=True)
    
    def set_hyperparameters(self, params_dict):
        """Store hyperparameters as JSON"""
        self.hyperparameters = json.dumps(params_dict)
    
    def get_hyperparameters(self):
        """Retrieve hyperparameters as dictionary"""
        if self.hyperparameters:
            return json.loads(self.hyperparameters)
        return {}
    
    def __str__(self):
        return f"Policy {self.version} ({self.training_type}) - Avg Reward: {self.average_reward:.2f}"
    
    class Meta:
        ordering = ['-created_at']

class TrainingSession(models.Model):
    """Track training sessions and their progress"""
    session_id = models.CharField(max_length=50, unique=True)
    training_type = models.CharField(max_length=10, choices=PolicyModel.TRAINING_TYPE_CHOICES)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    episodes_completed = models.IntegerField(default=0)
    target_episodes = models.IntegerField()
    current_average_reward = models.FloatField(default=0.0)
    is_completed = models.BooleanField(default=False)
    resulting_policy = models.ForeignKey(PolicyModel, on_delete=models.SET_NULL, null=True, blank=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        status = "Completed" if self.is_completed else "In Progress"
        return f"Training Session {self.session_id[:8]}... ({status})"
    
    class Meta:
        ordering = ['-start_time']