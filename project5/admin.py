from django.contrib import admin
from .models import GameState, Trajectory, HumanFeedback, PolicyModel, TrainingSession

@admin.register(GameState)
class GameStateAdmin(admin.ModelAdmin):
    list_display = ('trajectory_id_short', 'step_number', 'mouse_position', 'action_taken', 'reward', 'created_at')
    list_filter = ('step_number', 'reward', 'action_taken', 'created_at')
    search_fields = ('trajectory_id',)
    ordering = ('trajectory_id', 'step_number')
    readonly_fields = ('created_at',)
    
    def trajectory_id_short(self, obj):
        return f"{obj.trajectory_id[:8]}..."
    trajectory_id_short.short_description = "Trajectory ID"

@admin.register(Trajectory)
class TrajectoryAdmin(admin.ModelAdmin):
    list_display = ('trajectory_id_short', 'total_reward', 'total_steps', 'cheese_collected', 
                   'organic_cheese_collected', 'traps_hit', 'end_reason', 'policy_version', 'created_at')
    list_filter = ('policy_version', 'end_reason', 'episode_ended', 'created_at')
    search_fields = ('trajectory_id', 'end_reason')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)
    
    def trajectory_id_short(self, obj):
        return f"{obj.trajectory_id[:8]}..."
    trajectory_id_short.short_description = "Trajectory ID"

@admin.register(HumanFeedback)
class HumanFeedbackAdmin(admin.ModelAdmin):
    list_display = ('trajectory_a_short', 'trajectory_b_short', 'preferred_trajectory', 
                   'confidence_level', 'short_reason', 'submitted_at')
    list_filter = ('preferred_trajectory', 'confidence_level', 'submitted_at')
    search_fields = ('trajectory_a_id', 'trajectory_b_id', 'feedback_reason')
    ordering = ('-submitted_at',)
    readonly_fields = ('submitted_at',)
    
    def trajectory_a_short(self, obj):
        return f"{obj.trajectory_a_id[:8]}..."
    trajectory_a_short.short_description = "Trajectory A"
    
    def trajectory_b_short(self, obj):
        return f"{obj.trajectory_b_id[:8]}..."
    trajectory_b_short.short_description = "Trajectory B"
    
    def short_reason(self, obj):
        if obj.feedback_reason:
            return obj.feedback_reason[:50] + "..." if len(obj.feedback_reason) > 50 else obj.feedback_reason
        return "No reason provided"
    short_reason.short_description = "Feedback Reason"

@admin.register(PolicyModel)
class PolicyModelAdmin(admin.ModelAdmin):
    list_display = ('version', 'training_type', 'training_episodes', 'average_reward', 
                   'is_active', 'created_at')
    list_filter = ('training_type', 'is_active', 'created_at')
    search_fields = ('version',)
    ordering = ('-created_at',)
    readonly_fields = ('model_data', 'created_at')  # Don't allow editing binary data directly
    
    def get_readonly_fields(self, request, obj=None):
        # Make model_data readonly but allow viewing hyperparameters
        return self.readonly_fields

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id_short', 'training_type', 'episodes_completed', 'target_episodes', 
                   'current_average_reward', 'is_completed', 'start_time')
    list_filter = ('training_type', 'is_completed', 'start_time')
    search_fields = ('session_id', 'notes')
    ordering = ('-start_time',)
    readonly_fields = ('start_time', 'end_time')
    
    def session_id_short(self, obj):
        return f"{obj.session_id[:8]}..."
    session_id_short.short_description = "Session ID"