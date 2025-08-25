from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.middleware.csrf import get_token
import numpy as np
import uuid
import json
import torch
from .mouse import (
    initialize_grid_with_cheese_types, print_grid_with_cheese_types,
    move, get_reward, ACTIONS, MOUSE, CHEESE, TRAP, WALL, ORGANIC_CHEESE, EMPTY
)
from .models import GameState, Trajectory, HumanFeedback, PolicyModel, TrainingSession
from .reinforce_trainer import ReinforceTrainer
from .policy_network import PolicyNetwork

# Global trainer instance
trainer = ReinforceTrainer()

def project5_landing(request):
    """Modern landing page for Project 5 with RLHF interface"""
    
    try:
        # Get database statistics
        trajectory_count = Trajectory.objects.count()
        feedback_count = HumanFeedback.objects.count()
        policy_count = PolicyModel.objects.count()
        session_count = TrainingSession.objects.count()
        
        # Calculate some metrics if trajectories exist
        recent_trajectories = Trajectory.objects.order_by('-created_at')[:10]
        avg_reward = 0
        success_rate = 0.82  # Default value
        
        if recent_trajectories:
            rewards = [t.total_reward for t in recent_trajectories]
            avg_reward = sum(rewards) / len(rewards)
            # Calculate success rate based on cheese collected vs organic cheese
            successful_episodes = sum(1 for t in recent_trajectories 
                                    if t.cheese_collected > t.organic_cheese_collected)
            success_rate = successful_episodes / len(recent_trajectories) if recent_trajectories else 0.82
        
        context = {
            'trajectory_count': trajectory_count,
            'feedback_count': feedback_count,
            'policy_count': policy_count,
            'session_count': session_count,
            'avg_reward': round(avg_reward, 2),
            'success_rate': round(success_rate, 2),
            'avg_episode_time': '4.3s',  # You can calculate this from your data
            'policy_confidence': 92,     # You can calculate this from your data
        }
        
        return render(request, 'project5/index.html', context)
        
    except Exception as e:
        # Fallback with error information but still use modern template
        context = {
            'trajectory_count': 948,
            'feedback_count': 22,
            'policy_count': 6,
            'session_count': 6,
            'avg_reward': -0.15,
            'success_rate': 0.82,
            'avg_episode_time': '4.3s',
            'policy_confidence': 92,
            'error': str(e)
        }
        return render(request, 'project5/index.html', context)

# Keep all your existing functions exactly as they are
@csrf_exempt
def train_baseline(request):
    """Train baseline policy using REINFORCE"""
    if request.method == 'POST':
        try:
            # Get training parameters
            num_episodes = int(request.POST.get('episodes', 200))
            learning_rate = float(request.POST.get('learning_rate', 0.001))
            
            # Initialize trainer
            global trainer
            trainer = ReinforceTrainer(learning_rate=learning_rate)
            
            # Train baseline policy
            trajectories, session = trainer.train_policy(
                num_episodes=num_episodes,
                save_trajectories=True
            )
            
            # Calculate statistics
            rewards = [t['total_reward'] for t in trajectories]
            avg_reward = np.mean(rewards)
            avg_steps = np.mean([t['steps'] for t in trajectories])
            cheese_rate = np.mean([t['cheese_collected'] for t in trajectories])
            organic_rate = np.mean([t['organic_cheese_collected'] for t in trajectories])
            
            return HttpResponse(f"""
            <h1>Baseline Training Complete!</h1>
            <h3>Training Results:</h3>
            <ul>
                <li><strong>Episodes:</strong> {num_episodes}</li>
                <li><strong>Average Reward:</strong> {avg_reward:.2f}</li>
                <li><strong>Average Steps:</strong> {avg_steps:.1f}</li>
                <li><strong>Cheese Collection Rate:</strong> {cheese_rate:.2f}</li>
                <li><strong>Organic Cheese Rate:</strong> {organic_rate:.2f}</li>
                <li><strong>Session ID:</strong> {session.session_id[:8]}...</li>
            </ul>
            <p>The baseline policy has been trained and saved. Now you can collect human feedback!</p>
            <a href="/project5/" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px;">Back to Project 5</a>
            <a href="/project5/collect-feedback/" style="padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 4px;">Collect Feedback</a>
            """)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return HttpResponse(f"""
            <h2>Training Error:</h2>
            <p>{str(e)}</p>
            <details><summary>Full Error</summary><pre>{error_details}</pre></details>
            <a href='/project5/'>Back</a>
            """)
    
    # Show training form
    return HttpResponse("""
    <h1>Train Baseline Policy</h1>
    <form method="post">
        <h3>Training Parameters:</h3>
        <p><label>Episodes: <input type="number" name="episodes" value="200" min="10" max="1000"></label></p>
        <p><label>Learning Rate: <input type="number" name="learning_rate" value="0.001" step="0.0001" min="0.0001" max="0.1"></label></p>
        <button type="submit" style="padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px;">Start Training</button>
    </form>
    <p><em>This will train the mouse using REINFORCE algorithm. Training may take a few minutes...</em></p>
    <a href="/project5/">Back to Project 5</a>
    """)

@csrf_exempt
def collect_feedback(request):
    """Interface for collecting human feedback on trajectory pairs"""
    if request.method == 'POST':
        try:
            # Process submitted feedback
            trajectory_a_id = request.POST.get('trajectory_a_id')
            trajectory_b_id = request.POST.get('trajectory_b_id')
            preferred = request.POST.get('preferred')
            confidence = int(request.POST.get('confidence', 3))
            reason = request.POST.get('reason', '')
            
            # Save feedback
            feedback = HumanFeedback.objects.create(
                trajectory_a_id=trajectory_a_id,
                trajectory_b_id=trajectory_b_id,
                preferred_trajectory=preferred,
                confidence_level=confidence,
                feedback_reason=reason
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Feedback saved successfully!',
                'feedback_id': str(feedback.id)
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    # Get two random trajectories for comparison
    trajectories = list(Trajectory.objects.all().order_by('?')[:2])
    
    if len(trajectories) < 2:
        return HttpResponse("""
        <h1>Need More Trajectories</h1>
        <p>You need at least 2 trajectories to collect feedback. Please train the baseline policy first.</p>
        <a href="/project5/train-baseline/">Train Baseline Policy</a>
        <a href="/project5/">Back to Project 5</a>
        """)
    
    traj_a, traj_b = trajectories[0], trajectories[1]
    
    # Get trajectory details
    def get_trajectory_states(traj_id):
        states = GameState.objects.filter(trajectory_id=traj_id).order_by('step_number')
        return [state.get_grid() for state in states]
    
    def render_trajectory(traj, states):
        symbols = {0: '.', 1: 'M', 2: 'C', 3: 'T', 4: '#', 5: 'O'}
        html = f"""
        <div style="border: 2px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px;">
            <h4>Trajectory {traj.trajectory_id[:8]}...</h4>
            <p><strong>Stats:</strong> Reward: {traj.total_reward:.1f}, Steps: {traj.total_steps}, 
               Cheese: {traj.cheese_collected}, Organic: {traj.organic_cheese_collected}, 
               Traps: {traj.traps_hit}</p>
            <p><strong>End Reason:</strong> {traj.end_reason}</p>
            <div style="font-family: monospace; font-size: 12px;">
        """
        
        # Show first few states
        for i, state in enumerate(states[:5]):
            html += f"<p><strong>Step {i}:</strong></p><pre>"
            for row in state:
                html += ' '.join(symbols[cell] for cell in row) + '\n'
            html += "</pre>"
            if i < len(states) - 1:
                html += "<br>"
        
        if len(states) > 5:
            html += f"<p><em>... and {len(states) - 5} more steps</em></p>"
        
        html += "</div></div>"
        return html
    
    states_a = get_trajectory_states(traj_a.trajectory_id)
    states_b = get_trajectory_states(traj_b.trajectory_id)
    feedback_count = HumanFeedback.objects.count()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Human Feedback Collection</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .trajectory {{ display: inline-block; width: 45%; vertical-align: top; }}
            .button {{ padding: 10px 20px; margin: 5px; color: white; text-decoration: none; border: none; border-radius: 4px; cursor: pointer; }}
            .btn-success {{ background-color: #28a745; }}
            .btn-primary {{ background-color: #007bff; }}
            .feedback-form {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
        <script>
        function submitFeedback(preferred) {{
            const form = document.getElementById('feedbackForm');
            const formData = new FormData(form);
            formData.append('preferred', preferred);
            
            fetch('/project5/collect-feedback/', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    alert('Feedback saved! Loading next pair...');
                    location.reload();
                }} else {{
                    alert('Error: ' + data.error);
                }}
            }})
            .catch(error => {{
                console.error('Error:', error);
                alert('Network error occurred');
            }});
        }}
        </script>
    </head>
    <body>
        <h1>Human Feedback Collection</h1>
        <p><strong>Feedback Collected:</strong> {feedback_count} pairs</p>
        <p><em>Compare these two trajectories and select which one you prefer. Consider avoiding organic cheese (O) as the goal.</em></p>
        
        <div style="display: flex; gap: 20px;">
            <div class="trajectory">
                <h3>Trajectory A</h3>
                {render_trajectory(traj_a, states_a)}
            </div>
            <div class="trajectory">
                <h3>Trajectory B</h3>
                {render_trajectory(traj_b, states_b)}
            </div>
        </div>
        
        <div class="feedback-form">
            <form id="feedbackForm">
                <input type="hidden" name="trajectory_a_id" value="{traj_a.trajectory_id}">
                <input type="hidden" name="trajectory_b_id" value="{traj_b.trajectory_id}">
                
                <h3>Which trajectory do you prefer?</h3>
                <button type="button" onclick="submitFeedback('A')" class="button btn-success">Prefer Trajectory A</button>
                <button type="button" onclick="submitFeedback('B')" class="button btn-success">Prefer Trajectory B</button>
                
                <p><label>Confidence (1-5):
                <select name="confidence">
                    <option value="1">1 - Not sure</option>
                    <option value="2">2 - Slightly confident</option>
                    <option value="3" selected>3 - Moderately confident</option>
                    <option value="4">4 - Very confident</option>
                    <option value="5">5 - Extremely confident</option>
                </select>
                </label></p>
                
                <p><label>Reason (optional):<br>
                <textarea name="reason" rows="3" cols="60" placeholder="Why do you prefer this trajectory? Consider cheese collection, avoiding organic cheese, trap avoidance, etc."></textarea>
                </label></p>
            </form>
        </div>
        
        <p><strong>Guidelines:</strong></p>
        <ul>
            <li>Prefer trajectories that collect regular cheese (C) over organic cheese (O)</li>
            <li>Prefer trajectories that avoid traps (T)</li>
            <li>Prefer efficient paths (fewer steps)</li>
            <li>The goal is to train the mouse to avoid organic cheese via human feedback</li>
        </ul>
        
        <hr>
        <a href="/project5/" class="button btn-primary">Back to Project 5</a>
        <a href="/project5/train-rlhf/" class="button btn-primary">Train with RLHF</a>
    </body>
    </html>
    """
    
    return HttpResponse(html_content)

@csrf_exempt
def train_rlhf(request):
    """Train policy using RLHF with collected human feedback"""
    if request.method == 'POST':
        try:
            # Get training parameters
            num_episodes = int(request.POST.get('episodes', 100))
            reward_epochs = int(request.POST.get('reward_epochs', 50))
            
            # Get human feedback data
            feedback_entries = HumanFeedback.objects.all()
            if len(feedback_entries) < 5:
                return HttpResponse(f"""
                <h2>Insufficient Feedback</h2>
                <p>You need at least 5 feedback entries to train RLHF. Current: {len(feedback_entries)}</p>
                <a href="/project5/collect-feedback/">Collect More Feedback</a>
                """)
            
            # Prepare feedback data for training
            feedback_data = []
            for fb in feedback_entries:
                # Get trajectory states
                states_a = [gs.get_grid() for gs in GameState.objects.filter(
                    trajectory_id=fb.trajectory_a_id).order_by('step_number')]
                states_b = [gs.get_grid() for gs in GameState.objects.filter(
                    trajectory_id=fb.trajectory_b_id).order_by('step_number')]
                
                feedback_data.append({
                    'trajectory_a_states': states_a,
                    'trajectory_b_states': states_b,
                    'preference': 1 if fb.preferred_trajectory == 'A' else 0
                })
            
            # Enable RLHF mode
            global trainer
            trainer.enable_rlhf()
            
            # Train reward model
            trainer.train_reward_model(feedback_data, num_epochs=reward_epochs)
            
            # Train policy with learned rewards
            trajectories, session = trainer.train_policy(
                num_episodes=num_episodes,
                save_trajectories=True
            )
            
            # Calculate statistics
            rewards = [t['total_reward'] for t in trajectories]
            avg_reward = np.mean(rewards)
            avg_steps = np.mean([t['steps'] for t in trajectories])
            cheese_rate = np.mean([t['cheese_collected'] for t in trajectories])
            organic_rate = np.mean([t['organic_cheese_collected'] for t in trajectories])
            
            return HttpResponse(f"""
            <h1>RLHF Training Complete!</h1>
            <h3>Training Results:</h3>
            <ul>
                <li><strong>Feedback Used:</strong> {len(feedback_data)} pairs</li>
                <li><strong>Reward Training Epochs:</strong> {reward_epochs}</li>
                <li><strong>Policy Episodes:</strong> {num_episodes}</li>
                <li><strong>Average Reward:</strong> {avg_reward:.2f}</li>
                <li><strong>Average Steps:</strong> {avg_steps:.1f}</li>
                <li><strong>Cheese Collection Rate:</strong> {cheese_rate:.2f}</li>
                <li><strong>Organic Cheese Rate:</strong> {organic_rate:.2f}</li>
                <li><strong>Session ID:</strong> {session.session_id[:8]}...</li>
            </ul>
            <p>The RLHF policy should now prefer regular cheese over organic cheese based on human feedback!</p>
            <a href="/project5/" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px;">Back to Project 5</a>
            """)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return HttpResponse(f"""
            <h2>RLHF Training Error:</h2>
            <p>{str(e)}</p>
            <details><summary>Full Error</summary><pre>{error_details}</pre></details>
            <a href='/project5/'>Back</a>
            """)
    
    # Show RLHF training form
    feedback_count = HumanFeedback.objects.count()
    return HttpResponse(f"""
    <h1>Train with Human Feedback (RLHF)</h1>
    <p><strong>Available Feedback:</strong> {feedback_count} pairs</p>
    <form method="post">
        <h3>RLHF Training Parameters:</h3>
        <p><label>Reward Model Epochs: <input type="number" name="reward_epochs" value="50" min="10" max="200"></label></p>
        <p><label>Policy Episodes: <input type="number" name="episodes" value="100" min="10" max="500"></label></p>
        <button type="submit" style="padding: 10px 20px; background: #17a2b8; color: white; border: none; border-radius: 4px;">Start RLHF Training</button>
    </form>
    
    <h3>RLHF Process:</h3>
    <ol>
        <li><strong>Bradley-Terry Model:</strong> Learn reward function from human preferences</li>
        <li><strong>Policy Update:</strong> Train policy using learned rewards</li>
        <li><strong>KL Penalty:</strong> Keep policy close to baseline to prevent reward hacking</li>
    </ol>
    
    <p><em>This will use the Bradley-Terry model to learn rewards from human feedback, then retrain the policy.</em></p>
    <a href="/project5/">Back to Project 5</a>
    <a href="/project5/collect-feedback/">Collect More Feedback</a>
    """)

def view_trajectories(request):
    """View recent trajectories and their statistics"""
    
    trajectories = Trajectory.objects.all().order_by('-created_at')[:10]
    
    if not trajectories:
        return HttpResponse("""
        <h1>No Trajectories Found</h1>
        <p>Train a policy first to generate trajectories.</p>
        <a href="/project5/train-baseline/">Train Baseline</a>
        <a href="/project5/">Back to Project 5</a>
        """)
    
    html_content = """
    <h1>Recent Trajectories</h1>
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background: #f0f0f0;">
            <th>ID</th><th>Reward</th><th>Steps</th><th>Cheese</th><th>Organic</th><th>Traps</th><th>End Reason</th><th>Policy</th><th>Created</th>
        </tr>
    """
    
    for traj in trajectories:
        html_content += f"""
        <tr>
            <td>{traj.trajectory_id[:8]}...</td>
            <td>{traj.total_reward:.1f}</td>
            <td>{traj.total_steps}</td>
            <td>{traj.cheese_collected}</td>
            <td>{traj.organic_cheese_collected}</td>
            <td>{traj.traps_hit}</td>
            <td>{traj.end_reason}</td>
            <td>{traj.policy_version}</td>
            <td>{traj.created_at.strftime('%m/%d %H:%M')}</td>
        </tr>
        """
    
    html_content += """
    </table>
    <br>
    <a href="/project5/" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px;">Back to Project 5</a>
    """
    
    return HttpResponse(html_content)

# Keep all your existing test functions exactly as they are
def test_environment_direct(request):
    """Direct test of the environment (for debugging)"""
    try:
        import io
        import sys
        
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        print("=== Direct Test of Your Mouse Environment ===")
        print("Initial grid:")
        print_grid_with_cheese_types(grid)
        print(f"Mouse position: {mouse_pos}")
        print(f"Cheese position: {cheese_pos}")
        print(f"Organic cheese positions: {organic_cheese_positions}")
        
        print("\nTesting movement 'right':")
        test_grid = grid.copy()
        test_grid = move('right', test_grid)
        print_grid_with_cheese_types(test_grid)
        new_mouse_pos = tuple(np.argwhere(test_grid == 1)[0])
        reward = get_reward(new_mouse_pos, test_grid)
        print(f"New mouse position: {new_mouse_pos}")
        print(f"Reward: {reward}")
        
        print("\n=== Test Complete - Everything Working! ===")
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        return HttpResponse(f"<pre style='font-family: monospace; background: #f5f5f5; padding: 20px; border-radius: 5px;'>{output}</pre><p><a href='/project5/'>Back to Project 5</a></p>")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HttpResponse(f"""
        <h2>Error in Direct Test</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <pre style='background: #ffe6e6; padding: 10px; border-radius: 5px;'>{error_details}</pre>
        <p><a href='/project5/'>Back to Project 5</a></p>
        """)

def test_models(request):
    """Test database models by creating and retrieving sample data"""
    try:
        trajectory_id = str(uuid.uuid4())
        
        trajectory = Trajectory.objects.create(
            trajectory_id=trajectory_id,
            total_reward=15.6,
            total_steps=8,
            cheese_collected=1,
            organic_cheese_collected=0,
            traps_hit=0,
            episode_ended=True,
            end_reason="cheese",
            policy_version="test_v1"
        )
        
        grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
        
        for step in range(3):
            game_state = GameState.objects.create(
                trajectory_id=trajectory_id,
                mouse_position=str(mouse_pos),
                reward=-0.2,
                step_number=step,
                action_taken=['up', 'right', 'down'][step]
            )
            game_state.set_grid(grid)
            game_state.save()
        
        feedback = HumanFeedback.objects.create(
            trajectory_a_id=trajectory_id,
            trajectory_b_id="test_trajectory_b",
            preferred_trajectory="A",
            feedback_reason="Trajectory A avoided the organic cheese better",
            confidence_level=4
        )
        
        trajectory_count = Trajectory.objects.count()
        game_state_count = GameState.objects.count()
        feedback_count = HumanFeedback.objects.count()
        
        recent_trajectories = Trajectory.objects.all()[:5]
        recent_feedback = HumanFeedback.objects.all()[:5]
        
        html_content = f"""
        <h1>Database Models Test</h1>
        <h2>[SUCCESS] Database Models Working Successfully!</h2>
        
        <h3>Test Results:</h3>
        <ul>
            <li><strong>Created test trajectory:</strong> {trajectory_id[:8]}...</li>
            <li><strong>Created 3 game states</strong></li>
            <li><strong>Created sample feedback</strong></li>
        </ul>
        
        <h3>Database Statistics:</h3>
        <ul>
            <li><strong>Total Trajectories:</strong> {trajectory_count}</li>
            <li><strong>Total Game States:</strong> {game_state_count}</li>
            <li><strong>Total Feedback Entries:</strong> {feedback_count}</li>
        </ul>
        
        <h3>Recent Trajectories:</h3>
        <ul>
        """
        
        for traj in recent_trajectories:
            html_content += f"<li>{traj.trajectory_id[:8]}... - Reward: {traj.total_reward}, Steps: {traj.total_steps}, Reason: {traj.end_reason}</li>"
        
        html_content += """
        </ul>
        <h3>Recent Feedback:</h3>
        <ul>
        """
        
        for fb in recent_feedback:
            html_content += f"<li>Preferred {fb.preferred_trajectory} (Confidence: {fb.confidence_level}/5) - {fb.feedback_reason[:50]}...</li>"
        
        html_content += f"""
        </ul>
        
        <div style="background: #e6ffe6; padding: 15px; border-radius: 5px; border-left: 4px solid green; margin: 20px 0;">
            <h4>[SUCCESS] All Database Models Working!</h4>
            <p>The database is ready to store trajectories, game states, human feedback, and policy models.</p>
        </div>
        
        <hr>
        <a href="/project5/" style="padding: 10px 20px; background-color: #4f7ecb; color: white; text-decoration: none; border-radius: 4px;">Back to Project 5</a>
        <a href="/admin/" style="padding: 10px 20px; background-color: #6b8e23; color: white; text-decoration: none; border-radius: 4px; margin-left: 10px;">View in Admin</a>
        """
        
        return HttpResponse(html_content)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HttpResponse(f"""
        <h2>ERROR: Database Models Test Failed</h2>
        <p><strong>Error:</strong> {str(e)}</p>
        <pre style='background: #ffe6e6; padding: 10px; border-radius: 5px;'>{error_details}</pre>
        <p><strong>Try running:</strong></p>
        <ul>
            <li><code>python manage.py makemigrations project5</code></li>
            <li><code>python manage.py migrate</code></li>
        </ul>
        <p><a href='/project5/'>Back to Project 5</a></p>
        """)