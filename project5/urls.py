from django.urls import path
from . import views

app_name = 'project5'

urlpatterns = [
    # Main pages
    path('', views.project5_landing, name='index'),
    
    # Testing and debugging
    path('test/', views.test_environment_direct, name='test'),
    path('test-models/', views.test_models, name='test_models'),
    
    # Training workflows
    path('train-baseline/', views.train_baseline, name='train_baseline'),
    path('collect-feedback/', views.collect_feedback, name='collect_feedback'),
    path('train-rlhf/', views.train_rlhf, name='train_rlhf'),
    
    # Analysis and visualization
    path('view-trajectories/', views.view_trajectories, name='view_trajectories'),
]