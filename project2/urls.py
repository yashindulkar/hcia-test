from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path('active/', views.active_learning_view, name='active_learning'),
    path('active/loop/', views.active_learning_loop, name='active_learning_loop'),
    path('active/reset/', views.reset_active_session, name='active_reset'),
]