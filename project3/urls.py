from django.urls import path
from . import views

app_name = 'project3'
urlpatterns = [
    path('', views.index, name='index'),
    path('update_model/', views.update_model, name='update_model'),
] 