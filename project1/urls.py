from django.urls import path
from . import views

app_name = 'project1'
urlpatterns = [ 
    path('', views.index, name='index'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
    path('train_model_ajax/', views.train_model_ajax, name='train_model_ajax'),

]