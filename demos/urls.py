from django.urls import path, include
from . import views

app_name = 'demos'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_csv, name='upload'), 
    path('plot/', views.generate_plot, name='plot'), 
    path('project4/', include('demos.project4.urls', namespace='project4')),
]
