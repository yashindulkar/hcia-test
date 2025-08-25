from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.project4_landing, name='index'),                    # Clean landing page
    path('start/', views.project4_study, name='study'),               # All-in-one study interface  
    path('download/', views.project4_download_pdf, name='download'),   # PDF download
    path('export/', views.project4_export_feedback, name='export_feedback'),  # CSV export
]