from django.urls import path
from .views import *

urlpatterns = [
    path('<int:file_id>', DataAnalysisView.as_view(), name='dashboard-file-analysis'),
]
