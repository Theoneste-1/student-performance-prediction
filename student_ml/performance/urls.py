from django.urls import path
from .views import predict_performance, performance_ui

urlpatterns = [
    path('', performance_ui, name='home'),
    path('predict/', predict_performance, name='predict_performance'),
]