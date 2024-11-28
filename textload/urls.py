from django.urls import path
from textload.views import hello_world

urlpatterns = [
    path('originaltext/', hello_world),
]