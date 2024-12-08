from django.urls import path
from textload.views import hello_world
from . import views

urlpatterns = [
    path('originaltext/', hello_world),
    path('content/',views.content)
]