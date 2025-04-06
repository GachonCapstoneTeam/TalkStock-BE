from django.urls import path
from textload.views import hello_world
from . import views
from .views import search_reports


urlpatterns = [
    path('originaltext/', hello_world),
    path('content/',views.content),
    path('search/', search_reports, name='search_reports'),
    path('stock/', views.stock, name='stock'),
    path('industry/', views.industry, name='industry')
]