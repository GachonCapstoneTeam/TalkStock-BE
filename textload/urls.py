from django.urls import path
from textload.views import hello_world
from textload.views import bond
from textload.views import industry

from . import views

urlpatterns = [
    path('originaltext/', hello_world),
    path('content/',views.content),
    path('bond/', views.bond),
    path('stock/', views.stock),
    path('investment/', views.investment),
    path('market/', views.market),
    path('industry/', views.industry),
    path('economic/', views.economic)

]