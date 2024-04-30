from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='homePage'),
    path('stopStream/', views.stopStream, name='stopStream'),
    path('video_feed/<int:isstream>', views.video_feed, name='video_feed'),
    path('fetchclasses/', views.fetchClasses, name='fetchClasses')
]
