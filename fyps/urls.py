"""定义fyps的URL模式"""
from django.urls import path
from . import views

app_name = 'fyps'

urlpatterns = [

    path('',views.index,name='index'),
    path('image/',views.upload_image,name='image'),
    path('analysis/', views.analyze_image, name='analysis'),
]