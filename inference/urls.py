from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_image, name='process_image'),
    #path('download/', views.download_image, name='download_image'),  
    path('get-models/', views.get_available_models, name='get_models'),
    path('switch-model/', views.switch_model, name='switch_model'),
]