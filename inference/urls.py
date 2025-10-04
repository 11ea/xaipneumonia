from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),    
    path('model/classes/', views.get_model_classes, name='model-classes'),
    path('models/', views.get_models_config, name='models-config'),
    path('models/<str:model_type>/', views.get_model_config, name='model-config'),
    path('mask-worker.js', views.mask_worker_view, name='mask-worker'),
    path('sample-images/', views.sample_images, name='sample-images'),
]