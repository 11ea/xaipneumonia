import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import CDNModel

def index_view(request):
    models = CDNModel.objects.filter(is_active=True)
    
    context = {
        'available_models': [
            {
                'id': model.model_type,
                'name': model.name,
            }
            for model in models
        ],
    }
    return render(request, 'index.html', context)



def get_models_config(request):
    """API endpoint to get all active models"""
    models = CDNModel.objects.filter(is_active=True)
    
    models_data = [
        {
            'id': model.model_type,
            'name': model.name,
            'cdn_url': model.cdn_url,
            'input_size': model.input_size,
            'feature_layers': model.feature_layers,
            'batch_size': model.batch_size,
            'classification_layer': model.classification_layer,
            'class_names': model.classes
        }
        for model in models
    ]
    
    return JsonResponse({'models': models_data})

@csrf_exempt
def get_model_config(request, model_type):
    """API endpoint to get specific model config"""
    try:
        model = CDNModel.objects.get(model_type=model_type, is_active=True)
        return JsonResponse({
            'id': model.model_type,
            'name': model.name,
            'cdn_url': model.cdn_url,
            'input_size': model.input_size,
            'feature_layers': model.feature_layers,
            'batch_size': model.batch_size,
            'classification_layer': model.classification_layer,
            'classes': model.classes
        })
    except CDNModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    
@csrf_exempt
def mask_worker_view(request):
    worker_path = os.path.join(os.path.dirname(__file__), 'static', 'js', 'mask-worker.js')
    
    try:
        with open(worker_path, 'r') as f:
            content = f.read()
        
        response = HttpResponse(content, content_type='application/javascript')
        response['Access-Control-Allow-Origin'] = '*'
        return response
        
    except FileNotFoundError:
        return HttpResponse('Worker file not found', status=404)
    

def get_model_classes(request, model_type):
    try:
        model = CDNModel.objects.get(model_type=model_type, is_active=True)
        return JsonResponse({'classes': model.classes})
    except CDNModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    
def sample_images(request):
    classes = list(CDNModel.c.all())
    samples = []

    for c in classes:        
        img = c.images.order_by("?").first()  # random 1 per class
        if img:
            samples.append({
                "class": c.name,
                "url": img.url
            })

    return JsonResponse({"samples": samples})