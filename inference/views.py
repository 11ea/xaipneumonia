import json
import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

MOCK_RESULTS = {
    'classification': 'Bacterial Pneumonia',
    'confidence': 0.87,
    'heatmap_available': True
}

def index(request):
    """Main page with image upload and sample selection"""
    sample_images = [
        {'name': 'Sample 1', 'path': '/static/images/sample1.jpg'},
        {'name': 'Sample 2', 'path': '/static/images/sample2.jpg'},
        {'name': 'Sample 3', 'path': '/static/images/sample3.jpg'},
    ]
    
    available_models = get_available_models_list()
    
    return render(request, 'index.html', {
        'sample_images': sample_images,
        'available_models': available_models
    })

from django.http import JsonResponse, HttpResponse
import base64
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def process_image(request):
    """Process image - client handles everything, server just returns mock data"""
    if request.method == 'POST':
        try:
            model_id = request.POST.get('model_id', 'default')
            
            import time
            time.sleep(1.5)
            
            available_models = get_available_models_list()
            model_info = next((m for m in available_models if m['id'] == model_id), available_models[0])
            
            result = {
                'success': True,
                'result': MOCK_RESULTS,
                'model_info': model_info,
                'model_id': model_id,
            }
            
            return JsonResponse(result)
            
        except Exception as e:
            import traceback
            print(f"Error in process_image: {e}")
            print(traceback.format_exc())
            
            return JsonResponse({
                'success': False, 
                'error': str(e),
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})





def get_available_models(request):
    """Get list of available models from CDN (mock implementation)"""
    models = get_available_models_list()
    return JsonResponse({'models': models})

@csrf_exempt
def switch_model(request):
    """Switch to a different model (mock implementation)"""
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        return JsonResponse({'success': True, 'message': f'Switched to model {model_id}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def get_available_models_list():
    """Helper function to get list of available models"""
    return [
        {'id': 'yolon-artirilmisVeri', 'name': 'Pneumonia Detector v2 Fast', 'size': '6.2MB', 'updated': '2025-09-02'},
        {'id': 'yolos-artirilmisVeri', 'name': 'Pneumonia Detector v2 Slow', 'size': '21.5MB', 'updated': '2025-09-02'},
        {'id': 'yolos-azVeri', 'name': 'Pneumonia Detector V1 Slow', 'size': '21.4MB', 'updated': '2025-09-02'},
    ]