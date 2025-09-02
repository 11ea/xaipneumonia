import json
import os
from django.conf import settings

def get_model_list():
    """Get list of available models from CDN (mock implementation)"""
    return [
        {'id': 'model_v1', 'name': 'Pneumonia Detector V1', 'size': '4.2MB', 'updated': '2023-04-15'},
        {'id': 'model_v2', 'name': 'Pneumonia Detector V2', 'size': '4.5MB', 'updated': '2023-06-20'},
        {'id': 'model_v3', 'name': 'Pneumonia Detector V3 (Quantized)', 'size': '2.1MB', 'updated': '2023-09-10'},
    ]
def get_local_models():
    """Get list of models available locally (mock implementation)"""
    return get_model_list() 