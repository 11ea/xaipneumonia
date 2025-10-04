
from django.contrib import admin
from .models import CDNModel

@admin.register(CDNModel)
class CDNModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'cdn_url', 'classification_layer', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'model_type', 'cdn_url']
    list_editable = ['is_active']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'model_type', 'cdn_url', 'is_active')
        }),
        ('Model Configuration', {
            'fields': ('input_size','classes', 'classification_layer', 'feature_layers', 'batch_size'),
            'description': 'Configure which layers to use for classification and Score-CAM. Format: {"conv1": {"sideLen": 28, "channels": 128}, "conv2": {"sideLen": 56, "channels": 64}}'
        }),
    )