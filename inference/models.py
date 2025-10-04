from django.db import models

# models.py
class CDNModel(models.Model):
    MODEL_TYPES = [
        ('yolon-artirilmisVeri-ONNX', 'YOLO N Enhanced Data ONNX Format'),
        ('yolon-artirilmisVeri-ORT', 'YOLO N Enhanced Data ORT Format'),
        ('yolos-artirilmisVeri', 'YOLO S Enhanced Data'),
        ('yolom-artirilmisVeri', 'YOLO M Enhanced Data'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES, unique=True)
    cdn_url = models.URLField()
    input_size = models.IntegerField(default=224)
    classification_layer = models.CharField(max_length=100, default='classification')
    classes = models.JSONField(
        default=list,
        blank=True,
        help_text="List of classes, format: ['CLASS1', 'CLASS2',...]"
    )
    feature_layers = models.JSONField(
        default=dict,
        help_text='JSON object with layer names as keys and spatial dimensions as values. Format: {"layer_name": {"sideLen": 14, "channels": 64}, ...}'
    )
    
    batch_size = models.IntegerField(default=8)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'CDN Model'
        verbose_name_plural = 'CDN Models'
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"
    
    def get_feature_layers_list(self):
        """Return feature layers as a list of layer names"""
        return list(self.feature_layers.keys())
    
    def get_layer_dimensions(self, layer_name):
        """Get dimensions for a specific layer"""
        return self.feature_layers.get(layer_name, {"width": 32, "height": 32})