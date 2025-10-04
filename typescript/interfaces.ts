
export interface ImageData {
    name: string;
    data: string;
    type: string;
}

export interface LayerDimensions {
    sideLen: number;
    channels: number;
}

export interface ModelConfig {
    id: string;
    name: string;
    model_type: string;
    cdn_url: string;
    input_size: number;
    class_names: string[];
    classification_layer: string;
    feature_layers: { [layerName: string]: HeatmapResult };
    batch_size: number;
    is_active: boolean;
}
export interface HeatmapResult {
    data: Float32Array;
    sideLen: number;
    channels: number;
}

export interface CAMConfig {
    targetLayers: { [layerName: string]: LayerDimensions };
    classificationLayer: string;
    batchSize: number;
    inputSize: number;
}
export interface ResultVisualizationSettings {
    opacity: number;
    activationThreshold: number;
    visibleLayers: Set<string>;
}

export interface ModelsResponse {
    models: ModelConfig[];
}
/* export interface ProcessingResult {
    success: boolean;
    imageName: string;
    predictions: { [outputLayerName: string]: number[] };
    predictedClassIndex: { [outputLayerName: string]: number };
    confidence: { [outputLayerName: string]: number };
    heatmaps?: {
        [layer: string]: { [outputLayerName: string]: { data: number[]; size: number; } }
    };
    error?: string;
} */
export interface ProcessingResult {
    success: boolean;
    imageName: string;
    predictions: { [key: string]: any } | null;
    predictedClassIndex: { [key: string]: number } | null;
    confidence: { [key: string]: number } | null;
    heatmaps?: {
        [featureLayer: string]: { [outputLayerName: string]: HeatmapResult }
    };
    error?: string;
    // Add these new properties:
    hasHeatmaps?: boolean;
    metrics?: InferenceMetrics;
}
export interface InferenceMetrics {
    totalTime: number;
    averageTime: number;
    minTime: number;
    maxTime: number;
    totalInferences: number;
    inferencesPerSecond: number;
    channelProcessingTimes: number[];
    maskGenerationTime: number;
    heatmapComputationTime: number;
}
export interface SerializedHeatmap {
    data: number[];
    sideLen: number;
}
export interface CamVisualizationSettings {
    opacity: number;
    visibleLayers: Set<string>;
    activationThreshold: number;
}
export interface ClassificationResult {
    predictions: { [layer: string]: number[] };
    predictedClassIndex: { [layer: string]: number };
    confidence: { [layer: string]: number };
    originalOutput: any;
    origWidth: number;
    origHeight: number;
    originalImageU8: Uint8Array;
    preprocessingParams: { mean: number[]; std: number[]; scale: number };
}
export interface BatchResult {
    fileName: string;
    result: ProcessingResult | null;
    classificationResult: ClassificationResult | null;
    imageData?: string;
    camReady: boolean;
    progress: number;
    status: 'pending' | 'processing' | 'completed' | 'failed';
}
export interface AppResources {
    images: HTMLImageElement[];
    canvases: HTMLCanvasElement[];
    eventListeners: { element: HTMLElement; type: string; listener: EventListener }[];
}

export interface PredictionResult {
    classIndex: number;
    confidence: number;
    className: string;
}