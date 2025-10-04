import { ONNXModelService } from './onnx-model-service';
import { ONNXCAM } from './onnx-cam';
import {
    ImageData,
    BatchResult,
    ModelConfig,
    ModelsResponse,
    LayerDimensions,
    HeatmapResult,
    CamVisualizationSettings,
    InferenceMetrics,
    ResultVisualizationSettings,
    ProcessingResult
} from './interfaces';

const CLASS_NAMES = ['COVID-19', 'Normal', 'Viral', 'Bacterial'];
const CLASS_COLORS = ['#dc3545', '#198954ff', '#ffc107', '#0dcaf0'];

export class App {
    private availableModels: ModelConfig[] = [];
    private currentModelConfig: ModelConfig | null = null;
    private currentImage: ImageData | null = null;
    private cachedOriginalImage?: HTMLImageElement;
    private cachedHeatmaps: { [layerName: string]: HeatmapResult } = {};
    private isBatchProcessing: boolean = false;
    private batchResults: BatchResult[] = [];
    private selectedFiles: File[] = [];
    private currentFileIndex: number = 0;

    // UI Elements
    private fileInput: HTMLInputElement;
    private dropZone: HTMLElement;
    private processBtn: HTMLButtonElement;
    private modelSelect: HTMLSelectElement;

    // Services
    private onnxModelService: ONNXModelService;
    private onnxCAM: ONNXCAM;

    // Settings
    private camSettings: CamVisualizationSettings = {
        opacity: 0.6,
        visibleLayers: new Set<string>(),
        activationThreshold: 0.7
    };
    private dragListeners: {
        dragover: (e: DragEvent) => void;
        dragleave: (e: DragEvent) => void;
        drop: (e: DragEvent) => void;
    } | null = null;
    private allResults: Array<{
        data: {
            id: string;
            fileName?: string;
            success: boolean;
            progress: number;
            classification: ProcessingResult;
            heatmaps?: {
                [featureLayer: string]: { [outputLayerName: string]: HeatmapResult }
            };
            imageData: string;
            metrics: InferenceMetrics;
            status: string
        };
        timestamp: Date;
        camReady: boolean;
    }> = [];
    private resultSettings: Map<string, ResultVisualizationSettings> = new Map();

    constructor() {

        this.fileInput = document.getElementById('fileInput') as HTMLInputElement;
        this.dropZone = document.getElementById('dropZone') as HTMLElement;
        this.processBtn = document.getElementById('processBtn') as HTMLButtonElement;
        this.modelSelect = document.getElementById('modelTypeSelect') as HTMLSelectElement;
        this.dragListeners = null;

        if (!this.validateElements()) {
            throw new Error("Missing essential UI elements.");
        }

        this.onnxModelService = new ONNXModelService();
        this.onnxCAM = new ONNXCAM(this.onnxModelService);
    }

    private validateElements(): boolean {
        const requiredElements = [
            this.fileInput, this.dropZone, this.processBtn,
            this.modelSelect
        ];

        return requiredElements.every(element => element !== null);
    }

    async initialize(): Promise<void> {
        try {
            this.initEventListeners();
            await this.loadModelsConfig();
            await this.loadInitialModel();
            this.updateUIState('ready');
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Failed to initialize application');
            this.updateUIState('error');
        }
    }

    private async loadModelsConfig(): Promise<void> {
        try {
            const response = await fetch('/models/');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data: ModelsResponse = await response.json();
            this.availableModels = data.models;
            this.populateModelSelect();
        } catch (error) {
            console.error('Failed to load models config:', error);
            throw error;
        }
    }

    private populateModelSelect(): void {
        this.modelSelect.innerHTML = '';
        this.availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name || model.model_type;
            this.modelSelect.appendChild(option);
        });
    }

    private async loadInitialModel(): Promise<void> {
        const initialModelId = this.modelSelect.value;
        await this.loadModel(initialModelId);
    }

    private async loadModel(modelId: string): Promise<void> {
        const modelConfig = this.availableModels.find(m => m.id === modelId);
        if (!modelConfig) throw new Error(`Model configuration not found: ${modelId}`);

        this.currentModelConfig = modelConfig;
        this.updateModelStatus('loading', `Loading ${modelConfig.name}...`);

        try {
            // init target layers
            const targetLayers: { [layerName: string]: LayerDimensions } = {};
            if (modelConfig.feature_layers) {
                for (const [layerName, layerConfig] of Object.entries(modelConfig.feature_layers)) {
                    targetLayers[layerName] = {
                        sideLen: layerConfig.sideLen,
                        channels: layerConfig.channels
                    };
                }
            }

            this.camSettings.visibleLayers = new Set(Object.keys(targetLayers));

            this.onnxCAM.updateConfig({
                targetLayers,
                classificationLayer: modelConfig.classification_layer || '',
                batchSize: modelConfig.batch_size || 8,
                inputSize: modelConfig.input_size || 224
            });

            await this.onnxModelService.loadModel(modelConfig.cdn_url);
            this.updateModelStatus('success', `${modelConfig.name} loaded successfully`);

            this.updateLayerCheckboxes();

        } catch (error) {
            console.error('Failed to load model:', error);
            this.updateModelStatus('error', `Failed to load ${modelConfig.name}`);
            throw error;
        }
    }

    private updateModelStatus(status: 'loading' | 'success' | 'error', message: string): void {
        const statusElement = document.getElementById('modelStatusText');
        const statusContainer = document.getElementById('modelStatus');

        if (statusElement) {
            statusElement.textContent = message;
        }

        const alertClass = {
            loading: 'alert-info',
            success: 'alert-success',
            error: 'alert-danger'
        }[status];

        const iconClass = {
            loading: 'bi-hourglass-split',
            success: 'bi-check-circle',
            error: 'bi-exclamation-triangle'
        }[status];

        if (statusContainer) {
            statusContainer.className = `alert ${alertClass} py-2 mb-0`;

            const icon = statusContainer.querySelector('i');
            if (icon) {
                icon.className = `bi ${iconClass}`;
            }
        }
    }

    private updateLayerCheckboxes(): void {
        const container = document.getElementById('layer-checkboxes');
        if (!container || !this.currentModelConfig?.feature_layers) return;

        container.innerHTML = '';
        Object.keys(this.currentModelConfig.feature_layers).forEach(layerName => {
            const isChecked = this.camSettings.visibleLayers.has(layerName);

            const button = document.createElement('button');
            button.type = 'button';
            button.className = `btn btn-sm ${isChecked ? 'btn-primary' : 'btn-outline-primary'} mb-1`;
            button.textContent = layerName;
            button.dataset.layer = layerName;

            button.addEventListener('click', () => {
                this.toggleLayerVisibility(layerName);
                button.className = `btn btn-sm ${this.camSettings.visibleLayers.has(layerName)
                    ? 'btn-primary' : 'btn-outline-primary'} mb-1`;
                this.updateHeatmapDisplay();
            });

            container.appendChild(button);
        });
    }

    private toggleLayerVisibility(layerName: string): void {
        if (this.camSettings.visibleLayers.has(layerName)) {
            this.camSettings.visibleLayers.delete(layerName);
        } else {
            this.camSettings.visibleLayers.add(layerName);
        }
    }
    private setupDropZoneClick(): void {
        this.dropZone.addEventListener('click', (e: MouseEvent) => {
            if (e.target === this.dropZone ||
                (e.target as HTMLElement).tagName === 'P' ||
                (e.target as HTMLElement).tagName === 'H5') {
                this.fileInput.click();
            }
        });
    }
    private initEventListeners(): void {
        this.removeEventListeners();

        this.modelSelect.addEventListener('change', (e: Event) => this.handleModelChange(e));

        this.fileInput.addEventListener('change', (e: Event) => this.handleFileInputChange(e));

        this.setupDragAndDrop();

        this.setupDropZoneClick();

        this.processBtn.addEventListener('click', () => this.processImages());

        this.setupSettingsControls();

        this.attachResultsEventListeners();

        document.body.addEventListener('click', (e: MouseEvent) => this.handleEventDelegation(e));
        window.addEventListener('beforeunload', () => this.cleanupResources());
    }

    private setupSettingsControls(): void {
        const opacitySlider = document.getElementById('heatmap-opacity');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', (e: Event) => {
                this.camSettings.opacity = parseFloat((e.target as HTMLInputElement).value);
                this.updateHeatmapDisplay();
            });
        }
        const thresholdSlider = document.getElementById('heatmap-threshold');
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', (e: Event) => {
                this.camSettings.activationThreshold = parseFloat((e.target as HTMLInputElement).value);
                this.updateHeatmapDisplay();
            });
        }
    }

    private handleModelChange(e: Event): void {
        const target = e.target as HTMLSelectElement;
        this.handleModelChangeInternal(target.value).catch(console.error);
    }
    private async handleModelChangeInternal(modelId: string): Promise<void> {
        try {
            this.updateUIState('loading');
            await this.loadModel(modelId);
            this.updateUIState('ready');
        } catch (error) {
            console.error('Failed to change model:', error);
            this.showError('Failed to switch model');
            this.updateUIState('error');
        }
    }




    /* private clearBatchResults(): void {
        this.batchResults = [];

        // Clear the results list
        const list = document.getElementById('batchResultsList');
        if (list) {
            list.innerHTML = '';
        }

        // Hide batch results section and show "no results" message
        const batchResultsSection = document.getElementById('batchResultsSection');
        const noResultsMessage = document.getElementById('noResultsMessage');

        if (batchResultsSection) {
            batchResultsSection.style.display = 'none';
        }
        if (noResultsMessage) {
            noResultsMessage.style.display = 'block';
        }

        // Reset the results count and hide action buttons
        this.updateBatchResultsUI();
    } */

    private updateBatchResultsUI(): void {
        const resultsCount = document.getElementById('batchResultsCount');
        const exportButtons = document.querySelectorAll('[data-action="export-batch-results"], [data-action="export-batch-json"]');
        const clearButton = document.querySelector('[data-action="clear-batch-results"]');

        // Update results count
        if (resultsCount) {
            if (this.batchResults.length > 0) {
                resultsCount.textContent = `${this.batchResults.length} result${this.batchResults.length !== 1 ? 's' : ''}`;
            } else {
                resultsCount.textContent = '0 results';
            }
        }

        // Show/hide action buttons based on whether there are results
        const hasResults = this.batchResults.length > 0;
        exportButtons.forEach(button => {
            (button as HTMLElement).style.display = hasResults ? 'inline-block' : 'none';
        });

        if (clearButton) {
            (clearButton as HTMLElement).style.display = hasResults ? 'inline-block' : 'none';
        }
    }


    private downloadFile(content: string, filename: string, type: string): void {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    private cancelProcessing(isBatch: boolean): void {
        if (isBatch) {
            this.isBatchProcessing = false;
            this.hideBatchProgress();
            this.showInfo('Batch processing cancelled');
        } else {
            this.hideProgress();
            this.updateUIState('ready');
            this.showInfo('Analysis cancelled');
        }
    }

    private hideBatchProgress(): void {
        const container = document.getElementById('batchProgressContainer');
        if (container) {
            container.style.display = 'none';
        }
    }

    private showInfo(message: string): void {
        this.showToast(message, 'info');
    }

    private executeAction(action: string | null, imgSrc: string | null): void {
        if (!action) return;

        switch (action) {
            case 'browse':
                this.fileInput.click();
                break;
            case 'prev-file':
                this.navigateToFile('prev');
                break;
            case 'next-file':
                this.navigateToFile('next');
                break;
            case 'process':
                this.processImages();
                break;
            case 'clear-selection':
                this.clearAllSelections();
                break;
            case 'load-sample':
                if (imgSrc) this.loadSampleImage(imgSrc);
                break;
            case 'cancel-process':
                this.cancelProcessing(this.isBatchProcessing);
                break;
            case 'export-all-results': // Updated from export-batch-results
                this.exportAllResults();
                break;
            case 'export-all-json': // Updated from export-batch-json
                this.exportAllJson();
                break;
            case 'clear-all-results': // Updated from clear-batch-results
                this.clearAllResults();
                break;
            case 'remove-result':
                if (imgSrc) {
                    this.removeResult(imgSrc);
                }
                break;
            default:
                console.warn('Unknown action:', action);
        }
    }
    private clearAllResults(): void {
        this.allResults = [];
        this.displayResults();
    }
    // Update the updateResultUI method to handle completion properly:
    private updateResultUI(resultId: string, progress?: number, status?: string): void {
        const resultIndex = this.allResults.findIndex(r => r.data.id === resultId);
        if (resultIndex === -1) return;

        const result = this.allResults[resultIndex];
        const listItem = document.querySelector(`[data-result-id="${resultId}"]`);

        if (!listItem) return;

        // Update progress if provided
        if (progress !== undefined) {
            const progressBar = listItem.querySelector('.progress-bar');
            if (progressBar) {
                (progressBar as HTMLElement).style.width = `${progress}%`;
                (progressBar as HTMLElement).setAttribute('aria-valuenow', progress.toString());
                progressBar.innerHTML = `<small>${progress}%</small>`;

                if (progress < 100) {
                    progressBar.classList.add('progress-bar-animated');
                } else {
                    progressBar.classList.remove('progress-bar-animated');
                }
            }
        }

        // Update status if provided
        if (status) {
            const statusText = listItem.querySelector('.text-muted');
            if (statusText) {
                statusText.textContent = this.getStatusText(status);
            }
        }

        // When CAM processing is complete
        if (result.camReady) {
            // Remove the entire progress container (progress bar + status text)
            const progressContainer = listItem.querySelector('.d-inline-block.ms-2');
            if (progressContainer) {
                progressContainer.remove();
            }

            // Check if completion badge already exists to avoid duplicates
            if (!listItem.querySelector('.badge.bg-success.ms-2')) {
                const diagnosisBadge = listItem.querySelector('.badge:not(.bg-info):not(.bg-secondary)');
                if (diagnosisBadge && diagnosisBadge.parentNode) {
                    const completionBadge = document.createElement('span');
                    completionBadge.className = 'badge bg-success ms-2';
                    completionBadge.innerHTML = '<i class="bi bi-check-circle"></i> Ready';
                    diagnosisBadge.parentNode.insertBefore(completionBadge, diagnosisBadge.nextSibling);
                }
            }

            // Enable the accordion button and ensure it has correct content
            const accordionButton = listItem.querySelector('.accordion-button');
            if (accordionButton) {
                accordionButton.removeAttribute('disabled');

                // Update accordion content if it's open
                const accordionContent = listItem.querySelector('.accordion-body');
                const isOpen = accordionContent && !accordionContent.classList.contains('collapse');

                if (isOpen) {
                    accordionContent.innerHTML = this.renderAnalysisContent(result.data, result.data.id);
                    this.renderHeatmapsForResult(result.data.id);
                    //this.renderLayerHeatmapsForResult(resultId);
                }
            }
        }
    }
    private updateResultProgress(resultId: string, progress: number, status: string): void {
        const resultIndex = this.allResults.findIndex(r => r.data.id === resultId);
        if (resultIndex === -1) return;

        // Update the result data
        this.allResults[resultIndex].data.progress = progress;
        this.allResults[resultIndex].data.status = status;

        // Update the UI
        const listItem = document.querySelector(`[data-result-id="${resultId}"]`);
        if (!listItem) return;

        // Update progress bar
        const progressBar = listItem.querySelector('.progress-bar');
        if (progressBar) {
            (progressBar as HTMLElement).style.width = `${progress}%`;
            (progressBar as HTMLElement).setAttribute('aria-valuenow', progress.toString());
            progressBar.innerHTML = `<small>${Math.round(progress)}%</small>`;

            if (progress < 100) {
                progressBar.classList.add('progress-bar-animated');
            } else {
                progressBar.classList.remove('progress-bar-animated');
            }
        }

        // Update status text
        const statusText = listItem.querySelector('.text-muted');
        if (statusText) {
            statusText.textContent = status;
        }
    }
    private attachResultsEventListeners(): void {
        console.log("attaching listeners");
        // Remove result buttons
        document.querySelectorAll('[data-action="remove-result"]').forEach(button => {
            button.addEventListener('click', (e) => {
                const resultId = (e.target as HTMLElement).getAttribute('data-result-id');
                if (resultId) {
                    this.removeResult(resultId);
                }
            });
        });

        // Accordion expansion handlers
        document.querySelectorAll('.accordion-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const resultElement = (e.target as HTMLElement).closest('.list-group-item');
                const resultId = resultElement?.getAttribute('data-result-id');
                if (resultId) {
                    // Find the result
                    const result = this.allResults.find(r => r.data.id === resultId);
                    if (result && result.camReady) {
                        this.setupResultControls(resultId); // Setup controls when accordion opens
                        this.renderHeatmapsForResult(resultId);
                    }
                }
            });
        });
    }

    private getAggregatedHeatmapForResult(data: any, settings: ResultVisualizationSettings): Float32Array | null {
        if (!data.heatmaps) return null;
        const visibleLayers = settings.visibleLayers;
        const processedLayers = Object.keys(data.heatmaps);
        const intersection = Array.from(visibleLayers).filter(layer => processedLayers.includes(layer));
        const selected = intersection
            .map(layer => {
                const classificationLayer = this.currentModelConfig!.classification_layer;
                const heatmapData = data.heatmaps[layer][classificationLayer];
                return new Float32Array(heatmapData.data as ArrayLike<number>);
            })
            .filter(Boolean) as Float32Array[];

        if (!selected.length) return null;

        const out = new Float32Array(selected[0].length);
        for (const h of selected) {
            for (let i = 0; i < out.length; i++) out[i] += h[i];
        }
        for (let i = 0; i < out.length; i++) out[i] = (out[i] / selected.length);

        return out;
    }
    private handleFiles(files: FileList): void {
        if (files.length === 0) return;

        // Filter only image files
        const imageFiles = Array.from(files).filter(file => file.type.match('image.*'));

        if (imageFiles.length === 0) {
            this.showError('Please select image files (JPEG, PNG, etc.)');
            return;
        }

        this.selectedFiles = imageFiles;
        this.currentFileIndex = 0;
        this.updateFileSelectionUI();

        // Load the first file for preview
        this.loadFile(this.selectedFiles[0]).catch(console.error);
    }
    private async loadFile(file: File): Promise<void> {
        if (!file.type.match('image.*')) {
            this.showError('Please select an image file (JPEG, PNG, etc.)');
            return;
        }

        try {
            this.updateUIState('loading');

            const imageData = await this.readFileAsDataURL(file);
            this.currentImage = {
                name: file.name,
                data: imageData,
                type: file.type
            };

            this.updateDropZoneWithImage(imageData, file.name);
            if (this.selectedFiles.length === 1)
                this.showSuccess(`Loaded: ${file.name}`);

            // Update file navigation UI if multiple files
            this.updateFileNavigationUI();

        } catch (error) {
            console.error('Failed to load file:', error);
            this.showError('Failed to load image file');
        } finally {
            this.updateUIState('ready');
        }
    }
    private handleFileInputChange(e: Event): void {
        const target = e.target as HTMLInputElement;
        if (target.files?.length) {
            this.handleFiles(target.files);
        }
    }

    private handleEventDelegation(e: MouseEvent): void {
        const target = e.target as HTMLElement;
        const actionElement = target.closest('[data-action]') as HTMLElement;
        if (!actionElement) return;

        const action = actionElement.getAttribute('data-action');
        const imgSrc = actionElement.getAttribute('data-src');

        // Only handle actions that aren't already handled by direct listeners
        if (action && action !== 'process') {
            e.preventDefault();
            this.executeAction(action, imgSrc);
        }
    }

    private updateFileNavigationUI(): void {
        const prevButton = this.dropZone.querySelector('[data-action="prev-file"]');
        const nextButton = this.dropZone.querySelector('[data-action="next-file"]');

        if (prevButton) {
            (prevButton as HTMLButtonElement).disabled = this.currentFileIndex === 0;
        }
        if (nextButton) {
            (nextButton as HTMLButtonElement).disabled = this.currentFileIndex === this.selectedFiles.length - 1;
        }
    }
    private async navigateToFile(direction: 'prev' | 'next'): Promise<void> {
        if (this.selectedFiles.length <= 1) return;

        let newIndex = this.currentFileIndex;

        if (direction === 'prev' && this.currentFileIndex > 0) {
            newIndex--;
        } else if (direction === 'next' && this.currentFileIndex < this.selectedFiles.length - 1) {
            newIndex++;
        } else {
            return;
        }

        this.currentFileIndex = newIndex;
        await this.loadFile(this.selectedFiles[this.currentFileIndex]);
    }

    private updateFileSelectionUI(): void {
        const fileCount = this.selectedFiles.length;
        const fileCountBadge = document.getElementById('fileCountBadge');
        const selectionInfo = document.getElementById('selectionInfo');
        const selectedFilesCount = document.getElementById('selectedFilesCount');
        const processBtn = document.getElementById('processBtn') as HTMLButtonElement;

        // Safely handle null elements
        if (fileCountBadge) {
            if (fileCount > 0) {
                fileCountBadge.style.display = 'inline-block';
                fileCountBadge.textContent = `${fileCount} file${fileCount > 1 ? 's' : ''}`;
            } else {
                fileCountBadge.style.display = 'none';
            }
        }

        if (selectionInfo && selectedFilesCount) {
            if (fileCount > 0) {
                selectionInfo.style.display = 'block';
                selectedFilesCount.textContent = fileCount.toString();
            } else {
                selectionInfo.style.display = 'none';
            }
        }

        if (processBtn) {
            processBtn.disabled = fileCount === 0;
            this.updateProcessButtonMode(); // Update button text
        }
    }
    private processImages(): void {
        if (this.selectedFiles.length === 0) return;
        if (this.selectedFiles.length === 1) {
            // Single image processing
            this.processImage(false);
        } else {
            // Batch processing
            this.processImage(true);
        }
    }

    private clearAllSelections(): void {
        this.selectedFiles = [];
        this.currentFileIndex = 0;
        this.currentImage = null;

        // Store the badge before clearing
        const fileCountBadge = document.getElementById('fileCountBadge');

        // Clear only the content, not the badge
        const initialContent = `
        <i class="bi bi-cloud-upload display-4 text-muted d-block mb-3"></i>
        <h5 class="text-muted">Drag & drop images here</h5>
        <p class="text-muted mb-3">or click to select single or multiple X-ray images</p>
    `;

        // If badge exists, preserve it and add initial content
        if (fileCountBadge) {
            // Remove everything except the badge
            const elementsToRemove = Array.from(this.dropZone.children).filter(
                child => child.id !== 'fileCountBadge'
            );
            elementsToRemove.forEach(element => element.remove());

            // Add initial content after the badge
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = initialContent;
            while (tempDiv.firstChild) {
                this.dropZone.appendChild(tempDiv.firstChild);
            }
        } else {
            // Fallback: just set the innerHTML
            this.dropZone.innerHTML = initialContent;
        }

        this.dropZone.style.cursor = 'pointer';
        this.updateFileSelectionUI();
        //this.hideResults();
        //this.clearCanvases();
    }


    private hideResults(): void {
        // Since we're using a unified system, we just need to clear the results list
        const resultsList = document.getElementById('resultsList');
        const noResultsMessage = document.getElementById('noResultsMessage');
        const actionButtons = document.getElementById('resultsActionButtons');

        if (resultsList) resultsList.innerHTML = '';
        if (noResultsMessage) noResultsMessage.style.display = 'block';
        if (actionButtons) actionButtons.style.display = 'none';

        // Clear the results array
        this.allResults = [];

        // Update results count
        const resultsCount = document.getElementById('resultsCount');
        if (resultsCount) {
            resultsCount.textContent = '0 results';
        }
    }
    private async loadSampleImage(imageUrl: string): Promise<void> {
        try {
            this.updateUIState('loading');

            // Create a mock file object from the sample URL
            const response = await fetch(imageUrl);
            if (!response.ok) throw new Error(`Failed to fetch sample: ${response.status}`);

            const blob = await response.blob();
            const file = new File([blob], 'sample_image.jpg', { type: blob.type });

            await this.loadSingleImage(file);
            this.showSuccess('Sample image loaded successfully');

        } catch (error) {
            console.error('Failed to load sample image:', error);
            this.showError('Failed to load sample image');
            this.updateUIState('ready');
        }
    }
    private removeEventListeners(): void {
        // Clean up drag listeners
        this.cleanupDragListeners();

        // Clone other elements to remove event listeners
        const cloneModelSelect = this.modelSelect.cloneNode(true) as HTMLSelectElement;
        this.modelSelect.parentNode?.replaceChild(cloneModelSelect, this.modelSelect);
        this.modelSelect = cloneModelSelect;

        const cloneFileInput = this.fileInput.cloneNode(true) as HTMLInputElement;
        this.fileInput.parentNode?.replaceChild(cloneFileInput, this.fileInput);
        this.fileInput = cloneFileInput;

        const cloneProcessBtn = this.processBtn.cloneNode(true) as HTMLButtonElement;
        this.processBtn.parentNode?.replaceChild(cloneProcessBtn, this.processBtn);
        this.processBtn = cloneProcessBtn;

    }
    private cleanupDragListeners(): void {
        if (this.dragListeners) {
            this.dropZone.removeEventListener('dragover', this.dragListeners.dragover);
            this.dropZone.removeEventListener('dragleave', this.dragListeners.dragleave);
            this.dropZone.removeEventListener('drop', this.dragListeners.drop);
            this.dragListeners = null;
        }
    }
    private async loadSingleImage(file: File): Promise<void> {
        if (!file.type.match('image.*')) {
            this.showError('Please select an image file');
            return;
        }

        try {
            const imageData = await this.readFileAsDataURL(file);
            this.currentImage = {
                name: file.name,
                data: imageData,
                type: file.type
            };

            this.updateDropZoneWithImage(imageData, file.name);
            this.processBtn.disabled = false;
            this.showSuccess('Image loaded successfully');

        } catch (error) {
            console.error('Failed to load image:', error);
            this.showError('Failed to load image');
        }
    }

    private readFileAsDataURL(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e: ProgressEvent<FileReader>) => {
                if (e.target?.result) {
                    resolve(e.target.result as string);
                } else {
                    reject(new Error('Failed to read file'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    private updateDropZoneWithImage(imageData: string, fileName: string): void {
        this.dropZone.innerHTML = `
        <div class="position-relative">
            <img src="${imageData}" class="img-fluid rounded mb-3" style="max-height: 200px;">
            <span class="position-absolute top-0 start-0 badge bg-dark m-2">Preview</span>
        </div>
        <p class="mb-2 text-truncate"><strong>${fileName}</strong></p>
        ${this.selectedFiles.length > 1 ? `
        <div class="btn-group">
            <button class="btn btn-outline-secondary btn-sm" data-action="prev-file">
                <i class="bi bi-arrow-left"></i>
            </button>
            <button class="btn btn-outline-secondary btn-sm" data-action="next-file">
                <i class="bi bi-arrow-right"></i>
            </button>
        </div>
        <div class="mt-2 small text-muted">
            File ${this.currentFileIndex + 1} of ${this.selectedFiles.length}
        </div>
        ` : ''}
        
        <!-- Add a change button at the bottom -->
        <div class="mt-3">
            <button class="btn btn-outline-primary btn-sm" data-action="browse">
                <i class="bi bi-arrow-repeat"></i> Change Image
            </button>
        </div>
    `;
    }

    private async processImage(isBatch: boolean = false): Promise<void> {
        if (isBatch) {
            await this.processBatchImages();
        } else {
            await this.processSingleImage();
        }
    }
    private async processSingleImage(): Promise<void> {
        if (!this.currentImage) {
            this.showError('Please select an image first');
            return;
        }

        if (!this.onnxModelService.isModelLoaded()) {
            this.showError('Model is not loaded yet');
            return;
        }
        let resultId: string | null = null;
        try {
            this.updateUIState('processing');
            this.showProgress(0, 'Initializing analysis...');

            const img = await this.loadImage(this.currentImage.data);

            resultId = this.addToResults('single', {
                fileName: this.currentImage.name,
                imageData: this.currentImage.data,
                classification: null,
                heatmaps: null,
                metrics: null,
                camReady: false,
                success: true,
                progress: 5, // initial
                status: 'processing'
            });
            this.displayResults();
            this.switchToResultsTab();

            this.updateResultProgress(resultId, 15, 'Classifying...');
            this.showProgress(15, 'Classifying...');
            const classification = await this.onnxCAM.computeClassification(img);

            this.updateResult(resultId, {
                classification: classification,
                progress: 25,
                status: 'Generating heatmaps...'
            });
            this.showProgress(25, 'Generating heatmaps...');
            this.displayResults();
            //const cams = await this.onnxCAM.computeCAMs(classification);
            const cams = await this.onnxCAM.computeCAMs(
                classification,
                undefined,
                undefined,
                undefined,
                (progress: number, currentLayer: string, totalLayers: number, heatmaps?: any) => {
                    const overallProgress = 25 + progress * 75;
                    this.updateResultProgress(
                        resultId!,
                        overallProgress,
                        `Layer: ${currentLayer}`
                    );
                    this.updateResult(resultId!, { heatmaps: heatmaps });
                    console.log("layer,", currentLayer, " heatmap:", heatmaps);
                    if (heatmaps) {
                        this.updateHeatmapDisplay(resultId!);
                    }
                }
            );
            this.updateResult(resultId, {
                heatmaps: cams.heatmaps,
                metrics: cams.metrics,
                camReady: true,
                progress: 100,
                status: 'completed'
            });

            this.showProgress(100, 'Analysis complete');
            this.displayResults();
            this.updateUIState('results-ready');

        } catch (error) {
            console.error('Processing failed:', error);
            this.showError('Processing failed: ' + (error as Error).message);

            if (resultId) {
                this.updateResult(resultId, {
                    success: false,
                    error: (error as Error).message,
                    progress: 100,
                    status: 'failed'
                });
            }
            this.showProgress(100, 'Failed to process image');
            this.updateUIState('ready');
        }
        finally {
            this.hideProgress();
        }
    }

    private addToResults(type: 'single' | 'batch', data: any): string {
        const resultId = `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        // init
        this.resultSettings.set(resultId, {
            opacity: 0.6,
            activationThreshold: 0.7,
            visibleLayers: new Set(Object.keys(this.currentModelConfig?.feature_layers || {}))
        });

        this.allResults.push({
            data: {
                ...data,
                id: resultId,
                success: data.success !== undefined ? data.success : true,
                progress: 0, // init
                status: 'pending' // init
            },
            timestamp: new Date(),
            camReady: data.camReady || false
        });

        return resultId;
    }

    private updateResult(resultId: string, updates: any): void {
        const resultIndex = this.allResults.findIndex(r => r.data.id === resultId);
        if (resultIndex !== -1) {
            this.allResults[resultIndex] = {
                ...this.allResults[resultIndex],
                data: {
                    ...this.allResults[resultIndex].data,
                    ...updates
                },
                camReady: updates.camReady || this.allResults[resultIndex].camReady
            };
            console.log('Updated result:', this.allResults[resultIndex]);
            this.updateResultUI(resultId);
        }
    }

    private removeResult(resultId: string): void {
        this.resultSettings.delete(resultId);
        this.allResults = this.allResults.filter(r => r.data.id !== resultId);
        this.displayResults();
    }

    private displayResults(): void {
        const resultsList = document.getElementById('resultsList');
        const noResultsMessage = document.getElementById('noResultsMessage');
        const resultsCount = document.getElementById('resultsCount');
        const actionButtons = document.getElementById('resultsActionButtons');
        console.log("return condition", resultsList, noResultsMessage, resultsCount, actionButtons);
        if (!resultsList || !noResultsMessage || !resultsCount || !actionButtons) return;

        if (this.allResults.length === 0) {
            resultsList.innerHTML = '';
            noResultsMessage.style.display = 'block';
            actionButtons.style.display = 'none';
            resultsCount.textContent = '0 results';
            return;
        }
        console.log("before resultsList.innerHTML");
        noResultsMessage.style.display = 'none';
        actionButtons.style.display = 'flex';
        resultsCount.textContent = `${this.allResults.length} result${this.allResults.length !== 1 ? 's' : ''}`;
        resultsList.innerHTML = this.allResults.map((result, index) => {
            const { data, timestamp, camReady } = result;
            const classificationLayer = this.currentModelConfig?.classification_layer;
            const classIndex = classificationLayer && data.classification ? data.classification.predictedClassIndex![classificationLayer] : -1;
            const confidence = classificationLayer && data.classification ? data.classification.confidence![classificationLayer] : 0;
            const className = classIndex >= 0 ? this.currentModelConfig?.class_names[classIndex] || 'Unknown' : 'Unknown';
            return `
            <div class="list-group-item" data-result-id="${data.id}">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center mb-2">
                            <strong class="text-truncate me-2">${data.fileName}</strong>
                            <span class="badge bg-${this.getDiagnosisColor(classIndex)}">
                                ${className} (${(confidence * 100).toFixed(1)}%)
                            </span>
                        </div>

                        <div class="mb-2">
                            ${camReady ? `
                            <span class="badge bg-success ms-2">
                                <i class="bi bi-check-circle"></i> Ready
                            </span>
                        ` : `
                            <div class="d-inline-block ms-2" style="width: 100%; height: 60px">
                                <div class="progress result-progress">
                                    <div class="progress-bar progress-bar-striped ${data.progress < 100 ? 'progress-bar-animated' : ''}" 
                                        role="progressbar" 
                                        style="width: ${data.progress}%;"
                                        aria-valuenow="${data.progress}" 
                                        aria-valuemin="0" 
                                        aria-valuemax="100">
                                        <small>${data.progress}%</small>
                                    </div>
                                </div>                                                                  
                            <small class="text-muted">${this.getStatusText(data.status)}</small>                            
                            </div>
                        `}
                        </div>
                    </div>
                    
                    <div class="btn-group ms-2">
                        <button class="btn btn-sm btn-outline-danger" data-action="remove-result" data-result-id="${data.id}">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                </div>

                <!-- Accordion for heatmap-->
                <div class="accordion mt-3" id="accordion-${data.id}">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" 
                                    data-bs-toggle="collapse" 
                                    data-bs-target="#collapse-${data.id}" 
                                    aria-expanded="false" 
                                    aria-controls="collapse-${data.id}"
                                <i class="bi bi-bar-chart me-2"></i> XAI Heatmap
                            </button>
                        </h2>
                        <div id="collapse-${data.id}" class="accordion-collapse collapse" 
                                data-bs-parent="#accordion-${data.id}">
                                <div class="accordion-body">
                                    ${!camReady ? `
                                        <div class="text-center py-3">
                                            <div class="spinner-border spinner-border-sm text-primary"></div>
                                            <p class="mt-2 text-muted">Processing heatmaps...</p>
                                        </div>
                                    `: ``}
                                    ${data.classification ? this.renderAnalysisContent(result.data, result.data.id) : ''}
                                </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        }).join('');
        console.log("before attachResultsEventListeners");
        this.attachResultsEventListeners();

    }
    private getStatusText(status: string): string {
        switch (status) {
            case 'pending': return 'Waiting...';
            case 'processing': return 'Processing...';
            case 'completed': return 'Complete';
            case 'failed': return 'Failed';
            default: return status;
        }
    }
    private async loadImage(src: string): Promise<HTMLImageElement> {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = src;
        });
    }
    private setupResultControls(resultId: string): void {
        const opacitySlider = document.getElementById(`opacity-${resultId}`) as HTMLInputElement;
        const opacityValue = document.getElementById(`opacity-value-${resultId}`);

        if (opacitySlider && opacityValue) {
            opacitySlider.addEventListener('input', (e) => {
                const value = parseFloat((e.target as HTMLInputElement).value);
                this.updateResultSetting(resultId, 'opacity', value);
                opacityValue.textContent = `${Math.round(value * 100)}%`;
                this.renderHeatmapsForResult(resultId);
            });
        }

        const thresholdSlider = document.getElementById(`threshold-${resultId}`) as HTMLInputElement;
        const thresholdValue = document.getElementById(`threshold-value-${resultId}`);

        if (thresholdSlider && thresholdValue) {
            thresholdSlider.addEventListener('input', (e) => {
                const value = parseFloat((e.target as HTMLInputElement).value);
                this.updateResultSetting(resultId, 'activationThreshold', value);
                thresholdValue.textContent = `${Math.round(value * 100)}%`;
                this.renderHeatmapsForResult(resultId);
            });
        }

        this.updateLayerButtons(resultId);
    }

    private updateResultSetting(resultId: string, setting: keyof ResultVisualizationSettings, value: any): void {
        const settings = this.resultSettings.get(resultId);
        if (settings) {
            (settings as any)[setting] = value;
            this.resultSettings.set(resultId, settings);
        }
    }

    private updateLayerButtons(resultId: string): void {
        const layerContainer = document.querySelector(`.layer-buttons[data-result-id="${resultId}"]`);
        const settings = this.resultSettings.get(resultId);

        if (!layerContainer || !settings || !this.currentModelConfig?.feature_layers) return;

        layerContainer.innerHTML = '';

        Object.keys(this.currentModelConfig.feature_layers).forEach(layerName => {
            const isVisible = settings.visibleLayers.has(layerName);

            const button = document.createElement('button');
            button.type = 'button';
            button.className = `btn btn-sm ${isVisible ? 'btn-primary' : 'btn-outline-primary'} layer-btn`;
            button.textContent = layerName;
            button.dataset.layer = layerName;
            button.dataset.resultId = resultId;

            button.addEventListener('click', () => {
                this.toggleResultLayerVisibility(resultId, layerName);
                button.className = `btn btn-sm ${settings.visibleLayers.has(layerName)
                    ? 'btn-primary' : 'btn-outline-primary'} layer-btn`;
                this.renderHeatmapsForResult(resultId);
            });

            layerContainer.appendChild(button);
        });
    }

    private toggleResultLayerVisibility(resultId: string, layerName: string): void {
        const settings = this.resultSettings.get(resultId);
        if (settings) {
            if (settings.visibleLayers.has(layerName)) {
                settings.visibleLayers.delete(layerName);
            } else {
                settings.visibleLayers.add(layerName);
            }
            this.resultSettings.set(resultId, settings);
        }
    }
    private updateProcessButtonMode(): void {
        //const processBtn = document.getElementById('processBtn') as HTMLButtonElement;
        const processBtnText = document.getElementById('processBtnText');
        const processModeInfo = document.getElementById('processModeInfo');

        if (this.selectedFiles.length > 1) {
            // batch
            processBtnText!.textContent = `Process All Images (${this.selectedFiles.length})`;
            processModeInfo!.innerHTML = '<small class="text-info"></small>';
        } else if (this.selectedFiles.length === 1) {
            // single
            processBtnText!.textContent = 'Analyze Image';
            processModeInfo!.innerHTML = '<small> </small>';
        } else {
            // no files
            processBtnText!.textContent = 'Analyze Image';
            processModeInfo!.innerHTML = '<small>Select images to begin</small>';
        }
    }

    private getDiagnosisColor(classIndex: number): string {
        const colors = ['danger', 'success', 'warning', 'info'];
        return colors[classIndex] || 'secondary';
    }


    private switchToResultsTab(): void {
        const resultsTab = document.getElementById('results-tab') as HTMLButtonElement;
        if (resultsTab) {
            resultsTab.click();
        }
    }

    private updateHeatmapDisplay(resultId?: string): void {
        if (!resultId)
            return;
        const resultElement = document.querySelector(`[data-result-id="${resultId}"]`);
        const isAccordionOpen = resultElement?.querySelector('.accordion-body:not(.collapse)');

        if (isAccordionOpen) {
            this.renderHeatmapsForResult(resultId);
        }
    }
    // !deprecated

    private async processBatchImages(): Promise<void> {
        console.log('Processing batch images...');
        if (this.selectedFiles.length === 0) return;

        this.isBatchProcessing = true;
        this.batchResults = [];
        this.updateBatchProgress(0, `Starting batch processing: 0/${this.selectedFiles.length}`);

        try {
            // classify and show results before cam calc
            for (let i = 0; i < this.selectedFiles.length; i++) {
                if (!this.isBatchProcessing) break;

                const file = this.selectedFiles[i];
                const fileProgress = ((i + 1) / this.selectedFiles.length) * 50;

                this.updateBatchProgress(fileProgress, `Classifying ${i + 1}/${this.selectedFiles.length}: ${file.name}`);

                try {
                    const img = await this.loadImageFromFile(file);
                    const imageData = await this.loadImageAsDataURL(file);
                    const existingResult = this.allResults.find(result => result.data.fileName === file.name &&
                        result.data.imageData === imageData);

                    if (existingResult?.camReady)
                        continue;
                    // add to batch results list with initial state
                    this.batchResults.push({
                        fileName: file.name,
                        result: {
                            success: true,
                            imageName: file.name,
                            predictions: null,
                            predictedClassIndex: null,
                            confidence: null,
                            heatmaps: {},
                            hasHeatmaps: false,
                            metrics: { totalTime: 0, totalInferences: 0 } as InferenceMetrics
                        },
                        classificationResult: null,
                        imageData: imageData,
                        camReady: false,
                        progress: 15,
                        status: 'processing'
                    });

                    const resultId = existingResult ? existingResult.data.id :
                        this.addToResults('batch', {
                            fileName: file.name,
                            imageData: await this.loadImageAsDataURL(file),
                            classification: null,
                            heatmaps: null,
                            metrics: null,
                            camReady: false,
                            success: true,
                            progress: 15,
                            status: 'processing'
                        });

                    this.updateResultProgress(resultId, 20, 'Classifying...');

                    const classification = await this.onnxCAM.computeClassification(img);
                    const batchResult = this.batchResults.find(result => result.fileName === file.name);
                    if (batchResult === undefined || !batchResult.result) {
                        console.log('batchResult.result is null');
                        continue;
                    }
                    batchResult.classificationResult = classification;
                    batchResult.result.predictions = classification.predictions;
                    batchResult.result.predictedClassIndex = classification.predictedClassIndex;
                    batchResult.result.confidence = classification.confidence;
                    batchResult.progress = 25;

                    this.updateResult(resultId, {
                        classification: classification,
                        progress: 25,
                        status: 'Waiting...'
                    });
                    console.log("result with id", resultId, "updated with classification:", this.allResults.find(r => r.data.id === resultId));
                } catch (error) {
                    console.error(`Failed to process ${file.name}:`, error);
                    // !!todo handle error
                }
            }

            this.displayResults();
            this.switchToResultsTab();
            this.updateBatchProgress(25, `Starting CAM processing...`);
            await this.processCAMsInBackground();

        } finally {
            this.isBatchProcessing = false;
            this.hideBatchProgress();
        }
    }
    private async processCAMsInBackground(): Promise<void> {
        const totalItems = this.batchResults.length;
        for (let i = 0; i < totalItems; i++) {
            if (!this.isBatchProcessing) break;

            const batchResult = this.batchResults[i];
            if (!batchResult.result) continue;

            if (!batchResult.result.success || batchResult.camReady || !batchResult.classificationResult) continue;

            try {
                const unifiedResult = this.allResults.find(r =>
                    r.data.fileName === batchResult.fileName
                );

                /*if (unifiedResult) {                 
                    const individualProgress = 60 + ((i + 1) / totalItems) * 30;
                    this.updateResultProgress(unifiedResult.data.id, individualProgress, 'processing');
                } */
                const cams = await this.onnxCAM.computeCAMs(
                    batchResult.classificationResult!,
                    undefined,
                    undefined,
                    undefined,
                    (progress: number, currentLayer: string, totalLayers: number, heatmaps?: any) => {
                        const overallProgress = 25 + progress * 75;
                        this.updateResultProgress(
                            unifiedResult?.data.id!,
                            overallProgress,
                            `Layer: ${currentLayer}`
                        );
                        if (heatmaps) {
                            this.updateHeatmapDisplay(unifiedResult?.data.id!);
                        }
                    }
                );
                batchResult.result.heatmaps = cams.heatmaps;
                batchResult.result.hasHeatmaps = true;
                batchResult.result.metrics = cams.metrics;
                batchResult.camReady = Object.entries(cams.heatmaps).length > 0;
                batchResult.progress = 100;
                batchResult.status = 'completed';
                console.log('Generated CAMs for', batchResult.fileName);
                if (unifiedResult) {
                    this.updateResult(unifiedResult.data.id, {
                        heatmaps: cams.heatmaps,
                        metrics: cams.metrics,
                        camReady: true,
                        progress: 100,
                        status: 'completed'
                    });

                    this.updateResultUI(unifiedResult.data.id, 100, 'completed');
                }
            } catch (error) {
                console.error(`Failed to generate CAMs for ${batchResult.fileName}:`, error);
                batchResult.result.error = (error as Error).message;
                batchResult.result.success = false;
                batchResult.progress = 100; // mark as complete (failed)
                batchResult.status = 'failed';

                // update result
                const unifiedResult = this.allResults.find(r =>
                    r.data.fileName === batchResult.fileName
                );

                if (unifiedResult) {
                    this.updateResult(unifiedResult.data.id, {
                        error: (error as Error).message,
                        success: false,
                        progress: 100,
                        status: 'failed'
                    });

                    this.updateResultUI(unifiedResult.data.id, 100, 'failed');
                }
            }
        }
        this.updateBatchResultsUI();
    }
    /* private createInitialProcessingResult(fileName: string): ProcessingResult {
        return {
            success: true,
            imageName: fileName,
            predictions: {},
            predictedClassIndex: {},
            confidence: {},
            heatmaps: {},
            hasHeatmaps: false,
        };
    }

    private createInitialClassificationResult(img: HTMLImageElement): any {
        return {
            predictions: {},
            predictedClassIndex: {},
            confidence: {},
            originalOutput: null,
            origWidth: img.naturalWidth,
            origHeight: img.naturalHeight,
            originalImageU8: new Uint8Array(),
            preprocessingParams: { mean: [], std: [], scale: 1 }
        };
    } */
    private async loadImageAsDataURL(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (e.target?.result) {
                    resolve(e.target.result as string);
                } else {
                    reject(new Error('Failed to read file'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    private async renderHeatmapsForResult(resultId: string): Promise<void> {
        const result = this.allResults.find(r => r.data.id === resultId);

        const resultElement = document.querySelector(`[data-result-id="${resultId}"]`);
        const isAccordionOpen = resultElement?.querySelector('.accordion-body:not(.collapse)');

        if (!result || !result.data.heatmaps || !isAccordionOpen) return;

        const settings = this.resultSettings.get(resultId) ?? {
            opacity: this.camSettings.opacity,
            activationThreshold: this.camSettings.activationThreshold,
            visibleLayers: this.camSettings.visibleLayers
        } as ResultVisualizationSettings;
        console.log('resultsettings', this.resultSettings.get(resultId));
        const mainHeatmapCanvas = document.getElementById(`heatmap-${resultId}`) as HTMLCanvasElement;
        if (mainHeatmapCanvas && result.data.heatmaps) {
            const aggregatedHeatmap = this.getAggregatedHeatmapForResult(result.data, settings);

            const image = await this.loadImage(result.data.imageData);

            const empty = new Float32Array().fill(0);
            let tempCanvas = this.overlayHeatmapOnImage(
                image,
                empty,
                this.currentModelConfig?.input_size || 224,
                this.currentModelConfig?.input_size || 224,
                {
                    opacity: settings.opacity,
                    colormap: "jet",
                    activationThreshold: settings.activationThreshold
                },
            );
            let overlayCanvas = tempCanvas;
            const visibleLayers = Array.from(settings.visibleLayers);
            const processedLayers = Object.keys(result.data.heatmaps);
            const intersection = Array.from(visibleLayers).filter(layer => processedLayers.includes(layer));
            console.log("layers", intersection);

            let combinedCanvas = document.createElement('canvas');
            combinedCanvas.width = overlayCanvas.width;
            combinedCanvas.height = overlayCanvas.height;
            const combinedCtx = combinedCanvas.getContext('2d')!;

            combinedCtx.drawImage(overlayCanvas, 0, 0);
            for (const layer in intersection) {

                const heatmapArr = result.data.heatmaps[intersection[layer]][this.currentModelConfig!.classification_layer].data;
                const layerCanvas = this.overlayHeatmapOnImage(
                    image,
                    heatmapArr,
                    this.currentModelConfig?.input_size || 224,
                    this.currentModelConfig?.input_size || 224,
                    {
                        opacity: settings.opacity,
                        colormap: "jet",
                        blendMode: "copy",
                        activationThreshold: settings.activationThreshold
                    },
                );


                combinedCtx.drawImage(layerCanvas, 0, 0);

                overlayCanvas = combinedCanvas;
            }
            const ctx = mainHeatmapCanvas.getContext('2d');
            if (ctx && overlayCanvas) {
                mainHeatmapCanvas.height = overlayCanvas.height;
                mainHeatmapCanvas.width = overlayCanvas.width;
                const canvasSource = aggregatedHeatmap ? overlayCanvas : image;
                ctx.drawImage(canvasSource, 0, 0);
            }

        }

    }
    private overlayHeatmapOnImage(
        baseImg: HTMLImageElement,
        heatmapData: Float32Array, // normalized [0,1]
        width: number,
        height: number,
        {
            opacity = 0.5,
            colormap = "jet",
            blendMode = "multiply",
            activationThreshold = 1.0,
        }: { opacity?: number; colormap?: string; blendMode?: GlobalCompositeOperation, activationThreshold?: number } = {}
    ): HTMLCanvasElement {
        const canvas = document.createElement("canvas");
        canvas.width = baseImg.naturalWidth || 224;
        canvas.height = baseImg.naturalHeight || 224;

        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);

        // build from flat normalized heatmap
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        //console.log("heatmapData", heatmapData);
        let filtered = heatmapData;
        if (!filtered)
            return canvas;

        if (activationThreshold !== undefined) {
            const total = filtered.reduce((a, b) => a + b, 0);
            const threshold = total * activationThreshold;
            let running = 0;
            const sorted = Array.from(filtered).map((v, i) => ({ v, i }))
                .sort((a, b) => b.v - a.v);
            const keepIndices = new Set<number>();
            for (const { v, i } of sorted) {
                if (running >= threshold) break;
                keepIndices.add(i);
                running += v;
            }
            const newFiltered = new Float32Array(filtered.length);
            keepIndices.forEach(idx => { newFiltered[idx] = filtered[idx]; });
            filtered = newFiltered;
        }

        for (let i = 0; i < filtered.length; i++) {
            const t = Math.min(Math.max(filtered[i], 0), 1); // clamp
            const [r, g, b] = this.getColorMapValue(t, colormap);
            const j = i * 4;
            data[j] = r;
            data[j + 1] = g;
            data[j + 2] = b;
            data[j + 3] = Math.floor(255 * t); //intensity
        }

        // heatmap
        const heatmapCanvas = document.createElement("canvas");
        heatmapCanvas.width = width;
        heatmapCanvas.height = height;
        const hctx = heatmapCanvas.getContext("2d")!;
        hctx.putImageData(imageData, 0, 0);

        // blend 
        ctx.globalAlpha = opacity;
        ctx.globalCompositeOperation = blendMode;
        ctx.drawImage(heatmapCanvas, 0, 0, canvas.width, canvas.height);

        // reset
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = "source-over";

        return canvas;
    }

    /**
     * colormaps
     */
    private getColorMapValue(t: number, cmap: string): [number, number, number] {
        t = Math.min(Math.max(t, 0), 1); // clamp

        switch (cmap) {
            case "jet": {
                const r = Math.floor(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 3), 0), 1));
                const g = Math.floor(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 2), 0), 1));
                const b = Math.floor(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 1), 0), 1));
                return [r, g, b];
            }
            case "viridis": {
                // Approximation
                const r = Math.floor(255 * (0.267 + 0.633 * t - 0.333 * t * t));
                const g = Math.floor(255 * (0.005 + 1.31 * t - 0.775 * t * t));
                const b = Math.floor(255 * (0.329 + 0.705 * (1 - t) - 0.504 * (1 - t) * (1 - t)));
                return [r, g, b];
            }
            case "plasma": {
                const r = Math.floor(255 * (0.050 + 2.404 * t - 2.915 * t * t + 1.177 * t * t * t));
                const g = Math.floor(255 * (0.030 + 0.442 * t + 1.033 * t * t - 1.246 * t * t * t));
                const b = Math.floor(255 * (0.527 - 0.279 * t + 0.016 * t * t + 0.738 * t * t * t));
                return [r, g, b];
            }
            case "inferno": {
                const r = Math.floor(255 * (0.001 + 2.279 * t - 1.941 * t * t + 0.236 * t * t * t));
                const g = Math.floor(255 * (0.000 + 0.571 * t + 0.691 * t * t - 1.474 * t * t * t));
                const b = Math.floor(255 * (0.013 + 0.055 * t + 0.204 * t * t + 1.057 * t * t * t));
                return [r, g, b];
            }
            case "gray":
            default: {
                const v = Math.floor(255 * t);
                return [v, v, v];
            }
        }
    }
    private redrawHeatmapCanvas(
        canvas: HTMLCanvasElement,
        heatmap: Float32Array,
        sideLen: number,
        options: ResultVisualizationSettings
    ): void {
        const ctx = canvas.getContext("2d")!;
        const imageData = ctx.createImageData(sideLen, sideLen);

        // normalize 
        let min = Infinity, max = -Infinity;
        for (let v of heatmap) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min;
        let filtered = heatmap;


        if (options.activationThreshold !== undefined) {
            const total = filtered.reduce((a, b) => a + b, 0);
            const threshold = total * options.activationThreshold;
            let running = 0;
            const sorted = Array.from(filtered).map((v, i) => ({ v, i }))
                .sort((a, b) => b.v - a.v);
            const keepIndices = new Set<number>();
            for (const { v, i } of sorted) {
                if (running >= threshold) break;
                keepIndices.add(i);
                running += v;
            }
            const newFiltered = new Float32Array(filtered.length);
            keepIndices.forEach(idx => { newFiltered[idx] = filtered[idx]; });
            filtered = newFiltered;
        }

        // draw 
        for (let i = 0; i < filtered.length; i++) {
            const normVal = range === 0 ? 0 : (filtered[i] - min) / range;
            const [r, g, b] = this.colormap(normVal);
            const offset = i * 4;
            imageData.data[offset] = r;
            imageData.data[offset + 1] = g;
            imageData.data[offset + 2] = b;
            imageData.data[offset + 3] = Math.floor(255 * options.opacity);
        }

        ctx.putImageData(imageData, 0, 0);
    }
    private colormap(value: number): [number, number, number] {
        let r = 0, g = 0, b = 0;
        if (value < 0.25) {
            b = 255;
            g = Math.floor(value * 4 * 255);
        } else if (value < 0.5) {
            g = 255;
            b = Math.floor((0.5 - value) * 4 * 255);
            r = Math.floor((value - 0.25) * 4 * 255);
        } else if (value < 0.75) {
            r = 255;
            g = 255;
            b = Math.floor((0.75 - value) * 4 * 255);
        } else {
            r = 255;
            g = Math.floor((1.0 - value) * 4 * 255);
        }
        return [r, g, b];
    }

    // !!deprecated
    private renderLayerHeatmapsForResult(resultId: string): void {

        const layerCanvases = document.querySelectorAll(`[data-result-id="${resultId}"].layer-heatmap`);
        const result = this.allResults.find(r => r.data.id === resultId);
        console.log("result", result);
        if (!result)
            return;
        const classificationLayer = this.currentModelConfig?.classification_layer;
        layerCanvases.forEach(canvas => {
            const layerName = canvas.getAttribute('data-layer');
            if (layerName && result.data.heatmaps && result.data.heatmaps[layerName]) {
                const heatmapData = result.data.heatmaps![layerName][classificationLayer!];
                const heatmapArray = new Float32Array(heatmapData.data);
                //console.log("layerName", layerName, " data array", heatmapArray);
                const layerCanvas = this.onnxCAM.createHeatmapCanvas(
                    heatmapArray,
                    layerName,
                    heatmapData.sideLen,
                );
                if (layerCanvas) {
                    const ctx = (canvas as HTMLCanvasElement).getContext('2d');
                    if (ctx) {
                        console.log("canvas:", canvas, "layerCanvas:", layerCanvas, " ctx:", ctx);
                        (canvas as HTMLCanvasElement).width = layerCanvas.width;
                        (canvas as HTMLCanvasElement).height = layerCanvas.height;
                        ctx.drawImage(layerCanvas, 0, 0);
                    }
                }
            }
        });
    }

    private async exportAllResults(): Promise<void> {
        if (this.allResults.length === 0) {
            this.showError('No results to export');
            return;
        }

        try {
            const classificationLayer = this.currentModelConfig?.classification_layer;
            let csvContent = 'FileName,Type,Diagnosis,Confidence,Status,ProcessingTime,HeatmapsAvailable,Timestamp\n';

            this.allResults.forEach(result => {
                const res = result.data;
                if (res.success && classificationLayer) {
                    const classIndex = res.classification.predictedClassIndex![classificationLayer];
                    const confidence = res.classification.confidence![classificationLayer];
                    const className = this.currentModelConfig?.class_names[classIndex] || 'Unknown';
                    const processingTime = res.metrics?.totalTime || 0;
                    const hasHeatmaps = res.heatmaps && Object.keys(res.heatmaps).length > 0;

                    csvContent += `"${res.fileName}","${className}",${(confidence * 100).toFixed(2)}%,Success,${processingTime}ms,${hasHeatmaps},${result.timestamp.toISOString()}\n`;
                } else {
                    csvContent += `"${res.fileName}",Error,0%,Failed,0ms,false,${result.timestamp.toISOString()}\n`;
                }
            });

            this.downloadFile(csvContent, `analysis_results_${new Date().toISOString().slice(0, 10)}.csv`, 'text/csv');
            this.showSuccess('Results exported successfully');

        } catch (error) {
            console.error('Export failed:', error);
            this.showError('Failed to export results');
        }
    }

    private async exportAllJson(): Promise<void> {
        if (this.allResults.length === 0) {
            this.showError('No results to export');
            return;
        }

        try {
            const exportData = {
                timestamp: new Date().toISOString(),
                model: this.currentModelConfig?.name,
                results: this.allResults.map(result => ({
                    timestamp: result.timestamp.toISOString(),
                    fileName: result.data.fileName,
                    ...result.data
                }))
            };

            const jsonContent = JSON.stringify(exportData, null, 2);
            this.downloadFile(jsonContent, `analysis_results_${new Date().toISOString().slice(0, 10)}.json`, 'application/json');
            this.showSuccess('JSON results exported successfully');

        } catch (error) {
            console.error('JSON export failed:', error);
            this.showError('Failed to export JSON results');
        }
    }
    private updateBatchProgress(percent: number, text: string): void {
        const progressBar = document.getElementById('batchProgressBar');
        const progressText = document.getElementById('batchProgressText');
        const progressPercent = document.getElementById('batchProgressPercent');
        const container = document.getElementById('batchProgressContainer');

        if (progressBar) progressBar.style.width = `${percent}%`;
        if (progressText) progressText.textContent = text;
        if (progressPercent) progressPercent.textContent = `${Math.round(percent)}%`;
        if (container) container.style.display = 'block';
    }



    private renderAnalysisContent(data: any, resultId: string): string {
        const classificationLayer = this.currentModelConfig?.classification_layer;
        const classIndex = classificationLayer ? data.classification.predictedClassIndex[classificationLayer] : -1;
        const confidence = classificationLayer ? data.classification.confidence[classificationLayer] : 0;
        const className = classIndex >= 0 ? this.currentModelConfig?.class_names[classIndex] || 'Unknown' : 'Unknown';
        const camReady = data.progress >= 100;
        // !!replaced by checkboxes for layer overlay.
        /*         const layerHeatmapsString =`        
                ${data.heatmaps && Object.keys(data.heatmaps).length > 0 ? `
                <div class="row mt-4">
                    <div class="col-12">
                        <h6 class="mb-3">
                            <i class="bi bi-layers"></i> Individual Heatmap Layers
                        </h6>
                        <div class="row g-2">
                            ${Object.entries(data.heatmaps).map(([layerName, heatmapData]: [string, any]) => `
                                <div class="col-md-3 col-sm-6">
                                    <div class="card h-100">
                                        <div class="card-header py-1">
                                            <small class="fw-bold text-truncate d-block">${layerName}</small>
                                        </div>
                                        <div class="card-body p-1 text-center">
                                            <canvas class="layer-heatmap" 
                                                    data-layer="${layerName}" 
                                                    data-result-id="${resultId}"
                                                    style="max-height: 100px; width: auto;"></canvas>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                ` : ''}
                `; */
        const analysisContentString =
            `
        <!-- Diagnosis Results -->                             
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div id="visualizationSettings">
                        <div class="card-body">
                            <div class="row g-3 .prevent-select">
                                <div class="col-md-6">
                                    <label for="opacity-${resultId}" class="form-label">
                                        Opacity: <span class="badge bg-secondary" id="opacity-value-${resultId}">60%</span>
                                    </label>
                                    <input id="opacity-${resultId}" type="range" class="form-range heatmap-opacity" 
                                        min="0" max="1" step="0.05" value="0.6" data-result-id="${resultId}">
                                </div>
                                <div class="col-md-6">
                                    <label for="threshold-${resultId}" class="form-label">
                                        Activation Focus: <span class="badge bg-secondary" id="threshold-value-${resultId}">70%</span>
                                    </label>
                                    <input id="threshold-${resultId}" type="range" class="form-range heatmap-threshold" 
                                        min="0" max="1" step="0.05" value="0.7" data-result-id="${resultId}">
                                </div>
                                <div class="col-12">
                                    <label class="form-label">Visible Layers</label>
                                    <div class="layer-buttons" data-result-id="${resultId}">
                                        <!-- Layers-->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Heatmap -->
        <div class="row mb-4">
            <div class="col-md-6 text-center">
                <div class="border rounded p-2 bg-white text-center">
                    <canvas id="heatmap-${resultId}" class="img-fluid rounded" 
                            style="max-height: 250px; width: auto;"></canvas>
                </div>
            </div>
        </div>
        <!-- Performance Metrics -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card bg-light border-0">
                    <div class="card-body">
                        <h5 class="card-title text-primary mb-3">
                            <i class="bi bi-heart-pulse"></i> Performance Metrics
                        </h5>
                        <div class="row">
                            <div class="col-md-6">
                                ${data.metrics ? `
                                    <div class="mb-2">
                                        <strong>Processing Time:</strong> ${data.metrics.totalTime}ms
                                    </div>
                                    <div class="mb-2">
                                        <strong>Total Inferences:</strong> ${data.metrics.totalInferences}
                                    </div>
                                    <div class="mb-2">
                                        <strong>Heatmaps:</strong> ${data.heatmaps ? Object.keys(data.heatmaps).length : 0} layers
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        `;

        return `${analysisContentString}`//${layerHeatmapsString}';
    }


    private setupDragAndDrop(): void {
        //remove existing listeners
        this.cleanupDragListeners();

        const dragOverHandler = (e: DragEvent) => {
            e.preventDefault();
            this.dropZone.classList.add('drag-over', 'border-primary');
        };

        const dragLeaveHandler = (e: DragEvent) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-over', 'border-primary');
        };

        const dropHandler = (e: DragEvent) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-over', 'border-primary');

            if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
                this.handleFiles(e.dataTransfer.files);
            }
        };

        this.dropZone.addEventListener('dragover', dragOverHandler);
        this.dropZone.addEventListener('dragleave', dragLeaveHandler);
        this.dropZone.addEventListener('drop', dropHandler);

        // for cleanup
        this.dragListeners = {
            dragover: dragOverHandler,
            dragleave: dragLeaveHandler,
            drop: dropHandler
        };
    }
    private async loadImageFromFile(file: File): Promise<HTMLImageElement> {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const reader = new FileReader();

            reader.onload = (e) => {
                img.src = e.target?.result as string;
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Failed to load image'));
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    private updateUIState(state: 'loading' | 'ready' | 'processing' | 'results-ready' | 'error'): void {
        const processBtn = document.getElementById('processBtn') as HTMLButtonElement;
        const processBtnText = document.getElementById('processBtnText');

        switch (state) {
            case 'loading':
                processBtn.disabled = true;
                processBtnText!.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Loading...';
                break;
            case 'ready':
                processBtn.disabled = this.selectedFiles.length === 0;
                this.updateProcessButtonMode(); // restore text
                break;
            case 'processing':
                processBtn.disabled = true;
                if (this.selectedFiles.length > 1) {
                    processBtnText!.innerHTML = '<i class="bi bi-collection-play me-2"></i>Processing Batch...';
                } else {
                    processBtnText!.innerHTML = '<i class="bi bi-gear-fill me-2"></i>Processing...';
                }
                break;
            case 'results-ready':
                processBtn.disabled = false;
                this.updateProcessButtonMode();
                break;
            case 'error':
                processBtn.disabled = true;
                processBtnText!.innerHTML = '<i class="bi bi-exclamation-triangle me-2"></i>Error';
                break;
        }
    }
    /*
    ** update the progress bar on "Analysis" tab.
    */
    private showProgress(percent: number, message: string): void {
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const progressPercent = document.getElementById('progress-percent');
        const container = document.getElementById('progress-container');

        if (progressBar) progressBar.style.width = `${percent}%`;
        if (progressText) progressText.textContent = message;
        if (progressPercent) progressPercent.textContent = `${percent}%`;
        if (container) container.style.display = 'block';
    }
    /*
    ** hide progress bar on "Analysis" tab.
    */
    private hideProgress(): void {
        const container = document.getElementById('progress-container');
        if (container) container.style.display = 'none';
    }

    private showError(message: string): void {
        this.showToast(message, 'danger');
    }

    private showSuccess(message: string): void {
        this.showToast(message, 'success');
    }

    private showToast(message: string, type: 'success' | 'danger' | 'info' | 'warning'): void {
        const toast = document.getElementById('appToast');
        const toastBody = document.getElementById('appToastBody');

        if (toast && toastBody) {
            const bgClass = `bg-${type}`;
            toast.className = `toast align-items-center text-white border-0 ${bgClass}`;
            toastBody.textContent = message;

            const bsToast = new (window as any).bootstrap.Toast(toast);
            bsToast.show();
        }
    }

    private cleanupResources(): void {

        //this.cleanupDragListeners();
        this.removeEventListeners();

        this.onnxModelService.dispose();
        this.cachedHeatmaps = {};
        this.batchResults = [];
        this.selectedFiles = [];
        this.currentFileIndex = 0;
        this.currentImage = null;
    }






}

declare global {
    interface Window {
        App: typeof App;
        loadSampleImage: (path: string) => void;
    }
}

// expose
window.App = App;
window.loadSampleImage = function (path: string) {
    console.log('Loading sample:', path);
};
