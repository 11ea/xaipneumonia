document.addEventListener('DOMContentLoaded', function () {

    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const processBtn = document.getElementById('processBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const modelSelect = document.getElementById('modelTypeSelect');

    let currentImage = null;
    let currentModel = 'yolon-artirilmisVeri';
    let processingHistory = [];

    let batchFiles = [];
    let currentBatchIndex = 0;
    let isBatchProcessing = false;
    let batchResults = [];

    const appResources = {
        images: [],
        canvases: [],
        eventListeners: []
    };
    window.onnxModelService = new ONNXModelService();

    let currentModelType = 'yolon-artirilmisVeri';
    if (modelSelect) {
        modelSelect.addEventListener('change', function (e) {
            const newModelType = e.target.value;
            console.log('Model type changed to:', newModelType);

            // Check if model service is available
            if (window.onnxModelService) {
                // Show loading state
                const loadingIndicator = document.createElement('span');
                loadingIndicator.className = 'spinner-border spinner-border-sm ms-2';
                loadingIndicator.id = 'modelLoadingIndicator';
                e.target.parentNode.appendChild(loadingIndicator);

                // Load new model
                window.onnxModelService.loadModel(newModelType)
                    .then(() => {
                        console.log('Model switched successfully');
                    })
                    .catch(error => {
                        console.error('Failed to switch model:', error);
                        alert('Failed to load model: ' + error.message);
                    })
                    .finally(() => {
                        const indicator = document.getElementById('modelLoadingIndicator');
                        if (indicator) {
                            indicator.remove();
                        }
                    });
            } else {
                console.error('ONNX Model Service not available');
            }
        });
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    initEventListeners();
    addKeyboardControls();
    function initEventListeners() {
        cleanupResources();

        document.body.addEventListener('click', function (e) {

            if (e.target.tagName === 'INPUT' && e.target.type === 'file') {
                return;
            }
            if (e.target.tagName === 'LABEL' && e.target.htmlFor === 'fileInput') {
                return;
            }

            const actionElement = e.target.closest('[data-action]');
            if (!actionElement) return;

            const action = actionElement.getAttribute('data-action');

            if (action === 'browse') {
                e.preventDefault();
                e.stopImmediatePropagation();
                fileInput.click();
                return;
            }

            const resultIndex = actionElement.getAttribute('data-result-index');
            const imgSrc = actionElement.getAttribute('data-src');

            switch (action) {
                case 'browse':
                    fileInput.click();
                    break;

                case 'load-sample':
                    if (imgSrc) loadSampleImage(imgSrc);
                    break;

                // Processing
                case 'process-image':
                    processImage();
                    break;

                case 'try-another-model':
                    resultsContainer.style.display = 'none';
                    processImage();
                    break;

                // Results handling
                case 'back-to-upload':
                    resultsContainer.style.display = 'none';
                    clearSelection();
                    break;

                case 'download-result':
                    if (currentResultsData) {
                        downloadResultWithHeatmap(currentResultsData.result.classification);
                    }
                    break;

                case 'save-session':
                    saveSessionToLocalStorage();
                    showToast('Session saved to local storage!', 'success');
                    break;

                // Batch processing
                case 'clear-batch':
                    clearBatchSelection();
                    break;
                case 'preview-batch':
                    showBatchPreview();
                    break;
                case 'cancel-batch':
                    cancelBatchProcessing();
                    break;

                case 'export-results':
                    exportBatchResults();
                    break;

                // Single result from batch
                case 'back-to-batch':
                    hideSingleResultView();
                    break;

                case 'show-single-result':
                    if (resultIndex !== null) {
                        showSingleResultFromBatch(parseInt(resultIndex));
                    }
                    break;

                case 'download-single':
                    if (resultIndex !== null) {
                        downloadSingleResult(parseInt(resultIndex));
                    }
                    break;

                case 'reprocess-single':
                    if (resultIndex !== null) {
                        reprocessSingleResult(parseInt(resultIndex));
                    }
                    break;

                // Clear/Reset actions
                case 'clear-selection':
                    clearSelection();
                    break;

                default:
                    console.warn('Unknown action:', action);
            }
        });
        fileInput.addEventListener('change', function (e) {
            console.log('File input changed');
            if (e.target.files.length > 0) {
                handleFiles(e.target.files);
            }
        });
        document.getElementById('modelTypeSelect').addEventListener('change', function (e) {
            currentModelType = e.target.value;
            console.log('Selected model type:', currentModelType);

            if (window.onnxModelService.isLoaded) {
                window.onnxModelService.dispose();
                window.onnxModelService.loadModel(currentModelType);
            }
        });
        window.addEventListener('beforeunload', function () {
            cleanupResources();
        });

        window.addEventListener('pagehide', function () {
            cleanupResources();
        });
        const batchInput = document.getElementById('batchInput');
        if (batchInput) {
            batchInput.addEventListener('change', handleBatchSelect);
            console.log('Batch input listener attached');
        } else {
            console.warn('Batch input element not found');
        }

        document.body.addEventListener('input', function (e) {
            if (e.target.id === 'heatmapOpacity') {
                const heatmapOverlay = document.getElementById('heatmapOverlay');
                if (heatmapOverlay) {
                    heatmapOverlay.style.opacity = e.target.value / 100;
                    drawMockHeatmap();
                }
            }
            else if (e.target.id === 'singleHeatmapOpacity') {
                const heatmapOverlay = document.getElementById('singleHeatmapOverlay');
                if (heatmapOverlay) {
                    heatmapOverlay.style.opacity = e.target.value / 100;
                    drawSingleResultHeatmap();
                }
            }
        });
    }
    function cleanupResources() {
        console.log('Cleaning up resources...');
        appResources.images.forEach(img => {
            if (img.src && img.src.startsWith('blob:')) {
                URL.revokeObjectURL(img.src);
            }
            img.src = '';
            img.remove();
        });
        appResources.images = [];

        appResources.canvases.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            canvas.width = 1;
            canvas.height = 1;
            canvas.remove();
        });
        appResources.canvases = [];

        appResources.eventListeners.forEach(({ element, type, handler }) => {
            if (element && element.removeEventListener) {
                element.removeEventListener(type, handler);
            }
        });
        appResources.eventListeners = [];
        if (window.gc) {
            window.gc();
        }

        console.log('Resources cleaned up');
    }

    // Track resource creation
    function trackImage(element) {
        appResources.images.push(element);
        return element;
    }

    function trackCanvas(element) {
        appResources.canvases.push(element);
        return element;
    }

    function trackEventListener(element, type, handler) {
        appResources.eventListeners.push({ element, type, handler });
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];

        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        if (files.length === 1) {
            batchFiles = [];
            const batchInput = document.getElementById('batchInput');
            if (batchInput) {
                batchInput.value = '';
            }
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            currentImage = {
                name: file.name,
                data: e.target.result,
                type: file.type
            };

            processBtn.disabled = false;

            dropZone.innerHTML = `
            <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 200px;">
            <div class="mt-2">${file.name}</div>
            <button class="btn btn-sm btn-outline-secondary mt-2" data-action="browse">Change Image</button>
        `;
        };
        reader.readAsDataURL(file);
    }
    function loadSampleImage(src) {
        currentImage = {
            name: src.split('/').pop(),
            data: src,
            type: 'image/jpeg'
        };


        processBtn.disabled = false;

        dropZone.innerHTML = `
            <img src="${src}" class="img-fluid rounded" style="max-height: 200px;">
            <div class="mt-2">Sample Image</div>
            <button class="btn btn-sm btn-outline-secondary mt-2" data-action="clear-selection">Change Image</button>
        `;
    }

    function clearSelection() {
        console.log('Clearing selection');

        currentImage = null;
        currentProcessedImage = null;

        if (isBatchProcessing) {
            cancelBatchProcessing();
        }
        batchFiles = [];
        clearBatchResultsUI();

        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';
        fileInput.value = '';

        resetDropZoneToDefault();

        const batchInput = document.getElementById('batchInput');
        if (batchInput) {
            batchInput.value = '';
        }

        console.log('Selection cleared');
    }

    async function processImage() {
        if (isBatchProcessing) {
            alert('Batch processing already in progress');
            return;
        }

        if (!window.onnxModelService) {
            alert('Model service not initialized');
            return;
        }

        // Clear previous results
        clearPreviousResults();

        let inputTensor;
        let imgElement;

        try {
            processBtn.disabled = true;
            processBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Loading Model...';

            if (!window.onnxModelService.isLoaded) {
                const modelType = document.getElementById('modelTypeSelect').value;
                await window.onnxModelService.loadModel(modelType);
            }

            processBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';

            if (batchFiles.length > 0) {
                await processBatch();
                return;
            } else if (currentImage) {
                imgElement = new Image();
                imgElement.src = currentImage.data;

                await new Promise((resolve, reject) => {
                    imgElement.onload = resolve;
                    imgElement.onerror = reject;
                    // Timeout to prevent hanging
                    setTimeout(() => reject(new Error('Image loading timeout')), 10000);
                });
                const imageHash = await getImageHash(imgElement);
                console.log('Processing image with hash:', imageHash);

                inputTensor = await window.onnxModelService.preprocessImage(imgElement);
                const results = await window.onnxModelService.predict(inputTensor);

                const { classification, confidence } = parseYOLOOutput(results, imageHash);

                console.log('New prediction:', { classification, confidence });

                displayResults({
                    result: {
                        classification: classification,
                        confidence: confidence
                    },
                    heatmap_available: true
                });



            } else {
                alert('Please select an image first');
            }

        } catch (error) {
            console.error('Processing error:', error);
            alert('Error: ' + error.message);
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';

            // Force cleanup
            if (inputTensor) {
                inputTensor.dispose();
                inputTensor = null;
            }
            if (imgElement) {
                imgElement.src = ''; // Clear image source
                imgElement = null;
            }

            // Force garbage collection
            forceGarbageCollection();
        }
    }

    function clearPreviousResults() {
        // Clear any previous result state
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
            resultsContainer.style.display = 'none';
        }
    }

    function forceGarbageCollection() {
        // Try to force garbage collection
        if (window.gc) {
            window.gc();
        }
        // Alternative method
        try {
            new ArrayBuffer(1024 * 1024 * 100); // Allocate and immediately discard
        } catch (e) { }
    }

    function inspectResults(results) {
        // Detailed inspection of model outputs
        const inspection = {};
        for (const key in results) {
            const tensor = results[key];
            inspection[key] = {
                dimensions: tensor.dims,
                type: tensor.type,
                data: Array.from(tensor.data).slice(0, 5), // First 5 values
                sum: Array.from(tensor.data).reduce((a, b) => a + b, 0)
            };
        }
        return inspection;
    }
    const cancelBatchBtn = document.getElementById('cancelBatchBtn');
    if (cancelBatchBtn) {
        cancelBatchBtn.addEventListener('click', cancelBatchProcessing);
    }
    function parseYOLOOutput(results, imageHash) {
        const cacheBuster = imageHash || Date.now();

        try {
            let predictions;
            let outputTensor;

            for (const key in results) {
                if (results[key].dims && results[key].dims.length > 0) {
                    outputTensor = results[key];
                    predictions = Array.from(outputTensor.data);
                    break;
                }
            }

            if (!predictions || predictions.length === 0) {
                throw new Error('No predictions found');
            }

            const predictionsHash = predictions.join(',');
            console.log('Predictions hash:', predictionsHash.substring(0, 50) + '...');
            console.log('Predictions:', predictions);
            const maxConfidence = Math.max(...predictions);
            const maxIndex = predictions.indexOf(maxConfidence);

            const confidence = maxConfidence + (Math.random() * 0.0001); // Tiny random factor
            const CLASS_NAMES = ['COVID-19', 'Normal', 'Viral', 'Bakteriyel'];
            return {
                classification: CLASS_NAMES[maxIndex],
                confidence: confidence,
                rawPredictions: predictions // For debugging
            };

        } catch (error) {
            console.error('Error parsing output:', error);

            return {
                classification: 'Error - ' + (Math.random() * 100).toFixed(0),
                confidence: Math.random(),
                isFallback: true
            };
        }
    }
    async function getImageHash(imageElement) {
        // Create a simple hash to verify different images are processed
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 16;
        canvas.height = 16;

        ctx.drawImage(imageElement, 0, 0, 16, 16);
        const imageData = ctx.getImageData(0, 0, 16, 16).data;

        // Simple hash from first few pixels
        let hash = 0;
        for (let i = 0; i < Math.min(100, imageData.length); i++) {
            hash = ((hash << 5) - hash) + imageData[i];
            hash |= 0; // Convert to 32-bit integer
        }

        return hash;
    }
    function cancelBatchProcessing() {
        console.log('Cancelling batch processing');

        isBatchProcessing = false;
        batchFiles = [];
        currentBatchIndex = 0;

        const progressContainer = document.getElementById('batchProgressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }

        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';

        batchResults = [];
        clearBatchResultsUI();

        console.log('Batch processing cancelled');
    }
    function sendProcessingRequest(formData) {
        fetch('/process/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    processingHistory.push({
                        image: currentImage,
                        result: data.result,
                        model: currentModel,
                        timestamp: new Date()
                    });

                    displayResults(data);
                } else {
                    alert('Error processing image: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while processing the image');
            })
            .finally(() => {
                processBtn.disabled = false;
                processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';
            });
    }
    let currentProcessedImage = null;

    function displayResults(data) {
        const cacheBuster = '?v=' + new Date().getTime();
        fetch('/static/templates/result.html' + cacheBuster)
            .then(response => {
                if (!response.ok) throw new Error('Template not found');
                return response.text();
            })
            .then(html => {
                resultsContainer.innerHTML = html;
                resultsContainer.style.display = 'block';

                currentProcessedImage = currentImage;

                document.getElementById('classificationResult').textContent = data.result.classification;
                document.getElementById('confidenceScore').textContent = Math.round(data.result.confidence * 100) + '%';
                //document.getElementById('modelName').textContent = data.model_info.name;
                //document.getElementById('modelSize').textContent = data.model_info.size;
                //document.getElementById('modelUpdated').textContent = data.model_info.updated;

                setTimeout(() => {
                    setupImageAndHeatmap();
                    setupResultEventListeners(data);
                }, 50);
            })
            .catch(error => {
                console.error('Error loading template:', error);
                createFallbackResultDisplay(data);
            });
    }
    function setupImageAndHeatmap() {
        let attempts = 0;
        const maxAttempts = 5;

        function trySetup() {
            attempts++;
            const originalImg = document.getElementById('originalImage');
            const heatmapCanvas = document.getElementById('heatmapOverlay');

            if (originalImg && heatmapCanvas && currentProcessedImage) {
                originalImg.src = currentProcessedImage.data;

                if (originalImg.complete) {
                    onImageLoaded(originalImg, heatmapCanvas);
                } else {
                    originalImg.onload = function () {
                        onImageLoaded(originalImg, heatmapCanvas);
                    };
                    originalImg.onerror = function () {
                        console.error('Image load error');
                        originalImg.src = '/static/images/placeholder.jpg';
                    };
                }
            } else if (attempts < maxAttempts) {
                setTimeout(trySetup, 100);
            } else {
                console.error('Failed to find elements after', maxAttempts, 'attempts');
            }
        }

        trySetup();
    }

    // Update the onImageLoaded function to pass dimensions
    function onImageLoaded(img, canvas) {
        console.log('Image loaded, dimensions:', img.naturalWidth, 'x', img.naturalHeight);

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';

        // Pass the dimensions to drawHeatmap
        drawHeatmap(null, img.naturalWidth, img.naturalHeight);

        const opacitySlider = document.getElementById('heatmapOpacity');
        if (opacitySlider) {
            canvas.style.opacity = opacitySlider.value / 100;
        }
    }
    function drawHeatmap(heatmapData, width, height) {
        const canvas = document.getElementById('heatmapOverlay');
        if (!canvas) return;

        // Use provided dimensions or fall back to canvas dimensions
        const finalWidth = width || canvas.width;
        const finalHeight = height || canvas.height;

        canvas.width = finalWidth;
        canvas.height = finalHeight;
        const ctx = canvas.getContext('2d');

        // Create mock heatmap data if none provided
        if (!heatmapData) {
            heatmapData = generateMockHeatmapData(finalWidth, finalHeight);
        }

        // Create image data
        const imageData = ctx.createImageData(finalWidth, finalHeight);

        // Apply color map to heatmap data
        for (let i = 0; i < heatmapData.length; i++) {
            const intensity = heatmapData[i];
            const color = getHeatmapColor(intensity);

            imageData.data[i * 4] = color.r;     // R
            imageData.data[i * 4 + 1] = color.g; // G
            imageData.data[i * 4 + 2] = color.b; // B
            imageData.data[i * 4 + 3] = color.a; // A
        }

        ctx.putImageData(imageData, 0, 0);
    }
    function generateMockHeatmapData(width, height) {
        const data = new Array(width * height);
        const centerX = width / 2;
        const centerY = height / 2;
        const maxDistance = Math.sqrt(centerX * centerX + centerY * centerY);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                const intensity = 1 - (distance / maxDistance);
                data[y * width + x] = Math.max(0, Math.min(1, intensity));
            }
        }

        return data;
    }

    function getHeatmapColor(intensity) {
        // Simple heatmap color gradient
        const r = Math.min(255, intensity * 2 * 255);
        const g = Math.min(255, intensity * 255);
        const b = Math.min(255, (1 - intensity) * 255);

        return { r: r, g: g, b: b, a: 200 }; // Semi-transparent
    }
    function setupResultEventListeners(data) {
        console.log('Setting up result event listeners');

        const opacitySlider = document.getElementById('heatmapOpacity');
        const heatmapOverlay = document.getElementById('heatmapOverlay');

        if (opacitySlider && heatmapOverlay) {
            console.log('Opacity controls found');

            heatmapOverlay.style.opacity = opacitySlider.value / 100;

            opacitySlider.addEventListener('input', function () {
                const opacity = this.value / 100;
                heatmapOverlay.style.opacity = opacity;
                drawMockHeatmap();
            });
        }

        setTimeout(() => {
            setupActionButtons(data);
        }, 100);
    }

    function setupActionButtons(data) {
        console.log('Setting up action buttons');

        const backBtn = document.getElementById('backBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const tryAnotherBtn = document.getElementById('tryAnotherBtn');
        const saveSessionBtn = document.getElementById('saveSessionBtn');

        console.log('Buttons found:', {
            backBtn: !!backBtn,
            downloadBtn: !!downloadBtn,
            tryAnotherBtn: !!tryAnotherBtn,
            saveSessionBtn: !!saveSessionBtn
        });

        if (backBtn) {
            console.log('Setting up back button listener');
            backBtn.addEventListener('click', function () {
                console.log('Back button clicked');
                resultsContainer.style.display = 'none';
                clearSelection();
            });
        } else {
            console.error('Back button not found');
            setTimeout(() => setupActionButtons(data), 100);
        }

        if (downloadBtn) {
            downloadBtn.addEventListener('click', function () {
                downloadResultWithHeatmap(data.result.classification);
            });
        }
        if (tryAnotherBtn) {
            tryAnotherBtn.addEventListener('click', function () {
                resultsContainer.style.display = 'none';
                processImage();
            });
        }

        if (saveSessionBtn) {
            saveSessionBtn.addEventListener('click', function () {
                saveSessionToLocalStorage();
                showToast('Session saved to local storage!');
            });
        }
    }

    function showToast(message, type = 'success') {
        const existingToasts = document.querySelectorAll('.custom-toast');
        existingToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `custom-toast alert alert-${type} alert-dismissible fade show`;
        toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1050;
        min-width: 250px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;

        toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

        document.body.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 3000);
    }

    function createFallbackResultDisplay(data) {
        resultsContainer.innerHTML = `
        <div class="card shadow">
            <div class="card-header bg-white">
                <h5 class="mb-0">Analysis Results</h5>
            </div>
            <div class="card-body">
                <div class="alert alert-info">Classification: ${data.result.classification}</div>
                <div class="alert alert-success">Confidence: ${Math.round(data.result.confidence * 100)}%</div>
                <button class="btn btn-primary" data-action="clear-selection">Back to Upload</button>
            </div>
        </div>
    `;
        resultsContainer.style.display = 'block';
    }

    function drawMockHeatmap() {
        const canvas = document.getElementById('heatmapOverlay');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        const opacitySlider = document.getElementById('heatmapOpacity');
        const displayOpacity = opacitySlider ? opacitySlider.value / 100 : 0.5;

        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2.5;

        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, radius
        );

        gradient.addColorStop(0, `rgba(255, 0, 0, ${0.8 * displayOpacity})`);
        gradient.addColorStop(0.3, `rgba(255, 100, 0, ${0.6 * displayOpacity})`);
        gradient.addColorStop(0.6, `rgba(255, 200, 0, ${0.4 * displayOpacity})`);
        gradient.addColorStop(1, `rgba(255, 255, 0, 0)`);

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        drawMedicalPatterns(ctx, width, height, displayOpacity);
    }

    function drawMedicalPatterns(ctx, width, height, opacity, patternIntensity = 1.0) {
        const patterns = [
            { x: width * 0.3, y: height * 0.4, radius: width * 0.08, intensity: 0.7 },
            { x: width * 0.7, y: height * 0.3, radius: width * 0.06, intensity: 0.6 },
            { x: width * 0.4, y: height * 0.7, radius: width * 0.1, intensity: 0.8 },
            { x: width * 0.6, y: height * 0.6, radius: width * 0.07, intensity: 0.5 },
            { x: width * 0.2, y: height * 0.2, radius: width * 0.05, intensity: 0.4 },
            { x: width * 0.8, y: height * 0.8, radius: width * 0.09, intensity: 0.6 }
        ];

        patterns.forEach(pattern => {
            const gradient = ctx.createRadialGradient(
                pattern.x, pattern.y, 0,
                pattern.x, pattern.y, pattern.radius
            );

            const finalOpacity = pattern.intensity * opacity * patternIntensity;
            gradient.addColorStop(0, `rgba(255, 50, 50, ${finalOpacity})`);
            gradient.addColorStop(1, `rgba(255, 150, 50, 0)`);

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(pattern.x, pattern.y, pattern.radius, 0, Math.PI * 2);
            ctx.fill();
        });
    }
    function downloadResultWithHeatmap(classification, imgElement, heatmapCanvas, filename = null) {
        if (!imgElement || !heatmapCanvas) {
            alert('Error: Could not find image elements');
            return;
        }

        if (!imgElement.complete) {
            alert('Please wait for the image to finish loading');
            return;
        }

        const combinedCanvas = document.createElement('canvas');
        combinedCanvas.width = imgElement.naturalWidth;
        combinedCanvas.height = imgElement.naturalHeight;
        const ctx = combinedCanvas.getContext('2d');

        ctx.drawImage(imgElement, 0, 0);

        const currentOpacity = parseFloat(heatmapCanvas.style.opacity || 0.5);
        ctx.globalAlpha = currentOpacity;
        ctx.drawImage(heatmapCanvas, 0, 0);
        ctx.globalAlpha = 1.0;

        try {
            const dataURL = combinedCanvas.toDataURL('image/jpeg', 0.9);
            const finalFilename = filename || `result_${classification.replace(/\s+/g, '_')}.jpg`;

            const link = document.createElement('a');
            link.download = finalFilename;
            link.href = dataURL;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

        } catch (error) {
            console.error('Download error:', error);
            alert('Error creating download. Please try again.');
        }
    }

    function saveSessionToLocalStorage() {
        const sessionData = {
            history: processingHistory,
            currentModel: currentModel,
            timestamp: new Date()
        };

        localStorage.setItem('mlXaiSession', JSON.stringify(sessionData));
    }

    function loadSessionFromLocalStorage() {
        const savedSession = localStorage.getItem('mlXaiSession');
        if (savedSession) {
            try {
                const sessionData = JSON.parse(savedSession);
                processingHistory = sessionData.history || [];
                currentModel = sessionData.currentModel || 'model_v1';

                if (modelSelect) {
                    modelSelect.value = currentModel;
                }

                console.log('Session loaded from local storage');
            } catch (e) {
                console.error('Error loading session from local storage:', e);
            }
        }
    }
    function addKeyboardControls() {
        document.addEventListener('keydown', function (e) {
            const opacitySlider = document.getElementById('heatmapOpacity');
            if (!opacitySlider) return;

            if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {

                opacitySlider.value = Math.min(100, parseInt(opacitySlider.value) + 10);
                opacitySlider.dispatchEvent(new Event('input'));
                e.preventDefault();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {

                opacitySlider.value = Math.max(0, parseInt(opacitySlider.value) - 10);
                opacitySlider.dispatchEvent(new Event('input'));
                e.preventDefault();
            }
        });
    }
    function processBatch() {
        if (batchFiles.length === 0) return;

        isBatchProcessing = true;
        currentBatchIndex = 0;
        batchResults = [];

        const progressContainer = document.getElementById('batchProgressContainer');
        const progressText = document.getElementById('batchProgressText');
        const progressPercent = document.getElementById('batchProgressPercent');
        const progressBar = document.getElementById('batchProgressBar');

        if (progressContainer) {
            progressContainer.style.display = 'block';
            console.log('Progress container shown');
        }

        if (progressText) {
            progressText.textContent = `Processing: 0/${batchFiles.length}`;
        }

        if (progressPercent) {
            progressPercent.textContent = '0%';
        }

        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
            progressBar.classList.add('bg-primary');
        }
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Processing Batch';
        clearBatchResultsUI();
        processNextBatchFile();
    }

    function processNextBatchFile() {
        if (currentBatchIndex >= batchFiles.length) {
            finishBatchProcessing();
            return;
        }

        const file = batchFiles[currentBatchIndex];
        const reader = new FileReader();

        reader.onload = function (e) {
            currentImage = {
                name: file.name,
                data: e.target.result,
                type: file.type
            };

            updateBatchProgress();


            const formData = new FormData();
            formData.append('model_id', currentModel);

            sendBatchProcessingRequest(formData, file.name);
        };

        reader.onerror = function () {
            console.error('Error reading batch file:', file.name);
            currentBatchIndex++;
            processNextBatchFile();
        };

        reader.readAsDataURL(file);
    }

    function sendBatchProcessingRequest(formData, fileName) {
        fetch('/process/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': getCookie('csrftoken'),
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    batchResults.push({
                        fileName: fileName,
                        result: data.result,
                        timestamp: new Date()
                    });
                } else {
                    console.error('Error processing batch file:', fileName, data.error);
                }

                currentBatchIndex++;
                processNextBatchFile();
            })
            .catch(error => {
                console.error('Error processing batch file:', fileName, error);
                currentBatchIndex++;
                processNextBatchFile();
            });
    }

    function updateBatchProgress() {
        const progress = Math.round((currentBatchIndex / batchFiles.length) * 100);
        const progressText = document.getElementById('batchProgressText');
        const progressPercent = document.getElementById('batchProgressPercent');
        const progressBar = document.getElementById('batchProgressBar');

        if (progressText) {
            progressText.textContent = `Processing: ${currentBatchIndex}/${batchFiles.length}`;
        }

        if (progressPercent) {
            progressPercent.textContent = `${progress}%`;
        }

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            if (progress >= 80) {
                progressBar.classList.remove('bg-primary', 'bg-warning');
                progressBar.classList.add('bg-success');
            } else if (progress >= 50) {
                progressBar.classList.remove('bg-primary', 'bg-success');
                progressBar.classList.add('bg-warning');
            }
        }

        console.log(`Batch progress: ${progress}%`);
    }

    function finishBatchProcessing() {
        isBatchProcessing = false;

        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';

        const progressContainer = document.getElementById('batchProgressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }

        showBatchResults();

        console.log('Batch processing complete');
    }
    function showBatchResults() {
        const resultsToggle = document.getElementById('batchResultsToggle');
        const resultsContainer = document.getElementById('batchResults');
        const resultsCollapse = document.getElementById('batchResultsCollapse');

        if (resultsToggle) {
            resultsToggle.style.display = 'block';
            const toggleButton = resultsToggle.querySelector('button');
            if (toggleButton) {
                toggleButton.textContent = `Show Batch Results (${batchResults.length})`;
            }
        }

        if (resultsContainer) {
            let resultsHTML = '<div class="list-group mt-2">';

            if (batchResults.length === 0) {
                resultsHTML += `
                <div class="list-group-item text-center text-muted">
                    <i class="bi bi-exclamation-circle"></i> No results to display
                </div>
            `;
            } else {
                batchResults.forEach((result, index) => {
                    const confidencePercent = Math.round(result.result.confidence * 100);
                    const badgeClass = confidencePercent >= 80 ? 'bg-success' :
                        confidencePercent >= 60 ? 'bg-warning' : 'bg-danger';

                    resultsHTML += `
                    <div class="list-group-item batch-result-item" data-action="show-single-result" data-result-index="${index}" style="cursor: pointer;">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <h6 class="mb-1">${result.fileName}</h6>
                                <p class="mb-1">${result.result.classification}</p>
                                <small class="text-muted">${result.timestamp.toLocaleTimeString()}</small>
                            </div>
                            <div class="ms-3 text-center">
                                <span class="badge ${badgeClass} fs-6">${confidencePercent}%</span>
                            </div>
                        </div>
                    </div>
                `;
                });
            }

            resultsHTML += '</div>';
            resultsContainer.innerHTML = resultsHTML;

            resultsContainer.style.display = 'block';
        }

        if (resultsCollapse && batchResults.length > 0) {
            resultsCollapse.classList.add('show');
        }

        const resultsCount = document.getElementById('batchResultsCount');
        if (resultsCount) {
            resultsCount.textContent = batchResults.length;
        }

        showBatchCompletionMessage();
    }
    function showBatchCompletionMessage() {
        const existingMessage = document.getElementById('batchCompletionMessage');
        if (existingMessage) {
            existingMessage.remove();
        }

        const successCount = batchResults.filter(r => r.result.success !== false).length;
        const totalCount = batchFiles.length;

        const completionMessage = document.createElement('div');
        completionMessage.id = 'batchCompletionMessage';
        completionMessage.className = 'alert alert-info mt-3';

        completionMessage.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <i class="bi bi-check-circle-fill"></i> 
                Batch processing complete: ${successCount}/${totalCount} successful
            </div>
            <button class="btn btn-sm btn-outline-primary" data-action="export-results">
                <i class="bi bi-download"></i> Export Results
            </button>
        </div>
    `;

        const progressContainer = document.getElementById('batchProgressContainer');
        if (progressContainer && progressContainer.parentNode) {
            progressContainer.parentNode.insertBefore(completionMessage, progressContainer.nextSibling);
        }
    }
    function showSingleResultFromBatch(resultIndex) {
        const result = batchResults[resultIndex];
        if (!result) return;

        let singleResultView = document.getElementById('singleResultView');

        if (!singleResultView) {
            singleResultView = document.createElement('div');
            singleResultView.id = 'singleResultView';
            singleResultView.className = 'card mt-4';
            resultsContainer.parentNode.appendChild(singleResultView);
        }

        singleResultView.innerHTML = `
        <div class="card-header bg-white d-flex justify-content-between align-items-center">
            <h5 class="mb-0">${result.fileName}</h5>
            <div>
                <button class="btn btn-sm btn-outline-secondary" data-action="back-to-batch">
                    <i class="bi bi-arrow-left"></i> Back to Batch
                </button>
                <button class="btn btn-sm btn-outline-primary ms-2" data-action="download-single" data-result-index="${resultIndex}">
                    <i class="bi bi-download"></i> Download
                </button>
            </div>
        </div>
        <div class="card-body">
            <!-- Classification Results -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Classification</h6>
                        </div>
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <span>${result.result.classification}</span>
                                <span class="badge bg-success">${Math.round(result.result.confidence * 100)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Image with Heatmap Overlay -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">Explainable AI Heatmap</h6>
                            <div class="d-flex align-items-center">
                                <span class="me-2 small">Opacity:</span>
                                <input type="range" class="form-range" id="singleHeatmapOpacity" min="0" max="100" value="50" style="width: 100px;">
                            </div>
                        </div>
                        <div class="card-body text-center">
                            <div class="position-relative d-inline-block" style="max-width: 100%;">
                                <img id="singleOriginalImage" src="${result.imageData}" class="img-fluid rounded" style="max-height: 400px; width: auto;">
                                <canvas id="singleHeatmapOverlay" class="position-absolute top-0 start-0"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="d-flex gap-2 flex-wrap">
                <button class="btn btn-primary" data-action="download-single" data-result-index="${resultIndex}">
                    <i class="bi bi-download"></i> Download Result
                </button>
                <button class="btn btn-outline-info" data-action="reprocess-single" data-result-index="${resultIndex}">
                    <i class="bi bi-arrow-repeat"></i> Try Different Model
                </button>
            </div>
        </div>
    `;
        const imgElement = document.getElementById('singleOriginalImage');
        if (imgElement) {
            if (imgElement.complete) {
                setupSingleResultCanvas();
                drawSingleResultHeatmap();
            } else {
                imgElement.onload = function () {
                    setupSingleResultCanvas();
                    drawSingleResultHeatmap();
                };
            }
        }

        const opacitySlider = document.getElementById('singleHeatmapOpacity');
        const heatmapOverlay = document.getElementById('singleHeatmapOverlay');

        if (opacitySlider && heatmapOverlay) {
            heatmapOverlay.style.opacity = opacitySlider.value / 100;
            opacitySlider.addEventListener('input', function () {
                heatmapOverlay.style.opacity = this.value / 100;
            });
        }
    }
    function loadSingleResultImage(resultIndex) {
        const result = batchResults[resultIndex];
        if (!result || !result.imageData) return;

        const imgElement = document.getElementById('singleOriginalImage');
        if (imgElement) {
            imgElement.src = result.imageData;

            imgElement.onload = function () {
                setTimeout(() => {
                    setupSingleResultCanvas();
                    drawSingleResultHeatmap();
                }, 100);
            };

            imgElement.onerror = function () {
                console.error('Failed to load image for single result');
            };
        }
    }
    function setupSingleResultCanvas() {
        const imgElement = document.getElementById('singleOriginalImage');
        const canvas = document.getElementById('singleHeatmapOverlay');
        const container = imgElement ? imgElement.parentElement : null;

        if (!imgElement || !canvas || !container) return;

        const imgStyle = window.getComputedStyle(imgElement);
        const imgWidth = parseInt(imgStyle.width);
        const imgHeight = parseInt(imgStyle.height);

        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;

        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';

        console.log('Canvas setup:', {
            naturalSize: { width: imgElement.naturalWidth, height: imgElement.naturalHeight },
            displayedSize: { width: imgWidth, height: imgHeight },
            canvasSize: { width: canvas.width, height: canvas.height }
        });
    }
    function drawSingleResultHeatmap() {
        const canvas = document.getElementById('singleHeatmapOverlay');
        const imgElement = document.getElementById('singleOriginalImage');

        if (!canvas || !imgElement) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const opacitySlider = document.getElementById('singleHeatmapOpacity');
        const displayOpacity = opacitySlider ? opacitySlider.value / 100 : 0.5;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(canvas.width, canvas.height) / 3;

        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, radius
        );

        gradient.addColorStop(0, `rgba(255, 0, 0, ${0.8 * displayOpacity})`);
        gradient.addColorStop(0.3, `rgba(255, 100, 0, ${0.6 * displayOpacity})`);
        gradient.addColorStop(0.6, `rgba(255, 200, 0, ${0.4 * displayOpacity})`);
        gradient.addColorStop(1, `rgba(255, 255, 0, 0)`);

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        drawMedicalPatterns(ctx, canvas.width, canvas.height, displayOpacity);
    }

    function hideSingleResultView() {
        const singleResultView = document.getElementById('singleResultView');
        if (singleResultView) {
            singleResultView.remove();
        }

        const singleImage = document.getElementById('singleOriginalImage');
        const singleHeatmap = document.getElementById('singleHeatmapOverlay');

        if (singleImage && singleImage.src && singleImage.src.startsWith('blob:')) {
            URL.revokeObjectURL(singleImage.src);
        }
        if (singleHeatmap) {
            const ctx = singleHeatmap.getContext('2d');
            ctx.clearRect(0, 0, singleHeatmap.width, singleHeatmap.height);
        }
    }

    function downloadSingleResult(resultIndex = null) {
        if (resultIndex === null) {
            const element = document.querySelector('[data-action="download-single"]');
            if (element) {
                resultIndex = parseInt(element.getAttribute('data-result-index'));
            }
        }

        const result = batchResults[resultIndex];
        if (!result) return;
        const filename = `result_${result.result.classification.replace(/\s+/g, '_')}.jpg`;

        const imgElement = document.getElementById('singleOriginalImage');
        const heatmapCanvas = document.getElementById('singleHeatmapOverlay');

        if (imgElement && heatmapCanvas) {
            downloadResultWithHeatmap(result.result.classification, imgElement, heatmapCanvas);
        }
    }

    function reprocessSingleResult(resultIndex) {
        const result = batchResults[resultIndex];
        if (!result) return;

        currentImage = {
            name: result.fileName,
            data: result.imageData,
            type: 'image/jpeg'
        };

        hideSingleResultView();
        processImage();
    }
    function processNextBatchFile() {
        if (currentBatchIndex >= batchFiles.length) {
            finishBatchProcessing();
            return;
        }

        const file = batchFiles[currentBatchIndex];
        const reader = new FileReader();

        reader.onload = function (e) {
            const imageData = e.target.result;

            currentImage = {
                name: file.name,
                data: imageData,
                type: file.type
            };

            updateBatchProgress();
            const formData = new FormData();
            formData.append('model_id', currentModel);

            sendBatchProcessingRequest(formData, file.name, imageData);
        };

        reader.readAsDataURL(file);
    }

    function sendBatchProcessingRequest(formData, fileName, imageData) {
        fetch('/process/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': getCookie('csrftoken'),
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    batchResults.push({
                        fileName: fileName,
                        result: data.result,
                        imageData: imageData,
                        timestamp: new Date()
                    });
                }

                currentBatchIndex++;
                processNextBatchFile();
            })
            .catch(error => {
                console.error('Error processing batch file:', fileName, error);
                currentBatchIndex++;
                processNextBatchFile();
            });
    }
    function exportBatchResults() {
        if (batchResults.length === 0) {
            alert('No results to export');
            return;
        }
        let csvContent = "File Name,Classification,Confidence,Timestamp\n";

        batchResults.forEach(result => {
            const confidencePercent = Math.round(result.result.confidence * 100);
            const timestamp = result.timestamp.toLocaleString();
            csvContent += `"${result.fileName}","${result.result.classification}",${confidencePercent}%,"${timestamp}"\n`;
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

        link.href = url;
        link.setAttribute('download', `batch_results_${timestamp}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    function handleBatchSelect(e) {
        if (e.target.files.length > 0) {
            batchFiles = Array.from(e.target.files);
            console.log('Batch files selected:', batchFiles.length);

            processBtn.disabled = false;

            updateDropZoneForBatch();
            showBatchInfo();

            currentImage = null;
            fileInput.value = '';

        } else {
            batchFiles = [];
            processBtn.disabled = currentImage ? false : true;
            resetDropZoneToDefault();
        }
    }
    function showBatchInfo() {
        if (batchFiles.length > 0) {
            const batchInfo = `Batch: ${batchFiles.length} file(s) selected`;

            let batchInfoElement = document.getElementById('batchInfo');
            if (!batchInfoElement) {
                batchInfoElement = document.createElement('div');
                batchInfoElement.id = 'batchInfo';
                batchInfoElement.className = 'alert alert-info mt-2';

                const batchInput = document.getElementById('batchInput');
                if (batchInput && batchInput.parentNode) {
                    batchInput.parentNode.insertBefore(batchInfoElement, batchInput.nextSibling);
                }
            }

            batchInfoElement.innerHTML = `
            <i class="bi bi-info-circle"></i> 
            ${batchFiles.length} file(s) selected for batch processing
            <button type="button" class="btn-close float-end" data-bs-dismiss="alert"></button>
        `;
        }
    }

    function updateDropZoneForBatch() {
        if (batchFiles.length === 0) {
            resetDropZoneToDefault();
            return;
        }

        const firstFile = batchFiles[0];
        const reader = new FileReader();

        reader.onload = function (e) {
            dropZone.innerHTML = `
            <img src="${e.target.result}" class="img-fluid rounded mb-2" style="max-height: 150px;">
            <p class="mt-2"><strong>Batch Ready:</strong> ${batchFiles.length} file(s)</p>
            <div class="small text-muted">
                ${batchFiles.slice(0, 3).map(file => file.name).join(', ')}
                ${batchFiles.length > 3 ? `... and ${batchFiles.length - 3} more` : ''}
            </div>
            <button class="btn btn-sm btn-outline-secondary mt-2" data-action="clear-batch">
                Clear Batch
            </button>
            <button class="btn btn-sm btn-outline-info mt-2" data-action="preview-batch">
                <i class="bi bi-images"></i> View All
            </button>
        `;

            dropZone.classList.add('batch-active');
        };

        reader.onerror = function () {
            dropZone.innerHTML = `
            <i class="bi bi-collection display-4 text-primary d-block"></i>
            <p class="mt-2"><strong>Batch Ready:</strong> ${batchFiles.length} file(s)</p>
            <div class="small text-muted">
                ${batchFiles.slice(0, 3).map(file => file.name).join(', ')}
                ${batchFiles.length > 3 ? `... and ${batchFiles.length - 3} more` : ''}
            </div>
            <button class="btn btn-sm btn-outline-secondary mt-2" data-action="clear-batch">
                Clear Batch
            </button>
        `;
        };

        reader.readAsDataURL(firstFile);
    }
    function showBatchPreview() {
        const previewHTML = `
        <div class="modal fade" id="batchPreviewModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Batch Preview (${batchFiles.length} files)</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            ${batchFiles.map((file, index) => `
                                <div class="col-md-4 mb-3">
                                    <div class="card">
                                        <div class="card-body text-center">
                                            <img src="${URL.createObjectURL(file)}" class="img-fluid rounded" style="max-height: 100px;">
                                            <div class="small text-truncate mt-2">${file.name}</div>
                                            <div class="text-muted">${formatFileSize(file.size)}</div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
        const existingModal = document.getElementById('batchPreviewModal');
        if (existingModal) {
            existingModal.remove();
        }

        document.body.insertAdjacentHTML('beforeend', previewHTML);

        const modal = new bootstrap.Modal(document.getElementById('batchPreviewModal'));
        modal.show();
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    function clearBatchSelection() {
        console.log('Clearing batch selection and results');

        batchFiles = [];
        currentBatchIndex = 0;
        batchResults = [];
        isBatchProcessing = false;


        const batchInput = document.getElementById('batchInput');
        if (batchInput) {
            batchInput.value = '';
        }

        processBtn.disabled = currentImage ? false : true;
        processBtn.innerHTML = '<i class="bi bi-gear-fill"></i> Process Image';
        resetDropZoneToDefault();

        clearBatchResultsUI();

        console.log('Batch selection and results cleared');
    }

    function clearBatchResultsUI() {
        const resultsToggle = document.getElementById('batchResultsToggle');
        if (resultsToggle) {
            resultsToggle.style.display = 'none';
        }
        const resultsContainer = document.getElementById('batchResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
        }
        const resultsCollapse = document.getElementById('batchResultsCollapse');
        if (resultsCollapse) {
            resultsCollapse.classList.remove('show');
        }
        const completionMessage = document.getElementById('batchCompletionMessage');
        if (completionMessage) {
            completionMessage.remove();
        }
        const batchInfoElement = document.getElementById('batchInfo');
        if (batchInfoElement) {
            batchInfoElement.remove();
        }
        hideSingleResultView();
    }

    function resetDropZoneToDefault() {
        const dropZone = document.getElementById('dropZone');
        if (!dropZone) return;

        dropZone.innerHTML = `
        <i class="bi bi-cloud-upload display-4 text-muted d-block"></i>
        <p class="mt-2">Drag & drop your image here or click to browse</p>
        <button class="btn btn-primary mt-2" data-action="browse">
            Select Image
        </button>
    `;

        dropZone.classList.remove('batch-active');
    }
    loadSessionFromLocalStorage();

    window.clearSelection = clearSelection;
});