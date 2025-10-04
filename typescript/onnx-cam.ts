// web/onnx-cam.ts
import { ONNXModelService } from './onnx-model-service';
import { CAMConfig, HeatmapResult, InferenceMetrics } from './interfaces';
import { MaskWorkerManager } from './mask-worker-manager';
import { PerformanceMetrics } from './performance-metrics';

// onnx-cam.ts
export class ONNXCAM {
    private config: CAMConfig;
    private maskWorkerManager: MaskWorkerManager;
    private performanceMetrics: PerformanceMetrics;


    constructor(private modelService: ONNXModelService, config?: Partial<CAMConfig>) {
        this.config = {
            targetLayers: { ['features']: { sideLen: 224, channels: 3 } },
            classificationLayer: 'classification',
            batchSize: 8,
            inputSize: 224,
            ...config
        };

        for (const [layerName, dimensions] of Object.entries(this.config.targetLayers)) {
            if (!dimensions.sideLen || !dimensions.channels) {
                console.warn(`Invalid dimensions for layer ${layerName}:`, dimensions);
                // Set fallback values
                this.config.targetLayers[layerName] = {
                    sideLen: dimensions.sideLen || 32,
                    channels: dimensions.channels || 64
                };
            }
        }
        this.maskWorkerManager = new MaskWorkerManager();
        this.performanceMetrics = new PerformanceMetrics();

    }


    updateConfig(newConfig: Partial<CAMConfig>): void {
        this.config = { ...this.config, ...newConfig };
        //this.camSettings.visibleLayers = new Set<string>(Object.keys(this.config.targetLayers));
        console.log('CAM config updated:', this.config);
        console.log(' classify layer ', this.config.classificationLayer);
    }
    async computeCAMs(
        classificationResult: Awaited<ReturnType<typeof this.computeClassification>>,
        classIndexOverride?: number,
        specificFeatureLayer?: string,
        activation?: number,
        onProgress?: (progress: number, currentLayer: string, totalLayers: number, heatmaps?: any) => void
    ): Promise<{
        heatmaps: {
            [featureLayer: string]: { [outputLayer: string]: HeatmapResult }
        },
        metrics?: InferenceMetrics;
    }> {
        this.performanceMetrics.startMeasurement();
        console.log(' Starting CAM computation...');

        const { originalOutput, predictions, predictedClassIndex } = classificationResult;
        const classificationLayers = Object.keys(predictions);
        const featureLayers = Object.keys(this.config.targetLayers);
        const totalLayers = featureLayers.length;

        const allHeatmaps: {
            [featureLayer: string]: { [outputLayer: string]: HeatmapResult }
        } = {};

        for (let layerIndex = 0; layerIndex < featureLayers.length; layerIndex++) {
            const featureLayerName = featureLayers[layerIndex];
            const dimensions = this.config.targetLayers[featureLayerName];
            if (specificFeatureLayer && featureLayerName !== specificFeatureLayer) continue;

            // Update progress for this layer
            onProgress?.(layerIndex / totalLayers, featureLayerName, totalLayers);

            const featureMapData = originalOutput[featureLayerName]?.data as Float32Array;
            if (!featureMapData) continue;

            const heatmapsForFeatureLayer = await this.generateHeatmaps(
                featureMapData,
                classificationResult.originalImageU8,
                classificationResult.preprocessingParams,
                Object.fromEntries(
                    classificationLayers.map(layerName => [
                        layerName,
                        predictions[layerName][
                        classIndexOverride ?? predictedClassIndex[layerName]
                        ]
                    ])
                ),
                featureLayerName,
                dimensions.sideLen,
                dimensions.channels,
                classificationLayers,
                Object.fromEntries(
                    classificationLayers.map(layerName => [
                        layerName,
                        classIndexOverride ?? predictedClassIndex[layerName]
                    ])
                ),
                this.config.batchSize,
                activation,
                (progress: number, stage: string, heatmapData?: Record<string, Float32Array>) => {
                    const layerProgress = (layerIndex + progress) / totalLayers;
                    const processedHeatmaps = heatmapData ? { [featureLayerName]: heatmapData } : undefined
                    onProgress?.(layerProgress, `${featureLayerName}: ${stage}`,
                        totalLayers, processedHeatmaps
                    );
                }
            );

            allHeatmaps[featureLayerName] = {};
            for (const outputLayerName of classificationLayers) {
                allHeatmaps[featureLayerName][outputLayerName] = {
                    data: heatmapsForFeatureLayer[outputLayerName],
                    sideLen: this.config.inputSize,
                    channels: dimensions.channels
                };

                onProgress?.((layerIndex + 1) / totalLayers, `${featureLayerName}: ${outputLayerName} complete`, totalLayers, allHeatmaps);
            }
        }

        const metrics = this.performanceMetrics.endMeasurement();
        this.performanceMetrics.logMetrics('CAM Computation');
        return { heatmaps: allHeatmaps, metrics: metrics };
    }
    /*  async computeCAMs(
         classificationResult: Awaited<ReturnType<typeof this.computeClassification>>,
         classIndexOverride?: number,
         specificFeatureLayer?: string,
         activation?: number,
     ): Promise<{
         heatmaps: {
             [featureLayer: string]: { [outputLayer: string]: HeatmapResult }
         },
         metrics?: InferenceMetrics;
     }> {
         this.performanceMetrics.startMeasurement();
         console.log(' Starting CAM computation...');
 
         const { originalOutput, predictions, predictedClassIndex } = classificationResult;
 
         const classificationLayers = Object.keys(predictions);
 
         const allHeatmaps: {
             [featureLayer: string]: { [outputLayer: string]: HeatmapResult }
         } = {};
 
         for (const [featureLayerName, dimensions] of Object.entries(this.config.targetLayers)) {
             if (specificFeatureLayer && featureLayerName !== specificFeatureLayer) continue;
 
             const featureMapData = originalOutput[featureLayerName]?.data as Float32Array;
             if (!featureMapData) continue;
 
             const heatmapsForFeatureLayer = await this.generateHeatmaps(
                 featureMapData,
                 classificationResult.originalImageU8,
                 classificationResult.preprocessingParams,
                 Object.fromEntries(
                     classificationLayers.map(layerName => [
                         layerName,
                         predictions[layerName][
                         classIndexOverride ?? predictedClassIndex[layerName]
                         ]
                     ])
                 ),
                 featureLayerName,
                 dimensions.sideLen,
                 dimensions.channels,
                 classificationLayers,
                 Object.fromEntries(
                     classificationLayers.map(layerName => [
                         layerName,
                         classIndexOverride ?? predictedClassIndex[layerName]
                     ])
                 ),
                 this.config.batchSize,
                 activation,
                 (progress: number, stage: string) => {
                     // Example: stage = "inference" | "mask" | "aggregation"
                     this.updateProgressUI(progress, stage);
                 }
             );
             allHeatmaps[featureLayerName] = {};
             for (const outputLayerName of classificationLayers) {
                 allHeatmaps[featureLayerName][outputLayerName] = {
                     data: heatmapsForFeatureLayer[outputLayerName],
                     sideLen: this.config.inputSize,
                     channels: dimensions.channels
                 };
             }
             this.updateProgressUI(1, "Generating heatmaps...");
         }
         const metrics = this.performanceMetrics.endMeasurement();
         this.performanceMetrics.logMetrics('CAM Computation');
         return { heatmaps: allHeatmaps, metrics: metrics };
     }
  */
    //deprecated
    async computeCAM(
        imageElement: HTMLImageElement,
        classIndex?: number,
        specificFeatureLayer?: string,
        targetClassificationOutputLayers: string[] = [],
        activation?: number
    ): Promise<{
        predictions: { [outputLayerName: string]: number[] };
        predictedClassIndex: { [outputLayerName: string]: number };
        confidence: { [outputLayerName: string]: number };
        heatmaps: {
            [featureLayerName: string]: {
                [outputLayerName: string]: HeatmapResult;
            }
        }; metrics?: InferenceMetrics;
    }> {
        try {
            this.performanceMetrics.startMeasurement();
            console.log(' Starting CAM computation...');

            console.log('Target layers:', this.config.targetLayers);
            for (const [layerName, dimensions] of Object.entries(this.config.targetLayers)) {
                console.log(`Layer ${layerName}: sideLen=${dimensions.sideLen}, channels=${dimensions.channels}`);
            }
            // metadata
            const originalImageData = this.extractImageDataAtInputSize(imageElement);
            const preprocessedResult = await this.preprocessImage(imageElement);
            const preprocessedData = preprocessedResult.data;
            const { mean, std, scale } = preprocessedResult;
            console.log(' Image preprocessed, data length:', preprocessedData.length);


            // First inference
            const originalOutput = await this.modelService.runInference(
                preprocessedData,
                [1, 3, this.config.inputSize, this.config.inputSize]
            );

            if (!originalOutput) {
                throw new Error('Model inference returned undefined output');
            }

            const allPredictions: { [outputLayerName: string]: number[] } = {};
            const allPredictedClassIndices: { [outputLayerName: string]: number } = {};
            const allConfidences: { [outputLayerName: string]: number } = {};
            const originalTargetScores: { [outputLayerName: string]: number } = {};

            const classificationLayersToProcess = targetClassificationOutputLayers.length > 0
                ? targetClassificationOutputLayers
                : [this.config.classificationLayer];

            console.log(`Classification layers to process: ${classificationLayersToProcess}`);
            for (const outputLayerName of classificationLayersToProcess) {
                if (!originalOutput[outputLayerName]) {
                    console.warn(`Output layer ${outputLayerName} not found in inference results.`);
                    continue;
                }
                const currentPredictions = Array.from(originalOutput[outputLayerName].data as Float32Array);
                allPredictions[outputLayerName] = currentPredictions;

                let currentPredictedClassIndex = 0;
                let currentMaxConfidence = 0;

                for (let i = 0; i < currentPredictions.length; i++) {
                    if (currentPredictions[i] > currentMaxConfidence) {
                        currentMaxConfidence = currentPredictions[i];
                        currentPredictedClassIndex = i;
                    }
                }
                allPredictedClassIndices[outputLayerName] = classIndex !== undefined ? classIndex : currentPredictedClassIndex;
                allConfidences[outputLayerName] = currentMaxConfidence;
                console.log(`For output layer ${outputLayerName}: Predicted class: ${allPredictedClassIndices[outputLayerName]}, confidence: ${allConfidences[outputLayerName]}`);

                const targetIdx = allPredictedClassIndices[outputLayerName];
                originalTargetScores[outputLayerName] = currentPredictions[targetIdx];
            }

            const primaryTargetClassForHeatmap = classIndex !== undefined
                ? classIndex
                : (classificationLayersToProcess.length > 0 && allPredictedClassIndices[classificationLayersToProcess[0]] !== undefined
                    ? allPredictedClassIndices[classificationLayersToProcess[0]]
                    : 0);

            if (primaryTargetClassForHeatmap === undefined) {
                throw new Error("Could not determine primary target class for heatmap generation.");
            }

            const allHeatmaps: {
                [featureLayerName: string]: {
                    [outputLayerName: string]: HeatmapResult;
                }
            } = {};

            for (const [featureLayerName, dimensions] of Object.entries(this.config.targetLayers)) {
                if (specificFeatureLayer && featureLayerName !== specificFeatureLayer) {
                    continue;
                }
                const featureMapSideLength = dimensions.sideLen;
                const featureMapChannels = dimensions.channels;

                if (!originalOutput[featureLayerName]) {
                    console.warn(`Feature layer ${featureLayerName} not found in inference results.`);
                    continue;
                }
                const currentFeatureLayerData = originalOutput[featureLayerName].data as Float32Array;
                const heatmapsForFeatureLayer = await this.generateHeatmaps(
                    currentFeatureLayerData,
                    originalImageData,
                    { mean, std, scale },
                    originalTargetScores,
                    featureLayerName,
                    featureMapSideLength,
                    featureMapChannels,

                    classificationLayersToProcess,
                    Object.fromEntries(
                        classificationLayersToProcess.map(layerName =>
                            [layerName, classIndex !== undefined ? classIndex : allPredictedClassIndices[layerName]]
                        )
                    ),
                    this.config.batchSize,
                    activation,
                    (progress, stage) => {
                        this.updateProgressUI(progress, stage);
                    }
                );

                allHeatmaps[featureLayerName] = {};
                for (const outputLayerName of Object.keys(heatmapsForFeatureLayer)) {
                    const heatmapData = heatmapsForFeatureLayer[outputLayerName];
                    if (heatmapData && heatmapData.length > 0) {
                        allHeatmaps[featureLayerName][outputLayerName] = {
                            data: heatmapsForFeatureLayer[outputLayerName],
                            sideLen: this.config.inputSize,
                            channels: featureMapChannels
                        };
                    } else {
                        console.warn(`Empty heatmap for feature layer ${featureLayerName}, output layer ${outputLayerName}`);
                        allHeatmaps[featureLayerName][outputLayerName] = {
                            data: new Float32Array(this.config.inputSize * this.config.inputSize).fill(0), // Initialize with zeros
                            sideLen: this.config.inputSize,
                            channels: featureMapChannels
                        };
                    }
                }
            }

            const metrics = this.performanceMetrics.endMeasurement();
            this.performanceMetrics.logMetrics('CAM Computation');

            return {
                predictions: allPredictions,
                predictedClassIndex: allPredictedClassIndices,
                confidence: allConfidences,
                heatmaps: allHeatmaps,
                metrics: metrics
            };

        } catch (error) {
            console.error(' CAM computation failed:', error);
            throw error;
        }
    }
    private normalizeWeights(weights: number[]): number[] {
        const reluWeights = weights.map(w => Math.max(0, w));
        const maxVal = Math.max(...reluWeights);
        const minVal = Math.min(...reluWeights);

        if (!isFinite(maxVal) || reluWeights.length === 0) return [];

        if (maxVal === minVal) {
            return reluWeights.map(() => 1 / (reluWeights.length || 1));
        }
        return reluWeights.map(w => (w - minVal) / (maxVal - minVal));
    }
    private computeWeightedHeatmap(
        featureMapsData: Float32Array,
        normalizedWeights: number[],
        importantChannels: number[],
        channels: number,
        spatialArea: number,
        spatialSideLength: number
    ): Float32Array {
        let finalHeatmap = new Float32Array(this.config.inputSize * this.config.inputSize).fill(0);

        for (let i = 0; i < importantChannels.length; i++) {
            const channelIndex = importantChannels[i];
            const weight = normalizedWeights[i] ?? 0; // fallback to 0

            const startIdx = channelIndex * spatialArea;
            const endIdx = startIdx + spatialArea;
            const singleChannelData = featureMapsData.subarray(startIdx, endIdx);
            const upscaledChannel = this.upscaleBilinear(singleChannelData, spatialSideLength);
            for (let j = 0; j < upscaledChannel.length; j++) {
                finalHeatmap[j] += upscaledChannel[j] * weight;
            }
        }
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < finalHeatmap.length; i++) {
            if (finalHeatmap[i] < minVal) minVal = finalHeatmap[i];
            if (finalHeatmap[i] > maxVal) maxVal = finalHeatmap[i];
        }
        const range = maxVal - minVal;
        if (range > 0) {
            for (let i = 0; i < finalHeatmap.length; i++) {
                finalHeatmap[i] = (finalHeatmap[i] - minVal) / range;
            }
        } else {
            for (let i = 0; i < finalHeatmap.length; i++) {
                finalHeatmap[i] = 0;
            }
        }

        return finalHeatmap;
    }
    async computeClassification(
        imageElement: HTMLImageElement,
        targetLayers: string[] = []
    ): Promise<{
        predictions: { [layer: string]: number[] };
        predictedClassIndex: { [layer: string]: number };
        confidence: { [layer: string]: number };
        originalOutput: any;
        origWidth: number;
        origHeight: number;
        originalImageU8: Uint8Array;
        preprocessingParams: { mean: number[], std: number[], scale: number };
    }> {
        const origWidth = imageElement.naturalWidth;
        const origHeight = imageElement.naturalHeight;

        const originalImageU8 = this.extractImageDataAtInputSize(imageElement);

        const preprocessedResult = await this.preprocessImage(imageElement);
        const preprocessedData = preprocessedResult.data;

        const originalOutput = await this.modelService.runInference(
            preprocessedData,
            [1, 3, this.config.inputSize, this.config.inputSize]
        );

        if (!originalOutput) {
            throw new Error('Model inference returned undefined output');
        }

        const predictions: { [layer: string]: number[] } = {};
        const predictedClassIndex: { [layer: string]: number } = {};
        const confidence: { [layer: string]: number } = {};

        const layersToProcess = targetLayers.length > 0
            ? targetLayers
            : [this.config.classificationLayer];

        for (const outputLayerName of layersToProcess) {
            const scores = Array.from(originalOutput[outputLayerName].data as Float32Array);
            predictions[outputLayerName] = scores;

            let bestIdx = 0, bestScore = -Infinity;
            scores.forEach((s, i) => {
                if (s > bestScore) {
                    bestScore = s;
                    bestIdx = i;
                }
            });

            predictedClassIndex[outputLayerName] = bestIdx;
            confidence[outputLayerName] = bestScore;
        }

        return {
            predictions,
            predictedClassIndex,
            confidence,
            originalOutput,
            origWidth,
            origHeight,
            originalImageU8: originalImageU8,
            preprocessingParams: preprocessedResult
        };
    }

    private upscaleBilinear(data: Float32Array, originalSideLength: number): Float32Array {
        const targetSideLength = this.config.inputSize;
        const upscaled = new Float32Array(targetSideLength * targetSideLength);

        if (originalSideLength === targetSideLength) {
            return data.subarray();
        }

        const scaleX = (originalSideLength - 1) / (targetSideLength - 1);
        const scaleY = (originalSideLength - 1) / (targetSideLength - 1);

        for (let y = 0; y < targetSideLength; y++) {
            for (let x = 0; x < targetSideLength; x++) {
                const srcX = x * scaleX;
                const srcY = y * scaleY;

                const x1 = Math.floor(srcX);
                const y1 = Math.floor(srcY);
                const x2 = Math.min(x1 + 1, originalSideLength - 1);
                const y2 = Math.min(y1 + 1, originalSideLength - 1);

                const fx = srcX - x1;
                const fy = srcY - y1;

                const val11 = data[y1 * originalSideLength + x1];
                const val12 = data[y1 * originalSideLength + x2];
                const val21 = data[y2 * originalSideLength + x1];
                const val22 = data[y2 * originalSideLength + x2];

                const interpolatedVal =
                    val11 * (1 - fx) * (1 - fy) +
                    val12 * fx * (1 - fy) +
                    val21 * (1 - fx) * fy +
                    val22 * fx * fy;

                upscaled[y * targetSideLength + x] = interpolatedVal;
            }
        }
        return upscaled;
    }
    private upscaleToInputSize(data: Float32Array, originalSideLength: number): Float32Array {
        const targetSideLength = this.config.inputSize;

        if (originalSideLength === targetSideLength) {
            return data.slice();
        }
        // large upscale
        if (targetSideLength / originalSideLength > 4) {
            return this.upscaleNearestNeighbor(data, originalSideLength, targetSideLength);
        }
        // small upscale
        return this.upscaleBilinear(data, originalSideLength);
    }

    private upscaleNearestNeighbor(data: Float32Array, originalSideLength: number, targetSideLength: number): Float32Array {
        const upscaled = new Float32Array(targetSideLength * targetSideLength);
        const scale = originalSideLength / targetSideLength;

        for (let y = 0; y < targetSideLength; y++) {
            for (let x = 0; x < targetSideLength; x++) {
                const srcX = Math.floor(x * scale);
                const srcY = Math.floor(y * scale);
                const srcIndex = srcY * originalSideLength + srcX;
                upscaled[y * targetSideLength + x] = data[srcIndex];
            }
        }
        return upscaled;
    }
    private async preprocessImage(imageElement: HTMLImageElement): Promise<{
        data: Float32Array;
        mean: number[];
        std: number[];
        scale: number;
    }> {
        const canvas = this.letterboxImage(imageElement, this.config.inputSize);
        canvas.width = this.config.inputSize;
        canvas.height = this.config.inputSize;

        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(imageElement, 0, 0, this.config.inputSize, this.config.inputSize);

        const imageData = ctx.getImageData(0, 0, this.config.inputSize, this.config.inputSize);
        const data = new Float32Array(3 * this.config.inputSize * this.config.inputSize);

        const mean = [0.485, 0.456, 0.406]; // ImageNet mean
        const std = [0.229, 0.224, 0.225];  // ImageNet std
        const scale = 1 / 255.0;              // Scale factor

        for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
            data[j] = (imageData.data[i] * scale - mean[0]) / std[0];     // R
            data[j + 1] = (imageData.data[i + 1] * scale - mean[1]) / std[1]; // G
            data[j + 2] = (imageData.data[i + 2] * scale - mean[2]) / std[2]; // B
        }

        return { data, mean, std, scale };
    }
    letterboxImage(img: HTMLImageElement, targetSize: number) {
        const ratio = Math.min(targetSize / img.width, targetSize / img.height);
        const newWidth = Math.round(img.width * ratio);
        const newHeight = Math.round(img.height * ratio);

        const canvas = document.createElement('canvas');
        canvas.width = targetSize;
        canvas.height = targetSize;
        const ctx = canvas.getContext('2d')!;

        // fill padding with black
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, targetSize, targetSize);

        const dx = Math.floor((targetSize - newWidth) / 2);
        const dy = Math.floor((targetSize - newHeight) / 2);

        ctx.drawImage(img, 0, 0, img.width, img.height, dx, dy, newWidth, newHeight);
        return canvas;
    }

    private extractImageDataAtInputSize(imageElement: HTMLImageElement): Uint8Array {
        const canvas = document.createElement('canvas');
        canvas.width = this.config.inputSize;
        canvas.height = this.config.inputSize;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(imageElement, 0, 0, this.config.inputSize, this.config.inputSize);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = new Uint8Array(imageData.data.length);
        for (let i = 0; i < imageData.data.length; i++) data[i] = imageData.data[i];
        return data; // RGBA, 0..255, length = inputSize*inputSize*4
    }

    private debugApplyMask(originalImageData: Float32Array, mask: Float32Array, size: number): Float32Array {
        const pixels = size * size;
        const maskedImage = new Float32Array(originalImageData.length);

        if (mask.length !== pixels) {
            console.warn('  Mask size mismatch in debug');

            const originalSide = Math.sqrt(mask.length);
            if (!Number.isInteger(originalSide)) {
                console.error('Cannot resize non-square mask, using fallback');
                return originalImageData.slice();
            }

            const resizedMask = new Float32Array(pixels);
            const scale = originalSide / size;

            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const srcX = Math.min(Math.floor(x * scale), originalSide - 1);
                    const srcY = Math.min(Math.floor(y * scale), originalSide - 1);
                    resizedMask[y * size + x] = mask[srcY * originalSide + srcX];
                }
            }
            for (let i = 0; i < pixels; i++) {
                const maskValue = Math.max(0, Math.min(1, resizedMask[i]));
                const rgbaIndex = i * 4;

                maskedImage[rgbaIndex] = originalImageData[rgbaIndex] * maskValue;
                maskedImage[rgbaIndex + 1] = originalImageData[rgbaIndex + 1] * maskValue;
                maskedImage[rgbaIndex + 2] = originalImageData[rgbaIndex + 2] * maskValue;
                maskedImage[rgbaIndex + 3] = originalImageData[rgbaIndex + 3];
            }

            return maskedImage;
        }
        for (let i = 0; i < pixels; i++) {
            const maskValue = Math.max(0, Math.min(1, mask[i]));
            const rgbaIndex = i * 4;

            maskedImage[rgbaIndex] = originalImageData[rgbaIndex] * maskValue;
            maskedImage[rgbaIndex + 1] = originalImageData[rgbaIndex + 1] * maskValue;
            maskedImage[rgbaIndex + 2] = originalImageData[rgbaIndex + 2] * maskValue;
            maskedImage[rgbaIndex + 3] = originalImageData[rgbaIndex + 3];
        }

        return maskedImage;
    }
    public async quickDebug(): Promise<void> {
        console.log(' Quick preprocessing check...');

        const size = 224;
        const pixels = size * size; // 50176
        const expectedRgba = pixels * 4; // 200704
        const expectedRgb = pixels * 3; // 150528

        console.log('Using size:', size, 'pixels:', pixels);
        console.log('Expected RGBA elements:', expectedRgba);
        console.log('Expected RGB elements:', expectedRgb);

        const testImage = new Float32Array(expectedRgba);

        for (let i = 0; i < pixels; i++) {
            const rgbaIndex = i * 4;
            const x = i % size;
            const y = Math.floor(i / size);

            testImage[rgbaIndex] = (x / size) * 255;
            testImage[rgbaIndex + 1] = (y / size) * 128;
            testImage[rgbaIndex + 2] = 64;
            testImage[rgbaIndex + 3] = 255;
        }

        console.log('Test image created:', {
            length: testImage.length,
            expected: expectedRgba,
            first10: Array.from(testImage.slice(0, 10))
        });

        try {
            const processed = this.preprocessImageData(testImage);

            // Test with model
            console.log('Testing with model...');
            const output = await this.modelService.runInference(
                processed,
                [1, 3, size, size]
            );

            console.log(' Model test successful!');
            console.log('Output keys:', Object.keys(output));

            // output layers
            for (const key of Object.keys(output)) {
                const tensor = output[key];
                if (tensor instanceof Float32Array) {
                    console.log(`Output "${key}": ${tensor.length} values`);
                    console.log(`  Range: ${Math.min(...tensor).toFixed(4)} to ${Math.max(...tensor).toFixed(4)}`);

                    if (key === 'classification' && tensor.length > 10) {
                        const scores = Array.from(tensor)
                            .map((score, index) => ({ index, score }))
                            .sort((a, b) => b.score - a.score)
                            .slice(0, 5);

                        console.log('  Top 5 classes:');
                        scores.forEach(item => {
                            console.log(`    Class ${item.index}: ${item.score.toFixed(6)}`);
                        });
                    }
                }
            }

        } catch (error) {
            console.error(' Preprocessing failed:', error);
            throw error;
        }
    }
    public debugConfig(): void {
        console.log(' Current configuration:');
        console.log('  Input size:', this.config.inputSize);
        console.log('  heatmap layers:', this.config.targetLayers);
        console.log('  Classification layer:', this.config.classificationLayer);
        console.log('  Batch size:', this.config.batchSize);

        if (this.config.inputSize <= 0) {
            console.error(' Invalid input size:', this.config.inputSize);
        }
        const pixels = this.config.inputSize * this.config.inputSize;
        console.log('  Expected pixels:', pixels);
        console.log('  Expected RGBA elements:', pixels * 4);
        console.log('  Expected RGB elements (processed):', pixels * 3);
    }
    private updateProgressUI(progress: number, stage: string) {
        const bar = document.getElementById("progress-bar") as HTMLProgressElement;
        const text = document.getElementById("progress-text")!;
        document.getElementById("progress-container")!.style.display = "block";
        bar.value = progress * 100;
        text.textContent = `${stage}: ${(progress * 100).toFixed(1)}%`;

        if (progress >= 1) {
            setTimeout(() => {
                document.getElementById("progress-container")!.style.display = "none";
            }, 500);
        }
    }

    private async generateHeatmaps(
        featureMapsData: Float32Array,
        originalImageData: Uint8Array,
        preprocessingParams: { mean: number[]; std: number[]; scale: number },
        originalTargetScores: { [outputLayerName: string]: number },
        layerName: string,
        spatialSideLength: number,
        channels: number,
        outputLayerNames: string[],
        targetClassesForOutputLayers: { [outputLayerName: string]: number },
        batchSize: number = 32,
        activation: number = 0.95,
        onProgress?: (progress: number, stage: string, heatmapData?: Record<string, Float32Array>) => void
    ): Promise<Record<string, Float32Array>> {

        onProgress?.(0.1, 'Finding important channels');
        this.performanceMetrics.startMaskGeneration();

        const importantChannels = this.getMostImportantChannelsWithActivations(
            featureMapsData,
            channels,
            spatialSideLength,
            activation
        );

        const classificationLayer = 'classification';
        const targetClass = targetClassesForOutputLayers[classificationLayer];
        const originalScore = originalTargetScores[classificationLayer];

        console.log(` Score-CAM for layer ${layerName}:`);
        console.log(` Target class: ${targetClass}, Original score: ${originalScore.toFixed(6)}`);

        const importantChannelIndices = importantChannels.map(item => item.index);

        onProgress?.(0.2, 'Generating masks');
        const maskResults = await this.maskWorkerManager.generateMasks(
            featureMapsData,
            importantChannelIndices,
            channels,
            spatialSideLength,
            this.config.inputSize
        );
        this.performanceMetrics.endMaskGeneration();

        maskResults.sort((a, b) => a.channelIndex - b.channelIndex);

        const channelImportanceScores: Record<string, number[]> = {};
        outputLayerNames.forEach(name => channelImportanceScores[name] = []);

        this.performanceMetrics.startBatch();

        const H = this.config.inputSize;
        const C = 3;
        const planeSize = H * H;
        const singleImageSize = C * planeSize;
        const B = Math.min(batchSize, maskResults.length);
        for (let bStart = 0; bStart < maskResults.length; bStart += B) {
            const bEnd = Math.min(bStart + B, maskResults.length);
            const actualB = bEnd - bStart;
            const batchBuffer = new Float32Array(actualB * singleImageSize);
            for (let bi = 0; bi < actualB; bi++) {
                const mr = maskResults[bStart + bi];

                this.writePreprocessedMaskedImageToBuffer(
                    batchBuffer,
                    bi * singleImageSize,
                    originalImageData,
                    mr.mask as Uint8Array,
                    preprocessingParams
                );
            }
            const shape = [actualB, C, H, H];
            const outputs = await this.modelService.runInference(batchBuffer, shape);
            this.performanceMetrics.recordInference();
            // parse outputs -> compute reluDiff batch
            for (const targetOutputLayerName of Object.keys(targetClassesForOutputLayers)) {
                //const targetClass = targetClassesForOutputLayers[outputLayerName];
                const originalScore = originalTargetScores[targetOutputLayerName];

                const tensorOutput = outputs?.[targetOutputLayerName];
                let classScores: Float32Array;

                if (tensorOutput && typeof tensorOutput === 'object' && 'cpuData' in tensorOutput) {
                    if (tensorOutput.cpuData instanceof Float32Array) {
                        classScores = tensorOutput.cpuData;
                    } else {
                        classScores = new Float32Array();
                    }
                } else if (tensorOutput instanceof Float32Array) {
                    classScores = tensorOutput;
                } else {
                    classScores = new Float32Array();
                }
                const numClasses = classScores.length / actualB;
                for (let bi = 0; bi < actualB; bi++) {
                    const offset = bi * numClasses; // stride into classScores
                    const maskedScore = classScores[offset + targetClass] ?? 0;
                    const reluDiff = Math.max(0, maskedScore - originalScore);
                    channelImportanceScores[targetOutputLayerName].push(reluDiff);
                }

            }
            const batchProgress = (bStart + 1) / maskResults.length;
            onProgress?.(0.2 + batchProgress * 0.6, `Processing channel ${bStart + 1}/${maskResults.length}`);
            //console.log(`Processed ${bStart + 1}/${maskResults.length} channels`);
        }
        onProgress?.(0.8, 'Computing final heatmaps');
        this.performanceMetrics.startHeatmapComputation();

        const results: Record<string, Float32Array> = {};

        for (const outputLayerName of outputLayerNames) {
            const weights = channelImportanceScores[outputLayerName];
            const normalizedWeights = this.normalizeWeights(weights);

            const heatmapData = this.computeWeightedHeatmap(
                featureMapsData,
                normalizedWeights,
                importantChannelIndices,
                channels,
                spatialSideLength * spatialSideLength,
                spatialSideLength
            );

            results[outputLayerName] = heatmapData;
            console.log("results in generateHeatmap", results);
            // Notify progress with current heatmap data
            onProgress?.(
                0.8 + (Object.keys(results).length / outputLayerNames.length) * 0.2,
                `Generated ${outputLayerName} heatmap`,
                { ...results }
            );
        }
        this.performanceMetrics.endHeatmapComputation();
        onProgress?.(1.0, 'Heatmaps complete');
        return results;
    }

    destroy(): void {
        this.maskWorkerManager.terminate();
    }
    // Uint8Array length H*H*4 
    private writePreprocessedMaskedImageToBuffer(
        dest: Float32Array,
        destOffset: number, // index in dest
        originalImageData: Uint8Array,
        maskUint8: Uint8Array,
        preprocessingParams: {
            mean: number[],
            std: number[],
            scale: number
        }
    ) {
        const mean = preprocessingParams.mean;
        const std = preprocessingParams.std;
        const scale = preprocessingParams.scale;

        const H = this.config.inputSize;
        const planeSize = H * H;
        //NCHW contiguous: [R-plane (planeSize), G-plane, B-plane]
        const rBase = destOffset + 0 * planeSize;
        const gBase = destOffset + 1 * planeSize;
        const bBase = destOffset + 2 * planeSize;

        const rgba = originalImageData;
        for (let i = 0; i < planeSize; i++) {
            const m = maskUint8[i] / 255; // 0..1
            const rgbaIdx = i * 4;
            // apply mask scale normalize
            const r = rgba[rgbaIdx] * m * scale;
            const g = rgba[rgbaIdx + 1] * m * scale;
            const b = rgba[rgbaIdx + 2] * m * scale;

            dest[rBase + i] = (r - mean[0]) / std[0];
            dest[gBase + i] = (g - mean[1]) / std[1];
            dest[bBase + i] = (b - mean[2]) / std[2];
        }
    }

    private getMostImportantChannelsWithActivations(
        featureMapsData: Float32Array,
        channels: number,
        spatialSideLength: number,
        topNOrPercent: number, // if < 1 -> cumulative fractioni If >=1 topN
    ): { index: number, activation: number }[] {
        const spatialArea = spatialSideLength * spatialSideLength;
        const channelActivations: { index: number, activation: number }[] = [];

        let totalActivationSum = 0;
        for (let c = 0; c < channels; c++) {
            const startIdx = c * spatialArea;
            let sum = 0;
            for (let i = 0; i < spatialArea; i++) sum += Math.abs(featureMapsData[startIdx + i]);
            const avg = sum / spatialArea;
            channelActivations.push({ index: c, activation: avg });
            totalActivationSum += avg;
        }

        channelActivations.sort((a, b) => b.activation - a.activation);

        if (topNOrPercent >= 1) {
            const topN = Math.min(Math.floor(topNOrPercent), channels);
            return channelActivations.slice(0, topN);
        } else {
            // cumulative fraction
            const target = totalActivationSum * Math.max(0, Math.min(1, topNOrPercent));
            const result: { index: number, activation: number }[] = [];
            let cum = 0;
            for (const entry of channelActivations) {
                result.push(entry);
                cum += entry.activation;
                if (cum >= target) break;
            }
            return result;
        }
    }
    projectHeatmapBack(
        heatmapCanvas: HTMLCanvasElement,
        origWidth: number,
        origHeight: number,
        inputSize: number
    ): HTMLCanvasElement {
        const scale = Math.min(inputSize / origWidth, inputSize / origHeight);
        const newW = Math.round(origWidth * scale);
        const newH = Math.round(origHeight * scale);
        const dx = Math.floor((inputSize - newW) / 2);
        const dy = Math.floor((inputSize - newH) / 2);

        // 1. Crop heatmap to the valid region (no padding)
        const croppedCanvas = document.createElement("canvas");
        croppedCanvas.width = newW;
        croppedCanvas.height = newH;
        const croppedCtx = croppedCanvas.getContext("2d")!;
        croppedCtx.drawImage(
            heatmapCanvas,
            dx, dy, newW, newH, // source rect inside padded heatmap
            0, 0, newW, newH    // target rect (cropped heatmap)
        );

        // 2. Resize cropped heatmap to original image size
        const finalCanvas = document.createElement("canvas");
        finalCanvas.width = origWidth;
        finalCanvas.height = origHeight;
        const finalCtx = finalCanvas.getContext("2d")!;
        finalCtx.drawImage(croppedCanvas, 0, 0, origWidth, origHeight);

        return finalCanvas;
    }

    //deprecated
    private getMostImportantChannels(
        featureMapsData: Float32Array,
        channels: number,
        spatialSideLength: number,
        topN: number
    ): number[] {
        const spatialArea = spatialSideLength * spatialSideLength;
        const channelActivations: { index: number, activation: number }[] = [];

        for (let c = 0; c < channels; c++) {
            const startIdx = c * spatialArea;
            const endIdx = startIdx + spatialArea;
            const channelData = featureMapsData.slice(startIdx, endIdx);

            let sum = 0;
            for (let i = 0; i < channelData.length; i++) {
                sum += Math.abs(channelData[i]);
            }
            const avgActivation = sum / channelData.length;

            channelActivations.push({ index: c, activation: avgActivation });
        }

        return channelActivations
            .sort((a, b) => b.activation - a.activation)
            .slice(0, topN > channels ? channels : topN)
            .map(item => item.index);
    }

    createHeatmapCanvas(heatmap: Float32Array | undefined | null,
        layerName?: string,
        spatialSideLength?: number,
        opacity: number = 1,
        activationThreshold: number = 0.9,
        width?: number,
        height?: number
    ): HTMLCanvasElement {
        console.log(`Creating heatmap for ${layerName}:`, {
            dataLength: heatmap?.length,
            spatialSideLength,
            first10Values: heatmap ? Array.from(heatmap.slice(0, 10)) : 'no data'
        });

        if (!heatmap) {
            console.error(`createHeatmapCanvas for ${layerName || 'unknown layer'}: heatmap is undefined or null`);
            return this.createErrorCanvas(`No heatmap data for ${layerName || 'layer'}`);
        }

        if (heatmap.length === 0) {
            console.error(`createHeatmapCanvas for ${layerName || 'unknown layer'}: heatmap is empty`);
            return this.createErrorCanvas(`Empty heatmap data for ${layerName || 'layer'}`);
        }

        // check range
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < heatmap.length; i++) {
            const v = heatmap[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        console.log(`Data range for ${layerName}: min=${min}, max=${max}`);

        const canvasSize = spatialSideLength && spatialSideLength > 0 ? spatialSideLength : this.config.inputSize;
        const canvas = document.createElement('canvas');
        canvas.width = canvasSize;
        canvas.height = canvasSize;

        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const normalizedHeatmap = new Float32Array(heatmap.length);
        const range = max - min;

        for (let i = 0; i < heatmap.length; i++) {
            normalizedHeatmap[i] = range === 0 ? 0 : (heatmap[i] - min) / range;
        }

        for (let i = 0; i < Math.min(normalizedHeatmap.length, imageData.data.length / 4); i++) {
            const value = normalizedHeatmap[i];
            const alpha = opacity * 255;
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
            imageData.data[i * 4] = r;
            imageData.data[i * 4 + 1] = g;
            imageData.data[i * 4 + 2] = b;
            imageData.data[i * 4 + 3] = alpha;

        }
        //console.log("imagedata", imageData.data);
        ctx.putImageData(imageData, 0, 0);

        // check empty       
        if (width && height) {
            console.log("Resizing heatmap canvas to", width, "x", height);
            canvas.replaceWith(this.projectHeatmapBack(canvas, width, height, canvasSize));
        }

        return canvas;
    }

    private preprocessImageData(imageData: Float32Array): Float32Array {
        const pixelsPerChannel = imageData.length / 4; // RGBA
        const processed = new Float32Array(pixelsPerChannel * 3);

        for (let i = 0; i < pixelsPerChannel; i++) {
            const r = imageData[i] / 255.0;
            const g = imageData[i + pixelsPerChannel] / 255.0;
            const b = imageData[i + 2 * pixelsPerChannel] / 255.0;

            processed[i] = (r - 0.485) / 0.229;
            processed[i + pixelsPerChannel] = (g - 0.456) / 0.224;
            processed[i + 2 * pixelsPerChannel] = (b - 0.406) / 0.225;
        }

        return processed;
    }
    private createErrorCanvas(message: string): HTMLCanvasElement {
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 100;
        const ctx = canvas.getContext('2d')!;

        ctx.fillStyle = '#ffebee';
        ctx.fillRect(0, 0, 200, 100);
        ctx.fillStyle = '#d32f2f';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(message, 100, 50);

        return canvas;
    }
    //deprecated
    private createMaskFromChannel(
        featureMapsData: Float32Array, // all feature map
        channelIndex: number,
        totalChannels: number,
        spatialArea: number, // spatialSideLength * spatialSideLength
        inputSize: number,
        spatialSideLength: number // feature map original side len
    ): Float32Array {
        if (!spatialSideLength || spatialSideLength <= 0) {
            console.error(`Invalid spatialSideLength: ${spatialSideLength}. Using fallback value.`);
            spatialSideLength = Math.sqrt(featureMapsData.length / totalChannels);
            if (!Number.isInteger(spatialSideLength)) {
                spatialSideLength = Math.floor(spatialSideLength);
            }
        }

        if (spatialArea <= 0) {
            spatialArea = spatialSideLength * spatialSideLength;
        }
        const startIdx = channelIndex * spatialArea;
        const endIdx = startIdx + spatialArea;
        const singleChannelData = featureMapsData.slice(startIdx, endIdx);

        const upscaledMask = this.upscaleToInputSize(singleChannelData, spatialSideLength);

        //  normalize
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < upscaledMask.length; i++) {
            if (upscaledMask[i] < minVal) minVal = upscaledMask[i];
            if (upscaledMask[i] > maxVal) maxVal = upscaledMask[i];
        }
        const range = maxVal - minVal;
        if (range > 0) {
            for (let i = 0; i < upscaledMask.length; i++) {
                upscaledMask[i] = (upscaledMask[i] - minVal) / range;
            }
        } else { // edge case
            for (let i = 0; i < upscaledMask.length; i++) {
                upscaledMask[i] = 0;
            }
        }

        return upscaledMask;
    }

    //deprecated
    applyMask(originalImageData: Float32Array, mask: Float32Array): Float32Array {
        const pixels = this.config.inputSize * this.config.inputSize;
        const maskedImage = new Float32Array(originalImageData.length);
        for (let i = 0; i < pixels; i++) {
            const maskValue = mask[i];
            const rgbaIndex = i * 4;

            maskedImage[rgbaIndex] = originalImageData[rgbaIndex] * maskValue;
            maskedImage[rgbaIndex + 1] = originalImageData[rgbaIndex + 1] * maskValue;
            maskedImage[rgbaIndex + 2] = originalImageData[rgbaIndex + 2] * maskValue;
            maskedImage[rgbaIndex + 3] = originalImageData[rgbaIndex + 3]; // alpha
        }
        return maskedImage;
    }

    private debugMaskApplication(original: Float32Array, mask: Float32Array, masked: Float32Array): void {
        const preview = document.createElement('div');
        preview.style.display = 'flex';
        preview.style.gap = '10px';
        preview.style.margin = '10px';

        const originalCanvas = this.arrayToCanvas(original, this.config.inputSize, 'Original');
        const maskCanvas = this.maskToCanvas(mask, this.config.inputSize, 'Mask');
        const maskedCanvas = this.arrayToCanvas(masked, this.config.inputSize, 'Masked');

        preview.appendChild(originalCanvas);
        preview.appendChild(maskCanvas);
        preview.appendChild(maskedCanvas);

        document.body.appendChild(preview);
    }

    private arrayToCanvas(data: Float32Array, size: number, title: string): HTMLCanvasElement {
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d')!;

        const imageData = ctx.createImageData(size, size);

        for (let i = 0; i < size * size; i++) {
            const rgbaIndex = i * 4;
            imageData.data[rgbaIndex] = data[rgbaIndex];     // R
            imageData.data[rgbaIndex + 1] = data[rgbaIndex + 1]; // G
            imageData.data[rgbaIndex + 2] = data[rgbaIndex + 2]; // B
            imageData.data[rgbaIndex + 3] = 255;             // A
        }

        ctx.putImageData(imageData, 0, 0);

        // Add title
        ctx.fillStyle = 'red';
        ctx.font = '12px Arial';
        ctx.fillText(title, 5, 15);

        return canvas;
    }

    private maskToCanvas(mask: Float32Array, size: number, title: string): HTMLCanvasElement {
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d')!;

        const imageData = ctx.createImageData(size, size);

        for (let i = 0; i < mask.length; i++) {
            const value = Math.floor(mask[i] * 255);
            const rgbaIndex = i * 4;
            imageData.data[rgbaIndex] = value;     // R
            imageData.data[rgbaIndex + 1] = value; // G
            imageData.data[rgbaIndex + 2] = value; // B
            imageData.data[rgbaIndex + 3] = 255;   // A
        }

        ctx.putImageData(imageData, 0, 0);

        // Add title
        ctx.fillStyle = 'red';
        ctx.font = '12px Arial';
        ctx.fillText(title, 5, 15);

        return canvas;
    }
}
