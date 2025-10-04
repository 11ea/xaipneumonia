export interface MaskResult {
    channelIndex: number;
    mask: Uint8Array | Float32Array;
    importance: number;
}
export class MaskWorkerManager {
    private workers: Worker[] = [];
    private taskCounter = 0;
    private pendingTasks: Map<number, {
        resolve: (value: MaskResult[]) => void;
        reject: (error: any) => void;
        totalChunks: number;
        collectedResults: MaskResult[];
        receivedChunks: number;
    }> = new Map();

    constructor(numWorkers: number = navigator.hardwareConcurrency || 4) {
        for (let i = 0; i < numWorkers; i++) {
            const worker = this.createFunctionWorker();
            worker.onmessage = this.handleWorkerMessage.bind(this);
            worker.onerror = this.handleWorkerError.bind(this);
            this.workers.push(worker);
        }
    }
    /*     private createFunctionWorker(): Worker {
            const fastSilu = function (x: number): number {
                if (x < -3) return 0;
                if (x > 3) return x;
                const x2 = x * x;
                const inner = 0.7978845608 * x * (1 + 0.044715 * x2);
                return x * (0.5 + 0.5 * Math.tanh(inner));
            };
    
            const upscaleBilinear = function (data: Float32Array, originalSideLength: number, targetSideLength: number): Float32Array {
                const upscaled = new Float32Array(targetSideLength * targetSideLength);
    
                if (originalSideLength === targetSideLength) {
                    return data;
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
            };
    
            const normalizeMask = function (mask: Float32Array): Float32Array {
                let minVal = Infinity, maxVal = -Infinity;
                for (let i = 0; i < mask.length; i++) {
                    if (mask[i] < minVal) minVal = mask[i];
                    if (mask[i] > maxVal) maxVal = mask[i];
                }
    
                const range = maxVal - minVal;
                if (range > 0) {
                    for (let i = 0; i < mask.length; i++) {
                        mask[i] = (mask[i] - minVal) / range;
                    }
                } else {
                    for (let i = 0; i < mask.length; i++) {
                        mask[i] = 0;
                    }
                }
                return mask;
            };
    
            const normalizeFeatureMap = function (data: Float32Array): Float32Array {
                console.log('Worker: Normalizing feature map data');
    
                const reluData = new Float32Array(data.length);
                for (let i = 0; i < data.length; i++) {
                    reluData[i] = Math.max(0, data[i]);
                }
    
                const maxVal = Math.max(...reluData);
    
                if (maxVal > 0) {
                    for (let i = 0; i < reluData.length; i++) {
                        reluData[i] = reluData[i] / maxVal;
                    }
                }
    
                return reluData;
            };
    
            const processChannel = function (
                featureMapsData: Float32Array,
                channelIndex: number,
                spatialSideLength: number,
                inputSize: number
            ): any {
                const spatialArea = spatialSideLength * spatialSideLength;
                const startIdx = channelIndex * spatialArea;
                const endIdx = startIdx + spatialArea;
    
                const channelData = featureMapsData.slice(startIdx, endIdx);
    
                const normalizedData = normalizeFeatureMap(channelData);
    
                let maxImportance = -Infinity;
                for (let i = 0; i < normalizedData.length; i++) {
                    if (normalizedData[i] > maxImportance) {
                        maxImportance = normalizedData[i];
                    }
                }
    
                const upscaledMask = upscaleBilinear(normalizedData, spatialSideLength, inputSize);
                const finalMask = normalizeMask(upscaledMask);
    
    
                return {
                    channelIndex: channelIndex,
                    mask: finalMask,
                    importance: maxImportance
                };
            };
    
            const mainWorkerFunction = function (e: MessageEvent) {
                const {
                    featureMapsData,
                    channelIndices,
                    spatialSideLength,
                    inputSize,
                    taskId
                } = e.data;
    
                const results = [];
                const importances = [];
    
                for (const channelIndex of channelIndices) {
                    try {
                        const result = processChannel(featureMapsData, channelIndex, spatialSideLength, inputSize);
                        results.push(result);
                        importances.push({ index: channelIndex, importance: result.importance });
                    } catch (error) {
                        console.error(`Worker: Error processing channel ${channelIndex}:`, error);
                    }
                }
    
                self.postMessage({
                    taskId: taskId,
                    masks: results,
                    channelImportances: importances
                });
            };
    
            const workerCode = `
            const fastSilu = ${fastSilu.toString()};
            const upscaleBilinear = ${upscaleBilinear.toString()};
            const normalizeMask = ${normalizeMask.toString()};
            const normalizeFeatureMap = ${normalizeFeatureMap.toString()};
            const processChannel = ${processChannel.toString()};
            
            self.onmessage = ${mainWorkerFunction.toString()};
        `;
    
            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            const worker = new Worker(workerUrl);
    
            setTimeout(() => URL.revokeObjectURL(workerUrl), 1000);
    
            return worker;
        } */
    private createFunctionWorker(): Worker {
        const workerCode = `
    function normalizeFeatureMapTo01(data) {
        // ReLU
        let maxVal = -Infinity;
        for (let i = 0; i < data.length; i++) {
            const v = data[i] > 0 ? data[i] : 0;
            data[i] = v;
            if (v > maxVal) maxVal = v;
        }
        if (maxVal > 0) {
            for (let i = 0; i < data.length; i++) data[i] = data[i] / maxVal;
        }
        return data;
    }

    function upscaleBilinearToUint8(srcFloat, srcSide, dstSide, outUint8) {
        const scaleX = srcSide / dstSide;
        const scaleY = srcSide / dstSide;
        
        for (let dstY = 0; dstY < dstSide; dstY++) {
            const srcY = dstY * scaleY;
            const y1 = Math.floor(srcY);
            const y2 = Math.min(y1 + 1, srcSide - 1);
            const fy = srcY - y1;
            
            for (let dstX = 0; dstX < dstSide; dstX++) {
                const srcX = dstX * scaleX;
                const x1 = Math.floor(srcX);
                const x2 = Math.min(x1 + 1, srcSide - 1);
                const fx = srcX - x1;
                
                // Get the four surrounding points
                const val11 = srcFloat[y1 * srcSide + x1];
                const val12 = srcFloat[y1 * srcSide + x2];
                const val21 = srcFloat[y2 * srcSide + x1];
                const val22 = srcFloat[y2 * srcSide + x2];
                
                // Bilinear interpolation
                const interp = 
                    val11 * (1 - fx) * (1 - fy) +
                    val12 * fx * (1 - fy) +
                    val21 * (1 - fx) * fy +
                    val22 * fx * fy;
                
                const idx = dstY * dstSide + dstX;
                outUint8[idx] = Math.round(Math.max(0, Math.min(255, interp * 255)));
            }
        }
    }

    function findMinMaxAndNormalizeInPlaceFloat32(arr) {
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        const range = maxVal - minVal;
        if (range > 0) {
            for (let i = 0; i < arr.length; i++) arr[i] = (arr[i] - minVal) / range;
        } else {
            for (let i = 0; i < arr.length; i++) arr[i] = 0;
        }
    }

    self.onmessage = function(e) {
        const { featureMapsBuffer, isShared, featureMapsByteOffset, featureMapsLength, channelIndices, spatialSideLength, inputSize, taskId } = e.data;

        const f32 = new Float32Array(featureMapsBuffer, featureMapsByteOffset || 0, featureMapsLength); // length in floats
        const spatialArea = spatialSideLength * spatialSideLength;

        const masksOut = []; // we will post back each mask as transferable Uint8Array
        for (let ch of channelIndices) {
            try {
                const start = ch * spatialArea;
                const channelView = f32.subarray(start, start + spatialArea);
                // avoid mutating global map
                const tmp = new Float32Array(channelView.length);
                tmp.set(channelView);
                normalizeFeatureMapTo01(tmp);
                // upscale
                const out = new Uint8Array(inputSize * inputSize);
                upscaleBilinearToUint8(tmp, spatialSideLength, inputSize, out);
                // final normalization 
                let maxImp = 0;
                for (let i = 0; i < tmp.length; i++) if (tmp[i] > maxImp) maxImp = tmp[i];

                masksOut.push({ channelIndex: ch, mask: out, importance: maxImp });
            } catch (err) {
                console.error('worker channel error', err);
            }
        }

        const transferList = masksOut.map(m => m.mask.buffer);
        self.postMessage({ taskId, masks: masksOut }, transferList);
    };
    `;
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);
        // revoke later or on unload
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        return worker;
    }

    /* async generateMasks(
        featureMapsData: Float32Array,
        channelIndices: number[],
        channels: number,
        spatialSideLength: number,
        inputSize: number
    ): Promise<MaskResult[]> {
        const taskId = this.taskCounter++;

        return new Promise((resolve, reject) => {
            const chunks = this.chunkArray(channelIndices, this.workers.length);
            const activeChunks = chunks.filter(chunk => chunk.length > 0); // Only send if chunk has data

            this.pendingTasks.set(taskId, {
                resolve,
                reject,
                totalChunks: activeChunks.length, // Store how many chunks we expect back
                collectedResults: [],
                receivedChunks: 0,
            });

            if (activeChunks.length === 0) { // Handle case where no channels are important
                resolve([]);
                this.pendingTasks.delete(taskId);
                return;
            }

            activeChunks.forEach((chunk, workerIndex) => {
                if (this.workers[workerIndex]) {
                    const workerFeatureMapsDataCopy = new Float32Array(featureMapsData.length);
                    workerFeatureMapsDataCopy.set(featureMapsData);

                    this.workers[workerIndex].postMessage({
                        featureMapsData: workerFeatureMapsDataCopy, // Pass the NEW copy
                        channelIndices: chunk,
                        spatialSideLength: spatialSideLength,
                        inputSize: inputSize,
                        taskId: taskId
                    });
                }
            });
        });
    } */
    async generateMasks(
        featureMapsData: Float32Array,
        channelIndices: number[],
        channels: number,
        spatialSideLength: number,
        inputSize: number
    ): Promise<MaskResult[]> {
        const taskId = this.taskCounter++;
        console.log('SAB available?', typeof SharedArrayBuffer !== 'undefined', 'crossOriginIsolated:', window.crossOriginIsolated);
        return new Promise((resolve, reject) => {
            const chunks = this.chunkArray(channelIndices, this.workers.length);
            const activeChunks = chunks.filter(chunk => chunk.length > 0);
            if (activeChunks.length === 0) {
                resolve([]);
                return;
            }

            this.pendingTasks.set(taskId, {
                resolve,
                reject,
                totalChunks: activeChunks.length,
                collectedResults: [],
                receivedChunks: 0,
            });

            const mainBuffer = featureMapsData.buffer;
            const isSharedArrayBufferAvailable = typeof SharedArrayBuffer !== 'undefined' && mainBuffer instanceof (SharedArrayBuffer as any);

            activeChunks.forEach((chunk, workerIndex) => {
                const worker = this.workers[workerIndex];
                if (!worker) return;

                if (isSharedArrayBufferAvailable) {
                    //0cpy
                    worker.postMessage({
                        featureMapsBuffer: mainBuffer,
                        isShared: true,
                        featureMapsByteOffset: featureMapsData.byteOffset,
                        featureMapsLength: featureMapsData.length,
                        channelIndices: chunk,
                        spatialSideLength,
                        inputSize,
                        taskId
                    });
                } else {
                    //fallback
                    const copyBuf = featureMapsData.slice().buffer;
                    worker.postMessage({
                        featureMapsBuffer: copyBuf,
                        isShared: false,
                        featureMapsByteOffset: 0,
                        featureMapsLength: featureMapsData.length,
                        channelIndices: chunk,
                        spatialSideLength,
                        inputSize,
                        taskId
                    }, [copyBuf]);
                }
            });
        });
    }

    private handleWorkerMessage(event: MessageEvent) {
        const { taskId, masks } = event.data;
        const task = this.pendingTasks.get(taskId);
        if (!task) return console.warn('Unknown task', taskId);
        for (const m of masks) {
            const u8: Uint8Array = m.mask;
            task.collectedResults.push({
                channelIndex: m.channelIndex,
                mask: u8,
                importance: m.importance
            });
        }

        task.receivedChunks++;
        if (task.receivedChunks === task.totalChunks) {
            task.collectedResults.sort((a, b) => a.channelIndex - b.channelIndex);
            task.resolve(task.collectedResults);
            this.pendingTasks.delete(taskId);
        }
    }

    private handleWorkerError(error: ErrorEvent): void {
        console.error('Mask worker error:', error);
        // reject all pending
        for (const [taskId, task] of this.pendingTasks.entries()) {
            task.reject(error);
            this.pendingTasks.delete(taskId);
        }
    }

    private chunkArray<T>(array: T[], chunks: number): T[][] {
        const result: T[][] = Array.from({ length: chunks }, () => []);
        for (let i = 0; i < array.length; i++) {
            result[i % chunks].push(array[i]);
        }
        return result;
    }

    terminate(): void {
        this.workers.forEach(worker => worker.terminate());
        this.workers = [];
        this.pendingTasks.clear();
    }
}