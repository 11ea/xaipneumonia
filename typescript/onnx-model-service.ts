import * as ort from 'onnxruntime-web';

export class ONNXModelService {
    private session: ort.InferenceSession | null = null;
    private currentModelPath: string = '';
    private executionProvider: string = '';
    private static wasmConfigured: boolean = false;
    private inferenceTimes: number[] = [];
    private totalInferences: number = 0;

    private inferenceQueue: Array<{
        resolve: (value: any) => void;
        reject: (error: Error) => void;
        inputData: Float32Array;
        dims: number[];
    }> = [];

    private isProcessingQueue = false;


    constructor() {
        if (!ONNXModelService.wasmConfigured) {
            this.configureONNXEnvironment();
            ONNXModelService.wasmConfigured = true;
        }
    }

    private configureONNXEnvironment(): void {
        try {
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            ort.env.wasm.numThreads = navigator.hardwareConcurrency || 1;
            ort.env.wasm.simd = true;
            ort.env.wasm.proxy = false;
            console.log('ONNX environment configured for single thread');
        } catch (error) {
            console.warn('Failed to configure ONNX environment:', error);
        }
    }

    async loadModel(modelPath: string): Promise<void> {
        try {
            // Release existing session first
            if (this.session) {
                await this.session.release();
                this.session = null;
            }

            const providers = await this.getAvailableProviders();
            console.log('Available providers:', providers);

            for (const provider of providers) {
                try {
                    console.log(`Trying provider: ${provider}`);

                    const sessionOptions: ort.InferenceSession.SessionOptions = {
                        executionProviders: [provider],
                        graphOptimizationLevel: 'all',
                        enableCpuMemArena: true,
                        enableMemPattern: true,
                    };

                    this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
                    this.currentModelPath = modelPath;
                    this.executionProvider = provider;
                    console.log(`Successfully loaded with ${provider}`);
                    //this.validateBatchProcessing();
                    return;
                } catch (providerError) {
                    console.warn(`Failed with ${provider}:`, providerError);
                    if (this.session) {
                        await this.session.release();
                        this.session = null;
                    }
                }
            }

            throw new Error('All execution providers failed to load the model.');

        } catch (error) {
            console.error('Failed to load model:', error);
            throw new Error(`Model loading failed: ${(error as Error).message}`);
        }
    }
    // Simple test to validate batch processing
    async validateBatchProcessing(): Promise<boolean> {
        const testInput = new Float32Array(3 * 224 * 224).fill(0.5);

        // Single inference
        const singleResult = await this.runInference(testInput, [1, 3, 224, 224]);

        // Batch inference (same input repeated)
        const batchInputs = [testInput, testInput, testInput, testInput];
        const batchResults = await this.runBatchInference(batchInputs, [batchInputs.length, 3, 224, 224]);

        // Compare results - should be identical for all batch elements
        const tolerance = 1e-6;
        const firstBatchResult = batchResults[0];

        for (const outputName of this.session!.outputNames) {
            const singleData = singleResult[outputName].data as Float32Array;
            const batchData = firstBatchResult[outputName];

            for (let i = 0; i < singleData.length; i++) {
                if (Math.abs(singleData[i] - batchData[i]) > tolerance) {
                    console.error('Batch processing alters results!');
                    return false;
                }
            }
        }

        console.log('Batch processing validated - results are consistent');
        return true;
    }
    async runInference(input: Float32Array, dims: number[]): Promise<Record<string, any>> {
        const startTime = performance.now();

        try {
            const result = await this._runInferenceInternal(input, dims);
            const inferenceTime = performance.now() - startTime;

            this.inferenceTimes.push(inferenceTime);
            this.totalInferences++;

            if (this.totalInferences % 10 === 0) {
                this.logInferenceStats();
            }

            return result;
        } catch (error) {
            const failedTime = performance.now() - startTime;
            console.error(`Inference failed after ${failedTime}ms:`, error);
            throw error;
        }
    }

    async runBatchInference(inputs: Float32Array[], dims: number[]): Promise<Record<string, any>[]> {
        const startTime = performance.now();

        try {
            const results = await this._runBatchInferenceInternal(inputs, dims);
            const batchTime = performance.now() - startTime;
            const avgTime = batchTime / inputs.length;

            // Record each inference in the batch
            for (let i = 0; i < inputs.length; i++) {
                this.inferenceTimes.push(avgTime);
                this.totalInferences++;
            }

            console.log(`Batch inference: ${inputs.length} inputs in ${batchTime.toFixed(2)}ms (avg ${avgTime.toFixed(2)}ms/inf)`);

            return results;
        } catch (error) {
            const failedTime = performance.now() - startTime;
            console.error(`Batch inference failed after ${failedTime}ms:`, error);
            throw error;
        }
    }

    private logInferenceStats(): void {
        if (this.inferenceTimes.length === 0) return;

        const total = this.inferenceTimes.reduce((sum, time) => sum + time, 0);
        const average = total / this.inferenceTimes.length;
        const min = Math.min(...this.inferenceTimes);
        const max = Math.max(...this.inferenceTimes);

        console.log(`📊 Model Inference Stats: ${average.toFixed(2)}ms avg, ${min.toFixed(2)}ms min, ${max.toFixed(2)}ms max, ${this.totalInferences} total`);
    }

    getInferenceMetrics() {
        if (this.inferenceTimes.length === 0) {
            return { average: 0, min: 0, max: 0, total: 0 };
        }

        const total = this.inferenceTimes.reduce((sum, time) => sum + time, 0);
        return {
            average: total / this.inferenceTimes.length,
            min: Math.min(...this.inferenceTimes),
            max: Math.max(...this.inferenceTimes),
            total: this.totalInferences,
            times: [...this.inferenceTimes]
        };
    }

    resetMetrics(): void {
        this.inferenceTimes = [];
        this.totalInferences = 0;
    }
    async _runInferenceInternal(inputData: Float32Array, dims: number[]): Promise<Record<string, ort.Tensor>> {
        return new Promise((resolve, reject) => {
            // Add to queue
            this.inferenceQueue.push({ resolve, reject, inputData, dims });

            // Process queue if not already processing
            if (!this.isProcessingQueue) {
                this.processInferenceQueue();
            }
        });
    }
    private async processInferenceQueue(): Promise<void> {
        if (this.isProcessingQueue || this.inferenceQueue.length === 0) {
            return;
        }

        this.isProcessingQueue = true;

        while (this.inferenceQueue.length > 0) {
            const task = this.inferenceQueue.shift();
            if (!task) continue;

            const { resolve, reject, inputData, dims } = task;

            try {
                if (!this.isModelLoaded() || !this.session) {
                    throw new Error('Model not loaded. Please load a model first.');
                }

                console.log(' Running inference, input shape:', dims);

                // Create input tensor
                const inputTensor = new ort.Tensor('float32', inputData, dims);

                // Prepare feeds
                const feeds: Record<string, ort.Tensor> = {};
                const inputName = this.session.inputNames[0];
                feeds[inputName] = inputTensor;

                // Run inference
                const results = await this.session.run(feeds);
                console.log(' Inference completed successfully');

                resolve(results);

            } catch (error) {
                console.error(' Inference failed:', error);
                reject(new Error('Inference error: ' + (error as Error).message));
            }

            //await new Promise(resolve => setTimeout(resolve, 1));
        }

        this.isProcessingQueue = false;
    }
    async _runMaskedBatchInference(
        maskedInputs: Float32Array[],   // all your masked images
        inputDims: number[]             // e.g. [1, 3, 224, 224]
    ): Promise<Record<string, ort.Tensor>[]> {

        const batchSize = maskedInputs.length;
        if (batchSize === 0) return [];

        // Number of elements per input
        const elementsPerInput = inputDims.reduce((a, b) => a * b, 1);

        // Validate each input length
        for (let i = 0; i < batchSize; i++) {
            if (maskedInputs[i].length !== elementsPerInput) {
                throw new Error(
                    `Input ${i} has incorrect size. Expected ${elementsPerInput}, got ${maskedInputs[i].length}`
                );
            }
        }

        // Create one big Float32Array with all masked inputs concatenated
        const batchData = new Float32Array(batchSize * elementsPerInput);
        for (let i = 0; i < batchSize; i++) {
            batchData.set(maskedInputs[i], i * elementsPerInput);
        }

        // Adjust dims: [batch, channels, H, W]
        const batchDims = [batchSize, ...inputDims.slice(1)];

        console.log(`Running CAM batch inference: ${batchSize} masks, shape:`, batchDims);

        // Run ONNX inference
        const batchResult = await this.runInference(batchData, batchDims);

        // Split back into per-mask results
        return this.splitMaskedBatchResults(batchResult, batchSize);
    }
    private splitMaskedBatchResults(
        batchResult: Record<string, ort.Tensor>,
        batchSize: number
    ): Record<string, ort.Tensor>[] {

        const outputs: Record<string, ort.Tensor>[] = Array.from({ length: batchSize }, () => ({}));

        for (const [key, tensor] of Object.entries(batchResult)) {
            const dims = tensor.dims;
            const perSampleSize = tensor.size / batchSize;

            for (let i = 0; i < batchSize; i++) {
                const offset = i * perSampleSize;
                const sliceData = tensor.data.slice(offset, offset + perSampleSize);

                const sliceDims = [1, ...dims.slice(1)]; // keep batch=1
                outputs[i][key] = new ort.Tensor(tensor.type, sliceData, sliceDims);
            }
        }

        return outputs;
    }

    async _runBatchInferenceInternal(
        inputs: Float32Array[],
        inputDims: number[]
    ): Promise<Record<string, any>[]> {

        // Validate that total elements match expected
        const expectedElements = inputDims.reduce((a, b) => a * b, 1);
        const actualElements = inputs.reduce((sum, arr) => sum + arr.length, 0);
        if (actualElements !== expectedElements) {
            console.error(`Batch size mismatch: Expected ${expectedElements}, got ${actualElements}`);
            console.error(`Input dims: [${inputDims.join(', ')}]`);
            console.error(`Input arrays: ${inputs.length} arrays with lengths: ${inputs.map(arr => arr.length).join(', ')}`);
            throw new Error(`Input has incorrect size. Expected ${expectedElements}, got ${actualElements}`);
        }

        // Use true batch processing
        const batchSize = inputs.length;

        if (batchSize === 0) return [];
        if (batchSize === 1) {
            const result = await this.runInference(inputs[0], inputDims);
            return [this.convertTensorToOutput(result)];
        }

        try {
            // Create batch tensor
            const elementsPerInput = actualElements / batchSize;
            const batchData = new Float32Array(actualElements);

            for (let i = 0; i < batchSize; i++) {
                if (inputs[i].length !== elementsPerInput) {
                    throw new Error(`Input ${i} has incorrect size. Expected ${elementsPerInput}, got ${inputs[i].length}`);
                }
                batchData.set(inputs[i], i * elementsPerInput);
            }

            const batchDims = [batchSize, ...inputDims.slice(1)];

            console.log(`Running batch inference: ${batchSize} items, shape:`, batchDims);

            // Use the queued runInference for batch processing too
            const batchResult = await this.runInference(batchData, batchDims);

            // Split batch results
            return this.splitBatchResults(batchResult, batchSize);

        } catch (error) {
            console.error('Batch inference failed:', error);
            throw error;
        }
    }

    private convertTensorToOutput(result: Record<string, ort.Tensor>): { [key: string]: Float32Array } {
        const output: { [key: string]: Float32Array } = {};
        Object.keys(result).forEach(name => {
            output[name] = result[name].data as Float32Array;
        });
        return output;
    }

    private splitBatchResults(batchResult: Record<string, ort.Tensor>, batchSize: number): { [key: string]: Float32Array }[] {
        const outputs: { [key: string]: Float32Array }[] = Array(batchSize).fill(null).map(() => ({}));

        Object.entries(batchResult).forEach(([outputName, tensor]) => {
            const tensorData = tensor.data as Float32Array;
            const elementsPerOutput = tensorData.length / batchSize;

            if (!Number.isInteger(elementsPerOutput)) {
                throw new Error(`Cannot split tensor evenly. Batch size: ${batchSize}, Tensor length: ${tensorData.length}`);
            }

            for (let i = 0; i < batchSize; i++) {
                outputs[i][outputName] = tensorData.slice(
                    i * elementsPerOutput,
                    (i + 1) * elementsPerOutput
                );
            }
        });

        return outputs;
    }

    // Simple provider detection
    private async getAvailableProviders(): Promise<string[]> {
        const providers: string[] = [];

        if (await this.isWebGPUAvailable()) {
            providers.push('webgpu');
        }

        if (await this.isWebGLAvailable()) {
            providers.push('webgl');
        }

        // Always include WASM as fallback
        providers.push('wasm');

        return providers;
    }

    // Browser capability checks (keep these)
    private async isWebGLAvailable(): Promise<boolean> {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext &&
                (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch {
            return false;
        }
    }

    private async isWebGPUAvailable(): Promise<boolean> {
        try {
            //@ts-ignore 
            return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
        } catch {
            return false;
        }
    }

    isModelLoaded(): boolean {
        return this.session !== null;
    }

    getCurrentProvider(): string {
        return this.executionProvider;
    }

    async release(): Promise<void> {
        if (this.session) {
            await this.session.release();
            this.session = null;
        }
    }


    async getPerformanceInfo(): Promise<{
        provider: string;
        supportedProviders: string[]; // Based on browser capabilities
        hardwareConcurrency: number;
        webGL: boolean;
        webGPU: boolean;
    }> {
        return {
            provider: this.executionProvider,
            supportedProviders: await this.getAvailableProvidersBrowserCapabilities(),
            hardwareConcurrency: navigator.hardwareConcurrency || 1,
            webGL: await this.isWebGLAvailable(),
            webGPU: await this.isWebGPUAvailable(),
        };
    }

    private async getAvailableProvidersBrowserCapabilities(): Promise<string[]> {
        const providers: string[] = [];

        if (await this.isWebGPUAvailable()) providers.push('webgpu');
        if (await this.isWebGLAvailable()) providers.push('webgl');
        providers.push('wasm'); // Basic WASM is always considered available by browser.

        return providers;
    }
    dispose(): void {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
    }

    getCurrentModel(): string {
        return this.currentModelPath;
    }
}