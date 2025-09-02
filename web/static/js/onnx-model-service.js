class ONNXModelService {
    constructor() {
        this.session = null;
        this.isLoaded = false;
        this.inputShape = [1, 3, 224, 224]; // Adjust based on your model
        this.modelUrls = {
            'yolon-artirilmisVeri': 'https://rawcdn.githack.com/11ea/pneumoniadetectmodels/a4881882e7b8bfd8baa7f989bf3852324ca05d67/n-artirilmisVeri.onnx',
            'yolos-artirilmisVeri': 'https://rawcdn.githack.com/11ea/pneumoniadetectmodels/a4881882e7b8bfd8baa7f989bf3852324ca05d67/s-artirilmisVeri.onnx',
            'yolos-azVeri': 'https://rawcdn.githack.com/11ea/pneumoniadetectmodels/a4881882e7b8bfd8baa7f989bf3852324ca05d67/s-azVeri.onnx'
        };
    }

    async loadModel(modelType = 'yolon-artirilmisVeri') {
        try {
            const modelUrl = this.modelUrls[modelType];
            if (!modelUrl) {
                throw new Error(`Unknown model type: ${modelType}`);
            }

            console.log('Loading ONNX model from:', modelUrl);

            this.session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.isLoaded = true;
            console.log('ONNX model loaded successfully');
            console.log('Input names:', this.session.inputNames);
            console.log('Output names:', this.session.outputNames);

            await this.warmUp();

        } catch (error) {
            console.error('Error loading ONNX model:', error);
            throw error;
        }
    }

    async warmUp() {
        const dummyInput = this.createDummyInput();
        await this.predict(dummyInput);
    }

    createDummyInput() {
        const [batch, height, width, channels] = this.inputShape;
        const data = new Float32Array(batch * height * width * channels).fill(0.5);
        return new ort.Tensor('float32', data, [batch, height, width, channels]);
    }

    async preprocessImage(imageElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.inputShape[2]; // width
        canvas.height = this.inputShape[1]; // height

        ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = this.applyGammaCorrection(this.applyCLAHE(imageData.data));

        const float32Data = new Float32Array(this.inputShape[1] * this.inputShape[2] * this.inputShape[3]);

        let offset = 0;
        for (let i = 0; i < data.length; i += 4) {
            float32Data[offset++] = data[i] / 255.0;     // R
            float32Data[offset++] = data[i + 1] / 255.0; // G
            float32Data[offset++] = data[i + 2] / 255.0; // B
        }

        return new ort.Tensor('float32', float32Data, this.inputShape);
    }

    async predict(inputTensor) {
        if (!this.isLoaded || !this.session) {
            throw new Error('Model not loaded');
        }

        try {
            const feeds = {};
            feeds[this.session.inputNames[0]] = inputTensor;
            //await this.clearSessionState();

            const results = await this.session.run(feeds);
            return results;

        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }
    async clearSessionState() {
        //  force cleanup 
        if (this.session && this.predictionCount > 10) {
            const currentModelType = this.currentModelType;
            this.dispose();
            await this.loadModel(currentModelType);
            this.predictionCount = 0;
        }
        this.predictionCount = (this.predictionCount || 0) + 1;
    }
    async getFeatureMaps(inputTensor, layerName) {

        console.warn('Feature map extraction requires model modification');
        return null;
    }
    async applyGammaCorrection(imageData, gamma = 1.0) {
        const outputData = new Uint8ClampedArray(imageData); // Create a copy for output
        const inverseGamma = 1 / gamma;

        for (let i = 0; i < outputData.length; i += 4) { // Iterate over RGBA pixels
            // Apply gamma correction to R, G, B channels
            outputData[i] = Math.min(255, Math.max(0, Math.round(255 * Math.pow(imageData[i] / 255, inverseGamma))));     // Red
            outputData[i + 1] = Math.min(255, Math.max(0, Math.round(255 * Math.pow(imageData[i + 1] / 255, inverseGamma)))); // Green
            outputData[i + 2] = Math.min(255, Math.max(0, Math.round(255 * Math.pow(imageData[i + 2] / 255, inverseGamma)))); // Blue
            // Alpha channel remains unchanged (outputData[i + 3] = imageData[i + 3])
        }
        return outputData;
    }
    async applyCLAHE(imageData, width = 224, height = 224, clipLimit = 2.0, tilesX = 8, tilesY = 8) {
        const outputData = new Uint8ClampedArray(imageData); // Create a copy for output

        const tileWidth = Math.ceil(width / tilesX);
        const tileHeight = Math.ceil(height / tilesY);

        for (let channel = 0; channel < 3; channel++) { // R, G, B channels
            for (let ty = 0; ty < tilesY; ty++) {
                for (let tx = 0; tx < tilesX; tx++) {
                    const startX = tx * tileWidth;
                    const startY = ty * tileHeight;
                    const endX = Math.min(startX + tileWidth, width);
                    const endY = Math.min(startY + tileHeight, height);

                    const tilePixels = [];
                    for (let y = startY; y < endY; y++) {
                        for (let x = startX; x < endX; x++) {
                            tilePixels.push(imageData[(y * width + x) * 4 + channel]);
                        }
                    }

                    const histogram = new Array(256).fill(0);
                    for (const pixel of tilePixels) {
                        histogram[pixel]++;
                    }

                    const numPixelsInTile = tilePixels.length;
                    const averagePixelsPerBin = numPixelsInTile / 256;
                    const actualClipLimit = clipLimit * averagePixelsPerBin;

                    let excess = 0;
                    for (let i = 0; i < 256; i++) {
                        if (histogram[i] > actualClipLimit) {
                            excess += histogram[i] - actualClipLimit;
                            histogram[i] = actualClipLimit;
                        }
                    }

                    const redistributionStep = Math.floor(excess / 256);
                    let remainingExcess = excess % 256;

                    for (let i = 0; i < 256; i++) {
                        histogram[i] += redistributionStep;
                        if (remainingExcess > 0 && histogram[i] < actualClipLimit) {
                            histogram[i]++;
                            remainingExcess--;
                        }
                    }

                    const cdf = new Array(256);
                    cdf[0] = histogram[0];
                    for (let i = 1; i < 256; i++) {
                        cdf[i] = cdf[i - 1] + histogram[i];
                    }

                    const minCdf = cdf[0];
                    for (let i = 0; i < 256; i++) {
                        cdf[i] = Math.round(((cdf[i] - minCdf) / (numPixelsInTile - minCdf)) * 255);
                    }

                    for (let y = startY; y < endY; y++) {
                        for (let x = startX; x < endX; x++) {
                            const originalPixelValue = imageData[(y * width + x) * 4 + channel];
                            outputData[(y * width + x) * 4 + channel] = cdf[originalPixelValue];
                        }
                    }
                }
            }
        }

        return outputData;
    }
    dispose() {
        if (this.session) {
            this.session.release();
        }
        this.isLoaded = false;
    }
}

// Singleton instance
window.onnxModelService = new ONNXModelService();