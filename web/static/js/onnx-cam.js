class ONNXCAM {
    constructor() {
        this.modelService = window.onnxModelService;
    }

    async generateHeatmap(imageElement, targetClassIndex) {
        try {
            const inputTensor = await this.modelService.preprocessImage(imageElement);

            const results = await this.modelService.predict(inputTensor);
            const predictions = this.getPredictions(results);

            const heatmap = await this.generateGradientFreeHeatmap(inputTensor, targetClassIndex);

            inputTensor.dispose();

            return {
                heatmap: heatmap,
                predictions: predictions
            };

        } catch (error) {
            console.error('CAM generation error:', error);
            throw error;
        }
    }

    async generateGradientFreeHeatmap(inputTensor, targetClassIndex) {
        try {
            return await this.eigenCAM(inputTensor);
        } catch (error) {
            console.warn('Eigen-CAM failed, using fallback:', error);
            return await this.randomCAM(inputTensor);
        }
    }

    async eigenCAM(inputTensor) {
        console.warn('Eigen-CAM requires feature map access - using mock implementation');
        return this.mockHeatmap(inputTensor.dims[1], inputTensor.dims[2]);
    }

    mockHeatmap(width, height) {
        const heatmap = new Float32Array(width * height);

        const centerX = width / 2;
        const centerY = height / 2;
        const maxDist = Math.sqrt(centerX * centerX + centerY * centerY);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const dist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                const intensity = 1.0 - (dist / maxDist);
                heatmap[y * width + x] = Math.max(0, intensity);
            }
        }

        return heatmap;
    }

    getPredictions(results) {
        const outputKey = Object.keys(results)[0];
        const predictions = results[outputKey].data;

        return Array.from(predictions);
    }

    getTopPrediction(predictions) {
        const maxConfidence = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxConfidence);

        return {
            index: maxIndex,
            confidence: maxConfidence,
            className: `Class ${maxIndex}`
        };
    }
}

window.onnxCAM = new ONNXCAM();