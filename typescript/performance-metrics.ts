import { InferenceMetrics } from './interfaces';

export class PerformanceMetrics {
    private startTime: number = 0;
    private metrics: InferenceMetrics = {
        totalTime: 0,
        averageTime: 0,
        minTime: Infinity,
        maxTime: 0,
        totalInferences: 0,
        inferencesPerSecond: 0,
        channelProcessingTimes: [],
        maskGenerationTime: 0,
        heatmapComputationTime: 0
    };

    private currentBatchStart: number = 0;
    private maskGenStart: number = 0;

    startMeasurement(): void {
        this.startTime = performance.now();
        this.metrics = {
            totalTime: 0,
            averageTime: 0,
            minTime: Infinity,
            maxTime: 0,
            totalInferences: 0,
            inferencesPerSecond: 0,
            channelProcessingTimes: [],
            maskGenerationTime: 0,
            heatmapComputationTime: 0
        };
    }

    startMaskGeneration(): void {
        this.maskGenStart = performance.now();
    }

    endMaskGeneration(): void {
        this.metrics.maskGenerationTime = performance.now() - this.maskGenStart;
    }

    startBatch(): void {
        this.currentBatchStart = performance.now();
    }

    recordInference(): void {
        const inferenceTime = performance.now() - this.currentBatchStart;
        this.metrics.channelProcessingTimes.push(inferenceTime);
        this.metrics.totalInferences++;

        this.metrics.minTime = Math.min(this.metrics.minTime, inferenceTime);
        this.metrics.maxTime = Math.max(this.metrics.maxTime, inferenceTime);

        this.currentBatchStart = performance.now(); // Reset for next inference
    }

    startHeatmapComputation(): void {
        this.currentBatchStart = performance.now();
    }

    endHeatmapComputation(): void {
        this.metrics.heatmapComputationTime = performance.now() - this.currentBatchStart;
    }

    endMeasurement(): InferenceMetrics {
        const totalTime = performance.now() - this.startTime;
        this.metrics.totalTime = totalTime;

        if (this.metrics.totalInferences > 0) {
            const totalInferenceTime = this.metrics.channelProcessingTimes.reduce((sum, time) => sum + time, 0);
            this.metrics.averageTime = totalInferenceTime / this.metrics.totalInferences;
            this.metrics.inferencesPerSecond = (this.metrics.totalInferences / (totalInferenceTime / 1000)) || 0;
        }

        return this.getMetrics();
    }

    getMetrics(): InferenceMetrics {
        return { ...this.metrics }; // Return copy
    }

    logMetrics(context: string = ''): void {
        const metrics = this.getMetrics();
        console.groupCollapsed(` Performance Metrics ${context}`);
        console.log(`Total Time: ${metrics.totalTime.toFixed(2)}ms`);
        console.log(`Mask Generation: ${metrics.maskGenerationTime.toFixed(2)}ms`);
        console.log(`Heatmap Computation: ${metrics.heatmapComputationTime.toFixed(2)}ms`);
        console.log(`Inference Time: ${(metrics.totalTime - metrics.maskGenerationTime - metrics.heatmapComputationTime).toFixed(2)}ms`);
        console.log(`Total Inferences: ${metrics.totalInferences}`);
        console.log(`Avg Inference Time: ${metrics.averageTime.toFixed(2)}ms`);
        console.log(`Min/Max Inference: ${metrics.minTime.toFixed(2)}ms / ${metrics.maxTime.toFixed(2)}ms`);
        console.log(`Inferences/sec: ${metrics.inferencesPerSecond.toFixed(2)}`);

        if (metrics.channelProcessingTimes.length > 0) {
            console.log('Channel Processing Times:');
            metrics.channelProcessingTimes.forEach((time, index) => {
                console.log(`  Channel ${index}: ${time.toFixed(2)}ms`);
            });
        }
        console.groupEnd();
    }

    getMetricsSummary(): string {
        const metrics = this.getMetrics();
        return `
Total: ${metrics.totalTime.toFixed(0)}ms | 
Masks: ${metrics.maskGenerationTime.toFixed(0)}ms | 
Inferences: ${(metrics.totalTime - metrics.maskGenerationTime - metrics.heatmapComputationTime).toFixed(0)}ms |
Heatmap: ${metrics.heatmapComputationTime.toFixed(0)}ms |
Avg: ${metrics.averageTime.toFixed(1)}ms/inf | 
${metrics.inferencesPerSecond.toFixed(1)} inf/sec
        `.trim();
    }
}