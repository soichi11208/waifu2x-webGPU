// Waifu2x WebGPU Inference Engine

const MODEL_BASE_URL = 'https://raw.githubusercontent.com/nagadomi/waifu2x/master/models/vgg_7';

const MODEL_URLS = {
    art: {
        noise1: `${MODEL_BASE_URL}/art/noise1_model.json`,
        noise2: `${MODEL_BASE_URL}/art/noise2_model.json`,
        noise3: `${MODEL_BASE_URL}/art/noise3_model.json`,
        scale2x: `${MODEL_BASE_URL}/art/scale2.0x_model.json`,
    },
    photo: {
        noise1: `${MODEL_BASE_URL}/photo/noise1_model.json`,
        noise2: `${MODEL_BASE_URL}/photo/noise2_model.json`,
        noise3: `${MODEL_BASE_URL}/photo/noise3_model.json`,
        scale2x: `${MODEL_BASE_URL}/photo/scale2.0x_model.json`,
    }
};

const CONV_SHADER = `
struct Params {
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
    in_channels: u32,
    out_channels: u32,
    use_relu: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let oc = gid.z;

    if (x >= params.output_w || y >= params.output_h || oc >= params.out_channels) {
        return;
    }

    var sum: f32 = biases[oc];
    let in_c = params.in_channels;

    for (var ic: u32 = 0u; ic < in_c; ic = ic + 1u) {
        for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
                let iy = y + ky;
                let ix = x + kx;
                let input_idx = ic * params.input_h * params.input_w + iy * params.input_w + ix;
                let weight_idx = oc * in_c * 9u + ic * 9u + ky * 3u + kx;
                sum = sum + input_data[input_idx] * weights[weight_idx];
            }
        }
    }

    if (params.use_relu == 1u) {
        sum = max(sum, sum * 0.1); // waifu2x models use LeakyReLU(0.1)
    }

    let output_idx = oc * params.output_h * params.output_w + y * params.output_w + x;
    output_data[output_idx] = sum;
}
`;

const DB_NAME = 'waifu2x-webgpu';
const DB_VERSION = 2; // Incremented to invalidate potentially corrupt old caches
const STORE_NAME = 'models';

export class Waifu2xGPU {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.modelCache = {};
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported');
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        if (!adapter) {
            throw new Error('Failed to get GPU adapter');
        }

        // Request device with larger buffer limits
        const requiredLimits = {};
        const adapterLimits = adapter.limits;
        requiredLimits.maxStorageBufferBindingSize = Math.min(
            adapterLimits.maxStorageBufferBindingSize,
            512 * 1024 * 1024
        );
        requiredLimits.maxBufferSize = Math.min(
            adapterLimits.maxBufferSize,
            512 * 1024 * 1024
        );

        this.device = await adapter.requestDevice({ requiredLimits });

        this.device.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
        });

        const shaderModule = this.device.createShaderModule({ code: CONV_SHADER });

        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });
    }

    // ===== IndexedDB Cache =====

    _openDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = () => {
                if (!req.result.objectStoreNames.contains(STORE_NAME)) {
                    req.result.createObjectStore(STORE_NAME);
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    async _getFromCache(key) {
        try {
            const db = await this._openDB();
            return new Promise((resolve) => {
                const tx = db.transaction(STORE_NAME, 'readonly');
                const store = tx.objectStore(STORE_NAME);
                const req = store.get(key);
                req.onsuccess = () => resolve(req.result || null);
                req.onerror = () => resolve(null);
            });
        } catch {
            return null;
        }
    }

    async _saveToCache(key, data) {
        try {
            const db = await this._openDB();
            return new Promise((resolve) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                const store = tx.objectStore(STORE_NAME);
                store.put(data, key);
                tx.oncomplete = () => resolve();
                tx.onerror = () => resolve();
            });
        } catch {
            // silently fail
        }
    }

    // ===== Model Loading =====

    async loadModel(style, type, onProgress) {
        const key = `${style}_${type}`;
        if (this.modelCache[key]) return this.modelCache[key];

        // Try cache
        const cached = await this._getFromCache(key);
        if (cached) {
            const model = this._deserializeModel(cached);
            this.modelCache[key] = model;
            return model;
        }

        // Fetch from GitHub
        if (onProgress) onProgress(`モデルをダウンロード中 (${type})...`);
        const url = MODEL_URLS[style]?.[type];
        if (!url) throw new Error(`Unknown model: ${style}/${type}`);

        const response = await fetch(url);
        if (!response.ok) throw new Error(`Model download failed: ${response.statusText}`);
        const modelJson = await response.json();

        const model = this._processModelWeights(modelJson);

        // Cache serialized
        await this._saveToCache(key, this._serializeModel(model));

        this.modelCache[key] = model;
        return model;
    }

    _processModelWeights(modelJson) {
        return modelJson.map(layer => {
            const { nInputPlane, nOutputPlane, weight, bias } = layer;
            const flatWeights = new Float32Array(nOutputPlane * nInputPlane * 9);
            let idx = 0;
            for (let oc = 0; oc < nOutputPlane; oc++) {
                for (let ic = 0; ic < nInputPlane; ic++) {
                    for (let ky = 0; ky < 3; ky++) {
                        for (let kx = 0; kx < 3; kx++) {
                            flatWeights[idx++] = weight[oc][ic][ky][kx];
                        }
                    }
                }
            }
            return {
                inChannels: nInputPlane,
                outChannels: nOutputPlane,
                weights: flatWeights,
                biases: new Float32Array(bias)
            };
        });
    }

    _serializeModel(model) {
        return model.map(layer => ({
            inChannels: layer.inChannels,
            outChannels: layer.outChannels,
            weights: Array.from(layer.weights),
            biases: Array.from(layer.biases)
        }));
    }

    _deserializeModel(data) {
        return data.map(layer => ({
            inChannels: layer.inChannels,
            outChannels: layer.outChannels,
            weights: new Float32Array(layer.weights),
            biases: new Float32Array(layer.biases)
        }));
    }

    // ===== Image Processing =====

    async process(imageData, mode, style, noiseLevel, onProgress) {
        const { width, height, data } = imageData;

        // Extract RGB channels as float [0..1], in CHW layout
        const nPixels = width * height;
        const nChannels = 3;
        const rgbCHW = new Float32Array(nChannels * nPixels);
        for (let i = 0; i < nPixels; i++) {
            rgbCHW[0 * nPixels + i] = data[i * 4 + 0] / 255.0; // R
            rgbCHW[1 * nPixels + i] = data[i * 4 + 1] / 255.0; // G
            rgbCHW[2 * nPixels + i] = data[i * 4 + 2] / 255.0; // B
        }

        let currentData = rgbCHW;
        let curW = width;
        let curH = height;

        // Noise reduction
        if (mode === 'noise' || mode === 'noise_scale') {
            onProgress('ノイズ除去モデルを読み込み中...', 5);
            const noiseModel = await this.loadModel(style, `noise${noiseLevel}`, (msg) => onProgress(msg, 10));

            onProgress('ノイズ除去処理中...', 20);
            currentData = await this._processTiled(noiseModel, currentData, curW, curH, (p) => {
                onProgress(`ノイズ除去処理中... ${Math.round(p * 100)}%`, 20 + p * 25);
            });
        }

        // Scale 2x
        if (mode === 'scale' || mode === 'noise_scale') {
            onProgress('拡大モデルを読み込み中...', 50);
            const scaleModel = await this.loadModel(style, 'scale2x', (msg) => onProgress(msg, 55));

            // Bicubic resize to 2x
            onProgress('Bicubic補間中...', 60);
            const newW = curW * 2;
            const newH = curH * 2;
            const upscaled = this._bicubicResizeCHW(currentData, curW, curH, newW, newH, 3);
            curW = newW;
            curH = newH;

            onProgress('拡大処理中...', 65);
            currentData = await this._processTiled(scaleModel, upscaled, curW, curH, (p) => {
                onProgress(`拡大処理中... ${Math.round(p * 100)}%`, 65 + p * 30);
            });
        }

        onProgress('画像を生成中...', 95);

        // Convert back to ImageData
        const outPixels = curW * curH;
        const resultData = new Uint8ClampedArray(outPixels * 4);
        for (let i = 0; i < outPixels; i++) {
            resultData[i * 4 + 0] = Math.round(Math.min(1, Math.max(0, currentData[0 * outPixels + i])) * 255);
            resultData[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, currentData[1 * outPixels + i])) * 255);
            resultData[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, currentData[2 * outPixels + i])) * 255);
            resultData[i * 4 + 3] = 255;
        }

        onProgress('完了！', 100);
        return { imageData: new ImageData(resultData, curW, curH), width: curW, height: curH };
    }

    // Process image in 256x256 tiles to avoid WebGPU memory limits
    async _processTiled(model, rgbCHW, width, height, onProgress) {
        const TILE_SIZE = 256;
        const nLayers = model.length;
        const padding = nLayers;
        const nChannels = model[0].inChannels;
        const outChannels = model[nLayers - 1].outChannels;

        const outCHW = new Float32Array(outChannels * width * height);

        const nTilesX = Math.ceil(width / TILE_SIZE);
        const nTilesY = Math.ceil(height / TILE_SIZE);
        const totalTiles = nTilesX * nTilesY;
        let tileCount = 0;

        for (let ty = 0; ty < nTilesY; ty++) {
            for (let tx = 0; tx < nTilesX; tx++) {
                const startX = tx * TILE_SIZE;
                const startY = ty * TILE_SIZE;
                const curW = Math.min(TILE_SIZE, width - startX);
                const curH = Math.min(TILE_SIZE, height - startY);

                const paddedW = curW + padding * 2;
                const paddedH = curH + padding * 2;
                const paddedInput = new Float32Array(nChannels * paddedW * paddedH);

                for (let c = 0; c < nChannels; c++) {
                    for (let py = 0; py < paddedH; py++) {
                        for (let px = 0; px < paddedW; px++) {
                            let sy = startY + py - padding;
                            let sx = startX + px - padding;

                            if (sy < 0) sy = Math.min(-sy, height - 1);
                            if (sx < 0) sx = Math.min(-sx, width - 1);
                            if (sy >= height) sy = Math.max(2 * height - sy - 2, 0);
                            if (sx >= width) sx = Math.max(2 * width - sx - 2, 0);
                            sy = Math.max(0, Math.min(height - 1, sy));
                            sx = Math.max(0, Math.min(width - 1, sx));

                            paddedInput[c * paddedW * paddedH + py * paddedW + px] = rgbCHW[c * width * height + sy * width + sx];
                        }
                    }
                }

                let currentInput = paddedInput;
                let curTileW = paddedW;
                let curTileH = paddedH;
                let curC = nChannels;

                for (let i = 0; i < nLayers; i++) {
                    const layer = model[i];
                    const outW = curTileW - 2;
                    const outH = curTileH - 2;
                    const useRelu = i < nLayers - 1 ? 1 : 0;

                    currentInput = await this._runConvLayer(
                        currentInput, layer.weights, layer.biases,
                        curTileW, curTileH, outW, outH,
                        curC, layer.outChannels, useRelu
                    );

                    curTileW = outW;
                    curTileH = outH;
                    curC = layer.outChannels;
                }

                for (let c = 0; c < outChannels; c++) {
                    for (let y = 0; y < curH; y++) {
                        for (let x = 0; x < curW; x++) {
                            outCHW[c * width * height + (startY + y) * width + (startX + x)] = 
                                currentInput[c * curTileH * curTileW + y * curTileW + x];
                        }
                    }
                }

                tileCount++;
                if (onProgress) onProgress(tileCount / totalTiles);
            }
        }

        return outCHW;
    }

    async _runConvLayer(input, weights, biases, inW, inH, outW, outH, inC, outC, useRelu) {
        const device = this.device;

        const inputBuffer = device.createBuffer({
            size: input.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(inputBuffer, 0, input);

        const weightsBuffer = device.createBuffer({
            size: weights.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(weightsBuffer, 0, weights);

        const biasBuffer = device.createBuffer({
            size: biases.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(biasBuffer, 0, biases);

        const outputSize = outC * outH * outW * 4;
        const outputBuffer = device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const params = new Uint32Array([inW, inH, outW, outH, inC, outC, useRelu, 0]);
        const paramsBuffer = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramsBuffer, 0, params);

        const bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: weightsBuffer } },
                { binding: 2, resource: { buffer: biasBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } },
                { binding: 4, resource: { buffer: paramsBuffer } },
            ]
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(outW / 8),
            Math.ceil(outH / 8),
            outC
        );
        passEncoder.end();

        const readBuffer = device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);

        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();

        // Destroy buffers
        inputBuffer.destroy();
        weightsBuffer.destroy();
        biasBuffer.destroy();
        outputBuffer.destroy();
        paramsBuffer.destroy();
        readBuffer.destroy();

        return result;
    }

    // ===== Bicubic Interpolation (CHW format) =====

    _bicubicResizeCHW(src, srcW, srcH, dstW, dstH, channels) {
        const dst = new Float32Array(channels * dstW * dstH);
        const scaleX = srcW / dstW;
        const scaleY = srcH / dstH;

        for (let c = 0; c < channels; c++) {
            const cOffset = c * srcW * srcH;
            const dOffset = c * dstW * dstH;

            for (let dy = 0; dy < dstH; dy++) {
                for (let dx = 0; dx < dstW; dx++) {
                    const fx = (dx + 0.5) * scaleX - 0.5;
                    const fy = (dy + 0.5) * scaleY - 0.5;
                    const ix = Math.floor(fx);
                    const iy = Math.floor(fy);
                    const tx = fx - ix;
                    const ty = fy - iy;

                    let val = 0;
                    for (let m = -1; m <= 2; m++) {
                        const wy = this._cubicWeight(ty - m);
                        const sy = Math.max(0, Math.min(srcH - 1, iy + m));
                        for (let n = -1; n <= 2; n++) {
                            const wx = this._cubicWeight(tx - n);
                            const sx = Math.max(0, Math.min(srcW - 1, ix + n));
                            val += src[cOffset + sy * srcW + sx] * wx * wy;
                        }
                    }

                    dst[dOffset + dy * dstW + dx] = val;
                }
            }
        }

        return dst;
    }

    _cubicWeight(x) {
        x = Math.abs(x);
        if (x <= 1) return 1.5 * x * x * x - 2.5 * x * x + 1;
        if (x < 2) return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
        return 0;
    }
}
