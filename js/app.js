// Waifu2x WebGPU — Main Application Logic
import { Waifu2xGPU } from './waifu2x.js';
//opus4.6のコードも読みやすくなったもんだ。
class App {
    constructor() {
        this.engine = new Waifu2xGPU();
        this.gpuReady = false;
        this.originalImage = null;     // HTMLImageElement
        this.originalImageData = null; // ImageData
        this.resultImageData = null;

        // Settings
        this.style = 'art';
        this.mode = 'scale';
        this.noiseLevel = 2;

        // DOM refs
        this.el = {
            gpuWarning: document.getElementById('gpu-warning'),
            uploadSection: document.getElementById('upload-section'),
            settingsSection: document.getElementById('settings-section'),
            progressSection: document.getElementById('progress-section'),
            resultSection: document.getElementById('result-section'),
            dropZone: document.getElementById('drop-zone'),
            fileInput: document.getElementById('file-input'),
            previewImg: document.getElementById('preview-img'),
            imgInfo: document.getElementById('img-info'),
            noiseLevelGroup: document.getElementById('noise-level-group'),
            processBtn: document.getElementById('process-btn'),
            progressText: document.getElementById('progress-text'),
            progressPercent: document.getElementById('progress-percent'),
            progressFill: document.getElementById('progress-fill'),
            originalCanvas: document.getElementById('original-canvas'),
            resultCanvas: document.getElementById('result-canvas'),
            resultOverlay: document.querySelector('.result-overlay'),
            sliderLine: document.getElementById('slider-line'),
            comparisonContainer: document.getElementById('comparison-container'),
            resultInfo: document.getElementById('result-info'),
            downloadBtn: document.getElementById('download-btn'),
            newBtn: document.getElementById('new-btn'),
        };

        this._bindEvents();
        this._initGPU();
    }

    async _initGPU() {
        try {
            await this.engine.init();
            this.gpuReady = true;
        } catch (e) {
            console.error('WebGPU init failed:', e);
            this.el.gpuWarning.classList.remove('hidden');
        }
    }

    _bindEvents() {
        // Drop zone
        const dz = this.el.dropZone;
        dz.addEventListener('click', () => this.el.fileInput.click());
        dz.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.el.fileInput.click(); }
        });
        dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
        dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
        dz.addEventListener('drop', (e) => {
            e.preventDefault();
            dz.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) this._loadFile(file);
        });

        this.el.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this._loadFile(file);
            e.target.value = '';
        });

        // Toggle buttons
        this._initToggleGroup('style-group', (val) => { this.style = val; });
        this._initToggleGroup('mode-group', (val) => {
            this.mode = val;
            this.el.noiseLevelGroup.style.display = val === 'scale' ? 'none' : '';
        });
        this._initToggleGroup('noise-group', (val) => { this.noiseLevel = parseInt(val); });

        // Process button
        this.el.processBtn.addEventListener('click', () => this._startProcessing());

        // Result actions
        this.el.downloadBtn.addEventListener('click', () => this._download());
        this.el.newBtn.addEventListener('click', () => this._reset());

        // Comparison slider
        this._initComparisonSlider();
    }

    _initToggleGroup(groupId, onChange) {
        const group = document.getElementById(groupId);
        group.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn-toggle');
            if (!btn) return;
            group.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onChange(btn.dataset.value);
        });
    }

    // ===== File Loading =====

    _loadFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this._showSettings(img, file);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    _showSettings(img, file) {
        this.el.previewImg.src = img.src;
        const sizeMB = (file.size / 1024 / 1024).toFixed(2);
        this.el.imgInfo.textContent = `${img.naturalWidth} × ${img.naturalHeight} — ${sizeMB}MB`;

        // Show/hide noise level
        this.el.noiseLevelGroup.style.display = this.mode === 'scale' ? 'none' : '';

        this._showSection('settings');
    }

    // ===== Processing =====

    async _startProcessing() {
        if (!this.gpuReady) {
            alert('WebGPUが利用できません。対応ブラウザをご使用ください。');
            return;
        }

        const img = this.originalImage;
        if (!img) return;

        // Limit size for GPU memory safety
        const MAX_DIM = 1024;
        let w = img.naturalWidth;
        let h = img.naturalHeight;
        if (w > MAX_DIM || h > MAX_DIM) {
            const ratio = Math.min(MAX_DIM / w, MAX_DIM / h);
            w = Math.round(w * ratio);
            h = Math.round(h * ratio);
        }

        // Draw to canvas to get ImageData
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = w;
        tmpCanvas.height = h;
        const ctx = tmpCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        this.originalImageData = ctx.getImageData(0, 0, w, h);

        this._showSection('progress');
        this._updateProgress('準備中...', 0);

        try {
            const result = await this.engine.process(
                this.originalImageData,
                this.mode,
                this.style,
                this.noiseLevel,
                (text, percent) => this._updateProgress(text, percent)
            );

            this.resultImageData = result.imageData;
            this._showResult();
        } catch (err) {
            console.error('Processing error:', err);
            alert(`処理中にエラーが発生しました: ${err.message}`);
            this._showSection('settings');
        }
    }

    _updateProgress(text, percent) {
        this.el.progressText.textContent = text;
        this.el.progressPercent.textContent = `${Math.round(percent)}%`;
        this.el.progressFill.style.width = `${percent}%`;
    }

    // ===== Result Display =====

    _showResult() {
        const origData = this.originalImageData;
        const resData = this.resultImageData;

        // Draw on canvases — both at result size for comparison
        const rw = resData.width;
        const rh = resData.height;

        // Original canvas (upscaled to match result size)
        const origCanvas = this.el.originalCanvas;
        origCanvas.width = rw;
        origCanvas.height = rh;
        const origCtx = origCanvas.getContext('2d');
        // Draw original stretched to result size
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = origData.width;
        tmpCanvas.height = origData.height;
        tmpCanvas.getContext('2d').putImageData(origData, 0, 0);
        origCtx.imageSmoothingEnabled = true;
        origCtx.imageSmoothingQuality = 'high';
        origCtx.drawImage(tmpCanvas, 0, 0, rw, rh);

        // Result canvas
        const resCanvas = this.el.resultCanvas;
        resCanvas.width = rw;
        resCanvas.height = rh;
        resCanvas.getContext('2d').putImageData(resData, 0, 0);

        // Info
        let infoText = `${origData.width}×${origData.height} → ${rw}×${rh}`;
        this.el.resultInfo.textContent = infoText;

        // Reset slider
        this._setSliderPosition(0.5);

        this._showSection('result');
    }

    // ===== Comparison Slider =====

    _initComparisonSlider() {
        const container = this.el.comparisonContainer;
        let dragging = false;

        const updateSlider = (e) => {
            const rect = container.getBoundingClientRect();
            let clientX;
            if (e.touches) {
                clientX = e.touches[0].clientX;
            } else {
                clientX = e.clientX;
            }
            const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            this._setSliderPosition(x);
        };

        container.addEventListener('mousedown', (e) => { dragging = true; updateSlider(e); });
        container.addEventListener('touchstart', (e) => { dragging = true; updateSlider(e); }, { passive: true });

        window.addEventListener('mousemove', (e) => { if (dragging) updateSlider(e); });
        window.addEventListener('touchmove', (e) => { if (dragging) updateSlider(e); }, { passive: true });

        window.addEventListener('mouseup', () => { dragging = false; });
        window.addEventListener('touchend', () => { dragging = false; });
    }

    _setSliderPosition(x) {
        const pct = x * 100;
        // Result overlay shows on the LEFT side (clip the right)
        this.el.resultCanvas.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        this.el.sliderLine.style.left = `${pct}%`;
    }

    // ===== Download =====

    _download() {
        if (!this.resultImageData) return;
        const canvas = document.createElement('canvas');
        canvas.width = this.resultImageData.width;
        canvas.height = this.resultImageData.height;
        canvas.getContext('2d').putImageData(this.resultImageData, 0, 0);

        const link = document.createElement('a');
        link.download = `waifu2x_${this.resultImageData.width}x${this.resultImageData.height}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    // ===== Navigation =====

    _showSection(sectionName) {
        this.el.uploadSection.classList.toggle('hidden', sectionName !== 'upload');
        this.el.settingsSection.classList.toggle('hidden', sectionName !== 'settings');
        this.el.progressSection.classList.toggle('hidden', sectionName !== 'progress');
        this.el.resultSection.classList.toggle('hidden', sectionName !== 'result');
    }

    _reset() {
        this.originalImage = null;
        this.originalImageData = null;
        this.resultImageData = null;
        this.el.previewImg.src = '';
        this.el.fileInput.value = '';
        this._updateProgress('準備中...', 0);
        this._showSection('upload');
    }
}

// Initialize
new App();
