/**
 * LapTrinhAmThanh – Audio Waveform Visualizer (Web Audio API)
 * Canvas-based real-time analyzer with idle / waveform preview modes.
 */

class AudioVisualizer {
  /**
   * @param {string} canvasId - ID of the <canvas> element
   */
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.animationId = null;
    this.audioCtx = null;
    this.analyser = null;
    this.source = null;
    this.dataArray = null;
    this.sampleWaveform = null; // static preview from backend

    if (this.canvas) {
      this._resizeCanvas();
      window.addEventListener('resize', () => this._resizeCanvas(), { passive: true });
      this.drawIdleState();
    }
  }

  _resizeCanvas() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.offsetWidth * (window.devicePixelRatio || 1);
    this.canvas.height = this.canvas.offsetHeight * (window.devicePixelRatio || 1);
    this.ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  }

  /** Draw idle "breathing" placeholder bars */
  drawIdleState() {
    if (!this.ctx) return;
    this._cancelAnimation();
    const w = this.canvas.offsetWidth;
    const h = this.canvas.offsetHeight;

    const draw = () => {
      this.ctx.clearRect(0, 0, w, h);
      const barCount = 48;
      const barW = (w - barCount * 2) / barCount;
      const now = performance.now() / 1000;

      for (let i = 0; i < barCount; i++) {
        const phase = (i / barCount) * Math.PI * 2;
        const height = Math.max(4, (Math.sin(now * 1.4 + phase) * 0.5 + 0.5) * h * 0.55);
        const x = i * (barW + 2);
        const y = (h - height) / 2;

        const grad = this.ctx.createLinearGradient(0, y, 0, y + height);
        grad.addColorStop(0, 'rgba(99,102,241,0.7)');
        grad.addColorStop(1, 'rgba(168,85,247,0.2)');
        this.ctx.fillStyle = grad;
        this._roundRect(x, y, barW, height, 2);
        this.ctx.fill();
      }
      this.animationId = requestAnimationFrame(draw);
    };
    draw();
  }

  /**
   * Start real-time visualization from an HTMLAudioElement.
   * @param {HTMLAudioElement} audioEl
   */
  startVisualization(audioEl) {
    if (!this.ctx) return;
    this._cancelAnimation();
    this._teardownAudio();

    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.78;
      this.source = this.audioCtx.createMediaElementSource(audioEl);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this._drawLive();
    } catch (err) {
      console.warn('[Visualizer] Web Audio API error:', err);
      this.drawIdleState();
    }
  }

  _drawLive() {
    const w = this.canvas.offsetWidth;
    const h = this.canvas.offsetHeight;

    const draw = () => {
      this.animationId = requestAnimationFrame(draw);
      if (!this.analyser) return;

      this.analyser.getByteFrequencyData(this.dataArray);
      this.ctx.clearRect(0, 0, w, h);

      const barCount = this.dataArray.length;
      const barW = (w / barCount) - 1.5;

      for (let i = 0; i < barCount; i++) {
        const value = this.dataArray[i] / 255;
        const height = Math.max(3, value * h);
        const x = i * (barW + 1.5);
        const y = h - height;

        const hue = 240 + value * 120; // indigo → magenta
        const grad = this.ctx.createLinearGradient(0, y, 0, h);
        grad.addColorStop(0, `hsla(${hue}, 90%, 70%, 0.95)`);
        grad.addColorStop(1, `hsla(${hue}, 70%, 45%, 0.3)`);
        this.ctx.fillStyle = grad;
        this._roundRect(x, y, barW, height, 2);
        this.ctx.fill();
      }
    };
    draw();
  }

  /** Stop real-time visualization and go back to idle */
  stopVisualization() {
    this._cancelAnimation();
    this._teardownAudio();
    this.drawIdleState();
  }

  /**
   * Draw a static waveform preview (from backend waveform_preview array).
   * @param {number[]} samples - Normalized amplitude values [-1,1] or [0,1]
   */
  setSampleWaveform(samples) {
    if (!this.ctx || !samples || samples.length === 0) return;
    this.sampleWaveform = samples;
    this._cancelAnimation();

    const w = this.canvas.offsetWidth;
    const h = this.canvas.offsetHeight;
    this.ctx.clearRect(0, 0, w, h);

    const step = w / samples.length;
    const midY = h / 2;

    this.ctx.beginPath();
    this.ctx.moveTo(0, midY);

    for (let i = 0; i < samples.length; i++) {
      const x = i * step;
      const amp = Math.abs(samples[i]);
      const barH = amp * midY * 0.9;
      this.ctx.lineTo(x, midY - barH);
    }
    for (let i = samples.length - 1; i >= 0; i--) {
      const x = i * step;
      const amp = Math.abs(samples[i]);
      const barH = amp * midY * 0.9;
      this.ctx.lineTo(x, midY + barH);
    }

    this.ctx.closePath();
    const grad = this.ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, 'rgba(99,102,241,0.7)');
    grad.addColorStop(0.5, 'rgba(168,85,247,0.85)');
    grad.addColorStop(1, 'rgba(236,72,153,0.7)');
    this.ctx.fillStyle = grad;
    this.ctx.fill();
  }

  _cancelAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  _teardownAudio() {
    try {
      if (this.source) { this.source.disconnect(); this.source = null; }
      if (this.analyser) { this.analyser.disconnect(); this.analyser = null; }
      if (this.audioCtx && this.audioCtx.state !== 'closed') {
        this.audioCtx.close();
        this.audioCtx = null;
      }
    } catch (_) { /* ignore */ }
    this.dataArray = null;
  }

  /** Helper: draw rounded rectangle path */
  _roundRect(x, y, w, h, r) {
    if (!this.ctx.roundRect) {
      this.ctx.beginPath();
      this.ctx.moveTo(x + r, y);
      this.ctx.lineTo(x + w - r, y);
      this.ctx.arcTo(x + w, y, x + w, y + r, r);
      this.ctx.lineTo(x + w, y + h - r);
      this.ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      this.ctx.lineTo(x + r, y + h);
      this.ctx.arcTo(x, y + h, x, y + h - r, r);
      this.ctx.lineTo(x, y + r);
      this.ctx.arcTo(x, y, x + r, y, r);
      this.ctx.closePath();
    } else {
      this.ctx.beginPath();
      this.ctx.roundRect(x, y, w, h, r);
    }
  }
}
