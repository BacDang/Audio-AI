/**
 * LapTrinhAmThanh – Main App Controller v2.0
 * Handles: Tab navigation, File upload, Demo samples,
 *          Audio player, API calls, Render results,
 *          Genre explorer, History & stats, Toasts.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── STATE ─────────────────────────────────────────────────────────
  let currentFile       = null;   // File object from user upload
  let currentDemo       = null;   // Demo sample object { genre, filename, audio_url }
  let audioPlayer       = new Audio();
  let visualizer        = null;
  let isAnalyzing       = false;
  let audioSourceBound  = false;  // prevent double AudioContext binding

  // ── DOM REFS ──────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const dropZone     = $('dropZone');
  const fileInput    = $('fileInput');
  const fileBadge    = $('fileBadge');
  const fileNameDisp = $('fileNameDisplay');
  const btnRemove    = $('btnRemoveFile');
  const demoContainer= $('demoChipsContainer');
  const demoCount    = $('demoCount');

  const playerSection  = $('playerSection');
  const btnPlayPause   = $('btnPlayPause');
  const iconPlay       = $('iconPlay');
  const iconPause      = $('iconPause');
  const progressSlider = $('progressSlider');
  const timeCurrent    = $('timeCurrent');
  const timeTotal      = $('timeTotal');

  const btnAnalyze      = $('btnAnalyze');
  const btnSpinner      = $('btnSpinner');
  const btnAnalyzeText  = $('btnAnalyzeText');

  const resultsPlaceholder = $('resultsPlaceholder');
  const resultsContainer   = $('resultsContainer');
  const heroGenreCard      = $('heroGenreCard');
  const heroBgGlow         = $('heroBgGlow');
  const heroConfidenceBadge= $('heroConfidenceBadge');
  const heroGenreIcon      = $('heroGenreIcon');
  const heroGenreName      = $('heroGenreName');
  const heroGenreNameVi    = $('heroGenreNameVi');
  const heroGenreDesc      = $('heroGenreDesc');
  const heroSampleArtists  = $('heroSampleArtists');
  const probBarsList       = $('probBarsList');

  const metricTempo      = $('metricTempo');
  const metricRms        = $('metricRms');
  const metricCentroid   = $('metricCentroid');
  const metricHarmonic   = $('metricHarmonic');
  const metricPercussive = $('metricPercussive');
  const metricRolloff    = $('metricRolloff');
  const processingTime   = $('processingTime');

  const navBtns    = document.querySelectorAll('.nav-btn');
  const tabPanes   = document.querySelectorAll('.tab-pane');

  const genresGrid        = $('genresGrid');
  const historyTableBody  = $('historyTableBody');
  const statTotal         = $('statTotal');
  const statConfidence    = $('statConfidence');
  const statBpm           = $('statBpm');
  const genreDistBars     = $('genreDistBars');
  const btnRefreshHistory = $('btnRefreshHistory');

  // ── VISUALIZER INIT ───────────────────────────────────────────────
  visualizer = new AudioVisualizer('visualizerCanvas');

  // ══════════════════════════════════════════════════════════════════
  //  TAB NAVIGATION
  // ══════════════════════════════════════════════════════════════════
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      navBtns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
      tabPanes.forEach(p => { p.hidden = true; p.classList.remove('active'); });

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const pane = $(`tab-${target}`);
      if (pane) { pane.hidden = false; pane.classList.add('active'); }

      if (target === 'genres')  loadGenresExplorer();
      if (target === 'history') loadHistoryAndStats();
    });
  });

  // ══════════════════════════════════════════════════════════════════
  //  FILE UPLOAD & DRAG-DROP
  // ══════════════════════════════════════════════════════════════════
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

  ['dragenter', 'dragover'].forEach(evt => dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  }));
  dropZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) handleFile(f);
  });

  btnRemove.addEventListener('click', (e) => { e.stopPropagation(); resetAudio(); });

  function handleFile(file) {
    const validExts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'];
    if (!validExts.some(ext => file.name.toLowerCase().endsWith(ext))) {
      showToast('Định dạng không được hỗ trợ. Vui lòng chọn: WAV, MP3, OGG, FLAC', 'error');
      return;
    }
    currentFile = file;
    currentDemo = null;
    clearDemoChips();

    fileNameDisp.textContent = `${file.name}  (${(file.size / 1048576).toFixed(2)} MB)`;
    fileBadge.classList.add('show');
    btnAnalyze.disabled = false;

    setupPlayer(URL.createObjectURL(file));
    showToast(`Đã chọn: ${file.name}`, 'success');
  }

  function resetAudio() {
    currentFile = null;
    currentDemo = null;
    fileInput.value = '';
    fileBadge.classList.remove('show');
    btnAnalyze.disabled = true;
    clearDemoChips();
    stopPlayer();
    playerSection.hidden = true;
    audioSourceBound = false;
    visualizer.drawIdleState();
  }

  // ══════════════════════════════════════════════════════════════════
  //  DEMO SAMPLES
  // ══════════════════════════════════════════════════════════════════
  async function loadDemoSamples() {
    try {
      const res = await fetch('/api/demo-samples');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const samples = data.samples || [];

      demoCount.textContent = `${samples.length} file`;
      demoContainer.innerHTML = '';

      // Show at most 1 sample per genre (10 chips max)
      const seen = new Set();
      samples.filter(s => { if (seen.has(s.genre)) return false; seen.add(s.genre); return true; })
             .forEach(s => {
               const chip = document.createElement('button');
               chip.className = 'demo-chip';
               chip.type = 'button';
               chip.role = 'listitem';
               chip.textContent = `${s.icon} ${s.genre.toUpperCase()}`;
               chip.title = s.filename;
               chip.addEventListener('click', () => selectDemo(s, chip));
               demoContainer.appendChild(chip);
             });
    } catch (e) {
      demoCount.textContent = 'Không có mẫu';
      console.warn('[Demo] Lỗi tải mẫu:', e);
    }
  }

  function selectDemo(sample, chip) {
    currentDemo = sample;
    currentFile = null;
    fileInput.value = '';
    clearDemoChips();

    chip.classList.add('active');
    fileNameDisp.textContent = `${sample.icon} Demo: ${sample.genre.toUpperCase()} – ${sample.filename}`;
    fileBadge.classList.add('show');
    btnAnalyze.disabled = false;

    setupPlayer(sample.audio_url);
    showToast(`Đã chọn mẫu: ${sample.genre.toUpperCase()} (${sample.filename})`, 'success');
  }

  function clearDemoChips() {
    document.querySelectorAll('.demo-chip').forEach(c => c.classList.remove('active'));
  }

  // ══════════════════════════════════════════════════════════════════
  //  AUDIO PLAYER
  // ══════════════════════════════════════════════════════════════════
  function setupPlayer(src) {
    stopPlayer();
    audioSourceBound = false; // allow re-binding AudioContext on next play
    audioPlayer = new Audio(src);
    playerSection.hidden = false;
    progressSlider.value = 0;
    timeCurrent.textContent = '0:00';
    timeTotal.textContent = '0:00';

    audioPlayer.addEventListener('loadedmetadata', () => {
      timeTotal.textContent = fmtTime(audioPlayer.duration);
    });
    audioPlayer.addEventListener('timeupdate', () => {
      if (!audioPlayer.duration) return;
      progressSlider.value = (audioPlayer.currentTime / audioPlayer.duration) * 100;
      timeCurrent.textContent = fmtTime(audioPlayer.currentTime);
    });
    audioPlayer.addEventListener('ended', stopPlayer);
    audioPlayer.addEventListener('error', () => showToast('Không thể phát file âm thanh này', 'warning'));
    visualizer.drawIdleState();
  }

  btnPlayPause.addEventListener('click', () => {
    if (!audioPlayer.src) return;
    if (audioPlayer.paused) {
      audioPlayer.play().then(() => {
        iconPlay.hidden = true;
        iconPause.hidden = false;
        if (!audioSourceBound) {
          visualizer.startVisualization(audioPlayer);
          audioSourceBound = true;
        }
      }).catch(() => showToast('Không thể phát. Hãy thử file khác.', 'error'));
    } else {
      audioPlayer.pause();
      iconPlay.hidden = false;
      iconPause.hidden = true;
      visualizer.stopVisualization();
    }
  });

  progressSlider.addEventListener('input', () => {
    if (audioPlayer.duration) {
      audioPlayer.currentTime = (progressSlider.value / 100) * audioPlayer.duration;
    }
  });

  function stopPlayer() {
    if (audioPlayer) { audioPlayer.pause(); audioPlayer.currentTime = 0; }
    iconPlay.hidden = false;
    iconPause.hidden = true;
    if (visualizer) visualizer.stopVisualization();
  }

  function fmtTime(s) {
    if (!isFinite(s) || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec < 10 ? '0' : ''}${sec}`;
  }

  // ══════════════════════════════════════════════════════════════════
  //  ANALYZE / PREDICT
  // ══════════════════════════════════════════════════════════════════
  btnAnalyze.addEventListener('click', async () => {
    if (isAnalyzing) return;
    if (!currentFile && !currentDemo) {
      showToast('Vui lòng chọn hoặc tải lên file nhạc trước.', 'warning');
      return;
    }

    setAnalyzing(true);
    try {
      let data = null;

      if (currentFile) {
        const form = new FormData();
        form.append('file', currentFile);
        const res = await fetch('/api/predict', { method: 'POST', body: form });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Lỗi phân tích'); }
        data = await res.json();

      } else if (currentDemo) {
        const qs = `genre=${encodeURIComponent(currentDemo.genre)}&filename=${encodeURIComponent(currentDemo.filename)}`;
        const res = await fetch(`/api/predict-demo?${qs}`, { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Lỗi phân tích mẫu'); }
        data = await res.json();
      }

      if (data && data.success) {
        renderResults(data);
        showToast(`✅ Kết quả: ${data.prediction.predicted_genre.toUpperCase()} – ${data.prediction.confidence_percentage}%`, 'success');
      }
    } catch (err) {
      console.error('[Predict]', err);
      showToast(`Lỗi: ${err.message}`, 'error');
    } finally {
      setAnalyzing(false);
    }
  });

  function setAnalyzing(on) {
    isAnalyzing = on;
    btnAnalyze.disabled = on;
    btnAnalyze.setAttribute('aria-busy', on);
    btnSpinner.hidden = !on;
    btnAnalyzeText.textContent = on
      ? 'Đang phân tích 60 đặc trưng AI...'
      : '🔮 Bắt Đầu Phân Tích & Nhận Diện';
  }

  // ══════════════════════════════════════════════════════════════════
  //  RENDER RESULTS
  // ══════════════════════════════════════════════════════════════════
  function renderResults(data) {
    const pred    = data.prediction;
    const meta    = pred.genre_meta || {};
    const metrics = data.metrics    || {};

    resultsPlaceholder.hidden = true;
    resultsContainer.hidden = false;

    // ── Hero Card ──
    const color    = meta.color    || '#6366f1';
    const gradient = meta.gradient || 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)';
    heroGenreCard.style.setProperty('--genre-color', color);
    heroGenreCard.style.setProperty('--genre-grad', gradient);
    heroGenreCard.style.setProperty('--genre-glow', hexToRgba(color, 0.2));
    heroBgGlow.style.background = `radial-gradient(ellipse at 70% 10%, ${hexToRgba(color, 0.22)} 0%, transparent 65%)`;
    heroGenreIcon.textContent    = meta.icon || '🎵';
    heroGenreIcon.style.background = gradient;
    heroGenreName.textContent    = pred.predicted_genre.toUpperCase();
    heroGenreNameVi.textContent  = pred.predicted_genre_vi || '';
    heroGenreDesc.textContent    = meta.description || '...';
    heroGenreDesc.style.borderLeftColor = color;
    heroSampleArtists.textContent = meta.sample_artists ? `${meta.sample_artists}` : '';
    heroSampleArtists.style.display = meta.sample_artists ? 'block' : 'none';

    const lvlColor = pred.confidence_level === 'Cao' ? '#10b981'
                   : pred.confidence_level === 'Trung bình' ? '#f59e0b' : '#ef4444';
    heroConfidenceBadge.textContent = `⚡ ${pred.confidence_percentage}% – ${pred.confidence_level}`;
    heroConfidenceBadge.style.background   = hexToRgba(lvlColor, 0.12);
    heroConfidenceBadge.style.borderColor  = hexToRgba(lvlColor, 0.35);
    heroConfidenceBadge.style.color        = lvlColor;

    // ── Probability Bars ──
    probBarsList.innerHTML = '';
    (pred.probabilities || []).forEach(p => {
      const pMeta = p.meta || {};
      const row = document.createElement('div');
      row.className = 'prob-item';
      const isTop = p.genre === pred.predicted_genre;
      row.innerHTML = `
        <div class="prob-name">
          <span>${pMeta.icon || '🎵'}</span>
          <span style="${isTop ? `color:${pMeta.color || 'var(--accent)'};font-weight:800;` : ''}">${p.genre}</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" data-target="${Math.max(2, p.probability)}" style="background:${pMeta.gradient || 'var(--accent-grad)'}"></div>
        </div>
        <div class="prob-pct">${p.probability.toFixed(1)}%</div>`;
      probBarsList.appendChild(row);
      // Trigger CSS transition
      requestAnimationFrame(() => {
        const fill = row.querySelector('.prob-fill');
        if (fill) fill.style.width = `${Math.max(2, p.probability)}%`;
      });
    });

    // ── Acoustic Metrics ──
    metricTempo.textContent      = metrics.tempo_bpm ? `${metrics.tempo_bpm} BPM` : '--';
    metricRms.textContent        = metrics.rms_energy != null ? Number(metrics.rms_energy).toFixed(4) : '--';
    metricCentroid.textContent   = metrics.spectral_centroid_hz ? `${Math.round(metrics.spectral_centroid_hz)} Hz` : '--';
    metricHarmonic.textContent   = metrics.harmonic_ratio != null ? `${metrics.harmonic_ratio}%` : '--';
    metricPercussive.textContent = metrics.percussive_ratio != null ? `${metrics.percussive_ratio}%` : '--';
    metricRolloff.textContent    = metrics.spectral_rolloff_hz ? `${Math.round(metrics.spectral_rolloff_hz)} Hz` : '--';

    if (processingTime) processingTime.textContent = `⏱️ Thời gian xử lý: ${data.processing_time_ms} ms`;

    // ── Waveform Preview ──
    if (metrics.waveform_preview && visualizer) {
      visualizer.setSampleWaveform(metrics.waveform_preview);
    }
  }

  // Hex to rgba helper
  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16);
    const g = parseInt(hex.slice(3,5),16);
    const b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ══════════════════════════════════════════════════════════════════
  //  GENRE EXPLORER
  // ══════════════════════════════════════════════════════════════════
  let genresLoaded = false;
  async function loadGenresExplorer() {
    if (genresLoaded) return;
    try {
      const res = await fetch('/api/genres');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      renderGenres(data.genres || []);
      genresLoaded = true;
    } catch (e) {
      genresGrid.innerHTML = '<p style="color:var(--text-3);text-align:center;padding:40px;">Không thể tải thông tin thể loại.</p>';
    }
  }

  function renderGenres(genres) {
    genresGrid.innerHTML = '';
    genres.forEach(g => {
      const card = document.createElement('article');
      card.className = 'genre-card';
      card.style.setProperty('--genre-color', g.color || '#6366f1');
      card.innerHTML = `
        <div class="genre-card-header">
          <div class="genre-card-icon" style="background:${g.gradient}">${g.icon}</div>
          <div>
            <div class="genre-card-name" style="color:${g.color}">${g.id}</div>
            <div class="genre-card-vi">${g.name_vi}</div>
          </div>
        </div>
        <p class="genre-card-desc">${g.description}</p>
        <div class="genre-card-features">${g.key_features}</div>
        <div class="genre-card-artists">🎤 ${g.sample_artists}</div>`;
      genresGrid.appendChild(card);
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  HISTORY & ANALYTICS
  // ══════════════════════════════════════════════════════════════════
  btnRefreshHistory.addEventListener('click', () => loadHistoryAndStats(true));

  async function loadHistoryAndStats(force = false) {
    try {
      const [histRes, statsRes] = await Promise.all([
        fetch('/api/history?limit=100'),
        fetch('/api/stats')
      ]);
      const histData  = await histRes.json();
      const statsData = await statsRes.json();

      // Stats numbers
      statTotal.textContent      = statsData.total_predictions ?? 0;
      statConfidence.textContent = `${statsData.avg_confidence ?? 0}%`;
      statBpm.textContent        = `${statsData.avg_bpm ?? 0}`;

      // Genre distribution bars
      renderGenreDistribution(statsData.genre_distribution || {}, statsData.total_predictions || 1);

      // History table
      renderHistoryTable(histData.history || []);

    } catch (e) {
      console.warn('[History]', e);
      historyTableBody.innerHTML = '<tr><td colspan="8" class="empty-cell">Lỗi tải dữ liệu. Vui lòng thử lại.</td></tr>';
    }
  }

  function renderGenreDistribution(dist, total) {
    if (!genreDistBars) return;
    const entries = Object.entries(dist).sort(([,a],[,b]) => b - a);
    if (entries.length === 0) {
      genreDistBars.innerHTML = '<div style="color:var(--text-3);font-size:13px;">Chưa có dữ liệu thể loại.</div>';
      return;
    }
    genreDistBars.innerHTML = '';
    entries.forEach(([genre, count]) => {
      const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
      const row = document.createElement('div');
      row.className = 'dist-bar-item';
      row.innerHTML = `
        <span>${genre}</span>
        <div class="dist-bar-track"><div class="dist-bar-fill" data-w="${pct}"></div></div>
        <span class="dist-bar-count">${count}</span>`;
      genreDistBars.appendChild(row);
    });
    // Animate bars
    requestAnimationFrame(() => {
      genreDistBars.querySelectorAll('.dist-bar-fill').forEach(el => {
        el.style.width = `${el.dataset.w}%`;
      });
    });
  }

  function renderHistoryTable(items) {
    historyTableBody.innerHTML = '';
    if (items.length === 0) {
      historyTableBody.innerHTML = '<tr><td colspan="8" class="empty-cell">Chưa có lượt phân tích nào. Thử tải lên một bài hát!</td></tr>';
      return;
    }
    items.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>#${item.id}</strong></td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${item.filename}">${item.filename}</td>
        <td><span class="badge" style="background:rgba(99,102,241,.15);color:#818cf8;border:1px solid rgba(99,102,241,.3);">${item.predicted_genre}</span></td>
        <td><strong>${item.confidence_percentage}%</strong></td>
        <td>${item.tempo_bpm} BPM</td>
        <td style="font-size:12px;">${item.processing_time_ms ? item.processing_time_ms + ' ms' : '--'}</td>
        <td style="font-size:11px;color:var(--text-3);">${item.created_at}</td>
        <td><button class="btn-delete" data-id="${item.id}" title="Xóa bản ghi" aria-label="Xóa bản ghi #${item.id}">🗑️</button></td>`;

      tr.querySelector('.btn-delete').addEventListener('click', async () => {
        if (!confirm('Bạn có chắc muốn xóa bản ghi này?')) return;
        try {
          const res = await fetch(`/api/history/${item.id}`, { method: 'DELETE' });
          if (res.ok) { showToast('Đã xóa bản ghi.', 'success'); loadHistoryAndStats(true); }
          else showToast('Không thể xóa bản ghi.', 'error');
        } catch { showToast('Lỗi mạng khi xóa.', 'error'); }
      });
      historyTableBody.appendChild(tr);
    });
  }

  // ══════════════════════════════════════════════════════════════════
  //  TOAST NOTIFICATIONS
  // ══════════════════════════════════════════════════════════════════
  function showToast(message, type = 'info') {
    const container = $('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = { success:'✅', error:'❌', warning:'⚠️', info:'ℹ️' }[type] || 'ℹ️';
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('fade-out');
      setTimeout(() => toast.remove(), 320);
    }, 4500);
  }

  // ══════════════════════════════════════════════════════════════════
  //  INIT
  // ══════════════════════════════════════════════════════════════════
  loadDemoSamples();

  // Show health status briefly
  fetch('/api/health')
    .then(r => r.json())
    .then(data => {
      if (!data.model_loaded) {
        showToast('⚠️ Mô hình AI chưa nạp được. Kiểm tra file .pkl trong thư mục gốc.', 'warning');
      }
    })
    .catch(() => {});

}); // DOMContentLoaded
