// Home page: pick a photo (browse / drag-drop / paste), send it for a prediction, show the result.
(() => {
  const ALLOWED   = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_RECENT = 6;

  const $ = id => document.getElementById(id);
  const dropzone    = $('dropzone');
  const fileInput   = $('file-input');
  const preview     = $('preview');
  const previewImg  = $('preview-img');
  const fileName    = $('file-name');
  const removeBtn   = $('remove-btn');
  const changeBtn   = $('change-btn');
  const classifyBtn = $('classify-btn');
  const classifyLbl = $('classify-btn-label');
  const errorBox    = $('error-box');
  const errorTitle  = $('error-title');
  const errorMsg    = $('error-msg');
  const errorReset  = $('error-reset-btn');

  const result      = $('result');
  const resultLabel = $('result-text-label');
  const resultText  = $('result-text');
  const confValue   = $('confidence-value');
  const confBar     = $('confidence-bar');
  const confNote    = $('confidence-note');
  const resetBtn    = $('reset-btn');

  const recent      = $('recent');
  const recentList  = $('recent-list');
  const recentClear = $('recent-clear');

  let selectedFile = null;
  let previewUrl   = null;
  let busy         = false;
  const history    = [];

  // ── Helpers ────────────────────────────────────────────────────────────────

  const formatMB = b => `${(b / (1024 * 1024)).toFixed(1)} MB`;

  function showError(title, msg) {
    errorTitle.textContent = title;
    errorMsg.textContent = msg;
    errorBox.hidden = false;
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    previewImg.removeAttribute('src');
    preview.hidden = true;
    dropzone.hidden = false;
    dropzone.classList.remove('is-dragover');
  }

  function resetAll() {
    clearSelection();
    errorBox.hidden = true;
    result.hidden = true;
    dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    dropzone.focus({ preventScroll: true });
  }

  // ── Choosing a photo ───────────────────────────────────────────────────────

  function validate(file) {
    if (!file) return 'No photo was selected.';
    if (!ALLOWED.includes(file.type)) return 'Please use a JPEG, PNG, WEBP or BMP photo.';
    if (file.size > MAX_BYTES) return `That photo is ${formatMB(file.size)} — the limit is 10 MB.`;
    return null;
  }

  function selectFile(file) {
    const problem = validate(file);
    if (problem) { showError('That photo can’t be used', problem); return; }

    errorBox.hidden = true;
    result.hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl   = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    fileName.textContent = file.name || 'Pasted photo';

    dropzone.hidden = true;
    preview.hidden = false;
    classifyBtn.focus({ preventScroll: true });
  }

  fileInput.addEventListener('change', () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); });
  dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  ['dragenter', 'dragover'].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('is-dragover'); })
  );
  ['dragleave', 'dragend', 'drop'].forEach(evt =>
    dropzone.addEventListener(evt, () => dropzone.classList.remove('is-dragover'))
  );
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  // Dropping anywhere on the page also works (and never navigates away).
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => {
    e.preventDefault();
    if (dropzone.contains(e.target)) return;
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file && !busy) selectFile(file);
  });

  document.addEventListener('paste', e => {
    if (busy) return;
    const items = Array.from((e.clipboardData && e.clipboardData.items) || []);
    const item = items.find(i => i.kind === 'file' && i.type.startsWith('image/'));
    if (!item) return;
    const file = item.getAsFile();
    if (file) {
      e.preventDefault();
      const ext = (file.type.split('/')[1] || 'png').replace('jpeg', 'jpg');
      selectFile(new File([file], `pasted-photo.${ext}`, { type: file.type }));
    }
  });

  removeBtn.addEventListener('click', resetAll);
  changeBtn.addEventListener('click', () => fileInput.click());
  resetBtn.addEventListener('click', resetAll);
  errorReset.addEventListener('click', resetAll);

  // ── Checking ───────────────────────────────────────────────────────────────

  async function classify() {
    if (!selectedFile || busy) return;
    busy = true;
    classifyBtn.disabled = true;
    changeBtn.disabled = true;
    classifyLbl.textContent = 'Checking…';
    errorBox.hidden = true;
    result.hidden = true;

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: form });
      let data = null;
      try { data = await resp.json(); } catch (_) { /* non-JSON error body */ }
      if (!resp.ok) {
        const detail = data && typeof data.detail === 'string' ? data.detail : `The server responded with status ${resp.status}.`;
        throw new Error(detail);
      }
      renderResult(data);
      addRecent(data);
    } catch (err) {
      const offline = err instanceof TypeError;
      showError(
        offline ? 'Couldn’t reach TomatoSense' : 'We couldn’t check that photo',
        offline ? 'Check your internet connection and try again.' : (err.message || 'Please try again.')
      );
    } finally {
      busy = false;
      classifyBtn.disabled = false;
      changeBtn.disabled = false;
      classifyLbl.textContent = 'Check ripeness';
    }
  }

  classifyBtn.addEventListener('click', classify);
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && selectedFile && !busy && !preview.hidden && !(e.target instanceof HTMLButtonElement)) classify();
  });

  // ── Result ─────────────────────────────────────────────────────────────────

  const COPY = {
    ripe:   'Red and ready — this one looks good to pick or eat.',
    unripe: 'Still green — give it a few more days on the vine or the counter.',
  };

  function confidenceNote(c) {
    if (c >= 90) return 'Very confident.';
    if (c >= 70) return 'Fairly confident — a clearer photo could help.';
    return 'Not very confident — try better light or a closer shot.';
  }

  function renderResult(data) {
    const isRipe = data.label_index === 1;
    result.className = `card result fade-in ${isRipe ? 'is-ripe' : 'is-unripe'}`;
    resultLabel.textContent = data.label;
    resultText.textContent  = isRipe ? COPY.ripe : COPY.unripe;
    confValue.textContent   = `${Math.round(data.confidence)}%`;
    confNote.textContent    = confidenceNote(data.confidence);
    confBar.style.width     = '0%';
    result.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => { confBar.style.width = `${data.confidence}%`; }));
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ── Recent checks (kept only for this visit) ───────────────────────────────

  function addRecent(data) {
    history.unshift({ label: data.label, isRipe: data.label_index === 1, thumbUrl: URL.createObjectURL(selectedFile) });
    while (history.length > MAX_RECENT) URL.revokeObjectURL(history.pop().thumbUrl);
    renderRecent();
  }

  function renderRecent() {
    recent.hidden = history.length === 0;
    recentList.replaceChildren(...history.map(h => {
      const el = document.createElement('div');
      el.className = 'recent-item';
      const img = document.createElement('img');
      img.className = 'recent-thumb';
      img.src = h.thumbUrl;
      img.alt = `${h.label} tomato`;
      const badge = document.createElement('span');
      badge.className = `badge badge-${h.isRipe ? 'ripe' : 'unripe'}`;
      badge.textContent = h.label;
      el.append(img, badge);
      return el;
    }));
  }

  recentClear.addEventListener('click', () => {
    history.splice(0).forEach(h => URL.revokeObjectURL(h.thumbUrl));
    renderRecent();
  });
})();
