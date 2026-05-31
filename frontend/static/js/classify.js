(() => {
  const dropZone      = document.getElementById('drop-zone');
  const fileInput     = document.getElementById('file-input');
  const previewSection = document.getElementById('preview-section');
  const previewImg    = document.getElementById('preview-img');
  const fileNameEl    = document.getElementById('file-name');
  const removeBtn     = document.getElementById('remove-btn');
  const classifyBtn   = document.getElementById('classify-btn');
  const loading       = document.getElementById('loading');
  const resultCard    = document.getElementById('result-card');
  const resultHeader  = document.getElementById('result-header');
  const resultLabel   = document.getElementById('result-label');
  const resultConf    = document.getElementById('result-confidence');
  const probRipePct   = document.getElementById('prob-ripe-pct');
  const probRipeBar   = document.getElementById('prob-ripe-bar');
  const probUnripePct = document.getElementById('prob-unripe-pct');
  const probUnripeBar = document.getElementById('prob-unripe-bar');
  const resetBtn      = document.getElementById('reset-btn');
  const errorBox      = document.getElementById('error-box');
  const errorMsg      = document.getElementById('error-msg');
  const errorResetBtn = document.getElementById('error-reset-btn');

  let selectedFile = null;

  // ── Helpers ────────────────────────────────────────────────────────────────

  function show(el)  { el.classList.remove('hidden'); }
  function hide(el)  { el.classList.add('hidden'); }

  function reset() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    hide(previewSection);
    hide(loading);
    hide(resultCard);
    hide(errorBox);
    show(dropZone);
    dropZone.classList.remove('border-red-400', 'bg-red-50');
    classifyBtn.disabled = false;
  }

  // ── File selection ─────────────────────────────────────────────────────────

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      showError('Please select a valid image file (JPEG, PNG, WEBP, or BMP).');
      return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    fileNameEl.textContent = file.name;

    hide(dropZone);
    hide(errorBox);
    hide(resultCard);
    show(previewSection);
  }

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // Click on zone triggers the hidden input (input is absolutely positioned over zone)
  // The input's opacity-0 handles click; no extra listener needed.

  // ── Drag and drop ──────────────────────────────────────────────────────────

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('border-red-400', 'bg-red-50');
  });

  ['dragleave', 'dragend'].forEach(evt =>
    dropZone.addEventListener(evt, () =>
      dropZone.classList.remove('border-red-400', 'bg-red-50')
    )
  );

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('border-red-400', 'bg-red-50');
    const file = e.dataTransfer.files[0];
    handleFile(file);
  });

  // ── Remove / reset ─────────────────────────────────────────────────────────

  removeBtn.addEventListener('click', reset);
  resetBtn.addEventListener('click', reset);
  errorResetBtn.addEventListener('click', reset);

  // ── Classify ───────────────────────────────────────────────────────────────

  classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    classifyBtn.disabled = true;
    hide(previewSection);
    hide(errorBox);
    show(loading);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.detail || `Server error (${resp.status})`);
      }

      renderResult(data);
    } catch (err) {
      showError(err.message || 'Unexpected error. Please try again.');
    } finally {
      hide(loading);
    }
  });

  // ── Result rendering ───────────────────────────────────────────────────────

  function renderResult(data) {
    const isRipe = data.label_index === 1;

    resultHeader.className = resultHeader.className
      .replace(/bg-\S+/g, '')
      .trim();
    resultHeader.classList.add(isRipe ? 'bg-red-500' : 'bg-green-600');

    resultLabel.textContent     = data.label;
    resultConf.textContent      = `${data.confidence}%`;

    const ripe   = data.probabilities['Ripe']   ?? 0;
    const unripe = data.probabilities['Unripe'] ?? 0;

    probRipePct.textContent   = `${ripe}%`;
    probUnripePct.textContent = `${unripe}%`;

    // Animate bars after a short delay so the transition is visible
    requestAnimationFrame(() => {
      setTimeout(() => {
        probRipeBar.style.width   = `${ripe}%`;
        probUnripeBar.style.width = `${unripe}%`;
      }, 50);
    });

    show(resultCard);
  }

  // ── Error display ──────────────────────────────────────────────────────────

  function showError(msg) {
    hide(loading);
    hide(previewSection);
    hide(resultCard);
    errorMsg.textContent = msg;
    show(errorBox);
    show(dropZone);
    classifyBtn.disabled = false;
  }
})();
