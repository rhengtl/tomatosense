// Home page: pick one or many photos, shrink them in the browser, send for prediction, show results.
(() => {
  const ALLOWED    = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  const MAX_BYTES  = 10 * 1024 * 1024;           // server limit per image
  const MAX_BATCH  = Number(document.getElementById('upload-card').dataset.maxBatch) || 20;
  const MAX_EDGE   = 1280;                        // longest edge after shrinking (model uses 64×64)
  const SHRINK_MIN = 600 * 1024;                  // photos smaller than this are sent as-is
  const JPEG_Q     = 0.85;

  const $ = id => document.getElementById(id);
  const dropzone     = $('dropzone');
  const fileInput    = $('file-input');
  const preparing    = $('preparing');
  const preparingTxt = $('preparing-text');
  const preview      = $('preview');
  const previewImg   = $('preview-img');
  const fileName     = $('file-name');
  const removeBtn    = $('remove-btn');
  const batchPreview = $('batch-preview');
  const batchCount   = $('batch-count');
  const thumbGrid    = $('thumb-grid');
  const addMoreBtn   = $('add-more-btn');
  const actions      = $('actions');
  const classifyBtn  = $('classify-btn');
  const classifyLbl  = $('classify-btn-label');
  const changeBtn    = $('change-btn');
  const progress     = $('progress');
  const progressText = $('progress-text');
  const progressPct  = $('progress-pct');
  const progressBar  = $('progress-bar');
  const notice       = $('notice');
  const noticeMsg    = $('notice-msg');
  const errorBox     = $('error-box');
  const errorTitle   = $('error-title');
  const errorMsg     = $('error-msg');
  const errorReset   = $('error-reset-btn');

  const result       = $('result');
  const resultLabel  = $('result-text-label');
  const resultText   = $('result-text');
  const confValue    = $('confidence-value');
  const confBar      = $('confidence-bar');
  const confNote     = $('confidence-note');

  const batchResult  = $('batch-result');
  const sumRipe      = $('sum-ripe');
  const sumUnripe    = $('sum-unripe');
  const sumFailedWrap= $('sum-failed-wrap');
  const sumFailed    = $('sum-failed');
  const summaryText  = $('summary-text');
  const resultGrid   = $('result-grid');

  /** @type {{ id:number, file:File, url:string }[]} */
  let items = [];
  let nextId = 1;
  let busy = false;
  let adding = false;

  // ── Small helpers ──────────────────────────────────────────────────────────

  const formatMB = b => `${(b / (1024 * 1024)).toFixed(1)} MB`;
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

  function show(el) { el.hidden = false; }
  function hide(el) { el.hidden = true; }

  function showNotice(msg) { noticeMsg.textContent = msg; show(notice); }
  function showError(title, msg) { errorTitle.textContent = title; errorMsg.textContent = msg; show(errorBox); }

  function setProgress(text, pct) {
    progressText.textContent = text;
    progressPct.textContent = pct == null ? '' : `${Math.round(pct)}%`;
    progressBar.style.width = `${pct == null ? 100 : pct}%`;
    progressBar.classList.toggle('is-indeterminate', pct == null);
  }

  // ── Shrinking photos before upload ─────────────────────────────────────────
  // Phone photos are several MB; the model only needs a tiny version. Resizing in the
  // browser keeps uploads fast and keeps batches well under the server limits.

  async function shrink(file) {
    if (file.size <= SHRINK_MIN) return file;
    try {
      const bitmap = await createImageBitmap(file);
      const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
      const w = Math.max(1, Math.round(bitmap.width * scale));
      const h = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#fff';               // flatten transparency to white
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(bitmap, 0, 0, w, h);
      bitmap.close();
      const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', JPEG_Q));
      if (!blob || blob.size >= file.size) return file;
      return new File([blob], file.name.replace(/\.[^.]+$/, '') + '.jpg', { type: 'image/jpeg' });
    } catch (_) {
      return file; // browser couldn't decode it; let the server decide
    }
  }

  // ── Adding / removing photos ───────────────────────────────────────────────

  async function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (!incoming.length || busy || adding) return;
    adding = true;
    hide(errorBox); hide(notice); hide(result); hide(batchResult);

    const problems = [];
    let accepted = incoming.filter(f => {
      if (!ALLOWED.includes(f.type)) { problems.push(`${f.name || 'One file'} isn't a JPEG, PNG, WEBP or BMP photo.`); return false; }
      return true;
    });

    const room = MAX_BATCH - items.length;
    if (accepted.length > room) {
      problems.push(room > 0
        ? `You can check up to ${MAX_BATCH} photos at a time — we kept the first ${room}.`
        : `You can check up to ${MAX_BATCH} photos at a time.`);
      accepted = accepted.slice(0, Math.max(0, room));
    }

    if (accepted.length) {
      hide(dropzone); hide(preview); hide(batchPreview); hide(actions);
      show(preparing);
      for (let i = 0; i < accepted.length; i++) {
        preparingTxt.textContent = accepted.length > 1 ? `Preparing photo ${i + 1} of ${accepted.length}…` : 'Preparing photo…';
        const file = await shrink(accepted[i]);
        if (file.size > MAX_BYTES) { problems.push(`${file.name} is ${formatMB(file.size)} — the limit is 10 MB.`); continue; }
        items.push({ id: nextId++, file, url: URL.createObjectURL(file) });
      }
      hide(preparing);
    }

    if (problems.length) showNotice(problems.join(' '));
    fileInput.value = '';
    adding = false;
    render();
  }

  function removeItem(id) {
    const idx = items.findIndex(i => i.id === id);
    if (idx === -1) return;
    URL.revokeObjectURL(items[idx].url);
    items.splice(idx, 1);
    render();
  }

  function resetAll() {
    items.forEach(i => URL.revokeObjectURL(i.url));
    items = [];
    fileInput.value = '';
    hide(errorBox); hide(notice); hide(result); hide(batchResult); hide(progress);
    render();
    dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    dropzone.focus({ preventScroll: true });
  }

  // ── Rendering the selection ────────────────────────────────────────────────

  function render() {
    const n = items.length;
    dropzone.hidden = n > 0;
    preview.hidden = n !== 1;
    batchPreview.hidden = n < 2;
    actions.hidden = n === 0;

    if (n === 1) {
      previewImg.src = items[0].url;
      fileName.textContent = items[0].file.name || 'Pasted photo';
      classifyLbl.textContent = 'Check ripeness';
    } else if (n > 1) {
      batchCount.textContent = `${plural(n, 'photo')} selected`;
      addMoreBtn.hidden = n >= MAX_BATCH;
      classifyLbl.textContent = `Check ${n} tomatoes`;
      thumbGrid.replaceChildren(...items.map(item => {
        const el = document.createElement('div');
        el.className = 'thumb';
        const img = document.createElement('img');
        img.src = item.url; img.alt = item.file.name;
        const btn = document.createElement('button');
        btn.type = 'button'; btn.className = 'thumb-remove'; btn.title = 'Remove';
        btn.setAttribute('aria-label', `Remove ${item.file.name}`);
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
        btn.addEventListener('click', () => removeItem(item.id));
        el.append(img, btn);
        return el;
      }));
    }
    if (n > 0) classifyBtn.focus({ preventScroll: true });
  }

  // ── Input sources: browse, drag-drop, paste ────────────────────────────────

  fileInput.addEventListener('change', () => addFiles(fileInput.files));
  dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  addMoreBtn.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('is-dragover'); })
  );
  ['dragleave', 'dragend', 'drop'].forEach(evt =>
    dropzone.addEventListener(evt, () => dropzone.classList.remove('is-dragover'))
  );
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });

  document.addEventListener('paste', e => {
    const files = Array.from((e.clipboardData && e.clipboardData.items) || [])
      .filter(i => i.kind === 'file' && i.type.startsWith('image/'))
      .map(i => i.getAsFile()).filter(Boolean)
      .map(f => new File([f], `pasted-photo.${(f.type.split('/')[1] || 'png').replace('jpeg', 'jpg')}`, { type: f.type }));
    if (files.length) { e.preventDefault(); addFiles(files); }
  });

  removeBtn.addEventListener('click', resetAll);
  changeBtn.addEventListener('click', resetAll);
  errorReset.addEventListener('click', resetAll);
  document.querySelectorAll('.reset-btn').forEach(b => b.addEventListener('click', resetAll));

  // ── Sending ────────────────────────────────────────────────────────────────

  function upload(files) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      files.forEach(f => form.append('files', f));
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/predict-batch');
      xhr.responseType = 'json';
      xhr.upload.onprogress = e => {
        if (!e.lengthComputable) return;
        const pct = (e.loaded / e.total) * 100;
        setProgress(pct < 100 ? 'Uploading…' : 'Checking…', pct < 100 ? pct : null);
      };
      xhr.onload = () => {
        const body = xhr.response;
        if (xhr.status >= 200 && xhr.status < 300 && body) return resolve(body);
        const detail = body && typeof body.detail === 'string' ? body.detail : `The server responded with status ${xhr.status}.`;
        reject(new Error(detail));
      };
      xhr.onerror = () => reject(new TypeError('network'));
      xhr.send(form);
    });
  }

  async function classify() {
    if (!items.length || busy || adding) return;
    busy = true;
    classifyBtn.disabled = true;
    changeBtn.disabled = true;
    hide(errorBox); hide(notice); hide(result); hide(batchResult);
    setProgress('Uploading…', 0);
    show(progress);

    try {
      const data = await upload(items.map(i => i.file));
      if (items.length === 1) renderSingle(data.results[0]);
      else renderBatch(data);
    } catch (err) {
      const offline = err instanceof TypeError;
      showError(
        offline ? 'Couldn’t reach TomatoSense' : 'We couldn’t check those photos',
        offline ? 'Check your internet connection and try again.' : (err.message || 'Please try again.')
      );
    } finally {
      hide(progress);
      busy = false;
      classifyBtn.disabled = false;
      changeBtn.disabled = false;
    }
  }

  classifyBtn.addEventListener('click', classify);
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && items.length && !busy && !adding && !(e.target instanceof HTMLButtonElement)) classify();
  });

  // ── Results ────────────────────────────────────────────────────────────────

  const COPY = {
    ripe:   'Red and ready — this one looks good to pick or eat.',
    unripe: 'Still green — give it a few more days on the vine or the counter.',
  };

  function confidenceNote(c) {
    if (c >= 90) return 'Very confident.';
    if (c >= 70) return 'Fairly confident — a clearer photo could help.';
    return 'Not very confident — try better light or a closer shot.';
  }

  function renderSingle(r) {
    if (!r.ok) { showError('We couldn’t check that photo', r.error || 'Please try another photo.'); return; }
    const isRipe = r.label_index === 1;
    result.className = `card result fade-in ${isRipe ? 'is-ripe' : 'is-unripe'}`;
    resultLabel.textContent = r.label;
    resultText.textContent  = isRipe ? COPY.ripe : COPY.unripe;
    confValue.textContent   = `${Math.round(r.confidence)}%`;
    confNote.textContent    = confidenceNote(r.confidence);
    confBar.style.width     = '0%';
    show(result);
    requestAnimationFrame(() => requestAnimationFrame(() => { confBar.style.width = `${r.confidence}%`; }));
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderBatch(data) {
    sumRipe.textContent   = data.ripe_count;
    sumUnripe.textContent = data.unripe_count;
    sumFailed.textContent = data.failed_count;
    sumFailedWrap.hidden  = data.failed_count === 0;

    const checked = data.count - data.failed_count;
    summaryText.textContent =
      checked === 0 ? 'None of the photos could be checked — see the notes below.' :
      data.ripe_count === checked ? `All ${checked} look ripe and ready.` :
      data.unripe_count === checked ? `All ${checked} are still green.` :
      `${data.ripe_count} of ${checked} ${checked === 1 ? 'is' : 'are'} ready to eat.`;

    resultGrid.replaceChildren(...data.results.map((r, i) => {
      const item = items[i];
      const el = document.createElement('div');
      el.className = `result-tile ${r.ok ? (r.label_index === 1 ? 'is-ripe' : 'is-unripe') : 'is-failed'}`;
      const img = document.createElement('img');
      img.src = item ? item.url : ''; img.alt = r.filename || '';
      const body = document.createElement('div');
      body.className = 'result-tile-body';
      if (r.ok) {
        body.innerHTML = `<span class="badge badge-${r.label_index === 1 ? 'ripe' : 'unripe'}"></span><span class="result-tile-conf"></span>`;
        body.querySelector('.badge').textContent = r.label;
        body.querySelector('.result-tile-conf').textContent = `${Math.round(r.confidence)}% sure`;
      } else {
        body.innerHTML = `<span class="badge badge-failed">Couldn't check</span><span class="result-tile-conf"></span>`;
        body.querySelector('.result-tile-conf').textContent = r.error || '';
      }
      el.append(img, body);
      return el;
    }));

    batchResult.classList.add('fade-in');
    show(batchResult);
    batchResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
})();
