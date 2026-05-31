(() => {
  const statTotal   = document.getElementById('stat-total');
  const statRipe    = document.getElementById('stat-ripe');
  const statUnripe  = document.getElementById('stat-unripe');
  const recentList  = document.getElementById('recent-list');
  const recentEmpty = document.getElementById('recent-empty');

  function timeAgo(isoString) {
    const diff = Math.floor((Date.now() - new Date(isoString)) / 1000);
    if (diff < 5)  return 'just now';
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  }

  function renderRow(prediction) {
    const isRipe   = prediction.label_index === 1;
    const badgeCls = isRipe
      ? 'bg-red-100 text-red-700'
      : 'bg-green-100 text-green-700';

    const row = document.createElement('div');
    row.className = 'px-6 py-3 flex items-center justify-between gap-4';
    row.innerHTML = `
      <span class="inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold ${badgeCls}">
        ${prediction.label}
      </span>
      <div class="flex-1 mx-3">
        <div class="w-full bg-gray-100 rounded-full h-1.5">
          <div class="${isRipe ? 'bg-red-400' : 'bg-green-400'} h-1.5 rounded-full"
               style="width: ${prediction.confidence}%"></div>
        </div>
      </div>
      <span class="text-sm font-semibold text-gray-700 w-14 text-right shrink-0">
        ${prediction.confidence}%
      </span>
      <span class="hidden sm:block text-xs text-gray-400 w-16 text-right shrink-0">
        ${timeAgo(prediction.timestamp)}
      </span>
    `;
    return row;
  }

  async function loadStats() {
    try {
      const resp = await fetch('/stats');
      if (!resp.ok) return;
      const data = await resp.json();

      statTotal.textContent  = data.total;
      statRipe.textContent   = data.ripe_count;
      statUnripe.textContent = data.unripe_count;

      if (data.recent.length > 0) {
        recentEmpty.remove();
        // Clear existing rows (keep the container)
        recentList.innerHTML = '';
        data.recent.forEach(p => recentList.appendChild(renderRow(p)));
      }
    } catch (_) {
      // Silently ignore network errors on the dashboard
    }
  }

  loadStats();
})();
