(() => {
  const GRID_COLOR    = '#f3f4f6';
  const TICK_COLOR    = '#9ca3af';
  const KERNEL_COLORS = ['#ef4444', '#3b82f6', '#8b5cf6'];

  function baseOptions(yMax = 100, yMin = 0) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR } },
        y: {
          min: yMin, max: yMax,
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, callback: v => v + '%' },
        },
      },
    };
  }

  function renderKernelChart(data) {
    const kd = data.kernel_comparison;
    new Chart(document.getElementById('chart-kernels'), {
      type: 'bar',
      data: {
        labels: kd.kernels,
        datasets: [{
          data:            kd.accuracies,
          backgroundColor: kd.kernels.map((k, i) =>
            k === kd.best ? '#ef4444' : KERNEL_COLORS[i] + '99'
          ),
          borderColor: kd.kernels.map((k, i) =>
            k === kd.best ? '#dc2626' : KERNEL_COLORS[i]
          ),
          borderWidth: 2,
          borderRadius: 6,
        }],
      },
      options: {
        ...baseOptions(105, 40),
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: ctx => ` ${ctx.parsed.y}%` },
          },
        },
        scales: {
          ...baseOptions(105, 40).scales,
          y: {
            ...baseOptions(105, 40).scales.y,
            ticks: {
              color: TICK_COLOR,
              callback: v => v + '%',
            },
          },
        },
      },
    });
  }

  function renderTrialChart(data) {
    const td     = data.trials;
    const labels = td.accuracies.map((_, i) => i + 1);
    const meanLine = new Array(td.accuracies.length).fill(td.mean);

    new Chart(document.getElementById('chart-trials'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label:       'Accuracy',
            data:        td.accuracies,
            borderColor: '#ef4444',
            backgroundColor: '#ef444420',
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: '#ef4444',
            tension: 0.3,
            fill: true,
          },
          {
            label:       'Mean',
            data:        meanLine,
            borderColor: '#9ca3af',
            borderWidth: 1.5,
            borderDash:  [6, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        ...baseOptions(105, 65),
        plugins: {
          legend: {
            display: true,
            labels: { color: TICK_COLOR, boxWidth: 12 },
          },
          tooltip: {
            callbacks: { label: ctx => ` ${ctx.parsed.y}%` },
          },
        },
      },
    });

    // Summary pills
    const container = document.getElementById('trial-summary');
    [
      ['Mean', td.mean + '%'],
      ['Std',  td.std  + '%'],
      ['Min',  td.min  + '%'],
      ['Max',  td.max  + '%'],
    ].forEach(([k, v]) => {
      const el = document.createElement('span');
      el.className = 'bg-gray-100 rounded-full px-3 py-0.5';
      el.innerHTML = `<span class="text-gray-400">${k} </span><strong class="text-gray-700">${v}</strong>`;
      container.appendChild(el);
    });
  }

  function renderConfusionMatrix(data) {
    const cm = data.confusion_matrix;
    // [[TN, FP], [FN, TP]]
    const cells = [
      { id: 'cm-00', val: cm[0][0], label: 'TN', correct: true  },
      { id: 'cm-01', val: cm[0][1], label: 'FP', correct: false },
      { id: 'cm-10', val: cm[1][0], label: 'FN', correct: false },
      { id: 'cm-11', val: cm[1][1], label: 'TP', correct: true  },
    ];
    cells.forEach(({ id, val, label, correct }) => {
      const el = document.getElementById(id);
      el.classList.add(correct ? 'bg-green-500' : 'bg-red-400');
      el.innerHTML = `
        <span class="text-2xl">${val}</span>
        <span class="text-xs font-normal opacity-80">${label}</span>
      `;
    });
  }

  function renderMetricsChart(data) {
    const cr = data.classification_report;
    new Chart(document.getElementById('chart-metrics'), {
      type: 'bar',
      data: {
        labels: ['Unripe', 'Ripe'],
        datasets: [
          {
            label: 'Precision',
            data:  [cr.Unripe.precision * 100, cr.Ripe.precision * 100],
            backgroundColor: '#3b82f699',
            borderColor:     '#3b82f6',
            borderWidth: 2, borderRadius: 4,
          },
          {
            label: 'Recall',
            data:  [cr.Unripe.recall * 100, cr.Ripe.recall * 100],
            backgroundColor: '#22c55e99',
            borderColor:     '#22c55e',
            borderWidth: 2, borderRadius: 4,
          },
          {
            label: 'F1-Score',
            data:  [cr.Unripe.f1 * 100, cr.Ripe.f1 * 100],
            backgroundColor: '#f9731699',
            borderColor:     '#f97316',
            borderWidth: 2, borderRadius: 4,
          },
        ],
      },
      options: {
        ...baseOptions(100, 80),
        plugins: {
          legend: {
            display: true,
            labels: { color: TICK_COLOR, boxWidth: 12 },
          },
          tooltip: {
            callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)}%` },
          },
        },
      },
    });
  }

  function populateDataset(data) {
    const ds = data.dataset;
    document.getElementById('ds-total').textContent   = ds.total;
    document.getElementById('ds-ripe').textContent    = ds.ripe;
    document.getElementById('ds-unripe').textContent  = ds.unripe;
    document.getElementById('ds-imgsize').textContent = `${ds.img_size} × ${ds.img_size} px`;
    document.getElementById('ds-pca').textContent     = ds.pca_components;
    document.getElementById('ds-var').textContent     = `${ds.explained_variance}%`;
  }

  async function init() {
    try {
      const resp = await fetch('/analytics-data');
      if (!resp.ok) throw new Error('Failed to load analytics data');
      const data = await resp.json();

      renderKernelChart(data);
      renderTrialChart(data);
      renderConfusionMatrix(data);
      renderMetricsChart(data);
      populateDataset(data);
    } catch (err) {
      console.error('Analytics init error:', err);
    }
  }

  init();
})();
