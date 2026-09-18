// Wires the UI to the viewer and the activity model: loads the shells, lazily loads
// the neuron index and connectivity graph, searches, streams skeletons in from
// Janelia's bucket, and runs the simulation loop.

import { Viewer } from './viewer.js';
import { parseFlymesh, parseIndex, fetchSkeleton } from './formats.js';
import { parseSim, Simulation, DEFAULTS, heatColor } from './sim.js';

const FETCH_CONCURRENCY = 12;
const STEPS_PER_SECOND = 12;

const $ = (id) => document.getElementById(id);
const viewer = new Viewer($('viewport'));

let index = null;
let indexPromise = null;
let sim = null;
let simPromise = null;
let searchResults = [];
let selected = null;
let inFlight = null;
let running = false;

function setStatus(text, busy = false) {
  $('status').textContent = text;
  $('status').classList.toggle('busy', busy);
}

const neuronLimit = () => {
  const value = $('limit').value;
  return value === 'all' ? 750 : Number(value);
};

// ---------------------------------------------------------------- shells

async function loadShells() {
  const shells = [
    { file: 'data/brain-shell.flymesh', color: 0x2f6fb5 },
    { file: 'data/vnc-shell.flymesh', color: 0x2a5f8f },
  ];
  await Promise.all(shells.map(async ({ file, color }) => {
    const response = await fetch(file);
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    viewer.addShell(parseFlymesh(await response.arrayBuffer()), { color });
  }));
  viewer.resetView();
}

// ---------------------------------------------------------------- data

function loadIndex() {
  if (!indexPromise) {
    indexPromise = (async () => {
      setStatus('Loading neuron index…', true);
      const response = await fetch('data/neurons.idx');
      if (!response.ok) throw new Error(`neuron index: HTTP ${response.status}`);
      index = parseIndex(await response.arrayBuffer());
      setStatus(`Index ready — ${index.neuronCount.toLocaleString()} neurons, ` +
                `${index.names.length.toLocaleString()} cell types.`);
      return index;
    })().catch((error) => { indexPromise = null; throw error; });
  }
  return indexPromise;
}

function loadSim() {
  if (!simPromise) {
    simPromise = (async () => {
      setStatus('Loading connectivity graph…', true);
      const response = await fetch('data/connectome.sim');
      if (!response.ok) throw new Error(`connectome: HTTP ${response.status}`);
      const graph = parseSim(await response.arrayBuffer());
      sim = new Simulation(graph);
      sim.gain = Number($('gain').value);
      $('sim-panel').hidden = false;
      setStatus(`Connectivity ready — ${graph.edgeCount.toLocaleString()} ` +
                `type-to-type connections.`);
      return sim;
    })().catch((error) => { simPromise = null; throw error; });
  }
  return simPromise;
}

/** Rank types by how directly they match: exact, then prefix, then substring. */
function searchTypes(query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];

  const scored = [];
  for (let i = 0; i < index.names.length; i++) {
    const name = index.names[i];
    const at = name.toLowerCase().indexOf(needle);
    if (at === -1) continue;
    const exact = name.toLowerCase() === needle ? 0 : at === 0 ? 1 : 2;
    scored.push({ typeIndex: i, name, rank: exact, count: index.membersByType[i].length });
  }
  scored.sort((a, b) => a.rank - b.rank || b.count - a.count || a.name.localeCompare(b.name));
  return scored.slice(0, 80);
}

// ---------------------------------------------------------------- loading neurons

async function withConcurrency(items, limit, worker) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) await worker(items[cursor++]);
  });
  await Promise.all(runners);
}

function recordFor(row) {
  return {
    bodyId: index.bodyIds[row],
    typeIndex: index.typeIndex[row],
    type: index.names[index.typeIndex[row]],
    synPre: index.synPre[row],
    synPost: index.synPost[row],
  };
}

async function loadType(typeIndex, { append = false, cap = neuronLimit() } = {}) {
  if (inFlight) inFlight.abort();
  const controller = new AbortController();
  inFlight = controller;

  if (!append) {
    viewer.clearNeurons();
    selected = null;
  }

  const members = index.membersByType[typeIndex];
  // Busiest neurons first, so a capped batch still shows the type's main cells.
  const ranked = [...members].sort(
    (a, b) => (index.synPre[b] + index.synPost[b]) - (index.synPre[a] + index.synPost[a]),
  );
  const batch = ranked.slice(0, cap).map(recordFor)
    .filter((record) => !viewer.neurons.has(record.bodyId));

  const name = index.names[typeIndex];
  const capped = members.length > cap;
  let done = 0;
  let missing = 0;

  const colorByBody = new Map(batch.map((record) => [record.bodyId, viewer.nextColor()]));

  await withConcurrency(batch, FETCH_CONCURRENCY, async (record) => {
    try {
      const skeleton = await fetchSkeleton(record.bodyId, controller.signal);
      if (skeleton) viewer.addNeuron(record, skeleton, colorByBody.get(record.bodyId));
      else missing++;
    } catch (error) {
      if (error.name === 'AbortError') return;
      missing++;
      console.warn('skeleton failed', record.bodyId, error);
    }
    done++;
    if (done % 4 === 0 || done === batch.length) {
      setStatus(`Loading ${name}: ${done}/${batch.length} neurons…`, true);
      renderNeuronList();
    }
  });

  if (controller.signal.aborted) return;
  inFlight = null;

  if (!append) viewer.frame();
  renderNeuronList();

  let message = `${name}: showing ${viewer.neurons.size} neuron` +
                `${viewer.neurons.size === 1 ? '' : 's'}`;
  if (capped) message += ` (capped at ${cap} of ${members.length.toLocaleString()})`;
  if (missing) message += ` · ${missing} without a skeleton`;
  setStatus(message);
}

// ---------------------------------------------------------------- simulation

function simLoop() {
  if (!running || !sim) return;
  const budget = Math.max(1, Math.round(STEPS_PER_SECOND / 60));
  for (let i = 0; i < budget; i++) sim.step();
  viewer.paintActivity(sim.rate);
  renderActivity();
  requestAnimationFrame(simLoop);
}

function setRunning(on) {
  running = on;
  $('run').textContent = on ? '❚❚ Pause' : '▶ Run';
  $('run').classList.toggle('active', on);
  if (on) requestAnimationFrame(simLoop);
}

async function stimulate(typeIndex, additive = false) {
  await loadSim();
  if (!additive) sim.clearStimulus();
  sim.setStimulus(typeIndex, 1);
  renderStimulusList();
  if (!running) setRunning(true);
  setStatus(`Stimulating ${index.names[typeIndex]}.`);
}

/**
 * Draw the types the model has driven hardest, excluding ones already on screen.
 * Only a few neurons of each, so a cascade stays legible instead of becoming a bush.
 */
async function loadTopResponders(count = 6, perType = 8) {
  if (!sim) return;
  const onScreen = new Set([...viewer.neurons.values()].map((e) => e.record.typeIndex));
  const candidates = sim.mostActive(count + onScreen.size + 6)
    .filter((typeIndex) => !onScreen.has(typeIndex))
    .slice(0, count);
  if (!candidates.length) {
    setStatus('No new responders to add yet — let the model run a moment.');
    return;
  }
  for (const typeIndex of candidates) {
    await loadType(typeIndex, { append: true, cap: perType });
  }
  viewer.frame();
  setStatus(`Added ${candidates.length} downstream types: ` +
            candidates.map((i) => index.names[i]).join(', '));
}

function resetSim() {
  if (!sim) return;
  sim.reset();
  viewer.paintActivity(sim.rate);
  renderActivity();
}

function renderStimulusList() {
  const holder = $('stim-list');
  holder.innerHTML = '';
  const active = sim ? sim.stimulated : [];
  if (!active.length) {
    holder.innerHTML = '<span class="empty">Nothing is being stimulated yet.</span>';
    return;
  }
  for (const typeIndex of active) {
    const chip = document.createElement('button');
    chip.className = 'stim-chip';
    chip.innerHTML = `<span></span>✕`;
    chip.querySelector('span').textContent = index.names[typeIndex];
    chip.title = 'Stop stimulating this type';
    chip.addEventListener('click', () => {
      sim.setStimulus(typeIndex, 0);
      renderStimulusList();
    });
    holder.appendChild(chip);
  }
}

function renderActivity() {
  if (!sim) return;
  $('sim-time').textContent = `t = ${sim.time.toFixed(1)}`;
  $('sim-active').textContent = `${sim.activeCount().toLocaleString()} types active`;

  const top = sim.mostActive(14);
  const list = $('activity');
  list.innerHTML = '';
  if (!top.length) {
    list.innerHTML = '<li class="empty">No activity yet — stimulate a type.</li>';
    return;
  }
  for (const typeIndex of top) {
    const value = sim.rate[typeIndex];
    const item = document.createElement('li');
    item.innerHTML = `<button class="act"><span class="bar"></span>` +
                     `<span class="name"></span><span class="val"></span></button>`;
    const colour = `#${heatColor(value).toString(16).padStart(6, '0')}`;
    const bar = item.querySelector('.bar');
    bar.style.width = `${Math.round(value * 100)}%`;
    bar.style.background = colour;
    item.querySelector('.name').textContent = index.names[typeIndex];
    item.querySelector('.val').textContent = value.toFixed(2);
    item.querySelector('button').title = 'Load this type into the view';
    item.querySelector('button').addEventListener('click', () => {
      loadType(typeIndex, { append: true }).catch(reportError);
    });
    list.appendChild(item);
  }
}

// ---------------------------------------------------------------- UI rendering

function renderResults(results) {
  const list = $('results');
  list.innerHTML = '';
  if (!results.length) {
    list.innerHTML = '<li class="empty">No cell type matches that name.</li>';
    return;
  }
  for (const result of results) {
    const item = document.createElement('li');
    item.innerHTML =
      `<button class="result"><span class="name"></span><span class="count"></span></button>` +
      `<button class="zap" title="Stimulate this cell type">⚡</button>`;
    item.querySelector('.name').textContent = result.name;
    item.querySelector('.count').textContent =
      `${result.count.toLocaleString()} neuron${result.count === 1 ? '' : 's'}`;
    item.querySelector('.result').addEventListener('click', () => {
      for (const node of list.querySelectorAll('.result')) node.classList.remove('active');
      item.querySelector('.result').classList.add('active');
      loadType(result.typeIndex).catch(reportError);
    });
    item.querySelector('.zap').addEventListener('click', () => {
      stimulate(result.typeIndex).catch(reportError);
    });
    list.appendChild(item);
  }
}

function renderNeuronList() {
  const list = $('neurons');
  list.innerHTML = '';
  // Insertion order reflects whichever concurrent fetch landed first, so sort the
  // list for a stable ordering that matches the order neurons were requested in.
  const entries = [...viewer.neurons.values()].sort(
    (a, b) => (b.record.synPre + b.record.synPost) - (a.record.synPre + a.record.synPost),
  );
  $('neuron-count').textContent = entries.length ? `${entries.length} loaded` : '';

  for (const { record, color } of entries) {
    const item = document.createElement('li');
    item.innerHTML = `<button class="neuron"><span class="swatch"></span>` +
                     `<span class="body"></span><span class="syn"></span></button>`;
    item.querySelector('.swatch').style.background = `#${color.toString(16).padStart(6, '0')}`;
    item.querySelector('.body').textContent = String(record.bodyId);
    item.querySelector('.syn').textContent = `${record.synPre}↑ ${record.synPost}↓`;
    item.querySelector('button').addEventListener('click', () => select(record.bodyId));
    item.dataset.bodyId = String(record.bodyId);
    list.appendChild(item);
  }
  paintSelection();
}

function paintSelection() {
  for (const node of $('neurons').querySelectorAll('li')) {
    node.querySelector('button').classList.toggle(
      'active', Number(node.dataset.bodyId) === selected,
    );
  }
  const entry = selected !== null ? viewer.neurons.get(selected) : null;
  const panel = $('detail');
  if (!entry) {
    panel.hidden = true;
    return;
  }
  const { record } = entry;
  panel.hidden = false;
  $('detail-type').textContent = record.type;
  $('detail-body').textContent = String(record.bodyId);
  $('detail-pre').textContent = record.synPre.toLocaleString();
  $('detail-post').textContent = record.synPost.toLocaleString();
  $('detail-neuprint').href =
    `https://neuprint.janelia.org/results?dataset=male-cns%3Av1.0&qt=findneurons&q=1` +
    `&qr[0][code]=fn&qr[0][ds]=male-cns:v1.0&qr[0][pm][dataset]=male-cns:v1.0` +
    `&qr[0][pm][neuron_id]=${record.bodyId}`;
  $('detail-stim').onclick = () => stimulate(record.typeIndex).catch(reportError);
}

function select(bodyId) {
  selected = bodyId !== null && viewer.neurons.has(bodyId) ? bodyId : null;
  viewer.highlight(selected);
  paintSelection();
}

// ---------------------------------------------------------------- neuroglancer

function neuroglancerUrl() {
  const segments = [...viewer.neurons.keys()].map(String);
  const state = {
    dimensions: { x: [8e-9, 'm'], y: [8e-9, 'm'], z: [8e-9, 'm'] },
    layers: [
      { type: 'image', source: 'precomputed://gs://flyem-male-cns/em/em-clahe-jpeg',
        name: 'em-clahe' },
      { type: 'segmentation', source: 'precomputed://gs://flyem-male-cns/v1.0/segmentation',
        segments, name: 'cns-seg' },
    ],
    layout: 'xy-3d',
  };
  return `https://neuroglancer-demo.appspot.com/#!${encodeURIComponent(JSON.stringify(state))}`;
}

// ---------------------------------------------------------------- bootstrap

function reportError(error) {
  console.error(error);
  setStatus(`Error: ${error.message}`);
}

async function runSearch() {
  const query = $('search').value;
  if (!query.trim()) {
    renderResults([]);
    return;
  }
  try {
    await loadIndex();
    searchResults = searchTypes(query);
    renderResults(searchResults);
  } catch (error) {
    reportError(error);
  }
}

let searchTimer = null;
$('search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 160);
});

$('reset-view').addEventListener('click', () => viewer.resetView());
$('clear').addEventListener('click', () => {
  viewer.clearNeurons();
  selected = null;
  renderNeuronList();
  setStatus('Cleared.');
});
$('toggle-shells').addEventListener('change', (event) => {
  viewer.setShellsVisible(event.target.checked);
});
$('open-ng').addEventListener('click', (event) => {
  event.preventDefault();
  window.open(neuroglancerUrl(), '_blank', 'noopener');
});

$('run').addEventListener('click', async () => {
  await loadSim();
  setRunning(!running);
});
$('reset-sim').addEventListener('click', resetSim);
$('gain').addEventListener('input', (event) => {
  const value = Number(event.target.value);
  $('gain-value').textContent = value.toFixed(2);
  if (sim) sim.gain = value;
});
$('load-responders').addEventListener('click', () => {
  loadTopResponders().catch(reportError);
});
$('clear-stim').addEventListener('click', () => {
  if (!sim) return;
  sim.clearStimulus();
  renderStimulusList();
});

for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', async () => {
    $('search').value = button.dataset.example;
    await runSearch();
    if (!searchResults.length) return;
    const exact = searchResults.find(
      (r) => r.name.toLowerCase() === button.dataset.example.toLowerCase(),
    ) || searchResults[0];
    loadType(exact.typeIndex).catch(reportError);
  });
}

// One-click demonstrations: load the driver type, then stimulate it.
for (const button of document.querySelectorAll('[data-demo]')) {
  button.addEventListener('click', async () => {
    const name = button.dataset.demo;
    try {
      await Promise.all([loadIndex(), loadSim()]);
      const typeIndex = index.names.indexOf(name);
      if (typeIndex === -1) throw new Error(`no cell type named ${name}`);
      sim.reset();
      await loadType(typeIndex, { cap: Math.min(neuronLimit(), 60) });
      await stimulate(typeIndex);
      // Let activity spread before asking which types it reached.
      await new Promise((resolve) => setTimeout(resolve, 1600));
      await loadTopResponders();
      setStatus(`${name}: stimulated, with its strongest downstream types drawn.`);
    } catch (error) {
      reportError(error);
    }
  });
}

viewer.onSelect = (bodyId) => select(bodyId);

$('gain-value').textContent = DEFAULTS.gain.toFixed(2);
$('gain').value = String(DEFAULTS.gain);

loadShells()
  .then(() => setStatus('Ready. Search for a cell type, or try a demo.'))
  .catch(reportError);
