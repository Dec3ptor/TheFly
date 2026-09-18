// Wires the UI to the viewer: loads the shells, lazily loads the neuron index,
// searches it, and streams the matching skeletons in from Janelia's bucket.

import { Viewer } from './viewer.js';
import { parseFlymesh, parseIndex, fetchSkeleton } from './formats.js';

const MAX_NEURONS_PER_LOAD = 48;   // a type like Tm3 has >2000 members; cap the batch
const FETCH_CONCURRENCY = 8;

const $ = (id) => document.getElementById(id);
const viewer = new Viewer($('viewport'));

let index = null;
let indexPromise = null;
let searchResults = [];
let selected = null;
let inFlight = null;

function setStatus(text, busy = false) {
  $('status').textContent = text;
  $('status').classList.toggle('busy', busy);
}

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

// ---------------------------------------------------------------- index

function loadIndex() {
  if (!indexPromise) {
    indexPromise = (async () => {
      setStatus('Loading neuron index (165,122 neurons)…', true);
      const response = await fetch('data/neurons.idx');
      if (!response.ok) throw new Error(`neuron index: HTTP ${response.status}`);
      index = parseIndex(await response.arrayBuffer());
      setStatus(`Index ready — ${index.neuronCount.toLocaleString()} neurons, ` +
                `${index.names.length.toLocaleString()} cell types.`);
      return index;
    })().catch((error) => {
      indexPromise = null;            // let a later search retry after a failure
      throw error;
    });
  }
  return indexPromise;
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

/** Run tasks with a bounded number of concurrent requests. */
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
    type: index.names[index.typeIndex[row]],
    synPre: index.synPre[row],
    synPost: index.synPost[row],
  };
}

async function loadType(typeIndex, { append = false } = {}) {
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
  const batch = ranked.slice(0, MAX_NEURONS_PER_LOAD).map(recordFor);

  const name = index.names[typeIndex];
  const capped = members.length > batch.length;
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
    setStatus(`Loading ${name}: ${done}/${batch.length} neurons…`, true);
  });

  if (controller.signal.aborted) return;
  inFlight = null;

  viewer.frame();
  renderNeuronList();

  const shown = viewer.neurons.size;
  let message = `${name}: showing ${shown} of ${members.length.toLocaleString()} neurons`;
  if (capped) message += ` (capped at ${MAX_NEURONS_PER_LOAD})`;
  if (missing) message += ` · ${missing} without a skeleton`;
  setStatus(message);
}

// ---------------------------------------------------------------- rendering the UI

function renderResults(results) {
  const list = $('results');
  list.innerHTML = '';
  if (!results.length) {
    list.innerHTML = '<li class="empty">No cell type matches that name.</li>';
    return;
  }
  for (const result of results) {
    const item = document.createElement('li');
    item.innerHTML = `<button class="result"><span class="name"></span>` +
                     `<span class="count"></span></button>`;
    item.querySelector('.name').textContent = result.name;
    item.querySelector('.count').textContent =
      `${result.count.toLocaleString()} neuron${result.count === 1 ? '' : 's'}`;
    item.querySelector('button').addEventListener('click', () => {
      for (const node of list.querySelectorAll('.result')) node.classList.remove('active');
      item.querySelector('button').classList.add('active');
      loadType(result.typeIndex).catch(reportError);
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
    item.querySelector('.body').textContent = record.bodyId.toLocaleString('en-US', {
      useGrouping: false,
    });
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
  $('detail-body').textContent = record.bodyId.toLocaleString('en-US', { useGrouping: false });
  $('detail-pre').textContent = record.synPre.toLocaleString();
  $('detail-post').textContent = record.synPost.toLocaleString();
  $('detail-neuprint').href =
    `https://neuprint.janelia.org/results?dataset=male-cns%3Av1.0&qt=findneurons&q=1` +
    `&qr[0][code]=fn&qr[0][ds]=male-cns:v1.0&qr[0][pm][dataset]=male-cns:v1.0` +
    `&qr[0][pm][neuron_id]=${record.bodyId}`;
}

function select(bodyId) {
  selected = bodyId !== null && viewer.neurons.has(bodyId) ? bodyId : null;
  viewer.highlight(selected);
  paintSelection();
}

// ---------------------------------------------------------------- neuroglancer hand-off

/**
 * Build a neuroglancer state for whatever is currently loaded, so the visitor can
 * jump from these skeletons to the full EM volume with the same neurons selected.
 */
function neuroglancerUrl() {
  const segments = [...viewer.neurons.keys()].map(String);
  const state = {
    dimensions: { x: [8e-9, 'm'], y: [8e-9, 'm'], z: [8e-9, 'm'] },
    layers: [
      {
        type: 'image',
        source: 'precomputed://gs://flyem-male-cns/em/em-clahe-jpeg',
        name: 'em-clahe',
      },
      {
        type: 'segmentation',
        source: 'precomputed://gs://flyem-male-cns/v1.0/segmentation',
        segments,
        name: 'cns-seg',
      },
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
    setStatus(`${searchResults.length} matching cell type${searchResults.length === 1 ? '' : 's'}.`);
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

for (const button of document.querySelectorAll('[data-example]')) {
  button.addEventListener('click', async () => {
    $('search').value = button.dataset.example;
    await runSearch();
    if (searchResults.length) {
      const exact = searchResults.find(
        (r) => r.name.toLowerCase() === button.dataset.example.toLowerCase(),
      ) || searchResults[0];
      loadType(exact.typeIndex).catch(reportError);
    }
  });
}

viewer.onSelect = (bodyId) => select(bodyId);

loadShells()
  .then(() => setStatus('Ready. Search for a cell type, or try one of the examples.'))
  .catch(reportError);
