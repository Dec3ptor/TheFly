// Polls the local simulation and draws the arena. Everything here is presentation;
// the brain, the body and the world all live in the Python process.

const $ = (id) => document.getElementById(id);
const canvas = $('arena');
const ctx = canvas.getContext('2d');

let state = null;
let tool = null;

async function post(name, payload = {}) {
  const response = await fetch(`/api/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

// ------------------------------------------------------------------ drawing

function resize() {
  const ratio = Math.min(window.devicePixelRatio, 2);
  canvas.width = canvas.clientWidth * ratio;
  canvas.height = canvas.clientHeight * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}
window.addEventListener('resize', resize);

/** Arena millimetres to canvas pixels. */
function view() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const extent = state ? state.world.extent : 150;
  const scale = Math.min(w, h) / (extent * 2.15);
  return { w, h, scale, cx: w / 2, cy: h / 2, extent };
}

const toScreen = (v, x, y) => [v.cx + x * v.scale, v.cy - y * v.scale];

function draw() {
  const v = view();
  ctx.clearRect(0, 0, v.w, v.h);
  if (!state) return;
  const world = state.world;

  // Arena floor.
  ctx.beginPath();
  ctx.arc(v.cx, v.cy, v.extent * v.scale, 0, Math.PI * 2);
  ctx.fillStyle = '#0b1020';
  ctx.fill();
  ctx.strokeStyle = '#1f2a44';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Odour plumes, drawn as the gaussian they actually are.
  for (const odour of world.odours) {
    const [x, y] = toScreen(v, odour.x, odour.y);
    const radius = odour.sigma * v.scale;
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    const tint = odour.valence >= 0 ? '128,237,153' : '247,37,133';
    gradient.addColorStop(0, `rgba(${tint},0.30)`);
    gradient.addColorStop(1, `rgba(${tint},0)`);
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, Math.max(odour.r * v.scale, 3), 0, Math.PI * 2);
    ctx.fillStyle = odour.valence >= 0 ? '#80ed99' : '#f72585';
    ctx.fill();
    ctx.fillStyle = '#8494b4';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(odour.receptor, x, y - Math.max(odour.r * v.scale, 3) - 5);
  }

  // Posts.
  for (const p of world.posts) {
    const [x, y] = toScreen(v, p.x, p.y);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(p.r * v.scale, 2), 0, Math.PI * 2);
    ctx.fillStyle = '#2f3b57';
    ctx.fill();
    ctx.strokeStyle = '#4a5878';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Path travelled.
  if (state.trail.length > 1) {
    ctx.beginPath();
    state.trail.forEach(([tx, ty], i) => {
      const [x, y] = toScreen(v, tx, ty);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = 'rgba(76,201,240,0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // The fly, with its two eye fields shaded by how hard each side's looming
  // detectors are firing.
  const body = world.body;
  const [fx, fy] = toScreen(v, body.x, body.y);
  const t = state.telemetry;
  const eyeScale = Math.max(t.looming_left, t.looming_right, 1);
  for (const [side, centre] of [['looming_left', -1], ['looming_right', 1]]) {
    const strength = t[side] / eyeScale;
    if (strength <= 0.02) continue;
    ctx.beginPath();
    ctx.moveTo(fx, fy);
    const mid = -body.heading + centre * (Math.PI / 3);
    ctx.arc(fx, fy, 46, mid - Math.PI / 2.4, mid + Math.PI / 2.4);
    ctx.closePath();
    ctx.fillStyle = `rgba(255,183,3,${0.05 + 0.28 * strength})`;
    ctx.fill();
  }

  ctx.save();
  ctx.translate(fx, fy);
  ctx.rotate(-body.heading);
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(-6, 5);
  ctx.lineTo(-3, 0);
  ctx.lineTo(-6, -5);
  ctx.closePath();
  ctx.fillStyle = state.world.reinforcement < 0 ? '#f72585'
                : state.world.reinforcement > 0 ? '#80ed99' : '#e8eefc';
  ctx.fill();
  ctx.restore();
}

// ------------------------------------------------------------------ state

function render() {
  if (!state) return;
  const t = state.telemetry;
  const m = state.motor;
  const body = state.world.body;

  $('run').textContent = state.running ? '❚❚ Pause' : '▶ Run';
  $('run').classList.toggle('on', state.running);
  $('rate').textContent = `${state.wall_rate.toFixed(2)}× real time`;

  $('m-fwd').textContent = `${m.forward.toFixed(2)} (${body.speed.toFixed(1)} mm/s)`;
  $('m-turn').textContent = m.turn.toFixed(2);
  $('m-pos').textContent = `${body.x.toFixed(0)}, ${body.y.toFixed(0)}`;
  $('m-time').textContent = `${(t.brain_ms / 1000).toFixed(1)} s`;

  $('t-spiking').textContent = `${t.spiking.toLocaleString()} above 1 Hz`;
  $('t-dn').textContent = `${t.dn_left.toFixed(2)} / ${t.dn_right.toFixed(2)} Hz`;
  $('t-motor').textContent = `${t.motor.toFixed(2)} Hz`;
  $('t-loom').textContent = `${t.looming_left.toFixed(1)} / ${t.looming_right.toFixed(1)} Hz`;
  $('t-orn').textContent = `${t.orn.toFixed(2)} Hz`;
  $('t-kc').textContent = `${t.kenyon.toFixed(2)} Hz`;
  $('t-mbon').textContent = `${t.mbon.toFixed(2)} Hz`;
  $('t-dan').textContent = `${t.dan.toFixed(2)} Hz`;

  $('l-depr').textContent = `${(state.learning.depression * 100).toFixed(2)}%`;
  $('l-syn').textContent = state.learning.synapses.toLocaleString();

  const list = $('types');
  list.innerHTML = '';
  const peak = Math.max(...state.active_types.map((a) => a.hz), 1);
  for (const entry of state.active_types) {
    const item = document.createElement('li');
    item.innerHTML = '<span class="fill"></span><span class="n"></span><span class="v"></span>';
    item.querySelector('.fill').style.width = `${(entry.hz / peak) * 100}%`;
    item.querySelector('.n').textContent = entry.name;
    item.querySelector('.v').textContent = `${entry.hz.toFixed(1)} Hz`;
    list.appendChild(item);
  }

  const reinforcement = state.world.reinforcement;
  $('status').textContent = reinforcement < 0
    ? 'On an aversive odour — dopamine is depressing active KC→MBON synapses'
    : reinforcement > 0
      ? 'On a rewarding odour — dopamine is depressing active KC→MBON synapses'
      : tool
        ? `Click the arena to place: ${tool}`
        : `${state.ticks.toLocaleString()} body updates · click the arena to move the fly`;
}

async function poll() {
  try {
    const response = await fetch('/api/state');
    state = await response.json();
    render();
    draw();
  } catch (error) {
    $('status').textContent = `lost the simulation: ${error.message}`;
  }
}

// ------------------------------------------------------------------ controls

$('run').addEventListener('click', async () => {
  await post('run', { on: !state.running });
  poll();
});
$('reset').addEventListener('click', () => post('reset').then(poll));
$('scatter').addEventListener('click', () => post('scatter').then(poll));
$('unlearn').addEventListener('click', () => post('reset_learning').then(poll));
$('clear').addEventListener('click', () => post('clear_objects').then(poll));
$('speed').addEventListener('input', (e) => {
  $('speedv').textContent = Number(e.target.value).toFixed(1);
  post('speed', { value: Number(e.target.value) });
});

for (const [id, name] of [['tool-post', 'post'], ['tool-odour', 'reward'],
                          ['tool-bad', 'punish']]) {
  $(id).addEventListener('click', () => {
    tool = tool === name ? null : name;
    for (const other of ['tool-post', 'tool-odour', 'tool-bad']) {
      $(other).classList.toggle('on', tool === { 'tool-post': 'post',
        'tool-odour': 'reward', 'tool-bad': 'punish' }[other]);
    }
    render();
  });
}

canvas.addEventListener('click', async (event) => {
  if (!state) return;
  const v = view();
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left - v.cx) / v.scale;
  const y = -(event.clientY - rect.top - v.cy) / v.scale;
  if (Math.hypot(x, y) > v.extent) return;

  if (tool === 'post') await post('add_post', { x, y, r: 5 });
  else if (tool === 'reward') await post('add_odour', { x, y, receptor: 'ORN_DA1', valence: 1 });
  else if (tool === 'punish') await post('add_odour', { x, y, receptor: 'ORN_DM1', valence: -1 });
  else await post('place', { x, y });
  poll();
});

$('zap').addEventListener('click', async () => {
  const name = $('stim').value.trim();
  if (!name) return;
  const result = await post('stimulate', { type: name, mv: 1.0 });
  $('stim-note').textContent = result.ok
    ? `Driving ${result.neurons} neurons of ${name}. Reset clears it.`
    : result.error;
});
$('stim').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('zap').click(); });

resize();
poll();
setInterval(poll, 120);
