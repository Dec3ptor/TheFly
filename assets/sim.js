// A rate model of the CNS, run over the real cell-type connectivity graph.
//
// Each cell type is one unit holding a membrane-like state v, driven by its
// presynaptic partners:
//
//     drive(b) = sum over a of  sign(a) * weight(a -> b) * rate(a)
//     v(b)    += dt/tau * ( -v(b) + gain * drive(b) + stimulus(b) )
//     rate(b)  = clamp(v(b), 0, 1)
//
// Weights are the fraction of b's input synapses arriving from a, so drive stays
// on [-1, 1] no matter how large the presynaptic population is. Signs come from
// each type's predicted transmitter.
//
// This is a coarse model, not a biophysical one: it has no spikes, no channel
// dynamics, no neuromodulation, and the connectome fixes wiring but not synaptic
// strength. It shows how activity spreads through real wiring, and no more.

export const DEFAULTS = { gain: 2.0, tau: 1.0, dt: 0.25 };

export function parseSim(buffer) {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== 'FLYSIM1\0') throw new Error(`not a sim file (magic "${magic}")`);

  const view = new DataView(buffer);
  const typeCount = view.getUint32(8, true);
  const edgeCount = view.getUint32(12, true);

  let offset = 16;
  const sign = new Int8Array(buffer.slice(offset, offset + typeCount));
  offset += typeCount;
  const offsets = new Uint32Array(buffer.slice(offset, offset + 4 * (typeCount + 1)));
  offset += 4 * (typeCount + 1);
  const targets = new Uint16Array(buffer.slice(offset, offset + 2 * edgeCount));
  offset += 2 * edgeCount;
  const quantised = new Uint16Array(buffer.slice(offset, offset + 2 * edgeCount));

  // Dequantise once; the step loop is hot enough that it should not divide.
  const weights = new Float32Array(edgeCount);
  for (let i = 0; i < edgeCount; i++) weights[i] = quantised[i] / 65535;

  return { typeCount, edgeCount, sign, offsets, targets, weights };
}

export class Simulation {
  constructor(graph) {
    this.graph = graph;
    this.gain = DEFAULTS.gain;
    this.tau = DEFAULTS.tau;
    this.dt = DEFAULTS.dt;

    const n = graph.typeCount;
    this.v = new Float32Array(n);
    this.rate = new Float32Array(n);
    this.stim = new Float32Array(n);
    this.drive = new Float32Array(n);
    this.time = 0;
    this.steps = 0;
  }

  reset() {
    this.v.fill(0);
    this.rate.fill(0);
    this.drive.fill(0);
    this.time = 0;
    this.steps = 0;
  }

  clearStimulus() {
    this.stim.fill(0);
  }

  setStimulus(typeIndex, amplitude) {
    this.stim[typeIndex] = amplitude;
  }

  get stimulated() {
    const out = [];
    for (let i = 0; i < this.stim.length; i++) if (this.stim[i] !== 0) out.push(i);
    return out;
  }

  step() {
    const { sign, offsets, targets, weights, typeCount } = this.graph;
    const { v, rate, stim, drive } = this;

    drive.fill(0);
    // Walk the graph by source so each row's sign is looked up once. Scattering
    // into the target accumulator costs a random write but avoids transposing.
    for (let source = 0; source < typeCount; source++) {
      const activity = rate[source];
      if (activity === 0) continue;                 // most units are silent
      const s = sign[source];
      if (s === 0) continue;                        // modulatory: no direct drive
      const contribution = s * activity;
      const end = offsets[source + 1];
      for (let e = offsets[source]; e < end; e++) {
        drive[targets[e]] += weights[e] * contribution;
      }
    }

    const k = this.dt / this.tau;
    const gain = this.gain;
    for (let i = 0; i < typeCount; i++) {
      let next = v[i] + k * (-v[i] + gain * drive[i] + stim[i]);
      // Bound the state so a runaway gain saturates instead of going infinite,
      // while still allowing units to sit below zero after strong inhibition.
      if (next > 2) next = 2; else if (next < -1) next = -1;
      v[i] = next;
      rate[i] = next < 0 ? 0 : next > 1 ? 1 : next;
    }

    this.time += this.dt;
    this.steps++;
  }

  /** Indices of the most active types, strongest first. */
  mostActive(limit = 12, threshold = 0.02) {
    const { rate } = this;
    const found = [];
    for (let i = 0; i < rate.length; i++) if (rate[i] > threshold) found.push(i);
    found.sort((a, b) => rate[b] - rate[a]);
    return found.slice(0, limit);
  }

  activeCount(threshold = 0.05) {
    let count = 0;
    for (let i = 0; i < this.rate.length; i++) if (this.rate[i] > threshold) count++;
    return count;
  }
}

/** Dark blue -> cyan -> amber -> white, as a packed 0xRRGGBB integer. */
export function heatColor(t) {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  const stops = [
    [0.00, 0x18, 0x22, 0x3a],
    [0.25, 0x1e, 0x7a, 0xa8],
    [0.50, 0x35, 0xd0, 0xd8],
    [0.75, 0xff, 0xc2, 0x4d],
    [1.00, 0xff, 0xff, 0xf0],
  ];
  let i = 0;
  while (i < stops.length - 2 && x > stops[i + 1][0]) i++;
  const [t0, r0, g0, b0] = stops[i];
  const [t1, r1, g1, b1] = stops[i + 1];
  const f = (x - t0) / (t1 - t0);
  const r = Math.round(r0 + (r1 - r0) * f);
  const g = Math.round(g0 + (g1 - g0) * f);
  const b = Math.round(b0 + (b1 - b0) * f);
  return (r << 16) | (g << 8) | b;
}
