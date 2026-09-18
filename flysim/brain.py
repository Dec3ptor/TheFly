"""Leaky integrate-and-fire over the whole MaleCNS connectome.

One unit per neuron — 165,122 of them — wired by the 25.5M measured connections.
A spike steps the postsynaptic membrane by `mv_per_synapse` times the number of
synapses in that connection, signed by the presynaptic transmitter. Between spikes
the membrane leaks back toward rest with time constant `tau_membrane`.

    v     += dt/tau * (v_rest - v)        leak
    v     += W[spiked]                    instantaneous PSPs from this step's spikes
    spike  = v >= v_threshold and not refractory
    v[spike] = v_reset,  refractory for `refractory_ms`

Delta-synapse PSPs rather than conductance kinetics: the connectome fixes wiring and
synapse counts but not conductances or reversal potentials, so modelling those in
detail would be inventing numbers. Voltages are in millivolts and the parameters sit
in the range used for connectome-scale fly models, but `mv_per_synapse` sets the whole
network's operating point and is calibrated, not measured — see the README.

The step loop is written to touch the full 165k-element arrays as few times as
possible; everything that can be confined to the few hundred neurons spiking on a
given step is.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    v_rest: float = -52.0          # mV
    v_threshold: float = -45.0     # mV
    v_reset: float = -55.0         # mV
    tau_membrane: float = 20.0     # ms
    refractory_ms: float = 2.2     # ms
    mv_per_synapse: float = 0.10   # mV of PSP per anatomical synapse
    inhibition_scale: float = 1.0  # inhibitory synapses are often the stronger ones
    adapt_mv: float = 1.0          # hyperpolarisation added per spike
    tau_adapt: float = 150.0       # ms, decay of that hyperpolarisation
    dt: float = 1.0                # ms per step
    rate_window_ms: float = 200.0  # window the reported firing rate averages over


class Brain:
    def __init__(self, data, params=None):
        self.p = params or Params()

        self.indptr = np.asarray(data['indptr'], dtype=np.int64)
        self.indices = np.asarray(data['indices'], dtype=np.int32)
        self.sign = np.asarray(data['sign'], dtype=np.int8)
        self.count = self.indptr.shape[0] - 1

        # Signed weight per edge in mV, precomputed so the hot loop only gathers.
        syn = np.asarray(data['syn'], dtype=np.uint16).astype(np.float32)
        source_of_edge = np.repeat(np.arange(self.count, dtype=np.int32),
                                   np.diff(self.indptr))
        signs = self.sign[source_of_edge].astype(np.float32)
        np.multiply(signs, np.where(signs < 0, self.p.inhibition_scale, 1.0), out=signs)
        self.weight = syn * signs * self.p.mv_per_synapse
        del syn, signs, source_of_edge

        # Precomputed so the leak is one multiply-add instead of a subtraction chain.
        self._leak = np.float32(1.0 - self.p.dt / self.p.tau_membrane)
        self._rest_term = np.float32(self.p.dt / self.p.tau_membrane * self.p.v_rest)

        self.v = np.full(self.count, self.p.v_rest, dtype=np.float32)
        self.external = np.zeros(self.count, dtype=np.float32)
        self._bias = np.full(self.count, self._rest_term, dtype=np.float32)
        self.extra = np.zeros(self.count, dtype=np.float32)

        # Refractory neurons are tracked as a short queue of recently fired indices
        # rather than a mask over every neuron.
        self._refractory_steps = max(1, int(round(self.p.refractory_ms / self.p.dt)))
        self._recent = [np.zeros(0, dtype=np.int64)] * self._refractory_steps

        # Spike-frequency adaptation. Without it the network is bistable: a strong
        # stimulus tips the whole brain into saturation at the refractory ceiling.
        # Real neurons accumulate a slow hyperpolarising current when they fire, and
        # it is what keeps the operating point stable over a usable parameter range.
        self.adapt = np.zeros(self.count, dtype=np.float32)
        self._adapt_decay = np.float32(np.exp(-self.p.dt / self.p.tau_adapt))

        self.spike_count = np.zeros(self.count, dtype=np.int32)
        self._window_steps = 0
        self.time_ms = 0.0
        self.spikes = np.zeros(0, dtype=np.int64)

    # ---------------------------------------------------------------- control

    def reset(self):
        self.v[:] = self.p.v_rest
        self.adapt[:] = 0
        self.extra[:] = 0
        self._recent = [np.zeros(0, dtype=np.int64)] * self._refractory_steps
        self.spike_count[:] = 0
        self._window_steps = 0
        self.time_ms = 0.0
        self.spikes = np.zeros(0, dtype=np.int64)

    def set_external(self, neurons, mv_per_step):
        """Standing drive in mV per step, applied until changed."""
        self.external[neurons] = mv_per_step
        np.add(self._rest_term, self.external, out=self._bias)

    def clear_external(self):
        self.external[:] = 0
        self._bias[:] = self._rest_term

    def inject(self, delta_v):
        """Add a one-step voltage delta — how the plastic pathway feeds back in."""
        self.extra += delta_v

    def silence(self, edges):
        """Zero specific edges so another module can own that pathway."""
        self.weight[edges] = 0.0

    # ---------------------------------------------------------------- readout

    @property
    def rate(self):
        """Firing rate in Hz, averaged over the window since the last `tick_window`."""
        elapsed = max(self._window_steps * self.p.dt, self.p.dt)
        return self.spike_count * np.float32(1000.0 / elapsed)

    def tick_window(self):
        """Start a fresh rate-averaging window."""
        self.spike_count[:] = 0
        self._window_steps = 0

    # ---------------------------------------------------------------- stepping

    def outgoing(self, sources):
        """Edge indices of every out-edge of `sources`."""
        starts = self.indptr[sources]
        lengths = self.indptr[sources + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        # Ragged arange: walk each row from its own start without a Python loop.
        ends = np.cumsum(lengths)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths)
        return offsets + np.repeat(starts, lengths)

    def step(self):
        v = self.v

        # Leak toward rest and apply standing drive in one pass.
        np.multiply(v, self._leak, out=v)
        np.add(v, self._bias, out=v)

        # Adaptation decays slowly and subtracts from the membrane.
        np.multiply(self.adapt, self._adapt_decay, out=self.adapt)
        np.subtract(v, self.adapt, out=v)

        # Postsynaptic potentials from the previous step's spikes.
        if self.spikes.size:
            edges = self.outgoing(self.spikes)
            if edges.size:
                # bincount yields float64; adding in place avoids a second buffer.
                np.add(v, np.bincount(self.indices[edges],
                                      weights=self.weight[edges],
                                      minlength=self.count), out=v, casting='unsafe')
        if self.extra.any():
            np.add(v, self.extra, out=v)
            self.extra[:] = 0

        # Hold the still-refractory neurons at reset. Only the last few steps'
        # worth of spikes can be refractory, so this touches a few hundred entries.
        for recent in self._recent:
            if recent.size:
                v[recent] = self.p.v_reset

        self.spikes = np.flatnonzero(v >= self.p.v_threshold)
        if self.spikes.size:
            v[self.spikes] = self.p.v_reset
            self.adapt[self.spikes] += self.p.adapt_mv
            self.spike_count[self.spikes] += 1
        self._recent.pop(0)
        self._recent.append(self.spikes)

        self._window_steps += 1
        self.time_ms += self.p.dt
        return self.spikes

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.rate
