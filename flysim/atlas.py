"""Which neurons are what: populations, sides, and the sensory/motor interfaces.

Everything here is derived from the dataset's own cell-type names and instance names
(which carry `_L` / `_R`), so nothing is hand-listed beyond the naming conventions the
MaleCNS annotators used.
"""
import re

import numpy as np

# Cell-type name patterns for the populations the simulator needs to reach into.
FAMILIES = {
    'photoreceptor': r'^R[1-8]',
    'orn': r'^ORN_',
    'lamina': r'^L[1-5]$',
    'looming': r'^(LC4|LPLC2|LPLC1|LC6|LC9|LC10|LC11|LC12|LC13|LC15|LC16|LC17|LC18)',
    'descending': r'^DN[apgbxme]',
    'motor': r'^(MN|hi1 MN|ps1 MN|b1 MN|b2 MN|b3 MN|i1 MN|i2 MN|iii1 MN|iii3 MN|tp1 MN|tp2 MN|tpN MN|nm MN|hg[1-4] MN|pr[1-6] MN)',
    'kenyon': r'^KC',
    'mbon': r'^MBON',
    'dopaminergic': r'^(PAM|PPL|PPM)',
    'central_complex': r'^(EPG|PEG|PEN|Delta7|ER[0-9]|FC[0-9]|PFL|PFN)',
}


class Atlas:
    def __init__(self, data):
        self.type_names = [str(n) for n in data['type_names']]
        self.type_id = np.asarray(data['type_id'])
        self.body_id = np.asarray(data['body_id'])
        self.side = np.asarray(data['side'])
        self.soma = np.asarray(data['soma'])
        self.count = self.type_id.shape[0]

        self._index_of = {name: i for i, name in enumerate(self.type_names)}
        self._name_of_neuron = np.array(self.type_names, dtype=object)[self.type_id]

        self.families = {name: self._match(pattern)
                         for name, pattern in FAMILIES.items()}

    # ---------------------------------------------------------------- lookups

    def _match(self, pattern):
        rx = re.compile(pattern)
        wanted = {i for i, name in enumerate(self.type_names) if rx.match(name)}
        if not wanted:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.isin(self.type_id, list(wanted)))

    def type_index(self, name):
        return self._index_of.get(name, -1)

    def neurons_of_type(self, name):
        index = self._index_of.get(name)
        if index is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.type_id == index)

    def neurons_matching(self, pattern):
        return self._match(pattern)

    def type_name_of(self, neuron):
        return self._name_of_neuron[neuron]

    def by_side(self, neurons):
        """Split a population into (left, right); midline neurons go to neither."""
        side = self.side[neurons]
        return neurons[side < 0], neurons[side > 0]

    def rate_by_type(self, rate):
        """Mean firing rate per cell type."""
        totals = np.bincount(self.type_id, weights=rate, minlength=len(self.type_names))
        counts = np.bincount(self.type_id, minlength=len(self.type_names))
        return totals / np.maximum(counts, 1)

    # ---------------------------------------------------------------- vision

    def eye_azimuth(self, neurons):
        """A coarse horizontal receptive-field angle for each of `neurons`, in [-1, 1].

        True retinotopy would come from the optic-lobe column each neuron belongs to,
        but the dataset's column assignments cover only a fraction of these cells and
        are not one-to-one, so this falls back to anatomy: position along the
        anterior-posterior axis within the neuron's own optic lobe. Cells sitting
        frontally get values near -1, caudal ones near +1. It is an approximation of
        where a neuron looks, not a measurement of it.
        """
        position = self.soma[neurons, 2]
        known = np.isfinite(position)
        azimuth = np.zeros(neurons.shape[0], dtype=np.float32)
        if known.sum() < 4:
            return azimuth
        low, high = np.percentile(position[known], [5, 95])
        if high - low < 1.0:
            return azimuth
        scaled = (position - low) / (high - low) * 2.0 - 1.0
        azimuth[known] = np.clip(scaled[known], -1.0, 1.0)
        # Unplaced cells sit at the middle of the field rather than biasing an edge.
        azimuth[~known] = 0.0
        return azimuth

    def summary(self):
        lines = [f"{self.count:,} neurons, {len(self.type_names):,} cell types"]
        for name, members in self.families.items():
            left, right = self.by_side(members)
            lines.append(f"  {name:<16} {members.shape[0]:>6,}  "
                         f"(L {left.shape[0]:,} / R {right.shape[0]:,})")
        return "\n".join(lines)
