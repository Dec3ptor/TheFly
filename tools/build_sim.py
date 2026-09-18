#!/usr/bin/env python3
"""Build the cell-type connectivity graph that drives the in-browser activity model.

The full connectome is 25.6M neuron-to-neuron edges (502 MB), which no web page can
hold. Aggregating to cell type gives 11,752 nodes and 3.85M edges, and pruning that
to the connections which carry most of the synaptic mass leaves a graph small enough
to ship while keeping ~88% of all synapses.

Edge weights are normalised by the TARGET's total input, so a weight is "the fraction
of this type's input that arrives from that type". Drive is then bounded on [-1, 1]
however large the presynaptic population is, which is what keeps the simulation stable
without hand-tuned gains per node.

Signs come from the predicted neurotransmitter, following the usual convention for
Drosophila: acetylcholine excites; GABA, glutamate (via GluCl) and histamine inhibit;
the aminergic transmitters are modulatory and are given no sign here.

Type indices match data/neurons.idx exactly, so the two files share one numbering and
the names are not stored twice.

Output format (little-endian), parsed by assets/sim.js:
    magic     char[8]     "FLYSIM1\\0"
    nTypes    uint32
    nEdges    uint32
    sign      int8[nTypes]        +1 excitatory, -1 inhibitory, 0 modulatory/unknown
    offsets   uint32[nTypes + 1]  CSR row starts, indexed by SOURCE type
    targets   uint16[nEdges]      target type index
    weights   uint16[nEdges]      normalised weight * 65535
"""
import collections
import json
import struct
import sys
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.feather as feather

WEIGHT_FLOOR = 0.002      # keep edges carrying >= 0.2% of the target's input
TOP_PER_SOURCE = 24       # ...and always keep each type's strongest outputs

EXCITATORY = {'acetylcholine'}
INHIBITORY = {'gaba', 'glutamate', 'histamine'}

BUCKET = 'flyem-male-cns'


def fetch_json(path):
    url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
           f"{urllib.parse.quote(path, safe='')}?alt=media")
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def main():
    weights_path, nt_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]

    # Same type ordering as build_index.py: by descending population, then name.
    types = fetch_json('v1.0/segmentation/type_property/info')['inline']
    type_of = dict(zip(types['ids'], types['properties'][0]['values']))
    population = collections.Counter(type_of.values())
    names = sorted(population, key=lambda n: (-population[n], n))
    index_of = {name: i for i, name in enumerate(names)}
    n_types = len(names)

    bodies = np.array([int(b) for b in type_of], dtype=np.int64)
    body_type = np.array([index_of[type_of[str(b)]] for b in bodies], dtype=np.int32)
    order = np.argsort(bodies)
    sorted_bodies, sorted_types = bodies[order], body_type[order]
    print(f"{len(bodies):,} neurons / {n_types:,} cell types")

    # ---- signs, by majority transmitter within each type -------------------
    nt = feather.read_table(nt_path, columns=['body', 'consensus_nt'])
    nt_body = np.asarray(nt.column('body'))
    nt_call = nt.column('consensus_nt').to_pylist()

    votes = [collections.Counter() for _ in range(n_types)]
    position = np.searchsorted(sorted_bodies, nt_body)
    position = np.clip(position, 0, len(sorted_bodies) - 1)
    matched = sorted_bodies[position] == nt_body
    for type_idx, call in zip(sorted_types[position[matched]],
                              (c for c, m in zip(nt_call, matched) if m)):
        votes[type_idx][call] += 1

    sign = np.zeros(n_types, dtype=np.int8)
    for i, counter in enumerate(votes):
        ranked = [(n, c) for c, n in counter.items() if c != 'unclear']
        if not ranked:
            continue
        winner = max(ranked)[1]
        sign[i] = 1 if winner in EXCITATORY else -1 if winner in INHIBITORY else 0
    tally = collections.Counter(sign.tolist())
    print(f"  signs: {tally[1]:,} excitatory, {tally[-1]:,} inhibitory, "
          f"{tally[0]:,} modulatory/unknown")

    # ---- aggregate the connectome to type level ----------------------------
    table = feather.read_table(weights_path, columns=['body_pre', 'body_post', 'weight'])
    pre = np.asarray(table.column('body_pre'))
    post = np.asarray(table.column('body_post'))
    synapses = np.asarray(table.column('weight')).astype(np.float64)

    def lookup(arr):
        pos = np.clip(np.searchsorted(sorted_bodies, arr), 0, len(sorted_bodies) - 1)
        return sorted_types[pos], sorted_bodies[pos] == arr

    src_type, src_ok = lookup(pre)
    dst_type, dst_ok = lookup(post)
    usable = src_ok & dst_ok
    print(f"  {usable.sum():,} of {len(synapses):,} edges map onto indexed neurons")

    key = src_type[usable].astype(np.int64) * n_types + dst_type[usable]
    unique_key, inverse = np.unique(key, return_inverse=True)
    totals = np.bincount(inverse, weights=synapses[usable])
    src = (unique_key // n_types).astype(np.int32)
    dst_idx = (unique_key % n_types).astype(np.int32)
    print(f"  aggregated to {len(src):,} type-to-type connections")

    # Fraction of the target's total input that comes along this edge.
    total_input = np.bincount(dst_idx, weights=totals, minlength=n_types)
    norm = totals / np.maximum(total_input[dst_idx], 1.0)

    # ---- prune ------------------------------------------------------------
    keep = norm >= WEIGHT_FLOOR
    # Rescue each source's strongest outputs so small types keep their pathways.
    strongest = np.lexsort((-norm, src))
    run_start = np.searchsorted(src[strongest], np.arange(n_types))
    run_end = np.searchsorted(src[strongest], np.arange(n_types), side='right')
    for start, end in zip(run_start, run_end):
        keep[strongest[start:min(start + TOP_PER_SOURCE, end)]] = True

    src, dst_idx, norm = src[keep], dst_idx[keep], norm[keep]
    retained = totals[keep].sum() / totals.sum()
    print(f"  pruned to {len(src):,} edges, retaining {100 * retained:.1f}% of synapses")

    # ---- write CSR --------------------------------------------------------
    row_order = np.lexsort((dst_idx, src))
    src, dst_idx, norm = src[row_order], dst_idx[row_order], norm[row_order]
    offsets = np.zeros(n_types + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(np.bincount(src, minlength=n_types))

    quantised = np.clip(np.rint(norm * 65535), 0, 65535).astype('<u2')
    with open(dst, 'wb') as fh:
        fh.write(b'FLYSIM1\0')
        fh.write(struct.pack('<II', n_types, len(src)))
        fh.write(sign.astype('<i1').tobytes())
        fh.write(offsets.astype('<u4').tobytes())
        fh.write(dst_idx.astype('<u2').tobytes())
        fh.write(quantised.tobytes())

    import gzip
    import os
    raw = os.path.getsize(dst)
    packed = len(gzip.compress(open(dst, 'rb').read(), 9))
    print(f"  {dst}: {raw/1e6:.1f} MB raw, {packed/1e6:.1f} MB gzipped")


if __name__ == '__main__':
    main()
