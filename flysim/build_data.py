#!/usr/bin/env python3
"""Build the neuron-level connectome the spiking simulator runs on.

The web viewer ships a 2.7 MB cell-type graph because a browser cannot hold more.
Running natively lifts that limit, so this keeps every neuron and every synapse:
~165k neurons and ~25.5M directed connections, stored as a CSR adjacency plus a
per-neuron sign from the predicted transmitter.

Emitted as a single .npz (~160 MB):
    indptr      int64[N + 1]   CSR row starts, indexed by presynaptic neuron
    indices     int32[E]       postsynaptic neuron index
    syn         uint16[E]      synapse count for that connection
    sign        int8[N]        +1 excitatory, -1 inhibitory, 0 modulatory/unknown
    body_id     int64[N]       MaleCNS body id, ascending
    type_id     int32[N]       index into type_names
    type_names  str[T]         cell type names, by descending population
"""
import argparse
import collections
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.feather as feather

EXCITATORY = {'acetylcholine'}
INHIBITORY = {'gaba', 'glutamate', 'histamine'}
BUCKET = 'flyem-male-cns'


def fetch_json(path):
    url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
           f"{urllib.parse.quote(path, safe='')}?alt=media")
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def neuron_table():
    """Every traced neuron, ascending by body id, with its cell type."""
    types = fetch_json('v1.0/segmentation/type_property/info')['inline']
    type_of = dict(zip(types['ids'], types['properties'][0]['values']))
    population = collections.Counter(type_of.values())
    # Same ordering the web build uses, so the two stay comparable.
    type_names = sorted(population, key=lambda name: (-population[name], name))
    type_index = {name: i for i, name in enumerate(type_names)}

    body_id = np.array(sorted(int(b) for b in type_of), dtype=np.int64)
    type_id = np.array([type_index[type_of[str(b)]] for b in body_id], dtype=np.int32)
    return body_id, type_id, type_names


def neuron_signs(nt_path, body_id):
    """Excitatory/inhibitory call per neuron, from the predicted transmitter."""
    table = feather.read_table(nt_path, columns=['body', 'consensus_nt'])
    nt_body = np.asarray(table.column('body'))
    nt_call = np.array(table.column('consensus_nt').to_pylist(), dtype=object)

    position = np.clip(np.searchsorted(body_id, nt_body), 0, body_id.shape[0] - 1)
    matched = body_id[position] == nt_body

    call_of = np.empty(body_id.shape[0], dtype=object)
    call_of[position[matched]] = nt_call[matched]

    sign = np.zeros(body_id.shape[0], dtype=np.int8)
    for i, call in enumerate(call_of):
        if call in EXCITATORY:
            sign[i] = 1
        elif call in INHIBITORY:
            sign[i] = -1
    return sign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', help='connectome-weights-*.feather')
    parser.add_argument('neurotransmitters', help='body-neurotransmitters-*.feather')
    parser.add_argument('out', help='destination .npz')
    args = parser.parse_args()

    body_id, type_id, type_names = neuron_table()
    count = body_id.shape[0]
    print(f"{count:,} neurons / {len(type_names):,} cell types")

    sign = neuron_signs(args.neurotransmitters, body_id)
    tally = collections.Counter(sign.tolist())
    print(f"  signs: {tally[1]:,} excitatory, {tally[-1]:,} inhibitory, "
          f"{tally[0]:,} modulatory/unknown")

    table = feather.read_table(args.weights,
                               columns=['body_pre', 'body_post', 'weight'])
    pre_body = np.asarray(table.column('body_pre'))
    post_body = np.asarray(table.column('body_post'))
    synapses = np.asarray(table.column('weight'))

    def to_index(bodies):
        position = np.clip(np.searchsorted(body_id, bodies), 0, count - 1)
        return position, body_id[position] == bodies

    source, source_ok = to_index(pre_body)
    target, target_ok = to_index(post_body)
    usable = source_ok & target_ok
    source, target, synapses = source[usable], target[usable], synapses[usable]
    print(f"  {usable.sum():,} of {usable.shape[0]:,} edges map onto indexed neurons")

    # CSR by presynaptic neuron: a spike then reads one contiguous slice.
    order = np.lexsort((target, source))
    source, target, synapses = source[order], target[order], synapses[order]
    indptr = np.zeros(count + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(source, minlength=count))

    np.savez(args.out,
             indptr=indptr,
             indices=target.astype(np.int32),
             syn=np.clip(synapses, 0, 65535).astype(np.uint16),
             sign=sign,
             body_id=body_id,
             type_id=type_id,
             type_names=np.array(type_names))

    degree = np.diff(indptr)
    print(f"  out-degree: mean {degree.mean():.1f}, median {int(np.median(degree))}, "
          f"max {degree.max():,}")
    print(f"  {args.out}: {os.path.getsize(args.out) / 1e6:.0f} MB")


if __name__ == '__main__':
    main()
