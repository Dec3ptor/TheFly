#!/usr/bin/env python3
"""One-shot builder for everything the simulator needs, into a single .npz (~160 MB).

Sources, all public and unauthenticated, from the MaleCNS v1.0 bucket:

  * connectome-weights-...-significant-only.feather  the 25.5M-edge connectome (502 MB)
  * body-neurotransmitters-...feather               per-neuron transmitter calls (43 MB)
  * segmentation/type_property                       cell type per neuron
  * segmentation/instance_property                   instance name, which carries side
  * malecns-v1.0-soma-points                         soma position per neuron (2.5 MB)

The two .feather files are large, so they are passed in as local paths; `flysim setup`
downloads them first. Everything else is fetched here.

Contents of the output:
    indptr      int64[N+1]   CSR row starts by presynaptic neuron
    indices     int32[E]     postsynaptic neuron
    syn         uint16[E]    synapse count
    sign        int8[N]      +1 excitatory, -1 inhibitory, 0 modulatory/unknown
    body_id     int64[N]     ascending MaleCNS body id
    type_id     int32[N]     index into type_names
    type_names  str[T]
    side        int8[N]      -1 left, +1 right, 0 midline/unknown
    soma        float32[N,3] soma position in nanometres, NaN where unknown
"""
import argparse
import collections
import gzip
import json
import os
import struct
import urllib.parse
import urllib.request

import numpy as np

from .dataset import save as save_dataset

try:
    import pyarrow.feather as feather
except ImportError:  # pragma: no cover - only needed to rebuild
    feather = None

EXCITATORY = {'acetylcholine'}
INHIBITORY = {'gaba', 'glutamate', 'histamine'}
BUCKET = 'flyem-male-cns'
SOMA_POINTS = 'v1.0/malecns-v1.0-soma-points'
VOXEL_NM = 8.0


def object_url(path):
    return (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
            f"{urllib.parse.quote(path, safe='')}?alt=media")


def fetch_json(path):
    with urllib.request.urlopen(object_url(path)) as resp:
        return json.load(resp)


def fetch_bytes(path):
    with urllib.request.urlopen(object_url(path)) as resp:
        return resp.read()


def neuron_table():
    """Every traced neuron, ascending by body id, with cell type and side."""
    types = fetch_json('v1.0/segmentation/type_property/info')['inline']
    type_of = dict(zip(types['ids'], types['properties'][0]['values']))

    instances = fetch_json('v1.0/segmentation/instance_property/info')['inline']
    instance_of = dict(zip(instances['ids'], instances['properties'][0]['values']))

    population = collections.Counter(type_of.values())
    type_names = sorted(population, key=lambda name: (-population[name], name))
    type_index = {name: i for i, name in enumerate(type_names)}

    body_id = np.array(sorted(int(b) for b in type_of), dtype=np.int64)
    type_id = np.empty(body_id.shape[0], dtype=np.int32)
    side = np.zeros(body_id.shape[0], dtype=np.int8)

    for i, body in enumerate(body_id):
        key = str(body)
        type_id[i] = type_index[type_of[key]]
        # Instance names end in _L / _R; anything else is midline or unlabelled.
        tail = instance_of.get(key, '').rsplit('_', 1)[-1]
        side[i] = -1 if tail == 'L' else 1 if tail == 'R' else 0

    return body_id, type_id, type_names, side


def soma_positions(body_id):
    """Soma point per neuron, in nanometres, NaN where the dataset has none.

    The points live in one neuroglancer sharded-annotation chunk. With shard_bits
    and minishard_bits both zero there is a single minishard holding a single chunk,
    so the whole thing decodes in one pass.
    """
    info = json.loads(fetch_bytes(f'{SOMA_POINTS}/info'))
    blob = fetch_bytes(f'{SOMA_POINTS}/by_spatial_level_0/0.shard')

    index_start, index_end = struct.unpack('<QQ', blob[:16])
    minishard = gzip.decompress(blob[16 + index_start:16 + index_end])
    table = np.frombuffer(minishard, dtype='<u8').reshape(3, -1)
    offset, size = int(table[1][0]), int(table[2][0])
    raw = gzip.decompress(blob[16 + offset:16 + offset + size])

    count = struct.unpack('<Q', raw[:8])[0]
    widths = {'uint32': 4, 'int32': 4, 'float32': 4,
              'uint16': 2, 'int16': 2, 'uint8': 1, 'int8': 1}
    property_bytes = sum(widths[p['type']] for p in info['properties'])
    # Point geometry is three float32; properties follow, padded to a 4-byte multiple.
    stride = 12 + (property_bytes + 3) // 4 * 4

    records = np.frombuffer(raw, dtype=np.uint8, count=count * stride,
                            offset=8).reshape(count, stride)
    position = records[:, :12].copy().view('<f4').reshape(count, 3)
    ids = np.frombuffer(raw[8 + count * stride:], dtype='<u8', count=count)

    soma = np.full((body_id.shape[0], 3), np.nan, dtype=np.float32)
    slot = np.clip(np.searchsorted(body_id, ids), 0, body_id.shape[0] - 1)
    matched = body_id[slot] == ids
    soma[slot[matched]] = position[matched] * VOXEL_NM
    return soma


def neuron_signs(nt_path, body_id):
    table = feather.read_table(nt_path, columns=['body', 'consensus_nt'])
    nt_body = np.asarray(table.column('body'))
    nt_call = np.array(table.column('consensus_nt').to_pylist(), dtype=object)

    slot = np.clip(np.searchsorted(body_id, nt_body), 0, body_id.shape[0] - 1)
    matched = body_id[slot] == nt_body
    call_of = np.empty(body_id.shape[0], dtype=object)
    call_of[slot[matched]] = nt_call[matched]

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

    if feather is None:
        raise SystemExit(
            "Rebuilding needs pyarrow to read the LZ4-compressed Arrow sources:\n"
            "    .venv/bin/pip install pyarrow\n"
            "The packaged dataset needs none of this — you only need pyarrow if you\n"
            "are rebuilding from Janelia's originals.")

    body_id, type_id, type_names, side = neuron_table()
    count = body_id.shape[0]
    print(f"{count:,} neurons / {len(type_names):,} cell types")
    print(f"  sides: {int((side < 0).sum()):,} left, {int((side > 0).sum()):,} right, "
          f"{int((side == 0).sum()):,} midline or unlabelled")

    soma = soma_positions(body_id)
    print(f"  soma points: {int(np.isfinite(soma[:, 0]).sum()):,}")

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
        slot = np.clip(np.searchsorted(body_id, bodies), 0, count - 1)
        return slot, body_id[slot] == bodies

    source, source_ok = to_index(pre_body)
    target, target_ok = to_index(post_body)
    usable = source_ok & target_ok
    source, target, synapses = source[usable], target[usable], synapses[usable]
    print(f"  {usable.sum():,} of {usable.shape[0]:,} edges map onto indexed neurons")

    order = np.lexsort((target, source))
    source, target, synapses = source[order], target[order], synapses[order]
    indptr = np.zeros(count + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(source, minlength=count))

    save_dataset(args.out,
                 indptr=indptr,
                 indices=target.astype(np.int32),
                 syn=np.clip(synapses, 0, 65535).astype(np.uint16),
                 sign=sign,
                 body_id=body_id,
                 type_id=type_id,
                 type_names=np.array(type_names),
                 side=side,
                 soma=soma)

    degree = np.diff(indptr)
    print(f"  out-degree: mean {degree.mean():.1f}, max {degree.max():,}")
    print(f"  {args.out}: {os.path.getsize(args.out) / 1e6:.0f} MB")


if __name__ == '__main__':
    main()
