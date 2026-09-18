#!/usr/bin/env python3
"""Build the compact neuron index the viewer searches against.

neuPrint's own segment-property files are ~3 MB each and there are two of them, so
fetching them live would cost the visitor 6 MB before the first search returns. We
merge them once, here, into a single varint-packed blob that gzips to a fraction of
that and is fetched lazily on the first search.

Output format (little-endian), parsed by assets/index.js:
    magic     char[8]    "FLYIDX1\\0"
    nTypes    uint32
    nNeurons  uint32
    nameLen   uint32     byte length of the type-name block
    names     utf-8      type names joined by "\\n", ordered by type index
    records   varint*    nNeurons records of (idDelta, typeIdx, synPre, synPost),
                         ordered by ascending body id so idDelta stays small
"""
import json
import struct
import sys
import urllib.parse
import urllib.request

BUCKET = "flyem-male-cns"
PREFIX = "v1.0/segmentation"


def fetch(path):
    url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
           f"{urllib.parse.quote(path, safe='')}?alt=media")
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def varint(value, out):
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def main():
    dst = sys.argv[1]

    types = fetch(f"{PREFIX}/type_property/info")["inline"]
    numeric = fetch(f"{PREFIX}/numeric_properties/info")["inline"]

    type_of = dict(zip(types["ids"], types["properties"][0]["values"]))
    by_prop = {p["id"]: p["values"] for p in numeric["properties"]}
    pre_of = dict(zip(numeric["ids"], by_prop["syn_pre"]))
    post_of = dict(zip(numeric["ids"], by_prop["syn_post"]))

    # Type index is assigned in descending population order, so the commonest types
    # get the smallest varints.
    population = {}
    for name in type_of.values():
        population[name] = population.get(name, 0) + 1
    names = sorted(population, key=lambda n: (-population[n], n))
    index_of = {name: i for i, name in enumerate(names)}

    records = bytearray()
    previous = 0
    for body in sorted(type_of, key=int):
        body_id = int(body)
        varint(body_id - previous, records)
        previous = body_id
        varint(index_of[type_of[body]], records)
        varint(pre_of.get(body, 0), records)
        varint(post_of.get(body, 0), records)

    name_blob = "\n".join(names).encode("utf-8")
    with open(dst, "wb") as fh:
        fh.write(b"FLYIDX1\0")
        fh.write(struct.pack("<III", len(names), len(type_of), len(name_blob)))
        fh.write(name_blob)
        fh.write(records)

    import gzip
    import os
    raw = os.path.getsize(dst)
    packed = len(gzip.compress(open(dst, "rb").read(), 9))
    print(f"{len(type_of):,} neurons / {len(names):,} types")
    print(f"  {dst}: {raw/1024:.0f} KB raw, {packed/1024:.0f} KB gzipped "
          f"({records.__len__()/len(type_of):.1f} bytes per neuron)")


if __name__ == "__main__":
    main()
