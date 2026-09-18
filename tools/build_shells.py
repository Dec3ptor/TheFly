#!/usr/bin/env python3
"""Decimate MaleCNS ROI shell meshes into a compact quantised format for the web viewer.

Source meshes are neuroglancer "legacy mesh" (.ngmesh) files of 40-60 MB, which is far
too heavy to fetch in a browser. We cluster vertices onto a uniform grid, collapse each
cell to its centroid, drop the triangles that degenerate, and quantise the survivors to
16-bit coordinates. The result is a ~200 KB file that keeps the silhouette intact.

Output format (little-endian), parsed by assets/flymesh.js:
    magic    char[8]   "FLYMESH1"
    nv       uint32    vertex count
    nt       uint32    triangle count
    origin   float32[3] bounding-box minimum, in nanometres
    scale    float32[3] (bbox extent) / 65535, in nanometres per quantisation step
    verts    uint16[nv * 3]
    idx      uint32[nt * 3]
"""
import struct
import sys

import numpy as np


def read_ngmesh(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    (nv,) = struct.unpack("<I", blob[:4])
    vend = 4 + nv * 12
    verts = np.frombuffer(blob[4:vend], dtype="<f4").reshape(nv, 3)
    idx = np.frombuffer(blob[vend:], dtype="<u4").reshape(-1, 3)
    return verts.astype(np.float64), idx.astype(np.int64)


def cluster_decimate(verts, idx, cell_nm):
    """Grid-cluster vertices, then rebuild the index buffer against the centroids."""
    keys = np.floor(verts / cell_nm).astype(np.int64)
    # Pack the 3-D cell coordinate into one integer so np.unique can group on it.
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    packed = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]

    _, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    ncluster = counts.shape[0]

    # Centroid of every cluster, accumulated with bincount for speed.
    centroids = np.empty((ncluster, 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.bincount(inverse, weights=verts[:, axis], minlength=ncluster)
    centroids /= counts[:, None]

    tris = inverse[idx]
    # A triangle whose corners landed in the same cell has no area left; drop it.
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[keep]

    # Drop vertices that no surviving triangle references, then compact the indices.
    used, tris = np.unique(tris, return_inverse=True)
    return centroids[used], tris.reshape(-1, 3)


def largest_components(verts, tris, keep_fraction=0.02):
    """Keep only the substantial connected pieces of the surface.

    The source segmentation leaves small islands behind, and clustering can strand a
    few triangles of its own. Rendered additively they show up as bright specks
    floating inside the shell, so anything far smaller than the main surface goes.
    """
    parent = np.arange(verts.shape[0])

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in tris:
        for x, y in ((a, b), (b, c)):
            ra, rb = find(x), find(y)
            if ra != rb:
                parent[rb] = ra

    roots = np.array([find(i) for i in range(verts.shape[0])])
    labels, counts = np.unique(roots[tris[:, 0]], return_counts=True)
    keep = set(labels[counts >= counts.max() * keep_fraction].tolist())

    mask = np.array([root in keep for root in roots[tris[:, 0]]])
    tris = tris[mask]
    used, tris = np.unique(tris, return_inverse=True)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  dropped {dropped:,} triangles in {len(labels) - len(keep)} stray components")
    return verts[used], tris.reshape(-1, 3)


def smooth(verts, tris, iterations=12, strength=0.6):
    """Laplacian smoothing, to take the staircase out of the clustered surface.

    Clustering snaps vertices onto a coarse grid, which leaves the shell faceted
    enough that the facets read as speckle once it is drawn translucent. Averaging
    each vertex toward its neighbours a few times settles the surface down without
    moving the silhouette meaningfully.
    """
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])          # both directions
    src, dst = edges[:, 0], edges[:, 1]
    degree = np.bincount(src, minlength=verts.shape[0]).astype(np.float64)
    degree[degree == 0] = 1

    for _ in range(iterations):
        neighbour_sum = np.empty_like(verts)
        for axis in range(3):
            neighbour_sum[:, axis] = np.bincount(
                src, weights=verts[dst, axis], minlength=verts.shape[0])
        verts += strength * (neighbour_sum / degree[:, None] - verts)
    return verts


def write_flymesh(path, verts, tris):
    lo = verts.min(axis=0)
    extent = np.maximum(verts.max(axis=0) - lo, 1e-6)
    quant = np.rint((verts - lo) / extent * 65535.0).astype("<u2")

    with open(path, "wb") as fh:
        fh.write(b"FLYMESH1")
        fh.write(struct.pack("<II", verts.shape[0], tris.shape[0]))
        fh.write(np.asarray(lo, dtype="<f4").tobytes())
        fh.write(np.asarray(extent / 65535.0, dtype="<f4").tobytes())
        fh.write(quant.tobytes())
        fh.write(tris.astype("<u4").tobytes())


def main():
    src, dst, cell_nm = sys.argv[1], sys.argv[2], float(sys.argv[3])
    verts, idx = read_ngmesh(src)
    print(f"{src}: {len(verts):,} verts / {len(idx):,} tris")
    print(f"  bbox nm min={verts.min(axis=0).round(0)} max={verts.max(axis=0).round(0)}")

    dverts, dtris = cluster_decimate(verts, idx, cell_nm)
    dverts, dtris = largest_components(dverts, dtris)
    dverts = smooth(dverts, dtris)
    write_flymesh(dst, dverts, dtris)

    import os
    print(f"  -> {len(dverts):,} verts / {len(dtris):,} tris, "
          f"{os.path.getsize(dst) / 1024:.0f} KB at cell={cell_nm:.0f}nm")


if __name__ == "__main__":
    main()
