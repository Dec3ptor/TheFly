"""Storage for the built connectome, and where to find it.

The dataset ships inside the package (~64 MB) so a first run needs no download, no
build step, and no dependency beyond numpy. That matters more than it sounds: the
source files are LZ4-compressed Arrow, and reading them needs pyarrow, which has no
wheel for every Mac — on an Intel machine with an older macOS pip falls back to
building it from source and then asks for a Rust toolchain.

Postsynaptic indices are stored as deltas within each CSR row. They ascend inside a
row, so the deltas are small and compress to roughly half the size of the raw
indices.
"""
from pathlib import Path

import numpy as np

PACKAGED = Path(__file__).parent / 'data' / 'connectome.npz'

FIELDS = ('indptr', 'syn', 'sign', 'body_id', 'type_id', 'type_names', 'side', 'soma')


def default_path():
    """The dataset that ships with the package, or a user-built one if present."""
    import os
    override = os.environ.get('FLYSIM_DATA')
    if override:
        return Path(override)
    home = Path(os.environ.get('FLYSIM_HOME', Path.home() / '.flysim'))
    built = home / 'flysim_data.npz'
    return built if built.exists() else PACKAGED


def save(path, indptr, indices, **arrays):
    """Write the dataset, delta-encoding indices within each CSR row."""
    delta = indices.astype(np.int32, copy=True)
    delta[1:] = indices[1:] - indices[:-1]
    # The first entry of each row is absolute, not a delta from the previous row.
    row_starts = indptr[:-1][np.diff(indptr) > 0]
    delta[row_starts] = indices[row_starts]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indptr=indptr, indices_delta=delta, **arrays)


def load(path=None):
    """Read the dataset back, rebuilding absolute indices from the deltas."""
    path = Path(path) if path else default_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No connectome at {path}. The packaged one should be at {PACKAGED}; "
            f"if it is missing, run `./fly rebuild`.")

    raw = np.load(path, allow_pickle=True)
    out = {name: raw[name] for name in raw.files if name != 'indices_delta'}

    if 'indices_delta' in raw.files:
        delta = raw['indices_delta']
        indptr = raw['indptr']
        running = np.cumsum(delta, dtype=np.int64)
        # The running sum carries everything from earlier rows too. Each row's first
        # stored value was absolute, so subtracting the sum standing just before a
        # row start clears that carry for every entry in the row.
        lengths = np.diff(indptr)
        keep = lengths > 0
        starts, lengths = indptr[:-1][keep], lengths[keep]
        carry = running[starts] - delta[starts]
        out['indices'] = (running - np.repeat(carry, lengths)).astype(np.int32)
    return out
