# TheFly — MaleCNS Browser

A 3D browser for [Janelia FlyEM's male *Drosophila* CNS connectome][malecns], running
entirely as a static page. No server, no API key, no build step: neuron skeletons are
streamed straight from Janelia's public data bucket into WebGL.

Search a cell type, and its neurons are drawn inside translucent brain and ventral-nerve-cord
shells. Click one to see its body ID and synapse counts, or hand the whole selection off to
Neuroglancer for the full EM volume.

## Can this really run in the browser, on GitHub Pages?

Yes, and without a backend of any kind. Three things make that work:

1. **The data is public and unauthenticated.** MaleCNS v1.0 lives in the `flyem-male-cns`
   Google Cloud Storage bucket under CC-BY. Reading it needs no token.
2. **The bucket is reachable cross-origin — via the right API.** Requests go through the GCS
   *JSON* API (`storage.googleapis.com/storage/v1/b/<bucket>/o/<object>?alt=media`), which
   returns `Access-Control-Allow-Origin` for any origin and permits `Range` preflights. The
   XML API (`storage.googleapis.com/<bucket>/<object>`) sends no CORS headers for this bucket,
   so a browser blocks it. The distinction is the whole trick — see `assets/formats.js`.
3. **Skeletons are small.** A neuron's precomputed skeleton is 10–250 KB, against ~39 MB for
   its full-resolution mesh. Skeletons are what makes a responsive in-browser viewer possible;
   loading meshes at that size would not be.

What *cannot* be done this way is the heavy EM imagery — browsing the raw 8 nm volume means
Neuroglancer, which the page links out to with the current selection preserved.

## Running it

Any static file server works, because that is all the site needs:

```sh
python3 -m http.server 8777     # then open http://127.0.0.1:8777
```

### Deploying to GitHub Pages

`.github/workflows/pages.yml` publishes the repository root on every push to `main`. Enable it
once under **Settings → Pages → Build and deployment → Source: GitHub Actions**. (The
alternative, *Deploy from a branch*, also works — the repo is already a valid static site, and
`.nojekyll` keeps Jekyll out of the way.)

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | The page |
| `assets/app.js` | Search, loading, and UI wiring |
| `assets/viewer.js` | three.js scene, picking, camera framing |
| `assets/formats.js` | Parsers for skeletons, shells, and the neuron index |
| `assets/vendor/` | three.js r160, vendored so the page has no external dependencies |
| `data/*.flymesh` | Decimated neuropil shells (266 KB gzipped, from ~94 MB of source) |
| `data/neurons.idx` | All 165,122 neurons: type, body ID, synapse counts (791 KB gzipped) |
| `tools/` | The scripts that build the two data files |

Everything under `data/` is generated. To rebuild it:

```sh
pip install numpy
python3 tools/build_index.py data/neurons.idx

# Shells need the source meshes first (~94 MB, fetched once):
BASE=https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o
curl -o brain.ngmesh "$BASE/rois%2Fbrain-shell-v2.2%2Fmesh%2Fbrain-shell.ngmesh?alt=media"
curl -o vnc.ngmesh   "$BASE/rois%2Fvnc-shell-v2%2Fmesh%2Fvnc-shell.ngmesh?alt=media"
python3 tools/build_shells.py brain.ngmesh data/brain-shell.flymesh 10000
python3 tools/build_shells.py vnc.ngmesh   data/vnc-shell.flymesh   10000
```

## Notes and limits

- A cell type is capped at 48 neurons per load, busiest first. Types like `Tm3` have over
  2,000 members, and drawing them all would be neither fast nor legible.
- Not every segment has a skeleton; those are skipped and counted in the status line.
- Connectivity — who synapses onto whom — is not in this viewer. That needs the
  [neuPrint API][neuprint], which requires an account token and so cannot be driven from an
  anonymous static page. Per-neuron links to neuPrint are provided instead.

## Credits

Connectome data © Janelia Research Campus, released CC-BY as [MaleCNS v1.0][malecns]. This
viewer is independent and not affiliated with Janelia. three.js is MIT-licensed; its notice is
kept at `assets/vendor/LICENSE.three`.

[malecns]: https://male-cns.janelia.org/
[neuprint]: https://neuprint.janelia.org/
