# TheFly — MaleCNS Browser

A 3D browser **and activity model** for [Janelia FlyEM's male *Drosophila* CNS
connectome][malecns], running entirely as a static page. No server, no API key, no build
step: neuron skeletons are streamed straight from Janelia's public data bucket into WebGL,
and the simulation runs in your tab.

Search a cell type and its neurons are drawn inside translucent brain and ventral-nerve-cord
shells. Stimulate a type and watch activity spread through the real wiring — the types it
drives light up, and you can pull them into the view to see the pathway assemble itself.

**Live: <https://dec3ptor.github.io/TheFly/>**

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
   its full-resolution mesh. Skeletons are what make a responsive in-browser viewer possible.

What *cannot* be done this way is the heavy EM imagery — browsing the raw 8 nm volume means
Neuroglancer, which the page links out to with the current selection preserved.

## The activity model

### What it does

Each **cell type** is one unit holding a membrane-like state `v`, driven by its presynaptic
partners:

```
drive(b) = Σ over a of  sign(a) · weight(a → b) · rate(a)
v(b)    += dt/tau · ( −v(b) + gain · drive(b) + stimulus(b) )
rate(b)  = clamp(v(b), 0, 1)
```

`weight(a → b)` is **the fraction of b's input synapses that arrive from a**, so drive stays
bounded on [−1, 1] however large the presynaptic population is. That normalisation is what
keeps the model stable without per-node hand-tuning.

`sign(a)` comes from each type's predicted neurotransmitter, following the usual convention
for *Drosophila*: acetylcholine excites; GABA, glutamate (via GluCl) and histamine inhibit;
the aminergic transmitters are treated as modulatory and given no sign. 98% of neurons carry
a usable call — 5,984 excitatory types, 5,223 inhibitory, 545 modulatory or unknown.

### Does it produce the right answers?

It reproduces known circuits, which is the point of building it on real wiring:

| Stimulate | Strongest responders | Expected |
| --- | --- | --- |
| `LC4` (looming) | PVLP024, **DNp04**, **DNp02**, **DNp01** | looming → PVLP → giant-fiber escape |
| `LPLC2` (looming) | PVLP069, PVLP111, **DNp01**, **DNp04** | a parallel looming channel onto the same descending neurons |
| `ORN_D` (odour) | **D_adPN**, LHPV4k1, LHPV4a3 | ORN → its glomerulus' projection neuron → lateral horn |

`R1-R6` is a useful negative case: stimulating photoreceptors activates nothing downstream,
because histamine is **inhibitory** — they suppress L1/L2/L3 rather than driving them. That
is the biology, not a bug.

### What it is not

This is a coarse rate model, not a biophysical simulation. Be clear-eyed about the limits:

- **No spikes, no channels, no neuromodulation, no synaptic dynamics.** One scalar per type.
- **The connectome fixes wiring, not strength.** Synapse counts are a proxy for synaptic
  weight; real efficacy varies and is not in this data.
- **Neurotransmitters are ML predictions**, not measurements, for most neurons.
- **Activity is per cell type, not per neuron.** Every neuron of a type shares one value.
  Neuron-level dynamics would need all 25.6M edges (502 MB), which no web page can hold.
- **`gain` is a free parameter.** Below ~1.5 activity barely spreads; above ~2.2 the network
  runs away and everything saturates. The default of 2.0 sits in the useful band. The slider
  is exposed precisely so you can see that fragility for yourself.

It shows how activity spreads through real wiring. It does not tell you what the fly is
thinking.

## Running the fly natively

The browser version is a rate model over cell types, because that is what fits in a tab.
`flysim/` is the step past that: a native macOS/Linux app running **165,122 leaky
integrate-and-fire neurons over all 25.5M measured synapses**, at 1-2x real time, driving
a body around an arena — with mushroom-body plasticity so it can be conditioned.

```sh
pip install -e .   &&   flysim setup   &&   flysim run
```

See [`flysim/README.md`](flysim/README.md), which includes a frank account of where the
model is weak.

## Running it

Any static file server works, because that is all the site needs:

```sh
python3 -m http.server 8777     # then open http://127.0.0.1:8777
```

### Deploying to GitHub Pages

`.github/workflows/pages.yml` publishes the repository root on every push to `main`, and
turns Pages on itself the first time it runs (`enablement: true`), so no manual repository
setup is needed.

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | The page |
| `assets/app.js` | Search, loading, simulation loop, and UI wiring |
| `assets/viewer.js` | three.js scene, picking, camera framing, activity colouring |
| `assets/sim.js` | The rate model and its CSR sparse step |
| `assets/formats.js` | Parsers for skeletons, shells, and the neuron index |
| `assets/vendor/` | three.js r160, vendored so the page has no external dependencies |
| `data/*.flymesh` | Decimated neuropil shells (266 KB gzipped, from ~94 MB of source) |
| `data/neurons.idx` | All 165,122 neurons: type, body ID, synapse counts (791 KB gzipped) |
| `data/connectome.sim` | 11,752 types, 894k signed connections (2.7 MB gzipped, from 502 MB) |
| `tools/` | The scripts that build the three data files |

Everything under `data/` is generated and reproducible:

```sh
pip install numpy pyarrow
python3 tools/build_index.py data/neurons.idx

BASE=https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o
C=v1.0%2Fconnectome-data%2Fflat-connectome

# Shells (~94 MB)
curl -o brain.ngmesh "$BASE/rois%2Fbrain-shell-v2.2%2Fmesh%2Fbrain-shell.ngmesh?alt=media"
curl -o vnc.ngmesh   "$BASE/rois%2Fvnc-shell-v2%2Fmesh%2Fvnc-shell.ngmesh?alt=media"
python3 tools/build_shells.py brain.ngmesh data/brain-shell.flymesh 10000
python3 tools/build_shells.py vnc.ngmesh   data/vnc-shell.flymesh   10000

# Connectivity (~545 MB)
curl -o weights.feather "$BASE/$C%2Fconnectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather?alt=media"
curl -o nt.feather      "$BASE/$C%2Fbody-neurotransmitters-male-cns-v1.0.feather?alt=media"
python3 tools/build_sim.py weights.feather nt.feather data/connectome.sim
```

## Notes and limits

- **Neurons per type is capped**, selectable at 50 / 150 / 500 / as-many-as-possible
  (hard stop 750). Types like `Tm3` have over 2,000 members; each one is a separate HTTP
  request, so drawing them all is slow and illegible rather than impossible.
- Not every segment has a skeleton; those are skipped and counted in the status line.
- The connectivity graph is pruned to the 894k connections carrying ~88% of all synapses
  (of 3.85M type-to-type pairs, from 25.6M neuron-to-neuron edges). Each type also keeps its
  24 strongest outputs regardless, so small types do not lose their pathways.
- Per-neuron connectivity queries need the [neuPrint API][neuprint], which requires an
  account token and so cannot be driven from an anonymous static page. Per-neuron links to
  neuPrint are provided instead.

## Credits

Connectome data © Janelia Research Campus, released CC-BY as [MaleCNS v1.0][malecns]. This
viewer is independent and not affiliated with Janelia. three.js is MIT-licensed; its notice
is kept at `assets/vendor/LICENSE.three`.

[malecns]: https://male-cns.janelia.org/
[neuprint]: https://neuprint.janelia.org/
