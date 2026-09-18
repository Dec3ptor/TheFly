# flysim — the connectome, spiking, in a body

A native macOS (and Linux) app that runs **165,122 leaky integrate-and-fire neurons wired
by the measured MaleCNS connectome**, drives a body around an arena with them, and lets
that brain learn.

This is the step past the web viewer in the repository root. The browser version runs a
rate model over 11,752 *cell types* because that is what fits in a tab. Running natively
lifts the limit: every neuron, every one of the 25.5M measured synapses, spiking.

## Running it

One command, from the repository root:

```sh
./fly
```

That is the whole thing. It makes a virtual environment in `.venv`, installs numpy and
pyarrow into it, downloads the connectome on first run, builds the data file, and opens
the UI. Later runs skip straight to launching. On macOS you can also just **double-click
`Fly.command`** in Finder.

The virtual environment matters: on macOS a plain `pip install` into the Homebrew or
system Python is refused outright with `externally-managed-environment` (PEP 668), so
`./fly` keeps everything local and never touches your system Python.

| | |
| --- | --- |
| `./fly` | set up if needed, then run |
| `./fly probe --ticks 200` | run headless and print telemetry, no browser |
| `./fly rebuild` | force a rebuild of the data file |

First run downloads ~545 MB of public CC-BY data and builds a ~160 MB file into
`~/.flysim` (override with `FLYSIM_HOME`). It needs Python 3.10+; if you don't have one
`./fly` says so and tells you how to get it.

If you'd rather drive it yourself, `pip install -e .` gives you a `flysim` command with
the same `setup` / `run` / `probe` subcommands.

## The three layers

**Brain** (`brain.py`). One LIF unit per neuron. A spike steps each postsynaptic
membrane by `mv_per_synapse` × the anatomical synapse count, signed by the presynaptic
transmitter — acetylcholine excites; GABA, glutamate and histamine inhibit. Between
spikes the membrane leaks back to rest with τ=20 ms. Runs at **1–2× real time** on one
core, which is what makes a closed sensorimotor loop possible at all.

**Body and world** (`world.py`). A circular arena with dark posts and Gaussian odour
plumes. The fly is a point with a heading; the brain sets forward speed and turn rate.
No leg biomechanics — the point is the loop, not the gait.

**The loop** (`agent.py`). Every 20 ms of brain time:
- *Sense* — apparent size and looming of posts drives the looming-sensitive lobula
  columnar cells on the corresponding side; odour concentration at each antenna drives
  that odorant's own ORNs; reinforcement drives the dopaminergic neurons.
- *Act* — the left/right firing imbalance across the 1,280 descending neurons, the real
  anatomical bottleneck between brain and nerve cord, becomes a steering command.
- *Learn* — see below.

## Learning

The 61,210 Kenyon-cell→MBON synapses are lifted out of the fixed connectome into a
plastic matrix. Everything else stays exactly as measured. The rule is the canonical fly
one: **dopamine depresses the KC→MBON synapses of Kenyon cells that are active when
reinforcement arrives.**

Holding the fly on an aversive odour for ~5 s of simulated time:

```
before training: MBON 4.65 Hz, KC→MBON depression 1.17%
after  training: MBON 3.29 Hz, KC→MBON depression 7.59%
                 1,905 of 4,064 Kenyon cells depressed
```

The MBON response to that odour drops by ~29%. That is associative learning in the
circuit that actually does it in the animal, not a bolt-on.

## What is validated

Driving a population and seeing where activity goes reproduces known circuits:

| Stimulate | Response |
| --- | --- |
| `LC4` | PVLP024 → DNp04, DNp02, DNp01, and on to **motor neurons** (`hi1 MN`, `ps1 MN`, `MNwm36`) |
| `ORN_DA1` | its own projection neurons, then the lateral horn |
| `R1-R6` | nothing downstream — correct, histamine is inhibitory, so photoreceptors *suppress* L1/L2 |

Kenyon cells sit at ~0.1 Hz, which is right: they are famously sparse coders. Baseline
network activity is ~0.4 Hz.

## What is wrong with it

Read this part. The model is interesting, not correct.

- **The operating point is calibrated, not measured.** `mv_per_synapse` is a free
  parameter. The connectome gives wiring and synapse counts; it does not give
  conductances. Below ~0.12 mV nothing propagates, above ~0.16 the network tips into
  runaway. The useful band is narrow and the dynamics inside it are chaotic — nearby
  parameter values give visibly different behaviour.
- **Some populations saturate.** Strong odour input can pin antennal-lobe local neurons
  and projection neurons at the refractory ceiling (~350 Hz). When APL — the mushroom
  body's giant inhibitory neuron — saturates, it silences the Kenyon cells outright. The
  default `ODOUR_DRIVE` is tuned to mostly avoid this, not to make it impossible.
- **Retinotopy is approximate.** True receptive fields would come from each neuron's
  optic-lobe column. The dataset's column assignments cover only 14% of photoreceptors
  and are not one-to-one, so `atlas.eye_azimuth` falls back to anatomical position within
  the optic lobe. It is an approximation of where a neuron looks.
- **Vision is injected as features, not pixels.** The world computes looming and drives
  the looming detectors. It does not render an image onto photoreceptors — and it could
  not usefully, since photoreceptors are histaminergic and inhibit their targets.
- **The steering readout is a modelling choice.** That descending-neuron left/right
  imbalance maps to turn direction the way it does here is an assumption. I have not
  validated its *sign* against fly behaviour, and the resulting trajectories are noisy
  rather than a clean escape turn.
- **Transmitter calls are ML predictions** for most neurons, not measurements.
- **No neuromodulation, no synaptic dynamics, no gap junctions, no spike timing.**

What it does show: activity moving through real measured wiring, fast enough to close a
loop with a body, with plasticity in the right place.

## Files

| Path | What |
| --- | --- |
| `brain.py` | The LIF engine and its hot loop |
| `atlas.py` | Populations, sides, sensory/motor interfaces |
| `world.py` | Arena, body, posts, odour plumes |
| `agent.py` | Sense → act → learn |
| `server.py`, `ui/` | Local web UI |
| `build.py` | Fetches the public data and builds the .npz |
| `cli.py` | `setup`, `run`, `probe` |

Data © Janelia Research Campus, CC-BY, from [MaleCNS v1.0](https://male-cns.janelia.org/).
Independent project, not affiliated with Janelia.
