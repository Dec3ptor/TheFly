"""The closed loop: world into spikes, spikes into movement, and learning in between.

Sensing injects drive into the populations that actually carry those signals — ORNs for
odour, looming-sensitive lobula columnar cells for approaching objects. Acting reads the
descending neurons, which are the real anatomical bottleneck between brain and nerve
cord, and takes their left/right asymmetry as a steering command.

Learning is confined to the Kenyon-cell-to-MBON synapses, which is where associative
learning happens in the fly. Those edges are lifted out of the fixed connectome into a
small plastic matrix; everything else stays as measured.
"""
import math

import numpy as np

from .atlas import Atlas
from .brain import Brain, Params


class Agent:
    BODY_DT = 0.02          # s between body updates
    BRAIN_MS_PER_TICK = 20  # brain steps per body update, at dt = 1 ms

    # Drive strengths, in mV per step, for each sensory channel.
    LOOMING_DRIVE = 1.3
    ODOUR_DRIVE = 0.9
    DAN_DRIVE = 1.2

    LEARNING_RATE = 0.06
    WEIGHT_FLOOR = 0.05     # depression cannot take a synapse below this fraction

    def __init__(self, data, world, params=None):
        self.brain = Brain(data, params or Params(mv_per_synapse=0.15,
                                                  inhibition_scale=3.0))
        self.atlas = Atlas(data)
        self.world = world

        families = self.atlas.families
        self.looming_left, self.looming_right = self.atlas.by_side(families['looming'])
        self.dn_left, self.dn_right = self.atlas.by_side(families['descending'])
        self.motor = families['motor']
        self.kenyon = families['kenyon']
        self.mbon = families['mbon']

        # Reward and punishment are carried by different dopaminergic families.
        self.dan_reward = self.atlas.neurons_matching(r'^PAM')
        self.dan_punish = self.atlas.neurons_matching(r'^(PPL|PPM)')

        # ORNs grouped by receptor type, so an odour drives its own glomerulus.
        self.orn_by_receptor = {}
        for neuron in families['orn']:
            name = str(self.atlas.type_name_of(neuron))
            self.orn_by_receptor.setdefault(name, []).append(neuron)
        self.orn_by_receptor = {k: np.array(v) for k, v in self.orn_by_receptor.items()}

        self._setup_plasticity()

        self.forward = 0.0
        self.turn = 0.0
        self.baseline_walk = 0.35
        self.history = []

    # ---------------------------------------------------------------- plasticity

    def _setup_plasticity(self):
        """Lift KC->MBON out of the fixed graph into a plastic matrix."""
        self.kc_slot = {int(n): i for i, n in enumerate(self.kenyon)}
        self.mbon_slot = {int(n): i for i, n in enumerate(self.mbon)}
        # Neuron index -> row in the plastic matrix, or -1. Used every brain step,
        # so it is a lookup rather than a set intersection.
        self.kc_row = np.full(self.brain.count, -1, dtype=np.int32)
        self.kc_row[self.kenyon] = np.arange(self.kenyon.shape[0], dtype=np.int32)

        edges = self.brain.outgoing(self.kenyon)
        targets = self.brain.indices[edges]
        keep = np.isin(targets, self.mbon)
        edges, targets = edges[keep], targets[keep]

        # Which KC each surviving edge came from.
        lengths = self.brain.indptr[self.kenyon + 1] - self.brain.indptr[self.kenyon]
        source = np.repeat(self.kenyon, lengths)[keep]

        rows = np.array([self.kc_slot[int(s)] for s in source], dtype=np.int32)
        cols = np.array([self.mbon_slot[int(t)] for t in targets], dtype=np.int32)

        self.kc_mbon = np.zeros((self.kenyon.shape[0], self.mbon.shape[0]),
                                dtype=np.float32)
        np.add.at(self.kc_mbon, (rows, cols), self.brain.weight[edges])
        self.kc_mbon_initial = self.kc_mbon.copy()
        self.plastic_edges = edges

        # The fixed network must not also carry these synapses.
        self.brain.silence(edges)

    def reset_learning(self):
        self.kc_mbon[:] = self.kc_mbon_initial

    @property
    def learning_depression(self):
        """How far KC->MBON has been depressed overall, as a fraction."""
        total = float(np.abs(self.kc_mbon_initial).sum())
        if total == 0:
            return 0.0
        return 1.0 - float(np.abs(self.kc_mbon).sum()) / total

    # ---------------------------------------------------------------- sensing

    def sense(self):
        brain = self.brain
        brain.clear_external()

        # Computed by the world across the last move, so looming is a real derivative.
        vision = self.world.vision
        # Looming detectors on each side respond to expansion in that eye.
        for eye, population in (('left', self.looming_left),
                                ('right', self.looming_right)):
            signal = vision[eye]['looming'] * 90.0 + vision[eye]['size'] * 0.5
            if signal > 0 and population.size:
                brain.set_external(population,
                                   min(signal, 1.5) * self.LOOMING_DRIVE)

        odour = self.world.odour_field()
        for side, key in (('left', 'left'), ('right', 'right')):
            for receptor, concentration in odour[key].items():
                population = self.orn_by_receptor.get(receptor)
                if population is None or concentration <= 0:
                    continue
                left, right = self.atlas.by_side(population)
                target = left if side == 'left' else right
                if target.size:
                    brain.set_external(target, concentration * self.ODOUR_DRIVE)

        # Reinforcement drives the dopaminergic neurons, which is what gates learning.
        reinforcement = self.world.reinforcement
        if reinforcement > 0 and self.dan_reward.size:
            brain.set_external(self.dan_reward, abs(reinforcement) * self.DAN_DRIVE)
        elif reinforcement < 0 and self.dan_punish.size:
            brain.set_external(self.dan_punish, abs(reinforcement) * self.DAN_DRIVE)

        return vision

    # ---------------------------------------------------------------- acting

    def act(self):
        rate = self.brain.rate
        left = float(rate[self.dn_left].mean()) if self.dn_left.size else 0.0
        right = float(rate[self.dn_right].mean()) if self.dn_right.size else 0.0

        # Steering from the left/right imbalance of the descending population. More
        # drive on one side turns the fly toward the other, as the decussating
        # descending pathways do.
        imbalance = (right - left) / (right + left + 4.0)
        self.turn = float(np.tanh(imbalance * 3.0))

        # Forward speed: a spontaneous walk, slowed when descending drive is high
        # (strong looming means stop and turn, not run into the thing).
        drive = (right + left) / 2.0
        self.forward = float(np.clip(self.baseline_walk + 0.02 * drive
                                     - 0.06 * max(0.0, drive - 6.0), 0.0, 1.0))
        return self.forward, self.turn

    # ---------------------------------------------------------------- learning

    def learn(self):
        """Dopamine depresses the KC->MBON synapses of currently active Kenyon cells.

        This is the canonical fly rule: coincidence of odour-driven KC activity with
        dopaminergic reinforcement weakens that KC's drive onto the MBON, shifting the
        balance of MBON output for that odour.
        """
        # Dopaminergic neurons fire somewhat on their own in the recurrent network,
        # so plasticity is gated on reinforcement actually being delivered, not on
        # DAN spiking alone — otherwise the weights drift with no teaching signal.
        if self.world.reinforcement == 0.0:
            return 0.0
        rate = self.brain.rate
        dan = float(rate[self.dan_reward].mean() + rate[self.dan_punish].mean())
        if dan <= 0.1:
            return 0.0

        kc_activity = rate[self.kenyon]
        active = kc_activity > 0.5
        if not active.any():
            return 0.0

        scale = self.LEARNING_RATE * min(dan / 10.0, 1.0)
        factor = 1.0 - scale * (kc_activity[active] / max(kc_activity.max(), 1e-6))
        floor = self.WEIGHT_FLOOR * self.kc_mbon_initial[active]
        updated = self.kc_mbon[active] * factor[:, None]
        self.kc_mbon[active] = np.where(np.abs(updated) < np.abs(floor), floor, updated)
        return float(active.sum())

    # ---------------------------------------------------------------- the loop

    def tick(self):
        """One body update: run the brain, then sense and move."""
        brain = self.brain
        brain.tick_window()
        for _ in range(self.BRAIN_MS_PER_TICK):
            # Kenyon cells reach their MBONs through the plastic matrix, not the
            # fixed graph, so their contribution is injected each step.
            if brain.spikes.size and self.kenyon.size:
                rows = self.kc_row[brain.spikes]
                rows = rows[rows >= 0]
                if rows.size:
                    delta = np.zeros(brain.count, dtype=np.float32)
                    delta[self.mbon] = self.kc_mbon[rows].sum(axis=0)
                    brain.inject(delta)
            brain.step()

        self.sense()
        forward, turn = self.act()
        self.world.step(forward, turn, self.BODY_DT)
        learned = self.learn()

        return {
            'forward': round(forward, 3),
            'turn': round(turn, 3),
            'learned': learned,
            'depression': round(self.learning_depression, 4),
        }

    def telemetry(self):
        rate = self.brain.rate
        def mean(group):
            return round(float(rate[group].mean()), 2) if group.size else 0.0
        return {
            'brain_ms': round(self.brain.time_ms),
            'spiking': int((rate > 1).sum()),
            'mean_hz': round(float(rate.mean()), 3),
            'dn_left': mean(self.dn_left),
            'dn_right': mean(self.dn_right),
            'motor': mean(self.motor),
            'kenyon': mean(self.kenyon),
            'mbon': mean(self.mbon),
            'dan': round(mean(self.dan_reward) + mean(self.dan_punish), 2),
            'looming_left': mean(self.looming_left),
            'looming_right': mean(self.looming_right),
            'orn': mean(self.atlas.families['orn']),
        }
