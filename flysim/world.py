"""A small arena for the fly to live in, and the body it drives around.

Deliberately simple physics: the fly is a point with a heading, and the brain's motor
output sets forward speed and turn rate. There is no leg biomechanics here — the point
is the closed loop from world to spikes to movement, not walking gait.

Distances are in millimetres, angles in radians, time in seconds.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Post:
    """A dark vertical object. Visual only — this is what looming detectors see."""
    x: float
    y: float
    radius: float = 4.0


@dataclass
class Odour:
    """A diffusing odour source, tied to one olfactory receptor type."""
    x: float
    y: float
    receptor: str = 'ORN_DA1'
    strength: float = 1.0
    sigma: float = 45.0        # mm, plume width
    valence: float = 0.0       # +1 rewarding, -1 punishing, 0 neutral
    radius: float = 8.0        # within this distance the fly "arrives"


@dataclass
class Body:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0       # radians, 0 = +x
    speed: float = 0.0         # mm/s
    turn: float = 0.0          # rad/s

    max_speed: float = 22.0    # a walking fly does roughly this
    max_turn: float = 4.0      # rad/s


class World:
    # Each eye's field, as bearings relative to straight ahead. Fly eyes are close to
    # panoramic with a narrow binocular overlap in front.
    EYE_SPAN = math.radians(150.0)
    EYE_CENTRE = math.radians(60.0)
    ANTENNA_OFFSET = 0.35      # mm, half-separation used for bilateral odour sampling

    def __init__(self, extent=150.0, seed=0):
        self.extent = extent
        self.rng = np.random.default_rng(seed)
        self.body = Body()
        self.posts: list[Post] = []
        self.odours: list[Odour] = []
        self.time = 0.0
        self.reinforcement = 0.0   # set when the fly is on a valenced source
        self.arrivals: list[Odour] = []
        # Vision is computed across each move, so looming is the change in apparent
        # size caused by that move. Sensing reads this rather than recomputing, which
        # would compare a position against itself and always yield zero looming.
        self.vision = self.visual_field(previous=None)

    # ---------------------------------------------------------------- setup

    def reset(self):
        self.body = Body()
        self.time = 0.0
        self.reinforcement = 0.0
        self.arrivals = []
        self.vision = self.visual_field(previous=None)

    def scatter(self, posts=3, odours=2):
        """A default arena: a few posts and one attractive, one aversive odour."""
        self.posts = []
        self.odours = []
        for _ in range(posts):
            angle = self.rng.uniform(0, 2 * math.pi)
            distance = self.rng.uniform(0.35, 0.85) * self.extent
            self.posts.append(Post(distance * math.cos(angle),
                                   distance * math.sin(angle),
                                   radius=self.rng.uniform(3.0, 6.0)))
        recipes = [('ORN_DA1', 1.0), ('ORN_DM1', -1.0)]
        for i in range(min(odours, len(recipes))):
            receptor, valence = recipes[i]
            angle = self.rng.uniform(0, 2 * math.pi)
            distance = self.rng.uniform(0.4, 0.8) * self.extent
            self.odours.append(Odour(distance * math.cos(angle),
                                     distance * math.sin(angle),
                                     receptor=receptor, valence=valence))

    # ---------------------------------------------------------------- sensing

    def _bearing_to(self, x, y):
        """Angle of (x, y) relative to the fly's heading, wrapped to [-pi, pi]."""
        angle = math.atan2(y - self.body.y, x - self.body.x) - self.body.heading
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def visual_field(self, previous=None):
        """Per-eye apparent size and looming of the posts, plus the nearest bearing.

        `size` is summed angular width, so a near post dominates a far one. `looming`
        is the increase in that size since `previous`, which is the signal escape
        circuits actually respond to.
        """
        result = {}
        for eye, centre in (('left', -self.EYE_CENTRE), ('right', self.EYE_CENTRE)):
            size = 0.0
            nearest = None
            nearest_distance = float('inf')
            for post in self.posts:
                dx, dy = post.x - self.body.x, post.y - self.body.y
                distance = math.hypot(dx, dy)
                if distance < 1e-3:
                    continue
                bearing = self._bearing_to(post.x, post.y)
                # How far into this eye's field the object sits, 1 at the centre.
                offset = abs((bearing - centre + math.pi) % (2 * math.pi) - math.pi)
                if offset > self.EYE_SPAN / 2:
                    continue
                weight = math.cos(offset / (self.EYE_SPAN / 2) * (math.pi / 2))
                size += weight * 2.0 * math.atan2(post.radius, distance)
                if distance < nearest_distance:
                    nearest_distance, nearest = distance, bearing

            before = previous[eye]['size'] if previous else size
            result[eye] = {
                'size': size,
                'looming': max(0.0, size - before),
                'nearest': nearest,
                'distance': nearest_distance,
            }
        return result

    def odour_field(self):
        """Concentration per receptor type at each antenna."""
        field = {}
        for side, sign in (('left', -1.0), ('right', 1.0)):
            # Antennae sit slightly off the midline, which is what makes the
            # left-right concentration difference that drives chemotaxis.
            offset = sign * self.ANTENNA_OFFSET
            px = self.body.x + math.cos(self.body.heading + math.pi / 2) * offset
            py = self.body.y + math.sin(self.body.heading + math.pi / 2) * offset
            per_receptor = {}
            for odour in self.odours:
                distance2 = (px - odour.x) ** 2 + (py - odour.y) ** 2
                value = odour.strength * math.exp(-distance2 / (2 * odour.sigma ** 2))
                per_receptor[odour.receptor] = per_receptor.get(odour.receptor, 0.0) + value
            field[side] = per_receptor
        return field

    # ---------------------------------------------------------------- acting

    def step(self, forward, turn, dt):
        """Advance the body. `forward` and `turn` are in [0,1] and [-1,1]."""
        body = self.body
        body.speed = float(np.clip(forward, 0.0, 1.0)) * body.max_speed
        body.turn = float(np.clip(turn, -1.0, 1.0)) * body.max_turn

        body.heading = (body.heading + body.turn * dt) % (2 * math.pi)
        body.x += math.cos(body.heading) * body.speed * dt
        body.y += math.sin(body.heading) * body.speed * dt

        # The arena is a circular dish; bounce off the wall rather than sticking.
        distance = math.hypot(body.x, body.y)
        if distance > self.extent:
            body.x *= self.extent / distance
            body.y *= self.extent / distance
            body.heading = (body.heading + math.pi) % (2 * math.pi)

        # Posts are solid.
        for post in self.posts:
            dx, dy = body.x - post.x, body.y - post.y
            gap = math.hypot(dx, dy)
            if gap < post.radius:
                push = (post.radius - gap) / max(gap, 1e-3)
                body.x += dx * push
                body.y += dy * push

        self.vision = self.visual_field(previous=self.vision)

        # Reinforcement when sitting on a valenced odour source.
        self.reinforcement = 0.0
        self.arrivals = []
        for odour in self.odours:
            if math.hypot(body.x - odour.x, body.y - odour.y) < odour.radius:
                self.reinforcement += odour.valence
                self.arrivals.append(odour)

        self.time += dt
        return self.vision

    def snapshot(self):
        return {
            'time': round(self.time, 2),
            'extent': self.extent,
            'body': {'x': round(self.body.x, 2), 'y': round(self.body.y, 2),
                     'heading': round(self.body.heading, 3),
                     'speed': round(self.body.speed, 2),
                     'turn': round(self.body.turn, 3)},
            'posts': [{'x': p.x, 'y': p.y, 'r': p.radius} for p in self.posts],
            'odours': [{'x': o.x, 'y': o.y, 'r': o.radius, 'sigma': o.sigma,
                        'receptor': o.receptor, 'valence': o.valence}
                       for o in self.odours],
            'reinforcement': round(self.reinforcement, 3),
        }
