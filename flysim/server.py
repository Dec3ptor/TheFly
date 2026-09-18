"""Local web UI. The simulation runs in a background thread; the browser polls it.

Binds to localhost only — this is a desktop app that happens to use a browser for its
window, not a service.
"""
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .agent import Agent
from .dataset import load as load_dataset
from .world import Odour, Post, World

UI_DIR = Path(__file__).parent / 'ui'


class Simulation:
    """Owns the agent and steps it on its own thread."""

    def __init__(self, data_path):
        self.data = load_dataset(data_path)
        self.world = World(extent=150.0, seed=3)
        self.world.scatter(posts=3, odours=2)
        self.agent = Agent(self.data, self.world)
        self.lock = threading.Lock()
        self.running = False
        self.speed = 1.0
        self._stop = threading.Event()
        self.trail = []
        self.ticks = 0
        self.wall_rate = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.02)
                continue
            started = time.perf_counter()
            with self.lock:
                self.agent.tick()
                self.ticks += 1
                body = self.world.body
                self.trail.append((round(body.x, 1), round(body.y, 1)))
                if len(self.trail) > 600:
                    del self.trail[:200]
            elapsed = time.perf_counter() - started
            self.wall_rate = self.agent.BODY_DT / max(elapsed, 1e-6)
            # Hold real time when we can; `speed` scales the target.
            budget = self.agent.BODY_DT / max(self.speed, 0.05)
            if elapsed < budget:
                time.sleep(budget - elapsed)

    # ---------------------------------------------------------------- state

    def state(self):
        with self.lock:
            snapshot = self.world.snapshot()
            telemetry = self.agent.telemetry()
            rate = self.agent.brain.rate
            by_type = self.agent.atlas.rate_by_type(rate)
            top = np.argsort(-by_type)[:12]
            active = [{'name': self.agent.atlas.type_names[i],
                       'hz': round(float(by_type[i]), 1)}
                      for i in top if by_type[i] > 0.05]
            return {
                'running': self.running,
                'speed': self.speed,
                'ticks': self.ticks,
                'wall_rate': round(self.wall_rate, 2),
                'world': snapshot,
                'trail': self.trail[-300:],
                'telemetry': telemetry,
                'motor': {'forward': round(self.agent.forward, 3),
                          'turn': round(self.agent.turn, 3)},
                'learning': {
                    'depression': round(self.agent.learning_depression, 4),
                    'synapses': int((self.agent.kc_mbon != 0).sum()),
                },
                'active_types': active,
            }

    # ---------------------------------------------------------------- commands

    def command(self, name, payload):
        with self.lock:
            if name == 'run':
                self.running = bool(payload.get('on', True))
            elif name == 'speed':
                self.speed = float(payload.get('value', 1.0))
            elif name == 'reset':
                self.world.reset()
                self.agent.brain.reset()
                self.trail = []
                self.ticks = 0
            elif name == 'reset_learning':
                self.agent.reset_learning()
            elif name == 'scatter':
                self.world.scatter(posts=int(payload.get('posts', 3)),
                                   odours=int(payload.get('odours', 2)))
                self.world.reset()
                self.trail = []
            elif name == 'add_post':
                self.world.posts.append(Post(float(payload['x']), float(payload['y']),
                                             radius=float(payload.get('r', 5.0))))
            elif name == 'add_odour':
                self.world.odours.append(Odour(
                    float(payload['x']), float(payload['y']),
                    receptor=str(payload.get('receptor', 'ORN_DA1')),
                    valence=float(payload.get('valence', 1.0))))
            elif name == 'clear_objects':
                self.world.posts = []
                self.world.odours = []
            elif name == 'place':
                self.world.body.x = float(payload['x'])
                self.world.body.y = float(payload['y'])
                self.trail = []
            elif name == 'stimulate':
                target = str(payload.get('type', ''))
                neurons = self.agent.atlas.neurons_of_type(target)
                if neurons.size:
                    self.agent.brain.set_external(
                        neurons, float(payload.get('mv', 1.0)))
                    return {'ok': True, 'neurons': int(neurons.size)}
                return {'ok': False, 'error': f'no cell type named {target}'}
            else:
                return {'ok': False, 'error': f'unknown command {name}'}
        return {'ok': True}


def make_handler(sim):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep the console clean; the UI is the interface

        def _send(self, code, body, content_type='application/json'):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/api/state':
                self._send(200, sim.state())
                return
            if self.path == '/favicon.ico':
                # A fly, so the browser tab does not 404 looking for one.
                icon = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
                        "<text y='13' font-size='13'>\U0001FAB0</text></svg>")
                self._send(200, icon.encode(), 'image/svg+xml')
                return
            name = 'index.html' if self.path in ('/', '') else self.path.lstrip('/')
            target = (UI_DIR / name).resolve()
            if not str(target).startswith(str(UI_DIR.resolve())) or not target.is_file():
                self._send(404, {'error': 'not found'})
                return
            kind = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
            self._send(200, target.read_bytes(), kind)

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            try:
                payload = json.loads(self.rfile.read(length) or b'{}')
            except json.JSONDecodeError:
                self._send(400, {'ok': False, 'error': 'bad JSON'})
                return
            if not self.path.startswith('/api/'):
                self._send(404, {'ok': False, 'error': 'not found'})
                return
            self._send(200, sim.command(self.path[len('/api/'):], payload))

    return Handler


def serve(data_path, host='127.0.0.1', port=8799, open_browser=True):
    sim = Simulation(data_path)
    server = ThreadingHTTPServer((host, port), make_handler(sim))
    url = f'http://{host}:{port}/'
    print(f"flysim running at {url}")
    print(f"  {sim.agent.brain.count:,} neurons, "
          f"{sim.agent.brain.indices.shape[0]:,} synapses")
    print("  Ctrl-C to stop")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        sim._stop.set()
        server.server_close()
    return sim
