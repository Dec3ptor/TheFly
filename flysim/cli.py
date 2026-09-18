"""Command line: `flysim setup` to fetch and build, `flysim run` to fly."""
import argparse
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA_HOME = Path(os.environ.get('FLYSIM_HOME', Path.home() / '.flysim'))
DATA_FILE = DATA_HOME / 'flysim_data.npz'

BUCKET = 'flyem-male-cns'
FLAT = 'v1.0/connectome-data/flat-connectome'
SOURCES = {
    'weights.feather':
        f'{FLAT}/connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather',
    'nt.feather':
        f'{FLAT}/body-neurotransmitters-male-cns-v1.0.feather',
}


def object_url(path):
    return (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
            f"{urllib.parse.quote(path, safe='')}?alt=media")


def download(path, destination):
    if destination.exists():
        print(f"  {destination.name}: already have it "
              f"({destination.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  {destination.name}: downloading…", flush=True)
    seen = [0]

    def report(block, size, total):
        seen[0] += size
        if total > 0 and seen[0] % (40 * size or 1) == 0:
            done = min(block * size, total)
            sys.stdout.write(f"\r    {done/1e6:7.0f} / {total/1e6:.0f} MB")
            sys.stdout.flush()

    temporary = destination.with_suffix(destination.suffix + '.part')
    urllib.request.urlretrieve(object_url(path), temporary, reporthook=report)
    temporary.rename(destination)
    print(f"\r    {destination.stat().st_size/1e6:.0f} MB")


def setup(args):
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not args.force:
        print(f"{DATA_FILE} already built ({DATA_FILE.stat().st_size/1e6:.0f} MB)")
        print("Pass --force to rebuild.")
        return 0

    print("Fetching the MaleCNS connectome (public, CC-BY, no account needed).")
    print("This is ~545 MB once; the built file is ~160 MB.\n")
    cache = DATA_HOME / 'cache'
    cache.mkdir(exist_ok=True)
    for name, path in SOURCES.items():
        download(path, cache / name)

    print("\nBuilding…")
    from .build import main as build_main
    sys.argv = ['build', str(cache / 'weights.feather'),
                str(cache / 'nt.feather'), str(DATA_FILE)]
    build_main()

    if not args.keep_cache:
        for name in SOURCES:
            (cache / name).unlink(missing_ok=True)
        print(f"\nRemoved the {sum(1 for _ in SOURCES)} source files; "
              f"pass --keep-cache next time to keep them.")
    print(f"\nReady. Run: flysim run")
    return 0


def run(args):
    if not DATA_FILE.exists():
        print(f"No data at {DATA_FILE}. Run `flysim setup` first.")
        return 1
    from .server import serve
    serve(DATA_FILE, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def probe(args):
    """Run headless and report what the brain does — no browser involved."""
    import numpy as np
    from .agent import Agent
    from .world import World
    if not DATA_FILE.exists():
        print(f"No data at {DATA_FILE}. Run `flysim setup` first.")
        return 1
    data = np.load(DATA_FILE, allow_pickle=True)
    world = World(extent=150.0, seed=3)
    world.scatter()
    agent = Agent(data, world)
    print(agent.atlas.summary())
    for tick in range(args.ticks):
        agent.tick()
        if (tick + 1) % max(args.ticks // 10, 1) == 0:
            print(f"  t={agent.brain.time_ms/1000:5.1f}s  {agent.telemetry()}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flysim', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('setup', help='download the connectome and build the data file')
    p.add_argument('--force', action='store_true', help='rebuild even if it exists')
    p.add_argument('--keep-cache', action='store_true', help='keep the downloaded sources')
    p.set_defaults(func=setup)

    p = sub.add_parser('run', help='start the simulator and open the UI')
    p.add_argument('--port', type=int, default=8799)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--no-browser', action='store_true')
    p.set_defaults(func=run)

    p = sub.add_parser('probe', help='run headless and print telemetry')
    p.add_argument('--ticks', type=int, default=200)
    p.set_defaults(func=probe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
