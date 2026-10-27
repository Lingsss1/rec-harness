"""Run one registered method, retaining failures without automatic repetition."""
import argparse
import fcntl
import logging
from pathlib import Path
import traceback

from .bootstrap import bootstrap
bootstrap()
from .core import read, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method', choices=['mlevolve', 'mlestar', 'self_evolverec'], required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['root']) / args.method
    root.mkdir(exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'events.jsonl').exists() or (root / 'summary.json').exists():
        raise RuntimeError('Existing formal run: inspect retained evidence before any restart')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    try:
        if args.method == 'mlevolve':
            from .mlevolve import run
        elif args.method == 'mlestar':
            from .mlestar import run
        else:
            from .self_evolverec import run
        run(config)
    except BaseException:
        save(root / 'failure.json', {'error': traceback.format_exc(), 'status': 'failed'})
        raise


if __name__ == '__main__':
    main()
