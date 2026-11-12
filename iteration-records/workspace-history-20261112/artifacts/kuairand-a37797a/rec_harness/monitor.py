"""Local, read-only autoresearch dashboard. No agent calls or ledger recovery."""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .process_identity import identity

ROLES = ('researcher', 'experiment', 'reviewer', 'analyst')
TOKEN_KEYS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')
ASSETS = Path(__file__).with_name('monitor_assets')


def read(path, default=None):
    try:
        if path.stat().st_size > 8_000_000:
            return default
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path, size=32000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read(size).decode('utf-8', errors='replace')
    except OSError:
        return ''


def redact(text):
    text = re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]+)\b', '[REDACTED]', str(text))
    return re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', text)


def live(record):
    record = record.get('worker', record) if isinstance(record, dict) else {}
    if not record or 'pid' not in record:
        return None
    try:
        current = identity(record['pid'])
        if current.get('observable') is False:
            return None
        return all(current.get(k) == record.get(k) for k in ('pid', 'start_ticks', 'boot_id', 'session')) and current['state'] not in ('Z', 'X')
    except (OSError, ValueError):
        return False


def ledger(path):
    """One SQLite read transaction; never instantiate a writable Store."""
    if not path.exists():
        return [], {}
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    try:
        db.execute('BEGIN')
        events = [{'seq': s, 'kind': k, 'payload': json.loads(p), 'time': t}
                  for s, k, p, t in db.execute('SELECT seq,kind,payload,created_at FROM events ORDER BY seq')]
        rows = db.execute('SELECT id,reserved,actual,status FROM budget').fetchall()
        inherited = sum(a if a is not None else r for i, r, a, s in rows if i == 'predecessor-consumed')
        charged = sum(a if a is not None else r for i, r, a, s in rows)
        return events, {'charged_or_reserved': charged, 'inherited': inherited, 'current': charged - inherited,
                        'pending': sum(s == 'running' for i, r, a, s in rows)}
    finally:
        db.close()


def role_dirs(round_dir):
    # Do not recurse through model workspaces, raw data, artifacts or jail trees.
    for parent in [round_dir, *sorted((round_dir / 'review-repair').glob('revision-*'))]:
        for child in sorted(parent.iterdir()) if parent.is_dir() else []:
            if child.is_dir() and any(child.name == r or child.name.startswith(r + '-rejected-') for r in ROLES):
                yield child


def role_call(path, root):
    resources = read(path / 'resources.json', {})
    process = read(path / 'process.json', {})
    usage = resources.get('token_usage')
    # Receipts are authoritative. During execution, use completed-turn counters only.
    if not isinstance(usage, dict):
        usage = {}
        for line in tail(path / 'stdout.jsonl', 100_000).splitlines():
            try:
                item = json.loads(line)
                if item.get('type') == 'turn.completed':
                    for key in TOKEN_KEYS:
                        usage[key] = usage.get(key, 0) + (item.get('usage', {}).get(key) or 0)
            except (ValueError, AttributeError):
                continue
    running = live(read(path / 'launch.json', {}))
    status = 'accepted' if (path / 'accepted.json').exists() else ('finished' if process else 'running' if running else 'unconfirmed')
    if '-rejected-' in path.name:
        status = 'format_rejected'
    try:
        context_bytes = (path / 'context.json').stat().st_size
    except OSError:
        context_bytes = None
    return {'path': str(path.relative_to(root)), 'role': path.name.split('-rejected-')[0], 'status': status,
            'archived_revision': 'review-repair' in path.parts, 'usage': {k: usage.get(k) for k in TOKEN_KEYS},
            'context_bytes': context_bytes, 'wall_seconds': resources.get('wall_seconds', process.get('wall_seconds')),
            'started_at': resources.get('started_at', process.get('started_at')), 'live': running}


def training_jobs(root, datasets):
    jobs = []
    for path in sorted((root / 'study/jobs').glob('*')):
        if not path.is_dir():
            continue
        request = read(path / 'request.json', {})
        receipt = read(path / 'completed.json', {})
        proc = path / 'isolated/process'
        launch = read(proc / 'launch.json', {})
        running = live(launch)
        process = read(proc / 'process.json', {})
        status = receipt.get('status') or ('running' if running else 'finished_pending_receipt' if process else 'unconfirmed')
        dataset = receipt.get('dataset') or next((d for d in datasets if f'-{d}-' in path.name), None)
        # Read only published aggregate curves; never traverse an untrusted jail.
        curve = []
        for line in tail(path / 'curve.jsonl').splitlines():
            try:
                point = json.loads(line)
                curve.append({'epoch': point.get('epoch'), 'loss': point.get('loss'), 'hr10': point.get('metrics', {}).get('HR@10')})
            except (ValueError, AttributeError):
                pass
        # Worker stdout contains flushed aggregate epoch updates during training.
        if not curve:
            for line in tail(proc / 'stdout.jsonl').splitlines():
                try:
                    point = json.loads(line)
                    if 'epoch' in point:
                        curve.append({'epoch': point['epoch'], 'loss': point.get('loss'), 'hr10': point.get('metrics', {}).get('HR@10')})
                except (ValueError, AttributeError):
                    pass
        jobs.append({'id': path.name, 'dataset': dataset, 'gpu': request.get('gpu'), 'status': status,
                     'model_version': receipt.get('model_version') or (path.name.rsplit(f'-{dataset}-', 1)[0] if dataset else None),
                     'seed': receipt.get('seed'), 'epoch': receipt.get('completed_epoch', curve[-1]['epoch'] if curve else None),
                     'best_epoch': receipt.get('best_epoch'), 'hr10': receipt.get('metrics', {}).get('HR@10'),
                     'gpu_seconds': receipt.get('allocated_gpu_seconds'), 'curve': curve,
                     'pid': launch.get('worker', launch).get('pid'), 'live': running})
    return jobs


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3)
        if result.returncode:
            return {'available': False, 'devices': [], 'reason': 'GPU telemetry unavailable'}
        devices = []
        for line in result.stdout.splitlines():
            index, name, total, used, utilization = [s.strip() for s in line.split(',')]
            devices.append({'index': int(index), 'name': name, 'memory_total_mib': int(total), 'memory_used_mib': int(used), 'utilization': int(utilization)})
        return {'available': True, 'devices': devices}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'available': False, 'devices': [], 'reason': 'GPU telemetry unavailable'}


def snapshot(root, include_gpu=True):
    root = Path(root)
    cfg = read(root / 'research/amazon-auto.json', {})
    study = read(root / 'study/study.json', {})
    state = read(root / 'research/state.json', {})
    progress = read(root / 'progress.json', {})
    events, budget = ledger(root / 'research/memory/memory.sqlite')
    study_events, gpu_budget = ledger(root / 'study/memory/memory.sqlite')
    outcomes = {}
    for event in events:
        if event['kind'] == 'auto_round_completed':
            payload = event['payload']
            outcomes[payload['state']['round']] = payload.get('result', {})
            state = payload['state']
        elif event['kind'] == 'auto_round_reopened':
            state = event['payload']['state']
            outcomes = {n: v for n, v in outcomes.items() if n <= state.get('round', 0)}
    published = read(root / 'research/state.json', {})
    if published.get('round') == state.get('round') and published.get('status') == 'stopped':
        state = published
    rounds, calls = [], []
    for directory in sorted((root / 'research/rounds').glob('round-*'), key=lambda p: int(p.name.split('-')[-1])):
        number = int(directory.name.split('-')[-1])
        proposal = read(directory / 'researcher/response.json', {})
        readout = read(directory / 'development-readout.json', {})
        promotion = read(directory / 'promotion.json', {})
        review = read(directory / 'reviewer/response.json', {})
        outcome = outcomes.get(number, {})
        round_calls = [role_call(p, root) for p in role_dirs(directory)]
        calls.extend(round_calls)
        rounds.append({'number': number, 'status': 'failed' if outcome.get('status') == 'failed' else 'completed' if number in outcomes else 'in_progress',
                       'base_model': proposal.get('proposal', {}).get('base_model_id'),
                       'mechanism': redact(proposal.get('proposal', {}).get('mechanism', '')),
                       'means': readout.get('summary', {}).get('means', {}), 'promoted_model': promotion.get('promoted_model'),
                       'review_verdict': review.get('verdict'), 'review_findings': [redact(x) for x in review.get('blocking_findings', [])],
                       'revisions': read(directory / 'review-repair/state.json', {}).get('count', 0),
                       'error': redact(outcome.get('reason', '')), 'calls': round_calls})
    usage = {key: sum(call['usage'].get(key) or 0 for call in calls) for key in TOKEN_KEYS}
    usage['uncached_input_tokens'] = max(0, usage['input_tokens'] - usage['cached_input_tokens'])
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    usage['calls'] = len(calls)
    usage['calls_with_usage'] = sum(c['usage'].get('input_tokens') is not None for c in calls)
    jobs = training_jobs(root, study.get('datasets', []))
    launch = read(root / 'launch.json', {})
    alive = live(launch)
    error = read(root / 'runner-error.json')
    status = 'failed' if error else progress.get('stage', state.get('status', 'unknown'))
    if alive is False and status not in ('completed', 'completed_with_failed_rounds', 'failed'):
        status = 'process_stopped'
    safe_events = []
    for event in sorted(events + study_events, key=lambda e: e['time'])[-60:]:
        payload = event['payload']
        details = {k: payload[k] for k in ('role', 'task_id', 'job_id', 'round', 'status', 'error') if k in payload}
        safe_events.append({'time': event['time'], 'kind': event['kind'], 'details': redact(json.dumps(details, ensure_ascii=False))})
    return {'id': root.name, 'observed_at': time.time(), 'status': status, 'phase_updated_at': progress.get('updated_at'),
            'controller': {'pid': launch.get('pid'), 'live': alive}, 'error': redact(error) if error else None,
            'state': state, 'target_rounds': cfg.get('max_rounds'),
            'counts': {'attempted': len(outcomes), 'completed': sum(v.get('status') != 'failed' for v in outcomes.values()),
                       'failed': sum(v.get('status') == 'failed' for v in outcomes.values()), 'promoted': sum(bool(r['promoted_model']) for r in rounds)},
            'protocol': {'seeds': study.get('development_seeds'), 'metric': study.get('protocol', {}).get('metric'),
                         'selection_policy': study.get('selection_policy'), 'gpu_ids': study.get('gpu_ids'),
                         'model': cfg.get('agent_backend', {}).get('model'), 'review_revision_limit': cfg.get('review_revision_limit', 0)},
            'rounds': rounds, 'usage': usage, 'role_budget': budget, 'gpu_budget': gpu_budget, 'jobs': jobs,
            'job_counts': dict(Counter(j['status'] for j in jobs)), 'events': safe_events,
            'gpu': gpu_snapshot() if include_gpu else {'available': False, 'devices': []},
            'qualification': 'Development-only metrics. Token totals include cached input; reasoning is part of output. Current-campaign calls include rejected formats and archived revisions; inherited ledger charges are separate. No account quota estimate.'}


class Monitor:
    def __init__(self, runs):
        self.runs = {Path(p).resolve().name: Path(p).resolve() for p in runs}
        if len(self.runs) != len(runs):
            raise ValueError('campaign names must be unique')
        self.lock = threading.Lock()
        self.cache = {}

    def get(self, name):
        if name not in self.runs:
            raise KeyError(name)
        with self.lock:
            cached = self.cache.get(name)
            if cached and time.monotonic() - cached[0] < 5:
                return cached[1]
            value = snapshot(self.runs[name])
            self.cache[name] = (time.monotonic(), value)
            return value


def export_html(monitor, destination):
    """Self-contained server-local view, atomically replaced without opening a port."""
    snapshots = {name: monitor.get(name) for name in monitor.runs}
    payload = json.dumps(snapshots, ensure_ascii=False, allow_nan=False)
    # Research text is data, never executable HTML, including literal </script>.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    css = (ASSETS / 'style.css').read_text()
    js = (ASSETS / 'app.js').read_text()
    html = (ASSETS / 'index.html').read_text()
    html = html.replace('<link rel="stylesheet" href="/style.css">', '<style>' + css + '</style>')
    html = html.replace('<script src="/app.js" defer></script>', '')
    html = html.replace('</body>', '<script>window.__AUTORESEARCH_SNAPSHOTS__=' + payload + ';</script><script>' + js + '</script></body>')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(html)
    temporary.replace(destination)


def terminal_text(data):
    """Plain terminal view for servers without a desktop browser."""
    def number(value):
        return f'{value:,.0f}' if isinstance(value, (int, float)) else '—'
    def safe(value):
        return ''.join(c for c in str(value) if c.isprintable())
    count, usage = data['counts'], data['usage']
    lines = ['=' * 90, 'REC HARNESS / AUTORESEARCH 实时监控',
             safe(data['id']) + '    ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['observed_at'])),
             '-' * 90, f"阶段: {safe(data['status'])}    PID: {data['controller']['pid']}    存活: {data['controller']['live']}",
             f"完成实验: {count['completed']}/{data['target_rounds']}    已尝试: {count['attempted']}    失败: {count['failed']}    晋升: {count['promoted']}",
             f"当前模型: {safe(data['state'].get('incumbent', '—'))}",
             f"Codex 调用: {usage['calls']}    Token: {number(usage['total_tokens'])}    缓存输入: {number(usage['cached_input_tokens'])}",
             '思考 token 已包含在输出中；未返回用量不计入当前合计。', '-' * 90, 'GPU   显存使用 / 总量 (GiB)    利用率    * 为本任务配置的 GPU']
    for gpu in data['gpu']['devices']:
        assigned = '*' if gpu['index'] in (data['protocol']['gpu_ids'] or []) else ' '
        lines.append(f"{assigned}{gpu['index']:2}   {gpu['memory_used_mib']/1024:5.1f} / {gpu['memory_total_mib']/1024:4.0f}               {gpu['utilization']:3}%")
    if not data['gpu']['available']:
        lines.append('GPU 遥测不可用')
    active = [j for j in data['jobs'] if j['status'] != 'success']
    lines += ['-' * 90, f"训练任务: {len(data['jobs'])} 个 / 进行中或待确认: {len(active)}"]
    for job in (active or data['jobs'][-4:])[:8]:
        lines.append(f"GPU {job['gpu']}  {safe(job['dataset'] or job['id']):26}  {safe(job['status']):24}  epoch {job['epoch'] or '—'}")
    lines += ['-' * 90, '最近迭代']
    for item in data['rounds'][-5:]:
        lines.append(f"R{item['number']:02}  {safe(item['status']):14}  修订 {item['revisions']}  晋升 {safe(item['promoted_model'] or '—')}")
    if not data['rounds']:
        lines.append('尚未进入第一轮')
    if data['error']:
        lines.append('错误: ' + safe(data['error'])[:400])
    lines += ['=' * 90, '只读监控 · Ctrl+C 退出面板，不停止研究任务']
    return '\n'.join(lines)


def handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, data, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            # Loopback + Host validation prevents DNS rebinding to this read-only server.
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('localhost', '127.0.0.1'):
                return self.send(403, {'error': 'Loopback Host required'})
            url = urlsplit(self.path)
            try:
                if url.path == '/api/runs':
                    return self.send(200, {'runs': list(monitor.runs)})
                if url.path == '/api/snapshot':
                    name = parse_qs(url.query).get('run', [next(iter(monitor.runs))])[0]
                    return self.send(200, monitor.get(name))
                files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if url.path in files:
                    name, content_type = files[url.path]
                    return self.send(200, (ASSETS / name).read_bytes(), content_type)
                return self.send(404, {'error': 'Not found'})
            except KeyError:
                self.send(404, {'error': 'Unknown campaign'})
            except (OSError, ValueError, sqlite3.Error):
                self.send(503, {'error': 'Snapshot temporarily unavailable; retry shortly'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, type=Path, help='Campaign root; repeat for historical runs')
    parser.add_argument('--port', type=int, default=4318)
    parser.add_argument('--html', type=Path, help='Write a standalone local HTML dashboard instead of opening a port')
    parser.add_argument('--terminal', action='store_true', help='Display a local terminal dashboard without opening a port')
    parser.add_argument('--interval', type=int, default=0, help='Refresh HTML or terminal every N seconds; 0 renders once')
    args = parser.parse_args()
    for root in args.run:
        if not (root / 'research/amazon-auto.json').is_file():
            parser.error(f'Not an Amazon autoresearch campaign: {root}')
    if args.html and args.terminal:
        parser.error('choose either --html or --terminal')
    if args.interval < 0 or (args.interval and (not (args.html or args.terminal) or args.interval < 5)):
        parser.error('--interval requires --html or --terminal and must be 0 or at least 5 seconds')
    monitor = Monitor(args.run)
    if args.html or args.terminal:
        try:
            while True:
                try:
                    if args.html:
                        export_html(monitor, args.html)
                    else:
                        text = '\n\n'.join(terminal_text(monitor.get(name)) for name in monitor.runs)
                        print(('\033[2J\033[H' if sys.stdout.isatty() else '') + text, flush=True)
                except (OSError, ValueError, sqlite3.Error) as error:
                    if not args.interval:
                        raise
                    print('Monitor refresh deferred: ' + redact(error), flush=True)
                if not args.interval:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(monitor))
    print(f'Autoresearch monitor: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
