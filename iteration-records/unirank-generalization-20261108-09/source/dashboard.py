"""Read-only tmux view for data, all native tasks and paired transfer costs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from common import read


def optional(path):
    try:
        return read(path)
    except (FileNotFoundError, ValueError):
        return {}


def duration(seconds):
    return f'{seconds / 3600:.1f}h' if seconds >= 3600 else f'{seconds / 60:.0f}m'


def view(root):
    data = Table(title='官方数据获取（逐文件 SHA256）', expand=True)
    for column in ['数据集', '状态', 'GiB', '文件']:
        data.add_column(column)
    for folder in sorted((root / 'datasets').iterdir()):
        status = optional(folder / 'acquisition-status.json')
        total = status.get('total_bytes', status.get('bytes', 0))
        done = status.get('completed_bytes', status.get('bytes', 0))
        data.add_row(folder.name, status.get('phase', '?'), f'{done/2**30:.2f} / {total/2**30:.2f}',
                     f"{status.get('completed_files', 0)}/{status.get('total_files', 0)}")
    runs = Table(title='固定模型迁移 / DIN 配对对照', expand=True)
    for column in ['GPU', '数据集/模型', '阶段', '进度', '训练ETA', 'Val宏AUC', 'Test宏AUC']:
        runs.add_column(column)
    state = optional(root / 'supervisor-status.json')
    for name, job in state.get('jobs', {}).items():
        progress = optional(Path(job.get('output', '/nonexistent')) / 'progress.json')
        result = optional(Path(job.get('output', '/nonexistent')) / 'result.json')
        phase = job['phase']
        if phase in ['training', 'calibrating']:
            phase += '/' + progress.get('phase', 'starting')
        fraction = ''
        if 'total_steps' in progress:
            fraction = f"{progress.get('steps', 0)}/{progress['total_steps']}"
        elif 'total_samples' in progress:
            fraction = f"{progress.get('samples', 0)/1e6:.1f}/{progress['total_samples']/1e6:.1f}M"
        validation = result.get('validation', {}).get('auc')
        heldout = result.get('test', {}).get('auc')
        selection = optional(Path(job.get('output', '/nonexistent')) / 'selection.json')
        validation = validation if validation is not None else selection.get('validation', {}).get('auc')
        runs.add_row(str(job['gpu']), name, phase, fraction,
                     duration(progress['eta_training_seconds']) if progress.get('eta_training_seconds') else '',
                     f'{validation:.6f}' if validation is not None else '-',
                     f'{heldout:.6f}' if heldout is not None else '-')
    report = optional(root / 'report.json')
    results = Table(title='最终测试对照（完整配对后展示）', expand=True)
    for column in ['数据集', '候选', 'DIN', 'Δ宏AUC', '逐任务ΔAUC']:
        results.add_column(column)
    for row in report.get('datasets', []):
        if row['status'] == 'complete':
            results.add_row(row['dataset'], f"{row['candidate_auc']:.6f}", f"{row['din_auc']:.6f}",
                            f"{row['delta_macro_auc']:+.6f}",
                            ', '.join(f"{k} {v['delta_auc']:+.4f}" for k, v in row['per_task'].items()))
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = 'GPU telemetry unavailable'
    return Group(Panel('UniRank 跨数据集结构泛化 | g3-r0029-context_shared_primary | scratch / 1 epoch / seed20262027\n'
                       '选模：全部原生任务平均AUC；保留逐任务Logloss与成本。单种子结论为初步证据。\n' +
                       datetime.now(timezone.utc).isoformat(timespec='seconds') + ' | ' + str(root)),
                 data, runs, results, Panel(gpu, title='GPU 利用率 / 显存'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        Console().print(view(args.root.resolve()))
    else:
        with Live(view(args.root.resolve()), refresh_per_second=1 / 5, screen=True) as live:
            while True:
                time.sleep(5)
                live.update(view(args.root.resolve()))
