"""Read-only progress and three-model task-level comparisons for the click champion."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
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


def render(root):
    setup = optional(root / 'setup.json')
    progress = Table(title='GPU4–7：固定结构迁移（GPU0–3继续12h研究）', expand=True)
    for title in ['GPU', '数据集', '阶段', '步数/样本', '训练剩余', '峰值显存GiB']:
        progress.add_column(title)
    state = optional(root / 'supervisor-status.json')
    for dataset, job in state.get('jobs', {}).items():
        current = optional(Path(job.get('output', '/nonexistent')) / 'progress.json')
        phase = job['phase'] + ('/' + current['phase'] if current.get('phase') else '')
        count = f"{current.get('steps', 0)}/{current['total_steps']}" if 'total_steps' in current else (
            f"{current.get('samples',0)/1e6:.1f}/{current['total_samples']/1e6:.1f}M" if 'total_samples' in current else '')
        eta = current.get('eta_training_seconds')
        memory = current.get('peak_gpu_bytes')
        progress.add_row(str(job['gpu']), dataset, phase, count, f'{eta/3600:.2f}h' if eta else '-',
                         f'{memory/2**30:.2f}' if memory else '-')
    comparisons = Table(title='完整测试对照：首任务及全任务平均AUC', expand=True)
    for title in ['数据集/首任务', 'DIN 首任务', '上次模型 首任务', '当前最佳 首任务', 'DIN 平均', '上次模型 平均', '当前最佳 平均']:
        comparisons.add_column(title)
    report = optional(root / 'report.json')
    for row in report.get('datasets', []):
        latest = row.get('current_best', {})
        comparisons.add_row(row['dataset'] + '/' + row['primary_task'], f"{row['din']['primary_auc']:.6f}",
            f"{row['previous_macro_candidate']['primary_auc']:.6f}",
            f"{latest['primary_auc']:.6f}" if latest else '-', f"{row['din']['macro_auc']:.6f}",
            f"{row['previous_macro_candidate']['macro_auc']:.6f}", f"{latest['macro_auc']:.6f}" if latest else '-')
    header = (setup.get('source_model', 'preparing') + ' | Taobao test mean AUC0.829081 / click0.669158\n'
              '源模型按验证平均AUC选择，源域测试已经观察；单种子、从头训练；MerRec首任务Like不是点击。\n' +
              datetime.now(timezone.utc).isoformat(timespec='seconds'))
    return Group(Panel(header, title='当前验证最佳模型 / 多数据集泛化'), progress, comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        Console().print(render(args.root.resolve()))
    else:
        with Live(render(args.root.resolve()), screen=True, auto_refresh=False) as live:
            while True:
                time.sleep(5)
                live.update(render(args.root.resolve()), refresh=True)
