"""Read-only progress and three-model task-level comparisons for the click champion."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from common import read


def optional(path):
    try:
        return read(path)
    except (FileNotFoundError, ValueError):
        return {}


def render(root):
    setup = optional(root / 'setup.json')
    progress = Table(title='GPU4–7：最终第一名固定结构迁移 / 单种子', expand=True)
    for title in ['GPU', '数据集', '阶段', '步数/样本', '训练剩余', '峰值显存GiB']:
        progress.add_column(title)
    state = optional(root / 'supervisor-status.json')
    completed = sum(j.get('phase') == 'complete' for j in state.get('jobs', {}).values())
    for dataset, job in state.get('jobs', {}).items():
        current = optional(Path(job.get('output', '/nonexistent')) / 'progress.json')
        result = optional(Path(job.get('output', '/nonexistent')) / 'result.json')
        names = {'train': '训练中', 'valid': '验证中', 'test': '测试中',
                 'valid_metrics': '计算验证指标', 'test_metrics': '计算测试指标',
                 'calibration': '成本校准中', 'initializing': '初始化', 'loading_train': '加载训练数据'}
        phase = '已完成（正常结束）' if job['phase'] == 'complete' else names.get(current.get('phase'), job['phase'])
        if job['phase'] in ['failed', 'timeout', 'stopped']:
            phase = {'failed': '异常失败', 'timeout': '超时退出', 'stopped': '已终止'}[job['phase']]
        count = f"{current.get('steps', 0)}/{current['total_steps']}" if 'total_steps' in current else (
            f"{current.get('samples',0)/1e6:.1f}/{current['total_samples']/1e6:.1f}M" if 'total_samples' in current else '')
        if job['phase'] == 'complete' and 'test' in result:
            count = f"完整测试 {result['test']['samples']/1e6:.2f}M 样本"
        elif current.get('total_samples'):
            count += f" ({100*current.get('samples',0)/current['total_samples']:.1f}%)"
        elif current.get('total_steps'):
            count += f" ({100*current.get('steps',0)/current['total_steps']:.1f}%)"
        eta = current.get('eta_training_seconds')
        memory = current.get('peak_gpu_bytes', result.get('peak_gpu_bytes'))
        gpu_label = str(job['gpu']) + (' 已释放' if job['phase'] == 'complete' else ' 已退出' if job['phase'] in ['failed','timeout','stopped'] else ' 运行中')
        progress.add_row(gpu_label, dataset, phase, count, f'{eta/3600:.2f}h' if eta else '-',
                         f'{memory/2**30:.2f}' if memory else '-')
    comparisons = Table(title='完整测试对照：首任务及全任务平均AUC', expand=True)
    for title in ['数据集/首任务', 'DIN 首任务', 'g3-r0006 首任务', '最终第一名 首任务', 'DIN 平均', 'g3-r0006 平均', '最终第一名 平均']:
        comparisons.add_column(title)
    report = optional(root / 'report.json')
    for row in report.get('datasets', []):
        latest = row.get('current_best', {})
        comparisons.add_row(row['dataset'] + '/' + row['primary_task'], f"{row['din']['primary_auc']:.6f}",
            f"{row['previous_macro_candidate']['primary_auc']:.6f}",
            f"{latest['primary_auc']:.6f}" if latest else '-', f"{row['din']['macro_auc']:.6f}",
            f"{row['previous_macro_candidate']['macro_auc']:.6f}", f"{latest['macro_auc']:.6f}" if latest else '-')
    header = (setup.get('source_model', 'preparing') + ' | Taobao test mean AUC0.832094 / click0.670728\n'
              f'完成 {completed}/4 个数据集；完成任务正常退出并释放GPU。HeMix独立使用GPU0。\n'
              '各数据集原生任务等权平均AUC；单种子、从头训练；MerRec首任务为Like。\n' +
              datetime.now(timezone.utc).isoformat(timespec='seconds'))
    return Group(Panel(header, title='最终第一名 / 多数据集泛化'), progress, comparisons)


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
