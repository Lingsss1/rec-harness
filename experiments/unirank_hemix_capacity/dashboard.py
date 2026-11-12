"""Read-only live view of the one requested compressed HeMix fit."""
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
    status = optional(root / 'status.json')
    progress = optional(Path(status.get('output', '/nonexistent')) / 'progress.json')
    steps = f"{progress.get('steps',0)}/{progress['total_steps']}" if 'total_steps' in progress else '-'
    eta = progress.get('eta_training_seconds')
    header = (f"GPU0 | HeMix token_dim=80 | {status.get('phase','preflight')}/{progress.get('phase','-')} | steps {steps}\n"
              + (f"训练剩余约 {eta/3600:.2f}h\n" if eta else '')
              + '同原生 Embedding16 / 六任务平均AUC / 单种子 / 原版HeMix不训练\n'
              + datetime.now(timezone.utc).isoformat(timespec='seconds'))
    table = Table(title='KuaiRand：参数预算与完整测试结果', expand=True)
    for title in ['模型', '总参(M)', 'Embedding(M)', '稠密(M)', '平均AUC', '平均Logloss']:
        table.add_column(title)
    audit = optional(root / 'parameter-audit.json')
    rows = {r['model']: r for r in optional(root / 'report.json').get('rows', [])}
    for name, label in [('ours', 'ours_g3-r0018'), ('hemix_token80', 'hemix_token80')]:
        p = audit.get(name, {})
        metrics = rows.get(label, {})
        table.add_row(label, *(f"{p[k]/1e6:.3f}" if k in p else '-' for k in
            ['parameters', 'embedding_parameters', 'dense_parameters']),
            f"{metrics['macro_auc']:.8f}" if 'macro_auc' in metrics else 'pending',
            f"{metrics['macro_logloss']:.8f}" if 'macro_logloss' in metrics else 'pending')
    return Group(Panel(header, title='HeMix 稠密参数对齐实验'), table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.once:
        Console().print(render(root))
    else:
        with Live(render(root), screen=True, auto_refresh=False) as live:
            while True:
                time.sleep(5)
                live.update(render(root), refresh=True)
