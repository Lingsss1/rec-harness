"""One panel for the existing HeMix fit and both requested width-256 controls."""
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
    audit = optional(root / 'parameter-audit.json')
    state = optional(root / 'status.json')
    jobs = state.get('jobs', {})
    report = optional(root / 'report.json')
    results = {r['model']: r for r in report.get('rows', [])}
    header = ('KuaiRand | embedding16 | 六任务平均AUC | 相同数据/训练配方/seed\n'
              'HeMix token80、三层；我们 token256、一层/三层。稠密参数和成本分别展示。\n'
              + datetime.now(timezone.utc).isoformat(timespec='seconds'))
    table = Table(title='三个实验：训练、验证、测试自动推进', expand=True)
    for title in ['GPU', '模型', 'E/D/层', '总参/稠密(M)', '阶段', '进度', '训练剩余', '测试平均AUC', '平均Logloss']:
        table.add_column(title)
    for kind, gpu, label, dims in [('hemix80', 0, 'HeMix', '16/80/3'),
                                   ('joint1', 1, '我们一层', '16/256/1'), ('joint3', 2, '我们三层', '16/256/3')]:
        job = optional(Path(setup.get('hemix_campaign', '/nonexistent')) / 'status.json') if kind == 'hemix80' else jobs.get(kind, {})
        progress = optional(Path(job.get('output', '/nonexistent')) / 'progress.json')
        result = results.get(kind, {})
        phase = '已完成' if result.get('status') == 'complete' else {
            'train': '训练中', 'calibration': '校准中', 'valid': '验证中', 'test': '测试中',
            'valid_metrics': '计算验证指标', 'test_metrics': '计算测试指标',
            'initializing': '初始化', 'loading_train': '加载训练数据'}.get(progress.get('phase'), job.get('phase', '预检中'))
        count = (f"{progress.get('steps',0)}/{progress['total_steps']}" if 'total_steps' in progress else
                 f"{progress.get('samples',0)/1e6:.1f}/{progress['total_samples']/1e6:.1f}M" if 'total_samples' in progress else '-')
        eta = progress.get('eta_training_seconds')
        p = audit.get(kind, {})
        parameters = f"{p['parameters']/1e6:.2f}/{p['dense_parameters']/1e6:.2f}" if p else '-'
        table.add_row(str(gpu), label, dims, parameters, phase, count, f'{eta/3600:.2f}h' if eta else '-',
                      f"{result['test_macro_auc']:.8f}" if 'test_macro_auc' in result else '-',
                      f"{result['test_macro_logloss']:.8f}" if 'test_macro_logloss' in result else '-')
    baseline = report.get('ours32_reference', {}).get('test_macro_auc')
    foot = Panel((f'历史 ours token32/一层：测试平均AUC {baseline:.8f}；' if baseline else '')
                 + '两版256从头训练。三层为三个独立联合交互块；HeMix原版不训练。')
    return Group(Panel(header, title='KuaiRand 三模型对照'), table, foot)


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
