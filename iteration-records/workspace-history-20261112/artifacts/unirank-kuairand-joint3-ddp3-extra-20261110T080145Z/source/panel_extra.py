"""Observe all original experiments plus the one additional DDP3 depth control."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def optional(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def render(root):
    setup = optional(root / 'execution-freeze.json')
    base = Path(setup.get('base_campaign', '/nonexistent'))
    base_setup = optional(base / 'setup.json')
    hemix = Path(base_setup.get('hemix_campaign', '/nonexistent'))
    jobs = optional(base / 'status.json').get('jobs', {})
    audit = optional(base / 'parameter-audit.json')
    cases = [('HeMix', hemix, optional(hemix / 'status.json'), 'hemix80', '16/80/3', [0], hemix / 'runs/hemix_token80/fit-01'),
             ('我们一层/单卡', base, jobs.get('joint1', {}), 'joint1', '16/256/1', [1], base / 'runs/joint1/fit-01'),
             ('我们三层/单卡', base, jobs.get('joint3', {}), 'joint3', '16/256/3', [2], base / 'runs/joint3/fit-01'),
             ('新增三层/三卡', root, optional(root / 'status.json'), 'joint3', '16/256/3', [3, 4, 5], root / 'fit-01')]
    table = Table(title='原三组继续；只额外增加一组三卡训练', expand=True)
    for name in ['GPU', '实验', 'E/T/层', '总参/稠密(M)', '阶段', '进度', '训练剩余', '测试平均AUC', '平均Logloss']:
        table.add_column(name)
    for label, campaign, status, key, dims, gpus, fit in cases:
        progress = optional(Path(status.get('output', '/nonexistent')) / 'progress.json')
        result = optional(fit / 'result.json')
        phase = '已完成' if result else {'train': '训练中', 'calibration': '校准中', 'valid': '验证中',
            'test': '测试中', 'valid_metrics': '计算验证指标', 'test_metrics': '计算测试指标',
            'initializing': '初始化', 'initializing_ddp': '初始化三卡', 'loading_train': '加载训练数据'}.get(progress.get('phase'), status.get('phase', '等待'))
        if status.get('phase') in ['failed', 'timeout', 'stopped']:
            phase = status['phase']
        gpu = ','.join(map(str, gpus))
        if progress.get('phase') in ['valid', 'test', 'valid_metrics', 'test_metrics', 'complete']:
            gpu = str(gpus[0])
        count = (f"{progress.get('steps',0)}/{progress['total_steps']}" if 'total_steps' in progress else
                 f"{progress.get('samples',0)/1e6:.1f}/{progress['total_samples']/1e6:.1f}M" if 'total_samples' in progress else '-')
        eta = progress.get('eta_training_seconds')
        p = audit.get(key, {})
        table.add_row(gpu, label, dims, f"{p['parameters']/1e6:.2f}/{p['dense_parameters']/1e6:.2f}" if p else '-',
            phase, count, f'{eta/3600:.2f}h' if eta else '-',
            f"{result['test']['auc']:.8f}" if result else '-', f"{result['test']['logloss']:.8f}" if result else '-')
    header = ('KuaiRand 六任务平均AUC | 全局batch8192 | 同一数据/优化器/seed\n'
              '新增三层版：GPU3/4/5、DDP+SyncBN；验证/测试仍用原完整单卡评分程序。\n'
              + datetime.now(timezone.utc).isoformat(timespec='seconds'))
    return Group(Panel(header, title='KuaiRand 三模型＋额外多卡对照'), table,
                 Panel('原有三组没有停止或重启。多卡版保留所有参数，训练卡时单独报告；BF16不宣称逐位相同。'))


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
