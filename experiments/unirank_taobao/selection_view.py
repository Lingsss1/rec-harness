"""Read-only terminal view of validation-only historical selection auditing."""
import argparse
from pathlib import Path
import time

from rich.console import Console,Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from common import read


def render(report):
    tasks=report['audit_selection_policy']['tasks'];reference=report['original_reference']
    header=Text('新规则：四任务验证 AUC 等权平均；同分依次看平均 Logloss、参数量、模型 ID。\n',style='bold cyan')
    header.append('这是旧点击单目标搜索的只读重排审计，未重新训练、未重测，不改写原最终模型。\n',style='yellow')
    header.append(f"已审计 {report['trials']} 条验证记录；DIN 平均 AUC {reference['validation_macro_auc']:.6f}")
    ranking=Table(expand=True)
    for name in ['验证记录','平均 AUC','ΔDIN','点击','加购','收藏','购买','平均 Logloss']:ranking.add_column(name)
    def row(item):
        ranking.add_row(item['id'],f"{item['validation_macro_auc']:.6f}",
            f"{item['validation_macro_auc']-reference['validation_macro_auc']:+.6f}",
            *[f"{item['per_task'][task]['auc']:.6f}" for task in tasks],f"{item['validation_macro_logloss']:.6f}")
    row(reference)
    for item in report['top_validation'][:10]:row(item)
    best=report['audit_validation_best'];changes=Table(expand=True)
    for name in ['审计最优：'+best['id'],'AUC','ΔDIN AUC','Logloss','ΔDIN Logloss']:changes.add_column(name)
    for task in tasks:
        values=best['per_task'][task];delta=best['deltas_vs_din'][task]
        changes.add_row(task,f"{values['auc']:.6f}",f"{delta['auc']:+.6f}",f"{values['logloss']:.6f}",f"{delta['logloss']:+.6f}")
    return Panel(Group(header,ranking,changes),title='多任务选模规则 / 历史验证审计',border_style='cyan')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args();console=Console()
    if args.once:console.print(render(read(args.report)))
    else:
        with Live(render(read(args.report)),console=console,screen=True,auto_refresh=False) as live:
            while True:
                time.sleep(2);live.update(render(read(args.report)),refresh=True)
