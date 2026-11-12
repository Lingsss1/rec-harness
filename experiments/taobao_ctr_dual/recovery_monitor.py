"""Supervise resumed studies under the original deadline and preserve incidents."""
import argparse
import os
from pathlib import Path
import time

from common import read,write
from controller import comparison,stop_children


def alive(item):
    path=Path('/proc')/str(item['pid'])/'stat'
    if not path.exists():return False
    words=path.read_text().split()
    return words[2]!='Z' and (not item.get('start_ticks') or words[21]==item['start_ticks'])


def summarize(root):
    comparison(root)
    result=read(root/'comparison.json')
    result['protocol_incident']=read(root/'recovery.json')
    if (root/'knowledge-repair.json').exists():result['knowledge_runtime_repair']=read(root/'knowledge-repair.json')
    for mode in ['auto','enhanced']:
        marker=root/mode/'final-validation.json'
        if marker.exists():result['methods'][mode]['validation_only_final']=read(marker)
    write(root/'comparison.json',result)
    report=root/'REPORT.txt'
    text=report.read_text()
    prefix=('运行器故障与恢复：auto 的描述性引用被错误拒绝，且故障退出过早触发了封闭测试。\n'
            '原 auto 记录完整归档；当时仅从头重跑 auto，仍沿用原截止时间。\n'
            '重跑 auto 的研究上下文不含已产生的测试结果，其后续评估限于验证集与模型种子重复。\n'
            '本轮不能声称恢复了干净的双版本独立测试比较；需新未用评估数据复核。\n\n')
    if (root/'knowledge-repair.json').exists():
        prefix+=('知识接口修复：两个版本在完整轮次边界从检查点续跑；基线、数据、评分与截止时间未变。\n'
                 '修复后来源 ID 可读取版本化原文，按知识类别召回并记录采纳/拒绝及代码链接。\n'
                 '修复前后属于不同运行器版本，不能把其收益直接归因于行业知识。详见 knowledge-repair.json。\n\n')
    report.write_text(prefix+text)


def monitor(root):
    deadline=read(root/'budget.json')['deadline_unix']
    terminal={'completed','completed_validation_only','failed','failed_before_assessment'}
    while True:
        statuses={mode:read(root/mode/'status.json') for mode in ['auto','enhanced']}
        for mode,state in statuses.items():
            if state['phase'] not in terminal and not alive(read(root/mode/'worker-process.json')):
                write(root/mode/'status.json',{'phase':'failed','error':'Worker disappeared; preserve receipts for recovery','updated_unix':time.time()})
        summarize(root)
        if all(read(root/mode/'status.json')['phase'] in terminal for mode in ['auto','enhanced']):
            write(root/'status.json',{'phase':'completed_with_protocol_incident','pid':os.getpid(),'updated_unix':time.time(),
                  'independent_dual_test_comparison_valid':False,'incident_ref':'recovery.json'})
            break
        if time.time()>=deadline:
            stop_children(root)
            write(root/'status.json',{'phase':'budget_reached_with_protocol_incident','pid':os.getpid(),'updated_unix':time.time(),
                  'independent_dual_test_comparison_valid':False,'incident_ref':'recovery.json'})
            summarize(root);break
        write(root/'status.json',{'phase':'dual_research_running_after_auto_restart','pid':os.getpid(),'updated_unix':time.time(),
              'deadline_unix':deadline,'independent_dual_test_comparison_valid':False,'incident_ref':'recovery.json'})
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();monitor(args.root.resolve())
