#!/usr/bin/env python3
"""Offline evidence-delivery simulation over the published v5/v6 20-round prefix.

No model calls, training, promotion changes, or original archive writes. This is
a controlled cross-campaign retrieval comparison, not the original agent run.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rec_harness.core import digest,file_hash,read_json,write_json
from rec_harness.adapters.amazon_memory import validate_memory
from rec_harness.adapters.amazon_hypothesis_graph import project
from rec_harness.adapters.amazon_evidence_view import evidence_view,fact_id


def measurements(graph,reports,entries,failures):
    ids={n['id'] for n in graph['nodes']};visible=ids-{graph['nodes'][0]['id']}
    recent=[fact_id(r) for r in reports[-2:]]
    broken=[]
    for n in graph['nodes']:
        refs=set(n['attrs'].get('evidence_refs',[]))
        if n['attrs'].get('origin')=='imported_memory':refs.update(n['attrs'].get('evidence_dependency_ids',[]))
        broken.extend(sorted(refs-ids))
    for e in graph['edges']:
        broken.extend(sorted({e['source'],e['target'],*e['attrs'].get('evidence_refs',[])}-ids))
    allowed={r['hash'] for r in reports}
    leaks=[n['id'] for n in graph['nodes'] if n['attrs'].get('readout_hash') and n['attrs']['readout_hash'] not in allowed]
    leaks.extend(n['id'] for n in graph['nodes'] if n['attrs'].get('partition') not in (None,'development'))
    expected_failures={'execution-failure-'+digest(f) for f in failures}
    expected_memories={digest(e) for e in entries}
    leaks.extend(n['id'] for n in graph['nodes'] if n['attrs'].get('source_memory_hash') and n['attrs']['source_memory_hash'] not in expected_memories)
    leaks.extend(n['id'] for n in graph['nodes'] if n['attrs'].get('kind')=='execution_failure' and n['id'] not in expected_failures)
    fake_support=[e['id'] for e in graph['edges'] if e['relation'] in {'SUPPORTS','CONTRADICTS'} and e['attrs'].get('origin')=='imported_memory']
    counter=[fact_id(r) for r in reports if r['summary'].get('eligible_for_confirmation') is False]
    return {'nodes':len(ids),'edges':len(graph['edges']),'historical_nodes':len(visible),
            'recent_expected':recent,'recent_visible':sorted(set(recent)&ids),
            'measurement_gaps':sum(n['type']=='MeasurementGap' for n in graph['nodes']),
            'claims':sum(n['type']=='Claim' for n in graph['nodes']),
            'counter_readouts_available':len(counter),'counter_readouts_visible':len(set(counter)&ids),
            'execution_failures':sum(n['attrs'].get('kind')=='execution_failure' for n in graph['nodes']),
            'broken_references':broken,'prefix_leaks':leaks,'fabricated_support_edges':fake_support,
            'bytes':len(json.dumps(graph,ensure_ascii=False).encode()),'graph_hash':graph['hash']}


def simulate(root=ROOT,limit=64):
    summary_path=root/'docs/experiments/amazon-autoresearch-v5-v6-summary.json'
    memory_path=root/'docs/experiments/amazon-autoresearch-v7-runtime/records/development-memory.json'
    summary=read_json(summary_path);pack=read_json(memory_path)
    protocol=pack['protocol_hash'];validate_memory(pack,protocol,[42])
    mapping={e['round']:e for e in pack['entries']}
    reports=[];entries=[];failures=[];steps=[];availability=[]
    # An empty target study deliberately isolates cross-campaign read-back.
    # We do not pretend exported data are a native audited historical Store.
    snapshot={'version':0,'nodes':[],'edges':[]}
    for index,item in enumerate(summary['rounds'],1):
        campaign='amazon-next-v5' if item['campaign']=='v5' else 'amazon-next-v6-sol-low'
        name=f"{campaign}/round-{item['round']}"
        old=project(snapshot,reports,'selected_v3',protocol,[],limit=limit)
        new=evidence_view(snapshot,reports,entries,failures,'selected_v3',protocol,[],selected_history=reports,limit=limit)
        steps.append({'index':index,'before_round':name,'available_prior_rounds':index-1,
            'available_readout_hashes':[r['hash'] for r in reports],
            'legacy':measurements(old,reports,entries,failures),
            'repaired':measurements(new,reports,entries,failures),
            'selected_nodes':[{'id':n['id'],'type':n['type'],'origin':n['attrs'].get('origin'),
                               'reason':new['retrieval']['selection_reasons'].get(n['id'],'current_baseline')}
                              for n in new['nodes']],
            'omissions':new['retrieval']['omitted_requests']})
        # Reveal THIS round only after its pre-round measurement. Do not use the
        # recovery-only interpretation of v5/8: it is absent from this export.
        original=mapping.get(name);entry=copy.deepcopy(original) if original else {'round':name}
        report=entry.get('development')
        if item.get('readout_hash'):
            if not report or report['hash']!=item['readout_hash']:
                raise ValueError('missing or mismatched source readout: '+name)
            reports.append(report)
        if item.get('original_failure'):
            failure={'round':name,'reason':item['original_failure'],'status':'unresolved',
                     'scope':'Published execution failure; not mechanism refutation. Recovery timing is not reconstructed.'}
            failures.append(failure)
        entries.append(entry)
        availability.append({'round':name,'memory_entry_available':original is not None,
            'readout_available':report is not None,'interpretation_available':bool(entry.get('provisional_knowledge')),
            'original_failure':item.get('original_failure')})
    # Terminal read-back is separate: the 20th outcome cannot be used before round 20.
    terminal=evidence_view(snapshot,reports,entries,failures,'selected_v3',protocol,[],selected_history=reports,limit=limit)
    totals={}
    for policy in ('legacy','repaired'):
        rows=[step[policy] for step in steps]
        totals[policy]={
            'prefixes_with_history':sum(bool(r['historical_nodes']) for r in rows),
            'recent_readout_hits':sum(len(r['recent_visible']) for r in rows),
            'recent_readout_opportunities':sum(len(r['recent_expected']) for r in rows),
            'prefixes_with_measurement_gaps':sum(r['measurement_gaps']>0 for r in rows),
            'counter_readout_hits':sum(r['counter_readouts_visible'] for r in rows),
            'counter_readout_opportunities':sum(r['counter_readouts_available'] for r in rows),
            'max_nodes':max(r['nodes'] for r in rows),'max_bytes':max(r['bytes'] for r in rows),
            'broken_references':sum(len(r['broken_references']) for r in rows),
            'prefix_leaks':sum(len(r['prefix_leaks']) for r in rows),
            'fabricated_support_edges':sum(len(r['fabricated_support_edges']) for r in rows)}
    if len(steps)!=20:raise AssertionError('expected exactly 20 published rounds')
    for step in steps:
        check=step['repaired']
        if check['nodes']>limit or check['broken_references'] or check['prefix_leaks'] or check['fabricated_support_edges']:
            raise AssertionError('invalid repaired prefix: '+step['before_round'])
    result={'schema':'amazon-evidence-prefix-simulation-v1','node_limit':limit,
        'qualification':'Controlled cross-campaign read-back over actual exported v5/v6 records in round order; both policies receive identical prior reports. No native source graph or original per-round embedding scores reconstructed. No generated hypothesis, performance or causal benefit evaluated.',
        'availability_assumption':'Each exported record becomes available after its numbered round. Exact wall-clock/recovery timing and original call contexts are unavailable; this is round-prefix simulation, not exact historical replay. Before round 1 no historical metrics are supplied; the current baseline rows are intentionally empty in both arms.',
        'sources':[{'path':str(p.relative_to(root)),'sha256':file_hash(p)} for p in (summary_path,memory_path)],
        'implementation_sha256':{str(p.relative_to(root)):file_hash(p) for p in [
            root/'rec_harness/adapters/amazon_evidence_view.py',
            root/'rec_harness/adapters/amazon_hypothesis_graph.py',root/'rec_harness/retrieval.py',
            root/'rec_harness/adapters/amazon_autonomous.py',root/'rec_harness/adapters/amazon_compact.py',
            root/'rec_harness/adapters/amazon_memory.py',
            root/'scripts/replay_amazon_evidence_graph.py']},
        'availability':availability,'totals':totals,'steps':steps,
        'terminal_readback':measurements(terminal,reports,entries,failures)}
    return {**result,'hash':digest(result)}


def markdown(result):
    t=result['totals'];old=t['legacy'];new=t['repaired']
    lines=['# 证据图 v2：20 轮记录前缀模拟验证','',
        '数据：v5 的 8 轮 + v6 的 12 轮。每轮生成上下文后，才释放该轮的已记录结果。',
        '这是跨运行读回的受控模拟：目标本地图为空，两种策略收到完全相同的此前开发读出；没有复现原始语义排名或调用 LLM。',
        '记录按编号结束后可见是模拟假设；原始调用时序、恢复时间及完整原生图不齐，因此不能宣称精确重放历史决策。','',
        '| 检查 | 旧策略 | 修复策略 |','|---|---:|---:|',
        f"| 有历史节点的轮前上下文 | {old['prefixes_with_history']}/20 | {new['prefixes_with_history']}/20 |",
        f"| 最近两次可用读出的命中 | {old['recent_readout_hits']}/{old['recent_readout_opportunities']} | {new['recent_readout_hits']}/{new['recent_readout_opportunities']} |",
        f"| 包含测量缺口的轮前上下文 | {old['prefixes_with_measurement_gaps']}/20 | {new['prefixes_with_measurement_gaps']}/20 |",
        f"| 开发门槛未通过读出的命中（非机制反证） | {old['counter_readout_hits']}/{old['counter_readout_opportunities']} | {new['counter_readout_hits']}/{new['counter_readout_opportunities']} |",
        f"| 最大节点数 | {old['max_nodes']} | {new['max_nodes']} |",
        f"| 悬空引用 / 前缀越界 / 伪造支持边 | {old['broken_references']} / {old['prefix_leaks']} / {old['fabricated_support_edges']} | {new['broken_references']} / {new['prefix_leaks']} / {new['fabricated_support_edges']} |",'',
        '第一轮没有本次 20 轮中的先前证据，因此保持基线节点。开发读出不等于完整研究周期成功；执行错误始终只作为执行错误记录。',
        '外部解释按导出记忆呈现；缺失的原始训练收据、诊断节点和支持关系没有补造。上下文变大是显式代价，不能由检索命中推断研究效率提高。','',
        f"默认预算下修复视图最大 {new['max_bytes']:,} UTF-8 字节（JSON 序列化口径，不等于 token 数）。节点预算不等于字节预算，生产运行仍需观察实际上下文和调用成本。",'',
        '| 开始轮次 | 已有读出 | 修复后节点 / 边 | 测量缺口 | 最近读出命中 |',
        '|---|---:|---:|---:|---:|']
    for step in result['steps']:
        r=step['repaired'];lines.append(f"| {step['before_round']} | {len(step['available_readout_hashes'])} | {r['nodes']} / {r['edges']} | {r['measurement_gaps']} | {len(r['recent_visible'])}/{len(r['recent_expected'])} |")
    lines+=['','复现：`python3 scripts/replay_amazon_evidence_graph.py`。机器可读明细、输入和实现哈希见 [JSON](evidence-graph-prefix-v2.json)。',
            '验证结论仅限证据可达性、来源区分、预算与引用完整性；没有重训、改变旧晋升结果，也没有证明新假设质量或推荐指标改善。','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/experiments/evidence-graph-prefix-v2.json')
    parser.add_argument('--limit',type=int,default=64);args=parser.parse_args()
    result=simulate(limit=args.limit);write_json(args.output,result)
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps(result['totals'],ensure_ascii=False,indent=2))
