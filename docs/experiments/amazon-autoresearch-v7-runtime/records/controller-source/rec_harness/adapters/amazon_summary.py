"""Reproducible validation readout from audited pairs and the complete cost ledger."""
from pathlib import Path
import statistics

from rec_harness.core import Artifacts, ContractError, digest, file_hash, write_json
from rec_harness.store import Store
from .amazon_controls import load_config
from .amazon_report import paired_report


def summarize_contrast(report, metric):
    partitions = {}
    for name, part in report['partitions'].items():
        if not part['complete']:
            raise ContractError('cannot publish complete summary with missing experiment pairs')
        by_dataset = {}
        for row in part['rows']:
            by_dataset.setdefault(row['dataset'], []).append(row)
        slices = []
        for dataset, rows in sorted(by_dataset.items()):
            slices.append({'dataset': dataset,
                'baseline': statistics.mean(row['baseline'][metric] for row in rows),
                'candidate': statistics.mean(row['candidate'][metric] for row in rows),
                'effect': statistics.mean(row['delta'][metric] for row in rows),
                'seed_count': len(rows)})
        partitions[name] = {k: part[k] for k in ('baseline_mean', 'candidate_mean', 'mean_effect',
                           'seed_effects', 'minimum_effect', 'effect_gate_passed')}
        partitions[name].update({k:part[k] for k in ('effect_gate_scope','per_seed_effect_gate_passed',
            'all_seed_effect_gates_passed','all_seed_gate_scope') if k in part})
        partitions[name].update(datasets=slices,
            negative_slices=[{'dataset': row['dataset'], 'seed': row['seed'], 'effect': row['delta'][metric]}
                             for row in part['rows'] if row['delta'][metric] < 0])
    return {'baseline': report['baseline'], 'candidate': report['candidate'],
            'mechanism_status': report['mechanism_status'], 'partitions': partitions}


def render(report):
    lines = ['# Amazon NextItNet 验证报告', '',
             '本报告复核既有模型及固定机制消融，仅使用验证结果；不代表新 Codex 候选收益或最终测试成绩。', '',
             f"协议：`{report['protocol_hash']}`。主指标：{report['metric']}。类别等权平均，种子作为独立初始化单位。", '',
             '## 配对结果', '',
             '| 对比（后者减前者） | 阶段 | 基准 | 候选 | 差值（百分点） | 各种子差值（百分点） | 每个种子均超过门槛 |',
             '|---|---|---:|---:|---:|---|---|']
    for contrast in report['contrasts']:
        for partition, row in contrast['partitions'].items():
            effects = ', '.join(f'{seed}: {effect * 100:+.3f}' for seed, effect in row['seed_effects'].items())
            gate=row.get('all_seed_effect_gates_passed')
            gate_label='未提供' if gate is None else ('是' if gate else '否')
            lines.append(f"| {contrast['baseline']} → {contrast['candidate']} | {partition} | {row['baseline_mean']:.6f} | {row['candidate_mean']:.6f} | {row['mean_effect'] * 100:+.3f} | {effects} | {gate_label} |")
    for contrast in report['contrasts']:
        lines += ['', f"## {contrast['baseline']} → {contrast['candidate']}：确认阶段类别结果", '',
                  '| 类别 | 基准 | 候选 | 差值（百分点） |', '|---|---:|---:|---:|']
        for row in contrast['partitions']['confirmation']['datasets']:
            lines.append(f"| {row['dataset']} | {row['baseline']:.6f} | {row['candidate']:.6f} | {row['effect'] * 100:+.3f} |")
        lines += ['', '负向切片（包含开发阶段）：', '']
        negatives = [(partition, row) for partition, part in contrast['partitions'].items() for row in part['negative_slices']]
        lines += [f"- {partition} / {row['dataset']} / seed {row['seed']}: {row['effect'] * 100:+.3f} 个百分点。" for partition, row in negatives] or ['- 无负向切片。']
    budget = report['budget']
    lines += ['', '## 成本与推断范围', '',
              f"预算 {budget['limit']:.3f} GPU 秒；已计费或保留 {budget['charged_or_reserved']:.3f}；剩余 {budget['remaining']:.3f}。", '',
              '预算来自完整账本，包含诊断、失败与继承成本；本次派生报告不新增训练次数。', '',
              '- 确认种子使用不同初始化，但共享验证用户；不是新的隐藏测试集。',
              '- 既有模型同时改变多个设置，整体增益不能全部归因于前缀混合。',
              '- 固定前缀消融在既有模型结果已暴露后注册；其结论限于该对照和这些数据。',
              '- 阈值通过不等于统计显著；有限种子不支持普遍有效性结论。',
              '- effect_gate_passed 仅指平均增益达标；每个种子均达标另列。此报告本身不执行候选晋升。',
              '- 新 Codex 候选、正式候选准入和最终测试仍需完成。', '',
              '## 审计身份', '', f"报告摘要：`{report['hash']}`。事件链头：`{report['event_state']['head']}`。", '',
              '同目录 `report.json` 包含模型源码身份、配对报告 CAS 引用和原始预算读数。']
    return '\n'.join(lines) + '\n'


def build_summary(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ContractError('summary output must be a new directory')
    config = load_config(root)
    store = Store(root / 'memory')
    try:
        event_state = store.verify()
        sources = config['model_versions']
        for version in ('fixed_baseline', 'legacy_incumbent', 'legacy_no_prefix'):
            if file_hash(sources[version]['path']) != sources[version]['sha256']:
                raise ContractError('summary model source changed')
        output.mkdir(parents=True)
        artifacts = Artifacts(root / 'artifacts/objects')
        contrasts, refs = [], {}
        for baseline, candidate in [('fixed_baseline', 'legacy_incumbent'), ('legacy_no_prefix', 'legacy_incumbent')]:
            path = output / (baseline + '-versus-' + candidate + '.json')
            report = paired_report(root, baseline, candidate, output=path)
            contrasts.append(summarize_contrast(report, config['protocol']['metric']))
            refs[path.name] = artifacts.put(path)
        budget = store.budget()
        if store.verify() != event_state:
            raise ContractError('study events changed during summary; retry in a new directory')
        body = {'schema': 'amazon-validation-summary-v1', 'protocol_hash': config['protocol_hash'],
                'metric': config['protocol']['metric'], 'protocol': config['protocol'],
                'model_source_hashes': {v: sources[v]['sha256'] for v in ('fixed_baseline', 'legacy_incumbent', 'legacy_no_prefix')},
                'contrasts': contrasts, 'paired_reports': refs, 'budget': budget, 'event_state': event_state,
                'qualification': 'existing_model_validation_and_fixed_ablation_only',
                'new_codex_candidate_gain_verified': False, 'final_test_result_included': False}
        report = {**body, 'hash': digest(body)}
        write_json(output / 'report.json', report)
        (output / 'README.md').write_text(render(report))
        return report
    finally:
        store.close()
