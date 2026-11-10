"""Export auditable macro-AUC research and fixed-transfer records without data/weights."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(campaign, completed_transfer, running_transfer, output):
    campaign, completed_transfer, running_transfer = [p.resolve() for p in
                                                       [campaign, completed_transfer, running_transfer]]
    started = time.time()
    if output.exists():
        raise ValueError('Use a new archive directory')
    output.mkdir(parents=True)
    copies = []

    def copy(source, target, root):
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Read one immutable version even when snapshotting an atomically replaced progress file.
        destination.write_bytes(source.read_bytes())
        copies.append({'path': target.as_posix(), 'source_campaign': root.name,
                       'source_path': source.relative_to(root).as_posix(), 'sha256': sha(destination)})

    status = read(campaign / 'status.json')
    if status['phase'] != 'topic_research_stopped':
        raise ValueError('The research campaign must have finished')
    states = [read(p) for p in sorted((campaign / 'topics').glob('gpu-*/state.json'))]
    assessment = read(campaign / 'final-assessment-top3/report.json')
    freeze = read(campaign / 'final-assessment-top3/selection-freeze.json')
    if assessment['status'] != 'complete':
        raise ValueError('Final checkpoint assessment must be complete')

    # Preserve complete per-round proposal/context/tool streams and all attempted implementations.
    excluded = {'data', 'upstream', 'reference-cache', '__pycache__', 'dashboard'}
    suffixes = {'.json', '.jsonl', '.py', '.txt', '.yaml', '.log', '.patch'}
    for path in sorted(campaign.rglob('*')):
        relative = path.relative_to(campaign)
        if not path.is_file() or path.is_symlink() or excluded.intersection(relative.parts):
            continue
        if relative.parts[0] == 'prepared' and path.name not in {'feature_map.json', 'meta_data.json', 'block_manifest.json'}:
            continue
        if path.suffix in suffixes:
            copy(path, Path('research') / relative, campaign)

    trials = [row for state in states for row in state['trials']]
    references = read(campaign / 'references.json')['trials']
    catalog = {row['id']: row for row in references + trials}
    if len({row['id'] for row in trials}) != len(trials):
        raise ValueError('Duplicate research trial IDs')
    iteration_rows = []
    for row in trials:
        metrics = row.get('metrics', {})
        validation = metrics.get('validation', {})
        if row['status'] == 'success':
            source = Path(row['model_path'])
            if sha(source) != row['source_sha256']:
                raise ValueError('Training source changed: ' + row['id'])
            expected = math.fsum(v['auc'] for v in validation['per_task'].values()) / 4
            if len(validation['per_task']) != 4 or abs(expected - validation['macro_auc']) > 1e-12:
                raise ValueError('Research selection was not the four-task mean')
        parent = catalog.get(row.get('parent'))
        parent_score = parent.get('metrics', {}).get('validation', {}).get('macro_auc') if parent else None
        model_relative = Path('research') / Path(row['model_path']).relative_to(campaign)
        item = {'id': row['id'], 'gpu': row['gpu'], 'topic_id': row['topic_id'], 'round': row['round'],
                'parent': row.get('parent'), 'role': row['role'], 'status': row['status'], 'seed': row['seed'],
                'validation_macro_auc': validation.get('macro_auc'),
                'validation_macro_logloss': validation.get('macro_logloss'),
                'delta_macro_vs_parent': validation['macro_auc'] - parent_score if validation and parent_score is not None else None,
                'parameters': metrics.get('parameters'), 'train_seconds': metrics.get('train_seconds'),
                'job_wall_seconds': row.get('wall_seconds'), 'peak_gpu_bytes': metrics.get('peak_gpu_bytes'),
                'model_path': model_relative.as_posix(), 'source_sha256': row['source_sha256'],
                'checkpoint_sha256': metrics.get('checkpoint_sha256'), 'study_ref': row.get('study_ref')}
        for task, values in validation.get('per_task', {}).items():
            item[task + '_validation_auc'] = values['auc']
            item[task + '_validation_logloss'] = values['logloss']
        if parent:
            parent_path = Path(parent['model_path'])
            patch = ''.join(difflib.unified_diff(parent_path.read_text().splitlines(True),
                Path(row['model_path']).read_text().splitlines(True),
                fromfile=parent['id'] + '/model.py', tofile=row['id'] + '/model.py'))
            (output / model_relative.parent / 'changes-from-parent.patch').write_text(patch)
        iteration_rows.append(item)
    write(output / 'iterations.json', iteration_rows)
    csvwrite(output / 'iterations.csv', iteration_rows)
    csvwrite(output / 'topic-incumbents.csv', freeze['topic_incumbent_trace'])
    write(output / 'topic-representatives.json', freeze['topic_representatives'])

    calls = []
    for state in states:
        for call in state['agent_calls']:
            token_counts = Counter()
            for usage in call.get('usage', []):
                token_counts.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
            calls.append({'gpu': state['gpu'], 'round': call['round'], 'phase': call['phase'],
                'topic_id': call.get('topic_id'), 'status': call['status'], 'returncode': call.get('returncode'),
                'model': call.get('model'), 'reasoning_effort': call.get('reasoning_effort'),
                'wall_seconds': call.get('wall_seconds'), **token_counts})
    write(output / 'calls.json', calls)
    csvwrite(output / 'calls.csv', calls)
    write(output / 'topic-decisions.json', [{'gpu': state['gpu'], 'topics': state['topics'],
                                           'decisions': state['decisions'], 'memory': state['memory']} for state in states])
    lineage = []
    current = catalog[freeze['suite'][0]['id']]
    seen = set()
    while current:
        if current['id'] in seen:
            raise ValueError('Cyclic recorded parent lineage')
        seen.add(current['id'])
        lineage.append({'id': current['id'], 'parent': current.get('parent'),
                        'role': current['role'], 'validation_macro_auc': current['metrics']['validation']['macro_auc']})
        current = catalog.get(current.get('parent'))
    lineage.reverse()
    csvwrite(output / 'best-parent-lineage.csv', lineage)

    transfers = []
    for root, label, require_complete in [(completed_transfer, 'g3-r0006-completed', True),
                                           (running_transfer, 'g3-r0018-snapshot', False)]:
        state = read(root / 'supervisor-status.json')
        if require_complete and (state['phase'] != 'finished' or any(j['phase'] != 'complete' for j in state['jobs'].values())):
            raise ValueError('The completed transfer is incomplete')
        prefix = Path('generalization') / label
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or {'data', '__pycache__'}.intersection(relative.parts):
                continue
            if path.suffix not in suffixes:
                continue
            copy(path, prefix / relative, root)
        # Verify frozen sources, completed target metrics, original selection and reused target populations.
        for dataset in state['jobs']:
            folder = root / 'datasets' / dataset
            protocol = read(folder / 'protocol.json')
            for name, expected_hash in protocol['source_sha256'].items():
                if sha(root / 'source' / name) != expected_hash:
                    raise ValueError('Transfer frozen source drift: ' + label + '/' + name)
            run = root / 'runs' / dataset / 'candidate/fit-01'
            if not (run / 'result.json').exists():
                continue
            result = read(run / 'result.json')
            selection = read(run / 'selection.json')
            din = read(root / 'reference-runs' / dataset / 'din/result.json')
            if result['protocol_hash'] != protocol['identity'] or result['checkpoint_sha256'] != selection['checkpoint_sha256']:
                raise ValueError('Transfer result/selection mismatch')
            if selection['test_used_for_selection']:
                raise ValueError('Target test was used for checkpoint selection')
            for split, metric_key in [('valid', 'validation'), ('test', 'test')]:
                scores = result[metric_key]
                if scores['samples'] != protocol['sample_size'][split] or set(scores['per_task']) != set(protocol['tasks']):
                    raise ValueError('Target sample/task coverage mismatch')
                for metric in ['auc', 'logloss']:
                    if abs(scores[metric] - math.fsum(v[metric] for v in scores['per_task'].values()) / len(protocol['tasks'])) > 1e-12:
                        raise ValueError('Target macro metric mismatch')
            for task, metrics in result['test']['per_task'].items():
                b = din['test']['per_task'][task]
                if metrics['samples'] != b['samples'] or metrics['positives'] != b['positives']:
                    raise ValueError('Reused DIN target populations differ')
        transfers.append({'archive': label, 'source_model': read(root / 'setup.json')['source_model'],
                          'phase': state['phase'], 'jobs': {k: v['phase'] for k, v in state['jobs'].items()},
                          'completed_test_runs': sum((root / 'runs' / d / 'candidate/fit-01/result.json').exists() for d in state['jobs'])})

    test_rows, test_tasks = [], []
    for row in freeze['suite']:
        key = row['key']
        metrics = assessment['test'][key]['metrics']
        valid_replay = read(campaign / 'final-assessment-top3' / key / 'valid/result.json')
        raw_test = read(campaign / 'final-assessment-top3' / key / 'test/result.json')
        for task, values in metrics['per_task'].items():
            if abs(values['auc'] - raw_test['metrics']['per_task'][task]['auc']) > 1e-12:
                raise ValueError('Test summary differs from raw scorer result')
            if abs(row['validation']['per_task'][task]['auc'] - valid_replay['metrics']['per_task'][task]['auc']) > 1e-8:
                raise ValueError('Final checkpoint validation replay mismatch')
            test_tasks.append({'model': row['id'], 'rank_by_validation': row['rank_by_validation'], 'task': task,
                               'auc': values['auc'], 'logloss': values['logloss'],
                               'samples': values['samples'], 'positives': values['positives']})
        test_rows.append({'model': row['id'], 'rank_by_validation': row['rank_by_validation'],
            'validation_macro_auc': row['validation']['macro_auc'], 'test_macro_auc': metrics['macro_auc'],
            'test_macro_logloss': metrics['macro_logloss'], 'parameters': row['parameters'],
            'train_seconds': row['train_seconds'], 'test_inference_seconds': metrics['inference_seconds']})
    csvwrite(output / 'final-tests.csv', test_rows)
    csvwrite(output / 'final-test-tasks.csv', test_tasks)
    counts = {'completed_rounds': sum(s['completed_research_rounds'] for s in states),
              'research_trials': len(trials), 'trial_status_counts': dict(Counter(r['status'] for r in trials)),
              'agent_calls': len(calls), 'call_status_counts': dict(Counter(c['status'] for c in calls)),
              'topics': sum(len(s['topics']) for s in states),
              'baseline_results': len(list((campaign / 'baselines').rglob('result.json'))),
              'calibration_results': len(list((campaign / 'calibration').rglob('result.json')))}
    exported = datetime.now(timezone.utc).isoformat(timespec='seconds')
    write(output / 'summary.json', {'exported_at_utc': exported, 'source_campaign': campaign.name,
          'selection': 'validation equal mean of is_click/cart/fav/buy AUC', 'budget': read(campaign / 'budget.json'),
          'counts': counts, 'final_tests': test_rows, 'generalization': transfers,
          'snapshot_started_unix': started, 'snapshot_finished_unix': time.time(),
          'research_trials_exclude_separately_archived_baseline_and_calibration': True,
          'token_semantics': 'input_tokens includes cached_input_tokens; reasoning_output_tokens is part of output_tokens; do not add subsets twice',
          'limitations': ['single training seed', 'source/target test sets observed for earlier models; supplementary exploratory tests',
                         'g3-r0006 cross-dataset results are not evidence for final g3-r0018',
                         'parent diffs and score deltas are descriptive, not causal mechanism proof']})
    lines = ['# UniRank 淘宝：12 小时平均 AUC 自主研究迭代记录', '',
        f'归档时间：{exported}。研究 `{campaign.name}` 已结束；{counts["completed_rounds"]} 轮研究、'
        f'{counts["research_trials"]} 次课题训练尝试、{counts["agent_calls"]} 次 Agent 调用、{counts["topics"]} 个动态课题。'
        'DIN 基线和成本校准另列，仍计入原研究预算。', '',
        '从新的 DIN 起跑，四任务联合训练，验证集点击、加购、收藏、购买 AUC 的等权平均贯穿 checkpoint、父模型、课题收益和最终选模。'
        '本轮与之前以点击选模的 48 小时实验不同。每 GPU 一个课题，每 6 个完成轮次决策继续、切换或合并。'
        '研究模型为 gpt-6.1-sol / high；个人知识未启用，行业参考为 UniRank 的 15 种对比方法。', '',
        '## 最终淘宝测试', '',
        '前三从每个真实课题完成轮次后保留的最优模型中选出，按验证平均 AUC 排序，并对相同代码、配置和 checkpoint 去重。'
        '整组先冻结，再回放验证集，通过后执行测试；没有依据测试成绩重选。DIN 为额外参照。', '',
        '| 模型 | 验证平均 AUC | 测试平均 AUC | 测试平均 Logloss | 参数(M) | 训练(s) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in test_rows:
        lines.append(f'| {row["model"]} | {row["validation_macro_auc"]:.8f} | {row["test_macro_auc"]:.8f} | {row["test_macro_logloss"]:.8f} | {row["parameters"]/1e6:.3f} | {row["train_seconds"]:.1f} |')
    lines += ['', '逐任务 AUC、Logloss、正例数和样本数见 [final-test-tasks.csv](final-test-tasks.csv)。'
        '前三均为单种子；第一、第二名测试平均 AUC 仅相差约 0.000115，不能视为显著优势或跨数据集泛化证明。', '',
        '## 最优模型的记录路线', '',
        '以下按试验的 parent 字段回溯；它反映记录中的承接关系。跨课题导入和实际代码继承请结合 proposal 与源码差异查看，连线不构成机制归因。', '',
        '| 模型 | 验证平均 AUC | 角色 |', '|---|---:|---|']
    for row in lineage:
        lines.append(f'| {row["id"]} | {row["validation_macro_auc"]:.8f} | {row["role"]} |')
    lines += ['', '## 跨数据集评测状态', '',
        '- [g3-r0006 已完成结果](generalization/g3-r0006-completed/report.txt)：QK-Video、KuaiRand、TAAC-25、MerRec 的四次固定结构重训练全部完成；保留 DIN 和此前模型对照、逐任务退步及成本。它是中途模型，不是最终第一名。',
        '- [g3-r0018 当前快照](generalization/g3-r0018-snapshot/report.txt)：最终第一名的四路结构迁移正在运行，尚不能报告最终泛化成绩。20 步校准、正式训练进度和原始协议已归档；状态仅代表归档期间。',
        '- 两组均为各数据集原生词表/字段/任务头，从头训练；这不是把淘宝权重直接用于其他数据集的零样本迁移。任务平均 AUC 在各数据集内计算，不合并为跨数据集总分。', '',
        '## 轨迹入口', '',
        '- [iterations.csv](iterations.csv)、[iterations.json](iterations.json)：全部课题训练、角色、父模型、四任务验证指标和成本；未晋升或被否定的变体保留。',
        '- [topic-incumbents.csv](topic-incumbents.csv)、[topic-decisions.json](topic-decisions.json)：逐轮保留模型、课题切换和合并；原始状态在 research/topics/gpu-N/state.json。',
        '- research/topics/gpu-N/rounds/rXXXX/propose/proposal.json：前瞻问题、假设、相近方法、对照、知识采用与拒绝依据。',
        '- 每轮 codex.jsonl、runtime/task/workspace 导出、knowledge-access/usage：当时上下文、工具调用与知识访问证据。',
        '- research/topics/gpu-N/trials/TRIAL/：实际模型、配置、result 与 changes-from-parent.patch；最终前三代码在 research/final-assessment-top3/suite/。',
        '- [calls.csv](calls.csv)：Agent 调用和 Token 用量；input 包含 cached，reasoning 是 output 的子集，不能重复相加。',
        '- research/source/：冻结运行器、训练器、通用 Agent 指令与任务设计；research/config/、protocol.json、dataset-source-manifest.json：复现配置与数据来源。',
        '- [REPRODUCE.md](REPRODUCE.md)：重建与评测入口；copy-sources.json 和 manifest.json：来源与逐文件完整性。', '',
        '## 证据边界', '',
        '使用固定 UniRank 官方切分与 official_offline 历史反馈语义，最近 100 次历史。'
        '原测试集曾用于此前模型，本轮测试属于补充离线评估；研究期间用户要求的中途补测另有记录，不作为研究 Agent 的选模反馈。'
        '未完成跨数据集评测不计作成功证据；未做多种子或参数/算力匹配消融的结论不能解释为稳定泛化或单机制因果结论。', '',
        '原始绝对路径用于来源审计，迁移环境需重建路径。归档不含原始样本、词表大缓存、模型权重、逐样本预测、认证文件、环境目录或论文全文 HTML/PDF 缓存。'
        '运行中实验仅做只读快照；此次上传不会启动、停止或修改训练。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    (output / 'REPRODUCE.md').write_text('''# 复现入口

环境与版本见 research/environment*；UniRank 固定提交为 90bb99a60724e975ec565836266fd86dcad0039d。
数据 revision、文件校验和、固定切分、特征映射、历史口径见各 protocol 和 dataset-source-manifest。
权重及数据未上传，因此归档适合检查和同协议重训练，不能直接重放已训练模型。

淘宝模型：以 research/final-assessment-top3/suite/rank_1/model.py 和 model-config.json 为最终配置。
原训练器是 research/source/train.py，命令入口为 --campaign CAMPAIGN --model MODEL --config CONFIG --output OUTPUT --seed 20262027。
需要先按 research/config 和 source 中的数据准备程序重建目录，并把原绝对路径替换为本机路径。
选模使用 research/selection-policy.json；测试程序为 research/postprocess/assess_topk.py。

外部数据集：generalization/*/source/train.py 为实际执行代码。
先按 datasets/DATASET/dataset-source-manifest.json 获取完整固定版本，重建 data/、prepared/、parameters.json 中的本地路径。
需要重建协议中的 source_sha256 和 identity；不得改动评分、切分、任务顺序和训练配方。
运行：python source/train.py --root ROOT --dataset DATASET --kind candidate --output FRESH_OUTPUT。
训练器按验证集全任务平均 AUC 冻结 checkpoint 和选择凭据后再测试；不要通过测试成绩调参或选 checkpoint。
source/taobao-original.py 与 interface-adaptation.patch 记录原模型和字段/任务数适配。
历史 DIN 与上一版指标标记为复用；要重新对照应使用同一配方独立训练，不把历史耗时当作本轮新增成本。

每个复制文件的来源和 SHA256 见 copy-sources.json；manifest.json 覆盖整个归档的非 manifest 文件。
''')
    # Keep the exporter itself in the evidence bundle.
    copy(Path(__file__).resolve(), Path('exporter.py'), Path(__file__).resolve().parent)
    write(output / 'copy-sources.json', copies)
    write(output / 'export-validation.json', {'status': 'passed', 'counts': counts,
          'research_source_hashes_verified': True, 'four_task_macro_recomputed': True,
          'final_validation_replays_and_raw_tests_checked': True,
          'completed_target_native_tasks_samples_and_macro_checked': True,
          'historical_din_positive_counts_matched': True,
          'running_transfer_labeled_snapshot': True})
    files = {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'manifest.json', {'exported_at_utc': exported, 'files': files,
          'total_bytes': sum(v['bytes'] for v in files.values())})
    print(json.dumps({'output': str(output), 'files': len(files) + 1, 'MiB': sum(v['bytes'] for v in files.values()) / 2**20,
                      'counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--completed-transfer', type=Path, required=True)
    parser.add_argument('--running-transfer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.campaign, args.completed_transfer, args.running_transfer, args.output)
