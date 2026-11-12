"""Export completed fixed-transfer evidence without samples, weights or live research state."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

DATASETS = ['QK-Video', 'KuaiRand', 'TAAC-25', 'MerRec']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(root, output):
    root = root.resolve()
    if output.exists():
        raise ValueError('Use a new publication directory')
    output.mkdir(parents=True)
    copies = []

    def copy(path, relative, campaign=None):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copies.append({'path': relative, 'source_campaign': (campaign or root).name,
                       'source_path': str(path.relative_to(campaign or root)), 'sha256': digest(target)})

    state = read(root / 'supervisor-status.json')
    if state['phase'] != 'finished' or any(j['phase'] != 'complete' for j in state['jobs'].values()):
        raise ValueError('All eight fixed runs must have completed')
    setup = read(root / 'setup.json')
    for name in ['report.json', 'report.txt', 'setup.json', 'interface-checks.json', 'interface-checks.log',
                 'native-schema-checks.json', 'real-input-audit.json', 'environment.json', 'upstream-source-identity.json']:
        if (root / name).exists():
            copy(root / name, 'evidence/' + name)
    for path in sorted((root / 'source').glob('*.py')):
        copy(path, 'source/' + path.name)
    records = []
    task_rows = []
    for dataset in DATASETS:
        folder = root / 'datasets' / dataset
        protocol = read(folder / 'protocol.json')
        labels = protocol['tasks']
        if read(folder / 'acquisition-status.json')['phase'] != 'downloaded_and_verified':
            raise ValueError('Data verification missing: ' + dataset)
        for name, expected in protocol['source_sha256'].items():
            if digest(root / 'source' / name) != expected:
                raise ValueError('Frozen source drift: ' + name)
        for name in ['protocol.json', 'parameters.json', 'candidate-config.json', 'din-config.json',
                     'meta_data.json', 'dataset-source-manifest.json', 'acquisition-status.json', 'prepared/feature_map.json']:
            copy(folder / name, 'datasets/' + dataset + '/' + name)
        pair = {}
        for kind in ['candidate', 'din']:
            run = root / 'runs' / dataset / kind / 'fit-01'
            result = read(run / 'result.json')
            selected = read(run / 'selection.json')
            if result['status'] != 'success' or result['protocol_hash'] != protocol['identity']:
                raise ValueError('Run/protocol mismatch')
            source_name = 'model.py' if kind == 'candidate' else 'baseline.py'
            if result['source_sha256'] != digest(root / 'source' / source_name):
                raise ValueError('Run/source mismatch')
            if selected['checkpoint_sha256'] != result['checkpoint_sha256'] or selected['test_used_for_selection']:
                raise ValueError('Checkpoint freeze mismatch')
            for split, key in [('valid', 'validation'), ('test', 'test')]:
                metrics = result[key]
                if metrics['samples'] != protocol['sample_size'][split] or set(metrics['per_task']) != set(labels):
                    raise ValueError('Incomplete split/task coverage')
                for field, metric in [('auc', 'auc'), ('logloss', 'logloss')]:
                    average = math.fsum(metrics['per_task'][t][metric] for t in labels) / len(labels)
                    if abs(average - metrics[field]) > 1e-12:
                        raise ValueError('Macro metric differs from every-task mean')
            for name in ['result.json', 'selection.json']:
                copy(run / name, 'runs/' + dataset + '/' + kind + '/fit-01/' + name)
            calibration = root / 'runs' / dataset / kind / 'calibration-01/result.json'
            copy(calibration, 'runs/' + dataset + '/' + kind + '/calibration-01/result.json')
            pair[kind] = result
        candidate, baseline = pair['candidate'], pair['din']
        changes = {}
        for task in labels:
            c, b = candidate['test']['per_task'][task], baseline['test']['per_task'][task]
            if c['samples'] != b['samples'] or c['positives'] != b['positives']:
                raise ValueError('Paired target populations differ')
            changes[task] = {'candidate_auc': c['auc'], 'din_auc': b['auc'], 'delta_auc': c['auc'] - b['auc'],
                             'candidate_logloss': c['logloss'], 'din_logloss': b['logloss'],
                             'delta_logloss': c['logloss'] - b['logloss'], 'positives': c['positives'], 'samples': c['samples']}
            task_rows.append({'dataset': dataset, 'task': task, **changes[task]})
        row = {'dataset': dataset, 'task_count': len(labels), 'test_samples': candidate['test']['samples'],
               'candidate_auc': candidate['test']['auc'], 'din_auc': baseline['test']['auc'],
               'delta_macro_auc': candidate['test']['auc'] - baseline['test']['auc'],
               'candidate_logloss': candidate['test']['logloss'], 'din_logloss': baseline['test']['logloss'],
               'auc_improved_tasks': sum(v['delta_auc'] > 0 for v in changes.values()),
               'candidate_parameters': candidate['parameters'], 'din_parameters': baseline['parameters'],
               'candidate_train_hours': candidate['train_seconds'] / 3600, 'din_train_hours': baseline['train_seconds'] / 3600,
               'train_time_ratio': candidate['train_seconds'] / baseline['train_seconds'],
               'candidate_peak_gpu_gib': candidate['peak_gpu_bytes'] / 2**30,
               'din_peak_gpu_gib': baseline['peak_gpu_bytes'] / 2**30,
               'candidate_test_inference_seconds': candidate['test']['inference_seconds'],
               'din_test_inference_seconds': baseline['test']['inference_seconds'],
               'candidate_batch_latency_ms_p50': candidate['test']['batch_latency_ms_p50'],
               'din_batch_latency_ms_p50': baseline['test']['batch_latency_ms_p50']}
        records.append({**row, 'per_task': changes})
    for path in sorted((root / 'runs').rglob('failure.json')):
        copy(path, str(path.relative_to(root)))
    prior = Path(setup['source_campaign'])
    supplementary = read(prior / 'additional-assessments/multitask-audit-best-v1/report.json')
    if supplementary['selection_freeze']['id'] != setup['source_model']:
        raise ValueError('Taobao source model identity differs')
    for original, target in [
        ('additional-assessments/multitask-audit-best-v1/report.json', 'assessment-report.json'),
        ('best-model-multitask-audit/manifest.json', 'model-manifest.json'),
        ('best-model-multitask-audit/model-config.json', 'model-config.json'),
        ('audits/multitask-selection-v1/report.json', 'selection-audit.json'),
    ]:
        copy(prior / original, 'source-taobao/' + target, prior)
    exported_at = datetime.now(timezone.utc).isoformat()
    write(output / 'summary.json', {'exported_at_utc': exported_at, 'campaign': root.name,
          'source_model': setup['source_model'], 'selection': 'validation equal mean over all native task AUCs',
          'test_runs': 8, 'failed_runs': len(list((root / 'runs').rglob('failure.json'))),
          'seed': 20262027, 'datasets': records,
          'source_taobao': {'test_macro_auc': supplementary['test']['macro_auc'],
              'din_test_macro_auc': supplementary['din_test_reference']['macro_auc'],
              'evaluation_type': supplementary['evaluation_type']},
          'ongoing_macro_12h_taobao_run': 'not included: still in research and has no final test result'})
    for name, rows in [('summary.csv', [{k: v for k, v in x.items() if k != 'per_task'} for x in records]),
                       ('per-task.csv', task_rows)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# UniRank：固定模型跨数据集测试结果', '',
        f'导出时间：{exported_at}。实验 `{root.name}`，8/8 组完成。', '',
        '模型为淘宝验证集平均 AUC 审计选出的 `g3-r0029-context_shared_primary`。'
        '固定结构和超参数，在四个其他数据集从头训练，各与同协议 DIN 对照。'
        '这是结构迁移实验，使用各数据集自己的词表和任务头；本轮没有超参数搜索。', '',
        '## 测试集效果', '',
        '下表 AUC、Logloss 均为该数据集全部原生任务的等权平均；不同数据集的任务语义不同，不合并为一个总分。', '',
        '| 数据集 | 任务数 | 测试曝光数 | DIN AUC | 候选 AUC | ΔAUC | 提升任务数 | DIN Logloss | 候选 Logloss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['dataset']} | {row['task_count']} | {row['test_samples']:,} | {row['din_auc']:.8f} | {row['candidate_auc']:.8f} | {row['delta_macro_auc']:+.8f} | {row['auc_improved_tasks']}/{row['task_count']} | {row['din_logloss']:.8f} | {row['candidate_logloss']:.8f} |")
    lines += ['', '四个数据集的平均 AUC 和平均 Logloss 均改善；任务层面存在退步。'
              '以下列出所有 AUC 下降任务，完整逐任务指标见 [per-task.csv](per-task.csv)。', '',
              '| 数据集 / 任务 | DIN AUC | 候选 AUC | ΔAUC |', '|---|---:|---:|---:|']
    for row in task_rows:
        if row['delta_auc'] < 0:
            lines.append(f"| {row['dataset']} / {row['task']} | {row['din_auc']:.8f} | {row['candidate_auc']:.8f} | {row['delta_auc']:+.8f} |")
    lines += ['', '## 实测成本', '',
              '每组使用一张 A800 80GB。训练时间不含验证和测试，完整 wall time、校准成本与推理吞吐保存在各次 result.json。'
              '延迟为 batch8192 的模型前向 p50；计时包括预测检查及同步，不含数据加载，不等同于在线单条请求延迟。', '',
              '| 数据集 | DIN参数(M) | 候选参数(M) | DIN训练(h) | 候选训练(h) | 耗时倍数 | DIN/候选峰值显存(GiB) | DIN/候选前向批延迟(ms) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['dataset']} | {row['din_parameters']/1e6:.2f} | {row['candidate_parameters']/1e6:.2f} | {row['din_train_hours']:.3f} | {row['candidate_train_hours']:.3f} | {row['train_time_ratio']:.2f} | {row['din_peak_gpu_gib']:.2f} / {row['candidate_peak_gpu_gib']:.2f} | {row['din_batch_latency_ms_p50']:.2f} / {row['candidate_batch_latency_ms_p50']:.2f} |")
    lines += ['', '## 固定协议与字段适配', '',
        '- 各数据集固定作者发布 revision 和 train/valid/test 分块；完整测试样本覆盖已核查。',
        '- 选择依据为验证集全任务平均 AUC；同分用平均 Logloss。checkpoint 及其哈希先冻结，再进行一次完整测试。',
        '- 1 epoch、seed20262027、batch8192、最近100次历史、embedding16、AdamW1e-4 + Adagrad0.05、BF16；TF32与torch.compile关闭。',
        '- 官方离线历史反馈语义；延迟反馈可能尚未在目标曝光时成熟，结果不证明线上时点可见性。',
        '- 单 GPU，num_workers=0；验证和测试保留尾批，训练按分块丢弃不足整批的尾部。',
        '- 任务名称、数量、两个原始类别字段匹配槽参数化；保留继承的首任务梯度归属。首任务取作者元数据中的第一个任务。',
        '- QK：video_category / 缺失槽为零；KuaiRand：video_type / primary_tag；TAAC：匿名100 / 101；MerRec：c2_id / brand_id。没有把匿名字段或视频类型解释为品牌。',
        '- TAAC四个list用户字段采用官方FeatureProcessor默认的masked average pooling，两模型一致；词表使用发布元数据（104字段大小为3）。',
        '- 淘宝默认接口的初始化、非零门控下的预测和梯度完全等价；2/4/5/6任务及完整字段形状检查记录在 evidence/。', '',
        '## 来源与证据边界', '',
        f"源淘宝模型的补测平均 AUC 为 {supplementary['test']['macro_auc']:.8f}，DIN 为 {supplementary['din_test_reference']['macro_auc']:.8f}。"
        '该源模型来自旧点击目标搜索后的验证集重排，补测时此前测试结果已被观察；它不是一轮全新的平均AUC搜索或新的未见测试确认。完整凭据在 source-taobao/。', '',
        '本次四数据集均为单种子固定模型对照；没有重复种子或统计显著性证据。候选参数量和计算量更高，'
        '此处没有参数匹配消融，不能把全部收益归因于某一机制。TAAC转化、MerRec加购等反例保留。'
        '新启动的淘宝12h平均AUC研究仍在运行，尚无最终测试，本归档不将其中验证结果列作测试成绩。', '',
        '## 文件与复现', '',
        '- [summary.json](summary.json)、[summary.csv](summary.csv)：汇总；[per-task.csv](per-task.csv)：全部任务及目标计数。',
        '- runs/：8次完整训练的原始result、checkpoint选择凭据，以及独立20步成本校准；每次result包含历史长度切片。',
        '- datasets/：冻结协议、特征映射、配置、发布元数据和每个文件的SHA256。',
        '- source/：实际执行源码（含原淘宝模型），与协议记录的源码哈希一致。',
        '- evidence/、copy-sources.json、manifest.json：原始汇总、检查记录、环境、来源与导出文件完整性。',
        '- [REPRODUCE.md](REPRODUCE.md)：重建路径与同配方重训练。权重、原始样本、逐样本预测和认证文件不随报告提交。', '',
        '基准代码：[UniRank](https://github.com/salmon1802/UniRank/tree/90bb99a60724e975ec565836266fd86dcad0039d)；'
        '论文：[arXiv:2607.19987v2](https://arxiv.org/html/2607.19987v2)。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    write(output / 'copy-sources.json', copies)
    write(output / 'export-validation.json', {'status': 'passed', 'all_fixed_runs_complete': True,
          'runs': 8, 'checks': ['native task coverage and exact sample counts', 'macro AUC and Logloss recomputed',
          'paired positive-label counts identical', 'checkpoint selection hashes match results',
          'run source hashes and frozen protocol match', 'all negative task AUC deltas retained']})
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output)
