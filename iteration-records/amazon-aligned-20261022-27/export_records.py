#!/usr/bin/env python3
"""Publish an allowlisted snapshot of the four-category Amazon iteration records.

No training, model calls, checkpoint modifications, or source changes are performed.
Raw datasets, model weights, per-user simulator inputs and raw CLI events stay local.
"""
from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COPIES = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def copy(source, target):
    if not source.is_file():
        return
    data = source.read_bytes()
    text = data.decode('utf-8')
    for pattern in [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                    r'\bsk-[A-Za-z0-9_-]{24,}\b', r'\bgh[pousr]_[A-Za-z0-9]{24,}\b',
                    r'\bAKIA[0-9A-Z]{16}\b', r'(?i)Authorization:\s*(?:Bearer|Basic)\s+[A-Za-z0-9_./+=-]{20,}']:
        if re.search(pattern, text):
            raise RuntimeError('Potential credential in export source: ' + str(source))
    target = OUT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    COPIES[str(target.relative_to(OUT))] = {
        'source': str(source.relative_to(WORK)), 'sha256': digest(data), 'bytes': len(data)}


def selected(source, target, patterns):
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if '__pycache__' not in path.parts:
                copy(path, Path(target) / path.relative_to(source))


def csv_file(name, rows, fields):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    roots = {name: WORK / 'artifacts' / name for name in [
        'amazon-aligned-20261022', 'amazon-repeats-20261025', 'amazon-single-model-20261026',
        'amazon-reruns-20261026', 'self-evorec-resilient-20261027']}
    jobs = []
    base = roots['amazon-aligned-20261022']
    for method in ['mlevolve', 'mlestar', 'self_evolverec', 'recharness']:
        jobs.append((f'original-{method}', method, base / method,
                     base / method / 'config.json' if method == 'recharness' else base / 'config.json', 'original'))
    replica = base / 'recharness-replica-gpu0-3/recharness'
    jobs.append(('original-recharness-replica', 'recharness', replica, replica / 'config.json', 'original'))
    for group in ['amazon-repeats-20261025', 'amazon-reruns-20261026']:
        plan = read(roots[group] / 'plan.json')
        for queue in plan['queues']:
            for job in queue['jobs']:
                jobs.append((group.replace('amazon-', '') + '-' + Path(job['root']).name,
                             job['method'], Path(job['root']) / job['method'], Path(job['config']), group))
    latest = roots['self-evorec-resilient-20261027']
    for job in read(latest / 'plan.json')['jobs']:
        jobs.append((job['id'], 'self_evolverec', Path(job['root']) / 'self_evolverec',
                     Path(job['config']), 'resilient-20261027'))
    # Public protocol and source fingerprints, never the input data themselves.
    for label, root in roots.items():
        selected(root, 'protocol/' + label, [
            '*.json', 'queues/*/status.json', 'preflight/acceptance.json', 'preflight/regression.json',
            'preflight/tests.json', 'preflight/llm.json', 'preflight/native*.json',
            'preflight/constraint-gate.json', 'preflight/researcher.json',
            'preflight/tests.log', 'protocol-before/*.py', 'protocol-after/*.py'])
    selected(base, 'protocol/data', ['data/prepared.json', 'data/*.prepared.json',
             'data/trainval/dataset_stats.json', 'data/trainval/*_stats.json'])
    selected(WORK / 'artifacts/amazon-reviews-2023', 'protocol/raw-download', ['verification.json'])
    selected(WORK / 'amazon_aligned', 'source-at-export/amazon_aligned', ['*.py'])
    for package in ['mlevolve_codex', 'mle_star_codex']:
        selected(WORK / package, 'source-at-export/' + package, ['*.py', '*.md'])
    for package in ['RecHarness-6lyc', 'MLEvolve', 'Self-EvolveRec-code']:
        selected(WORK / 'artifacts' / package, 'protocol/licenses/' + package,
                 ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE'])
    iterations, comparisons, calls = [], [], []
    patterns = ['*.json', 'events.jsonl', 'resource-observations.jsonl',
        'native/**/*.json', 'native/**/*.py', 'rounds/**/*.json', 'recovery/*.json',
        'trials/*/train.py', 'trials/*/models.py', 'trials/*/diagnosis_tools.py',
        'trials/*/working/predict.py', 'trials/*/receipt.json', 'trials/*/result.json',
        'trials/*/request.json', 'trials/*/resources.json', 'trials/*/single-model-audit.json',
        'trials/*/feedback-error.json', 'trials/*/training.log', 'trials/*/worker.log',
        'calls/*/response.json', 'calls/*/receipt.json', 'calls/*/schema.json',
        'calls/*/attempts/*/failure.json', 'final-test/*.json', 'final-test/worker.log']
    for ident, method, root, config_path, group in jobs:
        target = Path('searches') / ident
        copy(config_path, target / 'config.json')
        selected(root, target, patterns)
        receipts = []
        for path in sorted(root.glob('trials/*/receipt.json')):
            record = read(path)
            receipts.append(record)
            iterations.append({'run': ident, 'method': method, 'trial': record['id'],
                'status': record['status'], 'validation_hr10': record.get('score'),
                'gpu_seconds': record.get('allocated_gpu_seconds'), 'gpu': record.get('gpu'),
                'record': str(target / path.relative_to(root)),
                'source': str(target / 'trials' / path.parent.name / 'train.py') if (path.parent / 'train.py').exists() else ''})
        for path in sorted(root.glob('calls/*/request.json')):
            request = read(path)
            prompt = request['prompt']
            response = path.parent / 'response.json'
            receipt = path.parent / 'receipt.json'
            # Preserve final public outputs, not raw CLI event streams or prompts
            # containing individual training users' histories and recommendations.
            calls.append({'run': ident, 'call': path.parent.name, 'completed': response.exists(),
                'prompt_sha256': digest(prompt.encode()), 'prompt_characters': len(prompt),
                'search': request.get('search', False),
                'role': re.search(r'NATIVE ROLE: ([^\n]+)', prompt).group(1) if 'NATIVE ROLE: ' in prompt else '',
                'wall_seconds': read(receipt).get('wall_seconds') if receipt.exists() else None,
                'response': str(target / response.relative_to(root)) if response.exists() else ''})
        budget = read(root / 'budget.json') if (root / 'budget.json').exists() else {}
        summary = read(root / 'summary.json') if (root / 'summary.json').exists() else {}
        incumbent = read(root / 'incumbent.json') if (root / 'incumbent.json').exists() else {}
        status_path = root / 'status.json'
        state = read(status_path).get('status') if status_path.exists() else None
        failure = read(root / 'failure.json') if (root / 'failure.json').exists() else {}
        if summary:
            state = 'complete'
        elif failure:
            state = 'interrupted'
        elif not receipts:
            state = state or 'prepared_or_preflight_only'
        counts = Counter(r['status'] for r in receipts)
        comparisons.append({'run': ident, 'group': group, 'method': method, 'status': state,
            'validation_hr10': summary.get('selected', incumbent).get('score'),
            'test_hr10': summary.get('test', {}).get('score'),
            'test_ndcg10': summary.get('test', {}).get('aggregate', {}).get('NDCG@10'),
            'spent_gpu_seconds': budget.get('spent_gpu_seconds'),
            'budget_gpu_seconds': budget.get('total_gpu_seconds', 43200),
            'successful_trials': counts.get('success', 0), 'trial_count': len(receipts),
            'single_model_search': read(config_path).get('single_model_search', False),
            'failure_type': 'LLM timeout' if 'TimeoutExpired' in failure.get('error', '') else
                            'native trajectory/feedback failure' if failure else '',
            'record': str(target)})
    single = roots['amazon-single-model-20261026']
    selected(single, 'single-model-reselection', ['*.json', 'audit-source-v1.py', '*/summary.json',
             '*/selection.json', '*/audits/*.json', '*/final-test/*.json'])
    single_rows = []
    for path in sorted(single.glob('*/summary.json')):
        record = read(path)
        selection = record['selection']
        single_rows.append({'run': path.parent.name, 'method': record['job']['method'],
            'original_search_status': selection['original_search_status'],
            'selected': selection['selected']['id'], 'validation_hr10': selection['selected']['score'],
            'test_hr10': record['test']['score'], 'test_ndcg10': record['test']['aggregate']['NDCG@10'],
            'record': 'single-model-reselection/' + str(path.relative_to(single))})
    save('comparisons.json', comparisons)
    save('single-model-results.json', single_rows)
    save('iterations.json', iterations)
    csv_file('comparisons.csv', comparisons, list(comparisons[0]))
    csv_file('iterations.csv', iterations, list(iterations[0]))
    csv_file('calls.csv', calls, list(calls[0]))
    save('copy-sources.json', COPIES)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save('snapshot.json', {'exported_at_utc': now, 'runs': len(comparisons), 'trials': len(iterations),
        'calls': len(calls), 'completed_searches': sum(r['status'] == 'complete' for r in comparisons),
        'interrupted_searches': sum(r['status'] == 'interrupted' for r in comparisons),
        'unstarted_runs': sum(r['status'] == 'prepared_or_preflight_only' for r in comparisons),
        'scope': 'snapshot of existing evidence only; no training or recovery performed during publication'})
    rows = ['| 实验 | 状态 | 验证 HR@10 | 测试 HR@10 | 已用 GPU 秒 |', '| --- | --- | ---: | ---: | ---: |']
    for row in comparisons:
        fmt = lambda v: '—' if v is None else f'{v:.4f}'
        rows.append(f"| [{row['run']}]({row['record']}) | {row['status']} | {fmt(row['validation_hr10'])} | {fmt(row['test_hr10'])} | {row['spent_gpu_seconds'] or 0:.1f} |")
    single_table = ['| 重评记录 | 原搜索状态 | 测试 HR@10 |', '| --- | --- | ---: |']
    for row in single_rows:
        single_table.append(f"| [{row['run']}]({row['record']}) | {row['original_search_status']} | {row['test_hr10']:.4f} |")
    text = '''# Amazon 四类别同预算迭代记录（2026-10-22 至 27）

本记录包归档 MLEvolve、MLE-STAR、Self-EvolveRec 和 RecHarness 的新增实验。它是现有运行证据的发布快照，不会启动、恢复或修改实验。完整状态以 [实验索引](comparisons.csv)、[逐候选索引](iterations.csv) 和 [调用索引](calls.csv) 为准。

## 协议

Amazon Reviews 2023，四类别为 Movies_and_TV、Industrial_and_Scientific、Electronics、CDs_and_Vinyl。预处理种子 0，训练和评测种子 42，SASRec per-dataset 冷启动。每次独立搜索预算为累计 43,200 GPU 秒，4 个单卡并行槽、64 CPU 核；训练、验证、诊断、失败和超时 worker 均计入，LLM 时间与最终测试推理开销分别记录。

LLM 为 `gpt-6.1-sol / medium`（用户所说 middle 对应配置值 medium）。公共宿主验证为每类别 200 用户、99 个负例，宏平均 HR@10；最终测试每类别最多 10,000 用户，Movies_and_TV 实际 7,467 用户。先冻结验证选择，再测试。不同重复保持相同数据与训练种子，使用独立 LLM 生成；这些重复不是多个训练随机种子的独立验证。

## 搜索运行

原始搜索允许集成，不能将其测试成绩直接当作单模型成绩。`reruns-20261026-mlevolve-single-*` 在搜索时关闭原生 fusion/aggregation，并在接受候选前检查预测器。Self-EvolveRec 保留原生搜索，最终按验证集过滤选择合规单模型。

''' + '\n'.join(rows) + '''

`complete` 表示有正式最终汇总；`interrupted` 表示异常退出，已有最佳验证成绩不等于完成一次搜索；`prepared_or_preflight_only` 表示只有配置或短预检，不能计作正式运行。最新两次 resilient 重跑在本快照时尚未正式启动。

## 原预算内的单模型重评

这一组是对已有搜索的事后资格筛选：保留全部原搜索消耗，包括集成候选，按宿主验证 HR@10 选最佳合规单模型，再重评或复用完全相同预测器的已有测试。它不是从头禁止集成的新搜索。每类别一个独立模型允许；跨模型、跨种子、检查点分数/排名融合，以及与独立拟合的转移/流行度分数相加均排除；一个网络内联合训练的特征和层允许。

''' + '\n'.join(single_table) + '''

Self-EvolveRec 旧重复中断后得到的单模型测试成绩仍保留 interrupted 标记，不应算作完整成功重复。仅凭这些少量重复不作统计显著性结论。

## 中断和修复

早期 Self-EvolveRec 的演化诊断文件遗漏 `diagnosis_interpreter_prompt`，另有诊断原始明细使提示词超过 1,048,576 字符。2026-10-26 修复了提示函数保护、诊断明细压缩和历史长度限制；其后两次重跑的退出原因为 LLM 1,200 秒调用超时。MLEvolve 去集成第 1 次也被 LLM 超时打断，另外两次正常完成。

2026-10-27 新增的 resilient 协议包含有界调用重试、成功请求/反馈缓存、原生种群和迭代检查点、待处理父节点/灵感持久化、进程组清理，以及两次运行共享的 4 路 LLM 并发上限。配置和已完成预检已归档；未启动的正式实验没有最终成绩。发布动作没有恢复中断搜索或追加计算预算。

## 源码和发布边界

[导出时适配器](source-at-export/amazon_aligned) 是上传时快照，不代表每次历史运行都用了相同源码。各历史冻结清单和 protocol-before/protocol-after 快照保留在 [协议目录](protocol)。原生方法固定提交、具体任务适配差异、预算账本和外部评测器哈希见各运行 config、frozen、budget、receipt 及汇总。

保留候选源代码、训练/评测日志、宿主指标、预算、选择、最终公开回答和结构化方案；原始 LLM 提示词仅保存哈希/长度索引。原始 CLI 事件、逐用户模拟输入/原始诊断明细、数据集、权重、二进制索引和凭据留在本地。JSON 中绝对路径只作溯源，不应在其他机器直接执行。模型权重未上传，因此记录包可审阅和核验，但不能独立重放已有预测。

每个逐字复制文件的来源和原始哈希见 [copy-sources.json](copy-sources.json)，整个发布包的文件校验见 [manifest.json](manifest.json)。离线核验：

```bash
python3 iteration-records/amazon-aligned-20261022-27/verify_records.py
```
'''
    (OUT / 'README.md').write_text(text)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p != OUT / 'manifest.json' and '__pycache__' not in p.parts]
    entries = {str(p.relative_to(OUT)): {'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size} for p in sorted(paths)}
    save('manifest.json', {'files': entries, 'total_files': len(entries), 'total_bytes': sum(v['bytes'] for v in entries.values())})
    print(json.dumps({'runs': len(jobs), 'trials': len(iterations), 'calls': len(calls),
        'files': len(entries), 'megabytes': round(sum(v['bytes'] for v in entries.values()) / 1e6, 2)}))


if __name__ == '__main__':
    main()
