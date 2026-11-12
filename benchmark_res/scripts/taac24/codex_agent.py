"""Real Codex file-producing adapter for the repository's TAAC process protocol."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recevobench.taac.engine import load_candidate
from recevobench.taac.knowledge import validate_design
from recevobench.taac.retrieval import validate_plan

CODEX = Path('/home/liuzhishan/lhr/rec-harness/portable/runtime/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex')
MODEL = 'gpt-6.1-sol'
EFFORT = 'high'

CONTRACT = '''You are the single coding researcher for this isolated TAAC study.
Read request.json, program.md, candidate-api.md and candidate.py in this workspace.
Perform ONLY request.phase and create every requested output as a real local file.
Do not spawn subagents. Do not train, install dependencies, use the network, or read
other studies, parent directories, private labels, datasets, checkpoints, gate/final
results, credentials or Codex sessions. Use only the supplied research context.
The host owns the training loop, sampler, splits and scorer; only candidate.py is
editable model code. For query/analyze do not alter candidate.py. Do not modify
program.md, request.json, candidate-api.md, or the controller. No clarification is
needed: make a bounded, evidence-based research decision and produce the files.
In plan, preserve a single literal SETTINGS dict and all host-callable interfaces.
When knowledge is empty, supply the basic design fields; enhanced-only assessments
and diffs are required only when context.retrieval is present. Past outcomes may be
used but do not invent experiments, scores, references or causal conclusions.
Use the fixed controls and protocol from request.context. Keep changes feasible
under its remaining_seconds; new candidates always train from scratch.
'''

API = '''Fixed host interface (do not edit the host):
SETTINGS: exactly one literal dictionary. Preserve hidden_units, num_blocks,
num_heads, dropout_rate, norm_first, lr, negatives (positive int), objective,
temperature (positive). Optional extra facts may be added as literal values.
BaselineModel(user_num,item_num,feat_statistics,feat_types,args): nn.Module.
args contains SETTINGS plus maxlen=history_limit+1, device and mm_emb_id=['81'].
model.log2feats(seq,mask,features)->[B,L,D]; training uses every valid next event.
model.feat2emb(ids,features,include_user=False)->[B,L,D] for positive, negative,
and the FULL 660000 candidate ads. model.predict(seq,features,mask)->[B,D].
mask: 0 padding, 1 history item, 2 user token; features is a list of lists of dicts.
History item dicts include history_action and history_timestamp (observed history
only). history_action can be None for an unknown publisher label; never treat it
as a known exposure/click. Padding/user dicts may lack those keys. Handle both.
make_optimizer(model)->torch optimizer. compute_loss(positive_scores [N],
negative_scores [N,SETTINGS['negatives']])->finite scalar differentiable loss.
The fixed host samples negatives excluding the user's full history. Candidates
can change their count and the loss, but cannot replace host sampling or training.
All ranking is exact inner product, all 660000 candidates, top 10 OIDs.
The frozen epochs/batch_size/seeds are in request.context.protocol. Training,
diagnostics, all agent phases, failures and scoring consume the study wall budget.
'''


def validate(request, original):
    phase, context = request['phase'], request['context']
    if phase == 'plan':
        design = json.loads(Path('design.json').read_text())
        validate_design(design, context, bool(context.get('retrieval')))
        load_candidate(Path('candidate.py'))
        current = hashlib.sha256(Path('candidate.py').read_bytes()).hexdigest()
        if any(t['source_sha256'] == current for t in context['history']) and design.get('kind') != 'repeat':
            raise ValueError('Candidate source already evaluated; edit the model or explicitly declare kind=repeat')
    elif phase == 'query':
        if Path('candidate.py').read_bytes() != original:
            raise ValueError('Query must not change candidate.py; restore it')
        validate_plan(json.loads(Path('retrieval-plan.json').read_text()), context)
    else:
        if Path('candidate.py').read_bytes() != original:
            raise ValueError('Analyze must not change candidate.py; restore it')
        analysis = json.loads(Path('analysis.json').read_text())
        for key in ['effect', 'alternatives', 'next_action', 'evidence_trials']:
            if key not in analysis:
                raise ValueError('Analysis missing ' + key)
        ids = {t['id'] for t in context['history']}
        refs = set(analysis['evidence_trials'])
        if not refs <= ids or context['history'][-1]['id'] not in refs:
            raise ValueError('Analysis must cite real IDs including latest trial')


def main():
    request_path = Path(sys.argv[1]).resolve()
    os.chdir(request_path.parent)
    request = json.loads(request_path.read_text())
    original = Path('candidate.py').read_bytes()
    Path('candidate-api.md').write_text(API)
    usage = []
    prompt = CONTRACT
    start = time.monotonic()
    for attempt in range(3):
        argv = [str(CODEX), 'exec', '--skip-git-repo-check', '--model', MODEL,
                '-c', 'model_reasoning_effort="high"', '-c', 'approval_policy="never"',
                '-c', 'web_search="disabled"', '-c', 'features.multi_agent=false',
                '--sandbox', 'workspace-write', '--json',
                '--output-last-message', f'codex-{attempt}.last.txt', '-']
        with Path(f'codex-{attempt}.jsonl').open('w') as stream:
            result = subprocess.run(argv, input=prompt, text=True, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=max(1, 1150-(time.monotonic()-start)))
        for line in Path(f'codex-{attempt}.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
                if event.get('type') == 'turn.completed':
                    usage.append(event.get('usage', {}))
            except json.JSONDecodeError:
                pass
        Path('model-usage.json').write_text(json.dumps({'model': MODEL, 'effort': EFFORT, 'calls': usage}, indent=2))
        try:
            if result.returncode:
                raise ValueError(f'Codex exited {result.returncode}; inspect codex-{attempt}.jsonl')
            validate(request, original)
            print(json.dumps({'status':'success','phase':request['phase'],'model':MODEL,'effort':EFFORT,'calls':len(usage)}))
            return
        except Exception as error:
            Path(f'validation-{attempt}.json').write_text(json.dumps({'error':str(error)}))
            if attempt == 2:
                raise
            prompt = CONTRACT + '\nYour previous output failed host validation:\n' + str(error) + '\nRead the current files and repair this error. Preserve legitimate prior research work.\n'


if __name__ == '__main__':
    main()
