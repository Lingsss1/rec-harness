"""Resume exact same-run requests and trials without resetting the GPU ledger."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from amazon_family_compare.controller import FamilyController
from amazon_family_compare.core import BudgetExhausted, read, save, schema, TEXT


class RecoveryController(FamilyController):
    def run_attempt(self, directory, prompt, transport, search):
        try:
            return super().run_attempt(directory, prompt, transport, search)
        except OSError as exc:
            # The historical transport did not retry Popen failures such as a
            # missing executable. Route them through its existing retry policy.
            raise RuntimeError('LLM CLI launch failed: ' + str(exc)) from exc

    def wave(self, proposals):
        if self.method != 'mlestar':
            return super().wave(proposals)
        if not proposals:
            return []
        if len(proposals) > len(self.c['gpus']):
            raise ValueError('A wave cannot exceed four GPU slots')
        receipts, pending = {}, []
        for proposal, gpu in zip(proposals, self.c['gpus']):
            trial = self.root/'trials'/proposal['id']
            if trial.exists():
                # FamilyController verifies the exact source and requires a
                # completed receipt. Replay must precede fresh-budget checks.
                receipts[proposal['id']] = self.trial(proposal['id'], proposal['source'], gpu)
                self.event('recovery_trial_replayed', id=proposal['id'])
            else:
                pending.append((proposal, gpu))
        if pending:
            allowance = min(self.c['trial_timeout_seconds'],
                            self.ledger.available()/len(pending)-3.)
            if allowance < self.c['minimum_trial_seconds']:
                raise BudgetExhausted('Insufficient aggregate allocation for another wave')
            def train(pair):
                p, gpu = pair
                return self.trial(p['id'], p['source'], gpu, requested=allowance,
                                  files=p.get('files'))
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                for record in pool.map(train, pending):
                    receipts[record['id']] = record
        return [receipts[p['id']] for p in proposals]

    def llm(self, instruction, output_schema=None, search=False, key=None):
        response = super().llm(instruction, output_schema, search, key)
        if self.method != 'mlestar' or not key or not key.startswith('select-'):
            return response
        return self.complete_selection(instruction, response, key, search)

    def complete_selection(self, instruction, response, key, search=False):
        # Parse the *same* retraining evidence and parent source that were sent
        # before interruption. No hidden-test data or new search direction.
        context = json.loads(instruction.split('\n', 1)[1])
        sources = context['source']
        if len(sources) != 4:
            raise ValueError('Expected four fixed MLE-STAR trajectories')
        allowed = [{node.name for node in ast.parse(code).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                   | {'hyperparameters'} for code in sources]
        choices = response.get('selections', [])
        accepted = {}
        for t in range(4):
            matches = [x for x in choices if type(x.get('trajectory')) is int
                       and x['trajectory'] == t]
            if len(matches) == 1 and matches[0].get('block') in allowed[t]:
                accepted[t] = matches[0]
        for t in range(4):
            if t in accepted:
                continue
            constraint = schema({'trajectory': {'type':'integer', 'enum':[t]},
                                 'block': {'type':'string', 'enum':sorted(allowed[t])},
                                 'reason':TEXT})
            prompt = (instruction + '\nThe earlier response omitted or invalidated trajectory '
                      + str(t) + '. Complete ONLY that trajectory using the same parent and '
                      'measured ablations. Return its exact identity and one allowed top-level '
                      'Python symbol. Earlier response:\n' + json.dumps(response))
            accepted[t] = super().llm(prompt, constraint, search,
                                     key=key+'-identity-repair-t'+str(t))
            if accepted[t]['trajectory'] != t or accepted[t]['block'] not in allowed[t]:
                raise ValueError('Identity-constrained block selection violated its schema')
        repaired = {'selections':[accepted[t] for t in range(4)]}
        if repaired != response:
            save(self.root/'recovery-selection'/(key+'.json'),
                 {'original_response':response, 'repaired_response':repaired,
                  'policy':'preserve valid selections; request only missing/invalid identities'})
            self.event('recovery_block_selection_completed', key=key,
                       trajectories=[x['trajectory'] for x in repaired['selections']])
        return repaired
