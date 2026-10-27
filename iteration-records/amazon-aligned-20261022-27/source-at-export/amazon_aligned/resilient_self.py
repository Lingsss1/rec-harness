"""Durable LLM requests, native checkpoints and exact training reuse within a run."""
from contextlib import contextmanager
import contextvars
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .core import TASK, TEXT, read, save, schema, strict_schema
from .prepare import WORKSPACE, sha
from .rerun_controller import FreshSelfController


class TransportUnavailable(RuntimeError):
    """Retry the pending native step after cooldown; preserve successful work."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ResilientSelfController(FreshSelfController):
    def __init__(self, config, method):
        super().__init__(config, method)
        if read(self.ledger.path)['reservations']:
            raise RuntimeError('Unsettled GPU workers require review before process recovery')
        self.scope = contextvars.ContextVar('native_request_scope', default='baseline')
        self.call_index = max([int(p.name) for p in (self.root / 'calls').glob('*') if p.name.isdigit()] or [0])
        for path in self.root.glob('trials/*/receipt.json'):
            receipt = read(path)
            if receipt['status'] == 'success' and (self.best is None or receipt['score'] > self.best['score']):
                self.best = receipt

    @contextmanager
    def request_scope(self, value):
        token = self.scope.set(value)
        try:
            yield
        finally:
            self.scope.reset(token)

    @contextmanager
    def llm_slot(self):
        folder = Path(self.c['llm_slot_directory'])
        folder.mkdir(parents=True, exist_ok=True)
        handles = [(folder / f'slot-{i}.lock').open('a') for i in range(self.c.get('global_llm_concurrency', 4))]
        acquired = None
        try:
            while acquired is None:
                for handle in handles:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = handle
                        break
                    except BlockingIOError:
                        pass
                if acquired is None:
                    time.sleep(.25)
            yield
        finally:
            if acquired is not None:
                fcntl.flock(acquired, fcntl.LOCK_UN)
            for handle in handles:
                handle.close()

    def run_attempt(self, directory, prompt, transport, search):
        """Own the CLI process group, including children left after a timeout."""
        save(directory / 'schema.json', transport)
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '--ignore-user-config',
                   '--ignore-rules', '--model', self.c['llm_model'], '--sandbox', 'read-only',
                   '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="' + self.c['llm_effort'] + '"',
                   '-c', 'web_search="live"' if search else 'web_search="disabled"', '--json',
                   '--output-schema', str(directory / 'schema.json'), '-o', str(directory / 'answer.json'),
                   '-C', str(directory), '-']
        save(directory / 'started.json', {'command': command, 'time': time.time()})
        with (directory / 'events.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, start_new_session=True)
            save(directory / 'pid.json', {'pid': process.pid, 'process_group': process.pid})
            try:
                process.communicate(input=prompt, timeout=self.c['llm_timeout'])
                if process.returncode or not (directory / 'answer.json').exists():
                    raise RuntimeError('LLM process failed or returned no structured answer')
                return read(directory / 'answer.json')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)

    def llm(self, instruction, output_schema=None, search=False, key=None):
        from staged_research.codex import remove_optional_nulls
        import jsonschema
        original = output_schema or schema({'text': TEXT})
        identity = {'instruction': instruction, 'schema': original, 'search': search,
                    'model': self.c['llm_model'], 'effort': self.c['llm_effort'], 'scope': self.scope.get()}
        cache_key = digest(identity)
        cache = self.root / 'request-cache' / (cache_key + '.json')
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.with_suffix('.lock').open('a') as cache_lock:
            fcntl.flock(cache_lock, fcntl.LOCK_EX)
            if cache.exists():
                saved = read(cache)
                if saved['request_sha256'] != cache_key:
                    raise RuntimeError('Request cache identity mismatch')
                jsonschema.validate(saved['response'], original)
                self.event('llm_cache_hit', request_sha256=cache_key, scope=self.scope.get())
                return saved['response'] if output_schema is not None else saved['response']['text']
            with self.call_lock:
                self.call_index += 1
                directory = self.root / 'calls' / f'{self.call_index:05d}'
                directory.mkdir(parents=True, exist_ok=False)
            prompt = TASK + '\nMETHOD: ' + self.method + '\n' + instruction
            if output_schema is None:
                prompt += '\nReturn the requested completion verbatim inside the JSON text field.'
            if len(prompt) > 950000:
                raise ValueError('LLM prompt exceeds bounded protocol; reduce raw diagnostic examples')
            save(directory / 'request.json', {'prompt': prompt, 'schema': original, 'search': search,
                 'scope': self.scope.get(), 'request_sha256': cache_key})
            save(directory / 'started.json', {'time': time.time(), 'model': self.c['llm_model'],
                                              'effort': self.c['llm_effort']})
            self.event('llm_started', call=directory.name, search=search, scope=self.scope.get())
            start = time.monotonic()
            for attempt in range(1, self.c.get('llm_attempts', 3) + 1):
                destination = directory / 'attempts' / f'{attempt:02d}'
                destination.mkdir(parents=True)
                try:
                    with self.llm_slot():
                        value = self.run_attempt(destination, prompt, strict_schema(original), search)
                    response = remove_optional_nulls(value, original)
                    jsonschema.validate(response, original)
                except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
                    error = {'type': type(exc).__name__, 'error': str(exc), 'attempt': attempt, 'time': time.time()}
                    save(destination / 'failure.json', error)
                    self.event('llm_attempt_failed', call=directory.name, **error)
                    if attempt == self.c.get('llm_attempts', 3):
                        raise TransportUnavailable('LLM retries exhausted; pending request retained: ' + str(directory)) from exc
                    time.sleep(self.c.get('retry_backoff_seconds', [5, 15])[min(attempt - 1, 1)])
                    continue
                save(directory / 'response.json', response)
                save(directory / 'receipt.json', {'wall_seconds': time.monotonic() - start,
                     'model': self.c['llm_model'], 'effort': self.c['llm_effort'], 'attempts': attempt})
                save(cache, {'request_sha256': cache_key, 'response': response, 'call': str(directory)})
                self.event('llm_completed', call=directory.name, attempts=attempt)
                return response if output_schema is not None else response['text']

    def trial(self, key, source, gpu, requested=None, files=None, smoke=False):
        if smoke:
            return super().trial(key, source, gpu, requested, files, smoke)
        identity = {'source': source, 'files': files or {}, 'scope': self.scope.get()}
        actual_key = 'baseline' if key == 'baseline' else f'candidate-gpu{gpu}-' + digest(identity)[:24]
        with self.gpu_locks[gpu]:
            folder = self.root / 'trials' / actual_key
            if folder.exists():
                if not (folder / 'receipt.json').exists():
                    raise RuntimeError('Interrupted GPU worker requires accounting review: ' + str(folder))
                if (folder / 'train.py').read_text() != source:
                    raise RuntimeError('Cached training source changed')
                for name, contents in (files or {}).items():
                    if (folder / name).read_text() != contents:
                        raise RuntimeError('Cached diagnostic source changed')
                self.event('trial_cache_hit', id=actual_key, scope=self.scope.get())
                return read(folder / 'receipt.json')
            return self._trial(actual_key, source, gpu, requested, files, False)

    def feedback(self, receipt, callback):
        folder = Path(receipt['output'])
        target = folder / 'directional-feedback.json'
        if target.exists():
            self.event('feedback_cache_hit', id=receipt['id'])
            return read(target)['feedback']
        with self.request_scope('feedback/' + receipt['id']):
            return callback(self, receipt)

    def configure_engine(self, engine, trajectory):
        durable = Path(engine.workspace) / 'durable'
        durable.mkdir(parents=True, exist_ok=True)
        latest = durable / 'latest.json'
        if latest.exists():
            state = read(latest)
            if any(sha(Path(path)) != expected for path, expected in state['files'].items()):
                raise RuntimeError('Native checkpoint changed since it was committed')
            engine.database.load(state['directory'])
            if len(engine.database.programs) != state['programs']:
                raise RuntimeError('Native checkpoint did not fully restore')
            self.event('native_checkpoint_loaded', trajectory=trajectory, iteration=engine.database.last_iteration)
        original_sample = engine.database.sample
        original_add = engine.database.add
        original_checkpoint = engine._save_checkpoint
        pending = durable / 'pending-sample.json'

        def snapshot(iteration):
            destination = durable / ('state-' + str(iteration) + '-' + uuid.uuid4().hex)
            engine.database.save(str(destination), iteration)
            save(latest, {'directory': str(destination), 'programs': len(engine.database.programs),
                'iteration': iteration, 'files': {str(p): sha(p) for p in destination.rglob('*.json')}})
            self.event('native_checkpoint_saved', trajectory=trajectory, iteration=iteration)

        def sample(*args, **kwargs):
            iteration = engine.database.last_iteration + 1
            self.scope.set(f'trajectory-{trajectory}/iteration-{iteration}')
            state = read(pending) if pending.exists() else None
            if state and state['iteration'] == iteration:
                ids = [state['parent'], *state['inspirations']]
                if not all(k in engine.database.programs for k in ids):
                    raise RuntimeError('Pending native parent/inspiration missing after restore')
                return engine.database.programs[ids[0]], [engine.database.programs[k] for k in ids[1:]]
            parent, inspirations = original_sample(*args, **kwargs)
            save(pending, {'iteration': iteration, 'parent': parent.id,
                           'inspirations': [p.id for p in inspirations]})
            return parent, inspirations

        def add(program, *args, **kwargs):
            value = original_add(program, *args, **kwargs)
            if program.id == 'root':
                snapshot(0)
            return value

        def checkpoint(iteration):
            original_checkpoint(iteration)
            snapshot(iteration)

        engine.database.sample = sample
        engine.database.add = add
        engine._save_checkpoint = checkpoint
        engine.config.checkpoint_interval = 1
