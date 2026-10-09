"""Direct Codex in an external, nonprivileged filesystem jail.

Private runtime/auth material stays outside task archives. Only public workspace
inputs and schema-checked outputs are copied across the boundary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from .agents import ROLES, run_process, schema_for, validate_schema, search_policy, search_instruction, search_receipt
from .core import ContractError, digest, public_only, read_json, safe_file, write_json


class JailedCodexAgent:
    def __init__(self, template, runtime_root, *, model=None, timeout=300, auth_root='/root/.codex',prompts=None,web_search=None,role_skills=None):
        self.template = Path(template).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.model, self.timeout = model, timeout
        self.auth_root = Path(auth_root)
        from .instruction_identity import role_skills as validate_role_skills
        self.role_skills=validate_role_skills(role_skills,ROLES)
        self.prompts=dict(prompts or {})
        self.web_search=search_policy(web_search)
        if not (self.template / 'jail-manifest.json').is_file():
            raise ContractError('prepared Codex jail template required')

    def role_prompt(self,role):
        if role not in ROLES:raise ContractError('unknown research role')
        from .instruction_identity import skill_suffix
        return (f'You are the rec-harness {role} agent. {ROLES[role]}\n'+self.prompts.get(role,'')+'\n'
                  'Read /workspace/context.json and inspect /workspace/model when supplied. '
                  'This is an OS-isolated public research workspace. Do not read credentials or unrelated paths. '
                  'Do not spawn other agents or execute GPU training. Return exactly the required JSON schema. '
                  'For changes, paths are relative to model/. Preserve the frozen protocol and unrelated code. '
                  + search_instruction(self.web_search[role],role)+skill_suffix(self.role_skills.get(role,[])))

    def run(self, role, context, directory):
        if role not in ROLES or os.geteuid() != 0:
            raise ContractError('known role and privileged external launcher required')
        public_only(context)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        jail = self.runtime_root / run_id
        jail.mkdir()
        # Different identities prevent same-UID ptrace or signal interference.
        uid = 200000 + int(run_id[:7], 16)
        for name in ('bin', 'usr', 'lib', 'lib64', 'etc', 'proc', 'root/.local'):
            source = self.template / name
            if source.exists():
                shutil.copytree(source, jail / name, symlinks=True, copy_function=os.link)
        # Only explicitly declared public runtime roots; never copy a host project.
        for relative in read_json(self.template / 'jail-manifest.json').get('public_runtime_roots', []):
            source = self.template / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ContractError('invalid public runtime root')
            shutil.copytree(source, jail / relative, symlinks=True, copy_function=os.link)
        for relative in ('tmp', 'workspace', 'output', 'root/.codex'):
            path = jail / relative; path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700); os.chown(path, uid, uid)
        import stat
        (jail / 'dev').mkdir()
        for name in ('null', 'zero', 'random', 'urandom'):
            info = os.stat('/dev/' + name)
            os.mknod(jail / 'dev' / name, stat.S_IFCHR | 0o666, info.st_rdev)
            (jail / 'dev' / name).chmod(0o666)
        for name in ('auth.json', 'config.toml'):
            source = self.auth_root / name
            if source.exists():
                target = jail / 'root/.codex' / name
                shutil.copyfile(source, target); target.chmod(0o600); os.chown(target, uid, uid)
        write_json(jail / 'jail-manifest.json', {'uid': uid, 'role': role, 'run_id': run_id, 'context_hash': digest(context)})
        workspace = jail / 'workspace'
        original_files = []
        if (directory / 'model').exists():
            (workspace / 'model').mkdir()
            for path in (directory / 'model').rglob('*'):
                if path.is_file():
                    name = str(path.relative_to(directory / 'model'))
                    source = safe_file(directory / 'model', name)
                    target = workspace / 'model' / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    original_files.append(name)
            for path in [workspace / 'model', *(workspace / 'model').rglob('*')]:
                os.chown(path, uid, uid)
        from .bound_output import bound_schema
        schema = bound_schema(role, context)
        write_json(directory / 'schema.json', schema)
        write_json(directory / 'context.json', context)
        for name in ('schema.json', 'context.json'):
            shutil.copyfile(directory / name, workspace / name)
            (workspace / name).chmod(0o444)
        prompt = self.role_prompt(role)
        argv = ['/root/.local/bin/codex', 'exec', '--json', '--ephemeral', '--skip-git-repo-check',
                '--sandbox', 'danger-full-access', '--cd', '/workspace', '--color', 'never',
                '--output-schema', '/workspace/schema.json', '--output-last-message', '/workspace/response.json']
        argv += ['-c', 'web_search="' + self.web_search[role] + '"']
        if self.model: argv += ['--model', self.model]
        argv += ['-']
        # CLI's external-sandbox mode is permitted only behind verified chroot+UID drop.
        request = {'role': role, 'argv': argv, 'prompt': prompt, 'context': context,
                   'isolation': {'kind': 'chroot-unprivileged', 'uid': uid, 'run_id': run_id}}
        from .instruction_identity import manifest
        skills=self.role_skills.get(role,[])
        request['instruction_identity']=manifest(prompt,schema,skills)
        if skills:request['role_skills']=skills
        write_json(directory / 'request.json', request)
        launch = [sys.executable, '-m', 'rec_harness.isolation', 'enter', str(jail), '--uid', str(uid), '--', *argv]
        # The isolated UID has no passwd entry. Resolve Codex's home to the
        # directory staged above, independent of the controller's environment.
        environment={**os.environ,'HOME':'/root','CODEX_HOME':'/root/.codex'}
        result = run_process(launch, Path(__file__).resolve().parents[1], directory,
                             stdin=prompt, timeout=self.timeout,env=environment)
        from .generation_workspace import capture
        capture(workspace,directory)
        search_receipt(directory, self.web_search[role])
        if result['status'] != 'success':
            write_json(directory / 'resources.json', {**result, 'token_usage': None, 'isolation': request['isolation']})
            raise ContractError(f'isolated Codex failed: {result["status"]}')
        response_path = safe_file(workspace, 'response.json')
        response = read_json(response_path)
        validate_schema(response, schema); public_only(response)
        write_json(directory / 'response.json', response)
        if role in {'experiment', 'tool'}:
            for name in sorted(set(original_files) | set(response['changed_files'])):
                source = safe_file(workspace / 'model', name)
                target = directory / 'model' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        usage = None
        for line in (directory / 'stdout.jsonl').read_text().splitlines():
            try: event = json.loads(line)
            except ValueError: continue
            if event.get('type') == 'turn.completed': usage = event.get('usage')
        write_json(directory / 'resources.json', {**result, 'token_usage': usage, 'isolation': request['isolation']})
        search_receipt(directory, self.web_search[role], response)
        return response
