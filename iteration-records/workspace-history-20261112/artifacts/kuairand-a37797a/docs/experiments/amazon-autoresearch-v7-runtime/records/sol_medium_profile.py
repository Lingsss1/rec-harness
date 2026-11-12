"""Task-local Codex profile; authentication remains outside research archives."""
from pathlib import Path
import os,re,shutil,tomllib

def prepare_profile(config):
    backend=config['agent_backend']
    assert backend['model']=='gpt-6-sol' and backend['reasoning_effort']=='medium'
    source=Path('/root/.codex');target=Path('/tmp/rec-harness-amazon-next-v7-sol-medium-profile')
    target.mkdir(mode=0o700,exist_ok=True);target.chmod(0o700)
    text=(source/'config.toml').read_text()
    first=re.search(r'^\s*\[',text,re.M);split=first.start() if first else len(text)
    header=re.sub(r'^\s*model_reasoning_effort\s*=.*$', '',text[:split],flags=re.M)
    text='model_reasoning_effort = "medium"\n'+header+text[split:]
    assert tomllib.loads(text)['model_reasoning_effort']=='medium'
    fd=os.open(target/'config.toml',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as out:out.write(text)
    shutil.copyfile(source/'auth.json',target/'auth.json');(target/'auth.json').chmod(0o600)
    return target
