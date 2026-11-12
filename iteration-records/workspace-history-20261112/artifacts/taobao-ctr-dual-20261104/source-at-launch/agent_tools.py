"""Generic local contract resolution and evidence search inside a public jail."""
import json
from pathlib import Path
import re
import sys

ROOT=Path('/workspace')


def tokens(text):
    return re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]',text.lower())


def main():
    operation=sys.argv[1]
    if operation=='resolve':
        registry=json.loads((ROOT/'registry.json').read_text())
        reference=sys.argv[2]
        if reference not in registry:raise ValueError('Unknown authorized reference')
        print((ROOT/registry[reference]).read_text())
    elif operation=='search':
        query=' '.join(sys.argv[2:])
        words=set(tokens(query))
        cards=[json.loads(line) for line in (ROOT/'evidence.jsonl').read_text().splitlines() if line.strip()]
        ranked=[]
        for card in cards:
            text=json.dumps(card,ensure_ascii=False)
            score=sum(text.lower().count(word) for word in words)
            if score:ranked.append((score,card))
        ranked.sort(key=lambda pair:pair[0],reverse=True)
        print(json.dumps({'query':query,'matches':[card for _,card in ranked[:4]],'ranking':'lexical overlap; applicability remains unverified'},ensure_ascii=False,indent=2))
    else:raise ValueError('Unknown capability')


if __name__=='__main__':main()
