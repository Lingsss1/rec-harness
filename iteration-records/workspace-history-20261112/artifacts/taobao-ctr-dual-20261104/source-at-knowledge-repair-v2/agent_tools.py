"""Registered public evidence access; no arbitrary host paths or private sources."""
import argparse
import json
from pathlib import Path
import time

from knowledge_engine import retrieve,rank,REVISION

ROOT=Path('/workspace')


def safe_path(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('Reference is outside the authorized public workspace')
    return path


def registered(reference):
    registry=json.loads((ROOT/'registry.json').read_text())
    if reference not in registry:raise ValueError('Unknown authorized reference: '+reference)
    entry=registry[reference]
    return {'json':entry} if isinstance(entry,str) else entry


def access(operation,**values):
    with (ROOT/'knowledge-access.jsonl').open('a') as stream:
        stream.write(json.dumps({'revision':REVISION,'operation':operation,'at_unix':time.time(),**values},ensure_ascii=False)+'\n')


def execute(operation,reference=None,query=None,source=None,top=6,find=None,start_line=1,lines=100):
    if operation=='search':
        cards=[json.loads(line) for line in (ROOT/'evidence.jsonl').read_text().splitlines() if line.strip()]
        if source:
            rows=rank(cards,query,source=source)[:max(1,min(top,6))]
            result={'revision':REVISION,'query':query,'source':source,'matches':[row['card'] for row in rows],
                    'scores':[{key:row[key] for key in ['id','knowledge_type','score','indexed_tokens']} for row in rows]}
        else:
            result=retrieve(cards,query,limit=max(1,min(top,6)))
            result['matches']=result.pop('cards')
        access(operation,query=query,source=source,matched_ids=[card['id'] for card in result['matches']])
        return result
    entry=registered(reference)
    metadata=json.loads(safe_path(entry['json']).read_text())
    if operation=='resolve':
        access(operation,reference=reference,source_id=metadata.get('id'),version=metadata.get('version'),text_sha256=metadata.get('text_sha256'))
        return metadata
    if operation!='read':raise ValueError('Unknown capability')
    if 'text' not in entry:
        if metadata.get('fallback_ref'):
            access(operation,reference=reference,status='no_detailed_text',fallback_ref=metadata['fallback_ref'])
            return {'id':metadata.get('id'),'status':'Detailed text unavailable; read the explicitly labeled fallback',
                    'fallback_ref':metadata['fallback_ref'],'access_scope':metadata.get('access_scope')}
        content=json.dumps(metadata,ensure_ascii=False,indent=2)
    else:content=safe_path(entry['text']).read_text()
    values=content.splitlines();begin=max(0,start_line-1);count=max(1,min(lines,160))
    if find:
        matches=[i for i,line in enumerate(values) if find.casefold() in line.casefold()]
        if not matches:
            access(operation,reference=reference,find=find,status='no_match')
            return {'id':metadata.get('id'),'status':'no matching text','total_lines':len(values)}
        begin=max(0,next((i for i in matches if i>=begin),matches[0])-2)
    selected=values[begin:begin+count]
    bounded='\n'.join(f'{begin+i+1}: {line}' for i,line in enumerate(selected))
    clipped=len(bounded)>24000
    access(operation,reference=reference,source_id=metadata.get('id'),version=metadata.get('version'),
           text_sha256=metadata.get('text_sha256'),start_line=begin+1,lines=len(selected),truncated=clipped,find=find)
    return {'id':metadata.get('id'),'url':metadata.get('url'),'version':metadata.get('version'),
            'access_scope':metadata.get('access_scope'),'text_sha256':metadata.get('text_sha256'),
            'total_lines':len(values),'start_line':begin+1,'next_line':begin+len(selected)+1,
            'truncated':clipped,'text':bounded[:24000]}


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='operation',required=True)
    for name in ['resolve','read']:
        command=sub.add_parser(name);command.add_argument('reference')
        if name=='read':
            command.add_argument('--find');command.add_argument('--start-line',type=int,default=1)
            command.add_argument('--lines',type=int,default=100)
    command=sub.add_parser('search');command.add_argument('query',nargs='+')
    command.add_argument('--source',choices=['paper','industry','agent']);command.add_argument('--top',type=int,default=6)
    args=vars(parser.parse_args())
    if 'query' in args:args['query']=' '.join(args['query'])
    print(json.dumps(execute(**args),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
