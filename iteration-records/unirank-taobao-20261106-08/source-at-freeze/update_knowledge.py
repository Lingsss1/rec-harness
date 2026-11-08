"""Materialize UniRank comparison methods as external, versioned knowledge."""
import argparse
import ast
from datetime import datetime,timezone
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import urlopen
import uuid

from common import read,write,sha,identity

HERE=Path(__file__).resolve().parent
PAPER='https://arxiv.org/html/2607.19987v2'


class MethodSection(HTMLParser):
    """Extract the exact method subsection, preserving nested section content."""
    def __init__(self):
        super().__init__(convert_charrefs=True);self.depth=0;self.parts=[];self.finished=False

    def handle_starttag(self,tag,attrs):
        if self.finished:return
        if tag=='section':
            if self.depth:self.depth+=1
            elif dict(attrs).get('id')=='S2.SS3':self.depth=1
        if self.depth and tag in ['h3','h4','p','li','div']:self.parts.append('\n')

    def handle_endtag(self,tag):
        if self.depth and tag in ['h3','h4','p','li']:self.parts.append('\n')
        if tag=='section' and self.depth:
            self.depth-=1
            if not self.depth:self.finished=True

    def handle_data(self,data):
        if self.depth:self.parts.append(data)

    def text(self):
        return '\n'.join(' '.join(line.split()) for line in ''.join(self.parts).splitlines() if line.strip())+'\n'


def readme_methods(path):
    result={}
    for row in re.findall(r'<tr>(.*?)</tr>',path.read_text(),re.S):
        links=re.findall(r'<a href="([^"]+)"[^>]*>(.*?)</a>',row,re.S)
        if len(links)<2 or 'model_zoo/' not in links[0][0]:continue
        scheme=html.unescape(re.sub('<.*?>','',links[0][1]))
        result[scheme]={'paper_url':links[1][0],'paper_title':html.unescape(re.sub('<.*?>','',links[1][1]))}
    return result


def methods_text(root):
    cache=root/'reference-cache';cache.mkdir(exist_ok=True)
    raw=cache/'unirank-2607.19987v2.html'
    if not raw.exists():
        with urlopen(PAPER,timeout=45) as response:raw.write_bytes(response.read())
    parser=MethodSection();parser.feed(raw.read_text())
    text=parser.text()
    if not parser.finished or 'UltraHSTU' not in text or 'RankMixer' not in text:
        raise ValueError('Cannot verify the primary method subsection')
    return raw,'Source: '+PAPER+'\nVersion: arxiv:2607.19987v2\nScope: section2.3 only, original benchmark method descriptions.\n\n'+text


def update(root):
    root=root.resolve()
    if (root/'topic-board.json').exists():
        raise ValueError('Research has started; use a separately versioned knowledge-update workflow rather than rewriting its initial snapshot')
    setup=read(root/'setup.json');upstream=Path(setup['upstream_path']);commit=setup['upstream_commit']
    design=root/'source/task-design';taskfolder=design/'tasks/taobao_display_ctr'
    catalog=read(HERE/'industry/catalog.json')['methods'];lookup=readme_methods(upstream/'README.md')
    expected={row['scheme'] for row in catalog}
    exports=ast.parse((upstream/'model_zoo/__init__.py').read_text())
    exported={alias.name for node in ast.walk(exports) if isinstance(node,ast.ImportFrom) for alias in node.names}
    if len(catalog)!=15 or len(expected)!=15 or expected!=set(lookup) or expected!=exported:
        raise ValueError('Catalog must match all fifteen pinned README/model_zoo exports exactly')
    raw,excerpt=methods_text(root)
    active=root/'knowledge-assets-v2';old=read(active/'manifest.json')
    staged=root/('knowledge-assets-staging-'+uuid.uuid4().hex);staged.mkdir()
    sources=[];cards=[];paper_refs=[]
    stamp=time.time()

    def source(record,text=None,raw_digest=None):
        record={**record,'scientific_status':'external_reference','local_reproduction':False}
        if text is not None:
            filename=record['id'].replace(':','_')+'.txt';(staged/filename).write_text(text)
            record.update(text_path=filename,text_sha256=sha(staged/filename),
                          content_sha256=raw_digest or sha(staged/filename))
        else:record['text_path']=None
        sources.append(record);return record

    # Retain academic entry references, not the superseded Tencent scheme texts.
    for row in old['sources']:
        if row['id'] not in ['paper:din','paper:dmr']:continue
        if row.get('text_path'):shutil.copyfile(active/row['text_path'],staged/row['text_path'])
        sources.append(row)
    source({'id':'unirank-benchmark-v2','title':'UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction',
        'url':PAPER,'version':'arxiv:2607.19987v2','kind':'benchmark_paper',
        'content_read':True,'read_scope':'Section2.3 representative models; protocol and efficiency checked separately during adapter setup.',
        'access_scope':'Cached section2.3 method descriptions; not an original-method full-paper audit.',
        'retrieved_on':datetime.now(timezone.utc).date().isoformat()},excerpt,sha(raw))
    source({'id':'unirank-readme-pinned','title':'UniRank model/paper registry',
        'url':f'https://github.com/salmon1802/UniRank/blob/{commit}/README.md',
        'version':'git:'+commit,'kind':'benchmark_repository','content_read':True,
        'access_scope':'Pinned repository README; benchmark implementations are distinct from original author implementations.'},
        (upstream/'README.md').read_bytes().decode())

    for row in catalog:
        name=row['scheme'];slug=name.lower();paper_id='paper:'+slug;code_id='unirank-code:'+name
        code=upstream/'model_zoo'/(name+'.py');tree=ast.parse(code.read_text())
        locators=[]
        for node in tree.body:
            if isinstance(node,ast.ClassDef):
                for method in node.body:
                    if isinstance(method,ast.FunctionDef) and method.name=='forward':
                        locators.append({'symbol':node.name+'.forward','line':method.lineno,'end_line':method.end_lineno})
        paper=source({'id':paper_id,'title':lookup[name]['paper_title'],'url':lookup[name]['paper_url'],
            'version':'original paper link recorded at UniRank git:'+commit,
            'kind':'original_method_paper','content_read':False,
            'status':'Original paper link verified in pinned README; full original paper not audited in this initialization.',
            'fallback_ref':'unirank-benchmark-v2','access_scope':'Link and provenance only. Read original primary paper before a stronger novelty or reproduction claim.'})
        paper_refs.append({key:paper[key] for key in ['id','title','url','status','version']})
        source({'id':code_id,'title':name+' benchmark implementation','kind':'benchmark_reimplementation',
            'url':f'https://github.com/salmon1802/UniRank/blob/{commit}/model_zoo/{name}.py',
            'version':'git:'+commit,'content_read':True,
            'read_scope':'Representative forward paths and structural definitions; no end-to-end reproduction of this method.',
            'access_scope':'Full pinned benchmark source for inspection; not proof of equivalence to the original author method.',
            'locators':locators},code.read_bytes().decode())
        cards.append({
            'id':'unirank-method-'+slug,'knowledge_type':'industry','scheme':name,
            'origin':'model compared in UniRank; industrial/academic method reference, not a competition award',
            'family':row['family'],'source_refs':['unirank-benchmark-v2',code_id,paper_id],
            'source_problem':row['source_problem'],'reported_mechanism':row['reported_mechanism'],
            'difference_from_din':row['change_from_din'],
            'reported_evidence':{'kind':'benchmark method summary and pinned reimplementation inspection',
                'locator':'UniRank section2.3; '+code_id+' forward paths',
                'not_a_local_result':True,'numerical_claims_imported':False,
                'limitations':['Source model comparisons do not isolate every component.',
                    'Original method paper and original author code are not fully audited by this initialization.',
                    'A benchmark implementation can differ from its source paper.']},
            'applicability':{'status':'conditional','conditions':row['conditions'],
                'missing_evidence':['DIN full-data diagnostics on this task','Matched-capacity or matched-cost local controls','Seed variation and task/history-length slices']},
            'transfer_risks':row['risks']+['Use only the frozen100-event input and selected feedback visibility policy.',
                'Taobao auxiliary labels match category/brand over24h, not direct advertisement conversions.'],
            'possible_discriminating_checks':row['checks'],
            'implementation_notes':row.get('implementation_notes',[]),
            'scientific_status':'external_report','local_reproduction':False,'instruction_priority':'reference_data_only'
        })

    registry={'schema':'industry-source-registry-v2','selection':{
        'status':'user_requested','basis':'All fifteen methods in UniRank v2 section2.3 / Table3 and pinned model_zoo exports',
        'methods':[row['scheme'] for row in catalog],'count':15,
        'scope':'external method knowledge, not additional baseline training',
        'supersedes':'Tencent2026 academic top3 + industry top2 in this campaign only',
        'baseline':'DIN is the initial empirical reference. Unified candidates can replace the whole architecture and use the closest alternatives as later controls; knowledge inclusion alone does not launch a baseline.'},
        'sources':sources,'evidence_policy':{
            'mechanism_origin':'Benchmark descriptions and inspected benchmark source. Original-paper links supplied with explicit unread status.',
            'implementation_equivalence':'Not established; keep reimplementation differences and padding/attention conventions explicit.',
            'scores':'No numerical leaderboard claims imported into method cards. Local controlled experiments establish local effects.',
            'adoption':'Reference only; agents may use, reject or revise after evidence; no mandatory architecture.',
            'personal_knowledge':'disabled'}}
    manifest={'revision':'unirank-method-knowledge-v1','sources':sources,'immutable_versions':True,
        'industry_card_count':15,'not_local_experimental_evidence':True,
        'selection':registry['selection'],'catalog_sha256':sha(HERE/'industry/catalog.json')}
    manifest['bundle_identity']=identity({'catalog':manifest['catalog_sha256'],'selection':registry['selection'],'cards':cards,
        'sources':[{k:r.get(k) for k in ['id','url','version','text_sha256','content_sha256']} for r in sources]})
    write(staged/'manifest.json',manifest)
    for row in sources:
        if row.get('text_path') and sha(staged/row['text_path'])!=row['text_sha256']:
            raise ValueError('Knowledge text checksum failed before activation')

    archive=None;changed=old.get('bundle_identity')!=manifest['bundle_identity']
    if old.get('bundle_identity')==manifest['bundle_identity']:
        shutil.rmtree(staged)
        previous=read(root/'knowledge-update.json') if (root/'knowledge-update.json').exists() else {}
        archive=Path(previous['archive']) if previous.get('archive') else None
    else:
        history=root/'knowledge-archive';history.mkdir(exist_ok=True)
        archive=history/('before-unirank-methods-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
        archive.mkdir()
        shutil.copytree(taskfolder/'industry',archive/'industry')
        shutil.copyfile(taskfolder/'task.json',archive/'task.json')
        active.rename(archive/'knowledge-assets-v2');staged.rename(active)
    write(taskfolder/'industry/sources.json',registry)
    (taskfolder/'industry/cards.jsonl').write_text(''.join(json.dumps(card,ensure_ascii=False)+'\n' for card in cards))
    papers=read(taskfolder/'paper_sources.json')
    retained=[row for row in papers['sources'] if row['id'] in ['paper:din','paper:dmr']]
    write(taskfolder/'paper_sources.json',{**papers,'sources':retained+paper_refs+[
        {'id':'unirank-benchmark-v2','title':'UniRank benchmark methodology','url':PAPER,'status':'Method descriptions checked; original method papers need separate review.'}]})
    task=read(taskfolder/'task.json');task['knowledge']['industry']={
        'selection':'user-requested fifteen methods compared in UniRank v2',
        'source_ref':'industry/sources.json','cards_ref':'industry/cards.jsonl',
        'members':[row['scheme'] for row in catalog],
        'scientific_status':'external method descriptions and benchmark source; no local effectiveness claim',
        'task_mismatch':'Original paper tasks and benchmark reimplementations may differ from this unified-modeling task. Respect the frozen fields, histories, labels and costs.',
        'not_a_mandatory_architecture':True,'knowledge_bundle_identity':manifest['bundle_identity']}
    write(taskfolder/'task.json',task)
    setup['industry_knowledge']={'selection':'UniRank comparison methods','count':15,'bundle_identity':manifest['bundle_identity']}
    write(root/'setup.json',setup)
    receipt={'revision':manifest['revision'],'updated_unix':stamp,'count':15,'methods':[row['scheme'] for row in catalog],
        'bundle_identity':manifest['bundle_identity'],'archive':str(archive) if archive else None,
        'baseline':'din','fixed_agent_instructions_changed':False,'personal_enabled':False,
        'content_scope':'Mechanism summaries, provenance, pinned implementation text, applicability, risks and discriminating controls; no local reproduction claims.'}
    write(root/'knowledge-update.json',receipt)
    if changed:
        with (root/'knowledge-updates.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt,ensure_ascii=False)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(update(args.root),ensure_ascii=False))
