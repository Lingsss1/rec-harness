"""Generic scoped evidence indexing, length-normalized retrieval and provenance."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

REVISION='knowledge-access-v2'
STOP={'the','a','an','and','or','of','to','in','for','from','with','is','are','as','by','on','at','this','that'}


def tokens(value):
    raw=str(value)
    expanded=re.sub(r'(?<=[a-z0-9])(?=[A-Z])',' ',raw)
    values=re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]',expanded.lower())
    values += [word.lower() for word in re.findall(r'\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b',raw)]
    return [word for word in values if word not in STOP]


def group(card):
    declared=card.get('knowledge_type')
    if declared:return declared
    if card.get('placement') and card.get('scheme'):return 'industry'
    if card.get('analysis') or card.get('kind')=='provisional_interpretation':return 'agent'
    return 'paper'


def clip(value,limit):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return {'text':text[:limit],'truncated':len(text)>limit}


def memory_card(record):
    """A bounded index view; full evidence and counterevidence remain resolvable."""
    analysis=record.get('analysis',{})
    fields={}
    limits={'observations':900,'interpretations':800,'alternatives':600,'falsifiers':600,
            'limitations':800,'next_action':900,'decision':600}
    for name,limit in limits.items():
        if name in analysis:fields[name]=clip(analysis[name],limit)
    updates=[]
    for item in analysis.get('memory_updates',[])[:6]:
        if isinstance(item,dict) and item.get('statement'):
            updates.append({'statement':clip(item['statement'],400),'evidence_refs':item.get('evidence_refs',[])[:20],
                            'source_maturity':item.get('maturity','unknown')})
    full=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return {'id':record['id'],'knowledge_type':'agent','maturity':record.get('maturity','hypothesis'),
            'task_scope':record.get('task_scope'),'protocol_hash':record.get('protocol_hash'),
            'summary':fields,'proposed_updates':updates,'source_trial_refs':record.get('source_trial_refs',[]),
            'full_ref':'memory://'+record['id'],'full_sha256':hashlib.sha256(full).hexdigest(),
            'summary_notice':'Bounded index view, not complete evidence. Read full_ref for original observations, alternatives and limitations. No maturity promotion is implied.'}


def scoped_cards(industry,papers,memory,data_identity,protocol_hash):
    cards=[{**card,'knowledge_type':'industry'} for card in industry]
    cards += [{**source,'knowledge_type':'paper','scientific_status':'external_reference'} for source in papers.get('sources',[])]
    for record in memory:
        if record.get('task_scope')==data_identity and record.get('protocol_hash')==protocol_hash:cards.append(memory_card(record))
    return cards


def searchable(card):
    kind=group(card)
    if kind=='industry':
        fields=['scheme','source_problem','reported_mechanism','applicability','transfer_risks','possible_discriminating_checks','reported_evidence']
    elif kind=='agent':fields=['summary','proposed_updates']
    else:fields=['title','abstract','mechanism','status','read_scope']
    return ' '.join(card.get(key,'') if isinstance(card.get(key,''),str) else json.dumps(card.get(key,''),ensure_ascii=False) for key in fields)


def rank(cards,query,source=None):
    """BM25 over scientific text fields; score magnitude does not grow with repeats."""
    documents=[tokens(searchable(card)) for card in cards]
    frequencies=[Counter(document) for document in documents]
    average=sum(map(len,documents))/max(1,len(documents))
    document_frequency=Counter(word for counts in frequencies for word in counts)
    terms=set(tokens(query));result=[]
    for card,document,counts in zip(cards,documents,frequencies):
        if group(card)=='personal' or (source and group(card)!=source):continue
        score=0.0
        for word in terms:
            count=counts.get(word,0)
            if not count:continue
            idf=math.log(1+(len(cards)-document_frequency[word]+0.5)/(document_frequency[word]+0.5))
            denominator=count+1.2*(0.25+0.75*len(document)/max(1,average))
            score+=idf*count*2.2/denominator
        if score>0:result.append({'id':card['id'],'knowledge_type':group(card),'score':score,'indexed_tokens':len(document),'card':card})
    return sorted(result,key=lambda item:(-item['score'],item['id']))


def retrieve(cards,query,requested=None,limit=6):
    ranked=rank(cards,query);lookup={card['id']:card for card in cards}
    requested=list(dict.fromkeys(requested or []))
    if not set(requested)<=set(lookup):raise ValueError('Retrieval names unknown evidence')
    by_source={name:[row for row in ranked if row['knowledge_type']==name] for name in ['paper','industry','agent']}
    selected=[];counts=Counter()
    # Each relevant knowledge family gets an opportunity to be considered.
    for name in ['paper','industry','agent']:
        if by_source[name] and len(selected)<limit:
            row=by_source[name][0];selected.append(row['id']);counts[name]+=1
    for identifier in requested+[row['id'] for row in ranked]:
        if identifier in selected or len(selected)>=limit:continue
        kind=group(lookup[identifier])
        if kind=='personal' or counts[kind]>=2:continue
        selected.append(identifier);counts[kind]+=1
    return {'revision':REVISION,'ranking':'BM25 over bounded scientific fields; separate recall by knowledge family',
            'query':query,'requested_ids':requested,'cards':[lookup[identifier] for identifier in selected],
            'selected_ids':selected,'candidates_by_source':{kind:[{key:row[key] for key in ['id','score','indexed_tokens']} for row in rows[:4]] for kind,rows in by_source.items()},
            'industry_relevance':'matching references offered for applicability review' if by_source['industry'] else 'no positive industry match; no architecture forced',
            'adoption_policy':'Retrieval is not adoption. Accept, reject or retain uncertainty with reasons; external reports do not establish local effects.'}


def usage_record(proposal,selected,trial_rows=None):
    available={card['id']:card for card in selected}
    assessments=proposal.get('knowledge_assessments',[])
    if not isinstance(assessments,list):assessments=[]
    decisions=[];documented=set()
    variants={variant['name'] for variant in proposal.get('variants',[])}
    for assessment in assessments:
        if not isinstance(assessment,dict):continue
        identifier=assessment.get('knowledge_id')
        if identifier not in available:continue
        decision=assessment.get('decision','unresolved')
        if decision not in {'used','considered','rejected','unresolved'}:decision='unresolved'
        names=assessment.get('variant_names',[])
        chosen=[name for name in names if isinstance(name,str) and name in variants] if isinstance(names,list) else []
        decisions.append({**assessment,'knowledge_id':identifier,'decision':decision,'variant_names':chosen,
                          'source_refs':available[identifier].get('source_refs',[]),
                          'executed_refs':[{'trial_id':row['id'],'source_sha256':row['source_sha256'],'status':row['status']}
                                           for row in (trial_rows or []) if row.get('variant_name') in chosen],
                          'evidence_class':'agent-declared usage; not mechanism confirmation'})
        documented.add(identifier)
    missing=[identifier for identifier,card in available.items() if group(card)=='industry' and identifier not in documented]
    return {'revision':REVISION,'retrieved_ids':list(available),'assessments':decisions,'undocumented_industry_ids':missing,
            'status':'documented' if not missing else 'incomplete; do not infer adoption',
            'executed_variants':[{'trial_id':row['id'],'variant_name':row.get('variant_name'),'source_sha256':row['source_sha256'],'status':row['status']} for row in (trial_rows or [])],
            'limits':'A declared source-to-code link is provenance, not a causal estimate of knowledge value or a novelty score.'}


def stage_resources(workspace,registry,resources):
    """Stage explicitly authorized source/record payloads; host paths never leak."""
    workspace=Path(workspace);folder=workspace/'resources';folder.mkdir(exist_ok=True)
    manifest=[]
    for resource in resources:
        identifier=resource['id'];stem=hashlib.sha256(identifier.encode()).hexdigest()[:24]
        descriptor=dict(resource['metadata']);entry={'json':'resources/'+stem+'.json'}
        text=resource.get('text_path')
        if text:
            raw=Path(text).read_bytes();target=folder/(stem+'.txt');target.write_bytes(raw)
            entry['text']='resources/'+stem+'.txt'
            descriptor.update(text_ref='source-text://'+identifier,text_sha256=hashlib.sha256(raw).hexdigest(),text_lines=len(raw.splitlines()))
        (folder/(stem+'.json')).write_text(json.dumps(descriptor,ensure_ascii=False,indent=2)+'\n')
        for alias in [identifier,*resource.get('aliases',[])]:
            if alias in registry:raise ValueError('Duplicate authorized resource reference: '+alias)
            registry[alias]=dict(entry)
        if text:registry['source-text://'+identifier]=dict(entry)
        manifest.append({'id':identifier,'aliases':resource.get('aliases',[]),'metadata_sha256':hashlib.sha256((folder/(stem+'.json')).read_bytes()).hexdigest(),
                         'text_sha256':descriptor.get('text_sha256'),'access_scope':descriptor.get('access_scope')})
    for path in folder.iterdir():path.chmod(0o444)
    folder.chmod(0o555)
    return manifest
