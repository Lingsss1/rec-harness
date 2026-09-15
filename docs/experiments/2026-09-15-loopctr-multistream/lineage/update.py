"""Generate experiment ancestry from append-only memory and explicit active plans."""
from pathlib import Path
import json,hashlib,fcntl,html
R=Path(__file__).resolve().parents[1];OUT=R/'lineage'
def main():
 lock=(OUT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 nodes={};edges=[];sources={}
 for dataset,mem in [('KuaiRec',R/'memory'),('Taobao',R/'taobao-transfer/memory')]:
  p=mem/'episodes.jsonl';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for line in p.read_text().splitlines():
   e=json.loads(line);eid=e['experiment_id'];key=dataset+':'+eid;primary=e.get('metrics',{}).get('primary',{});primary=primary if isinstance(primary,dict) else {}
   nodes[key]=dict(id=key,experiment_id=eid,dataset=dataset,family=e.get('family','unspecified'),status=e.get('decision','recorded'),auc=primary.get('value'),detail=e.get('hypothesis',''),source=str(p))
   parent=e.get('parent_experiment_id')
   if parent:edges.append(dict(source=dataset+':'+parent,target=key,kind='recorded_parent'))
 # Latest original Cross8 multistream episodes recorded the evaluation parent.
 for edge in edges:
  if edge['target'].startswith('Taobao:released-multistream--'):
   edge['kind']='comparison'
 for mode in ['static','dynamic_read','dynamic_write','dynamic_bypass','dynamic_all','collapsed_all']:
  target='Taobao:released-multistream--'+mode+'--s20262027'
  if target in nodes:edges.append(dict(source='Taobao:released-multistream--no_feedback--s20262027',target=target,kind='architecture'))
 for p in sorted((R/'taobao-transfer').glob('*/lineage-plan.json')):
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();plan=json.loads(p.read_text())
  for e in plan['experiments']:
   key='Taobao:'+e['experiment_id'];d=p.parent/'reports'/e['name'];state='planned'
   for phase in ('smoke','formal'):
    if (d/phase/'claim.json').exists():state=phase+'_running'
    if (d/phase/'exit.json').exists():state=phase+'_completed' if json.loads((d/phase/'exit.json').read_text())['exit_code']==0 else phase+'_failed'
   result=p.parent/'runs'/e['name']/'result.json';auc=json.loads(result.read_text())['macro_AUC'] if result.exists() else None
   if key not in nodes:nodes[key]=dict(id=key,experiment_id=e['experiment_id'],dataset='Taobao',family=plan['family'],status=state,auc=auc,detail=e['hypothesis'],source=str(result if result.exists() else p))
   for kind,parents in [('architecture',[e['architecture_parent']]),('comparison',e['comparison_parents'])]:
    for parent in parents:edges.append(dict(source='Taobao:'+parent,target=key,kind=kind))
 unique={(x['source'],x['target'],x['kind']):x for x in edges};edges=list(unique.values())
 data=dict(nodes=list(nodes.values()),edges=edges,sources=sources,note='recorded_parent means historical ledger parent; do not assume it distinguishes architecture and comparison. Dashed edges are explicit comparison controls. Protocols across families may differ.')
 def save(name,value):
  p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(value);t.replace(p)
 save('experiments.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 template='''<!doctype html><meta charset="utf-8"><title>实验血缘图</title><style>body{font:15px system-ui;margin:24px;color:#172238}select,input{padding:8px;margin:8px}svg{min-width:1100px}text{font:12px system-ui}.box{fill:#f3f7ff;stroke:#7492bb}.edge{stroke:#8d99aa;fill:none}.comparison{stroke-dasharray:5 4}#canvas{overflow:auto;border:1px solid #ddd}p{max-width:1000px}</style><h1>实验血缘图</h1><p>节点显示状态和 AUC；虚线表示指标对照，实线表示结构父模型或历史账本父模型。不同数据协议的 AUC 不可直接比较。选择实验族后同时显示直接父模型。</p><label>实验族 <select id="family"></select></label><label>搜索 <input id="query" placeholder="实验名 / 状态"></label><div id="count"></div><div id="canvas"></div><pre id="detail"></pre><script>const data=__DATA__;const select=document.querySelector('#family'),q=document.querySelector('#query');[...new Set(data.nodes.map(n=>n.dataset+' / '+n.family))].sort().forEach(f=>select.add(new Option(f,f)));select.value='Taobao / released_cross8_multistream';function render(){let chosen=data.nodes.filter(n=>n.dataset+' / '+n.family===select.value&&(!q.value||JSON.stringify(n).toLowerCase().includes(q.value.toLowerCase())));let ids=new Set(chosen.map(n=>n.id));data.edges.filter(e=>ids.has(e.target)).forEach(e=>ids.add(e.source));let ns=[...ids].map(id=>data.nodes.find(n=>n.id===id)||{id,experiment_id:id,status:'external reference'});let es=data.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));let ranks={};ns.forEach(n=>ranks[n.id]=0);for(let k=0;k<Math.min(ns.length,12);k++)es.filter(e=>e.kind!=='comparison').forEach(e=>ranks[e.target]=Math.min(12,Math.max(ranks[e.target],ranks[e.source]+1)));let cols={},pos={};ns.forEach(n=>{let c=ranks[n.id];cols[c]=(cols[c]||0)+1;pos[n.id]=[20+c*410,20+(cols[c]-1)*90]});let svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('width',Math.max(1100,...Object.values(pos).map(p=>p[0]+400)));svg.setAttribute('height',Math.max(180,...Object.values(pos).map(p=>p[1]+85)));function el(tag,attrs){let t=document.createElementNS(svg.namespaceURI,tag);Object.entries(attrs).forEach(([k,v])=>t.setAttribute(k,v));svg.append(t);return t}es.forEach(e=>{let a=pos[e.source],b=pos[e.target];el('path',{d:`M${a[0]+380},${a[1]+30} L${b[0]},${b[1]+30}`,class:'edge '+e.kind})});ns.forEach(n=>{let[x,y]=pos[n.id];let box=el('rect',{x,y,width:380,height:65,rx:6,class:'box'});box.onclick=()=>document.querySelector('#detail').textContent=JSON.stringify(n,null,2);let t=el('text',{x:x+8,y:y+20});t.textContent=n.experiment_id.length>52?n.experiment_id.slice(0,49)+'…':n.experiment_id;let t2=el('text',{x:x+8,y:y+44});t2.textContent=n.status+'  AUC: '+(n.auc==null?'—':Number(n.auc).toFixed(6));});document.querySelector('#canvas').replaceChildren(svg);document.querySelector('#count').textContent=`全库 ${data.nodes.length} 条实验；当前显示 ${ns.length} 个节点。点击节点查看详情。`}select.onchange=render;q.oninput=render;render();</script>'''
 save('index.html',template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
 save('README.md','# 实验血缘图\n\n打开 [交互血缘图](index.html)。完整机器可读节点与边见 [experiments.json](experiments.json)。\n\n每次实验启动及结束调用 `python3 qformer-loop/lineage/update.py`。失败也更新，不等整轮收尾。历史 parent 字段保持原意；当前研究单独记录结构父节点与指标对照边。\n')
 print(json.dumps(dict(nodes=len(nodes),edges=len(edges),sources=len(sources))))
if __name__=='__main__':main()
