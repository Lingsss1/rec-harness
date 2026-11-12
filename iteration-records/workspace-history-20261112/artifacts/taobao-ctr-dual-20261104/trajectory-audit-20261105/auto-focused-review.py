"""Review auto interventions only; separate score, mechanism and evidence value."""
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
OUT=ROOT/'auto-focused-review';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parents[1]/'experiments/taobao_ctr_dual'))
from common import read,write,sha

annotations={
0:('单次正对照','目标匹配次数','匹配次数胜过同参数匹配比例与非目标支持次数；目标对应信息的贡献较强。'),
1:('单次正对照','匹配与时间对应','对齐时间胜过活动时间、解耦时间与重复计数。'),
2:('对照胜出','联合类目品牌匹配','joint_pair 虽超过父模型，但未胜 factorized_pair/support_pair；保留支持与近期量对照，不能归功于联合匹配。'),
3:('主指标无增益/指标权衡','分行为语义池化','四个变体AUC均低于父模型，action_attention 的Logloss略好；不支持当前追加向量实现用于AUC选型。'),
4:('部分机制支持','字段对齐与独立路由','对齐胜过交叉字段；独立路由未胜共享路由。整体微小AUC增益伴随较差Logloss。'),
5:('对照胜出','目标条件上下文查询','query_blind 胜过 query_adaptive；支持学习型上下文的弱候选，未证明目标条件选择必要。'),
6:('主指标无增益/指标权衡','缺失字段抑制','全部AUC低于父模型，available_active 的Logloss略好；不推广为所有缺失感知方法无效。'),
7:('当前实现负结果','显式乘积读出','四个实现均未胜父模型；aligned_product 还未胜交叉及加法对照。'),
8:('当前实现负结果','目标分数强度读出','全部AUC低于父模型；raw目标条件未胜zero_raw，centered虽胜zero_centered仍未形成实用主指标增益。'),
9:('局部对照有信号/无实用AUC增益','学习衰减尺度','horizon_aligned 胜幅度控制，但仍低于父模型；Logloss改善。'),
10:('单次正对照/特异性较弱','最近匹配事件','match_latest 胜父模型及 mean；相对 activity_latest 的优势较小。'),
11:('单次正对照','最近非匹配事件','nonmatch_latest 胜 mean 与 activity_latest；保留该表示，不推出普遍的负兴趣因果解释。'),
12:('部分机制支持','相对时间顺序','rank 胜对应 decay；crossed_rank 胜 aligned_rank，未支持同字段锚点特异性；AUC与Logloss存在权衡。'),
13:('单次正对照','必要支持与顺序值','required_rank 胜较严支持父模型及 presence 替代；支持顺序数值而非仅支持标记。'),
14:('主指标无增益/指标权衡','全部锚点平均','all_rank 未胜 latest_rank 精确回放，但Logloss更好；不能称完全无价值。'),
15:('对照胜出/参数化证据','新增支持输入','raw_duplicate 胜 raw_support；重复已有输入不增加信息，却改变参数化和优化。'),
16:('负结果与回放','跨行为池化锚点','pooled_rank 低于 same_rank；same_rank 回放保留旧分数，不是新机制增益。'),
17:('微小单次收益/待复核','条件回退','仅缺支持时回退略胜 same_only；无条件池化仍差，Logloss略恶化。'),
18:('当前实现负结果','新增回退支持指示','四个变体都未胜父模型；最好的仍是重复特征控制。'),
19:('对照胜出','最近非匹配与活动','latest_activity 胜 latest_nonmatch；当前新读出不要求非匹配特异性。'),
20:('负结果与回放','连续时间间隔比值','ratio_oriented 未胜 ordinal；ordinal_oriented 精确复现父模型。'),
21:('当前实现负结果','同时间跨字段重叠','joint 同行见证未胜双支持控制，各变体未胜父模型。'),
22:('替换方式负结果','用次数替换原读出','四个替换均低于父模型；不能据此排除保留旧读出后增加次数。'),
23:('单次正对照/含参数化混杂','保留旧读出并补充量','complement_newer_count 胜 duplicate，但额外信息优势小于相对父模型总增益；all_count 未胜 duplicate。'),
24:('主指标无增益/指标权衡','按来源分开读出','provenance_count 未胜父模型；provenance_support 的Logloss稍好，仍不是AUC改进。'),
25:('当前实现负结果','编码ID去重宽度','distinct_newer/all 未胜各自 event_count 控制，全部未胜父模型。'),
26:('单次正对照','按时间排列的匹配切换','ordered_switch 胜 exchangeable_switch 和重复量；同时改善父模型AUC/Logloss。'),
27:('对照胜出','有方向的历史漂移','signed_drift 胜 folded_drift，但仍低于父模型和 latest_endpoint；latest_endpoint 是有用的胜出分支，不能把此结果归为方向性漂移成功。'),
28:('无增益/部分近似持平','最新连续匹配段','newest_run 不胜 oldestrun/noncontiguous 控制；oldest_run 与父模型AUC近似持平，无可靠新提升。'),
29:('主指标无增益/指标权衡','最新非匹配连续段','nonmatch_run 未胜 nonmatch_mass；后者Logloss更好但AUC较低。'),
30:('局部对照有信号/未胜主线','条件一阶持续性','ordered_conditional 胜同轮控制但略低于父模型；可作机制线索，不能计新增最优。'),
31:('主指标无增益/指标权衡','滞后一阶关联','ordered_contrast 未胜父模型；部分控制Logloss更好。'),
32:('中间分支/原机制未支持','目标条件跨行为邻接','target_ordered 未胜 activity_ordered；该活动控制成为r33实际父模型，必要性仍待消融。'),
33:('对照胜出/当前最高','行为转移与间隔耦合','factorized_symmetric 胜 aligned_directed；最佳不支持必须绑定逐事件间隔或新增箭头方向。'),
34:('负结果但有归因价值','邻接与组成替换','composition_relative 未胜现有adjacent_relative；这些替换用于检验原r33特征，非无信息的研究。'),
35:('主指标无增益/指标权衡','绝对节奏','cadence_joint 未胜 absolute 控制，均未胜父模型；joint 的Logloss略好。'),
36:('主指标无增益/指标权衡','更广支持的活动对','pair identity 未胜 mass，两者AUC低于父模型但Logloss改善。'),
37:('当前实现负结果','按支持状态追加读出','新增pairs未胜common_mass；全部未胜父模型，不支持当前分段追加实现。'),
38:('候选负结果/回放与对称性检查','等价特征基变换','残差基未胜 raw 回放；raw精确复现最高分、常数正负控制一致是正确性证据，不是独立种子复核；常数控制有Logloss权衡。'),
39:('当前实现负结果','类目品牌联合转移','joint_ordered 未胜 factorized_ordered；全部未胜父模型。'),
40:('主指标无增益/部分指标权衡','追加完整语义池化向量','四种AUC均低于主线且训练成本增加；uniform_mean 的Logloss更好，仍需区分精度目标。'),
41:('局部修复/未恢复主线/指标权衡','语义分支停止梯度','相对失败父分支r40有改善，但所有AUC仍低于r33；部分Logloss优于主线。不能把相对失败父模型涨分记成主线AUC成功，也不能抹掉指标权衡。'),
42:('未完成/不作科学判定','紧凑目标语义读出','用户中止了三项训练，没有完整协议结果；不能记为失败或有效。')}
state=read(ROOT/'auto/state.json');trials=state['trials'];lookup={t['id']:t for t in trials}
incumbent=lookup['baseline-din']['metrics']['validation']['auc'];rounds=[];variant_rows=[];records=[]
for index in range(43):
 directory=ROOT/'auto/rounds'/f'r{index:04d}';p=read(directory/'propose/proposal.json');parent=lookup[p['parent_trial']]
 group=[t for t in trials if t.get('round')==index];parent_v=parent['metrics']['validation'];category,name,interpretation=annotations[index]
 by_role={role:[t for t in group if t.get('role')==role] for role in ['candidate','control']}
 best=max(group,key=lambda t:t['metrics']['validation']['auc']) if group else None
 best_candidate=max(by_role['candidate'],key=lambda t:t['metrics']['validation']['auc']) if by_role['candidate'] else None
 best_control=max(by_role['control'],key=lambda t:t['metrics']['validation']['auc']) if by_role['control'] else None
 row={'round':index,'mechanism':name,'category':category,'parent_id':parent['id'],'parent_auc':parent_v['auc'],
      'incumbent_auc_before':incumbent,'completed_variants':len(group),'best_id':best['id'] if best else None,
      'best_role':best['role'] if best else None,'best_auc':best['metrics']['validation']['auc'] if best else None,
      'delta_parent_auc':best['metrics']['validation']['auc']-parent_v['auc'] if best else None,
      'delta_incumbent_auc':best['metrics']['validation']['auc']-incumbent if best else None,
      'refreshes_incumbent':bool(best and best['metrics']['validation']['auc']>incumbent),
      'best_candidate_id':best_candidate['id'] if best_candidate else None,
      'best_control_id':best_control['id'] if best_control else None,
      'candidate_minus_best_control_auc':best_candidate['metrics']['validation']['auc']-best_control['metrics']['validation']['auc'] if best_candidate and best_control else None,
      'interpretation':interpretation,'proposal_ref':str((directory/'propose/proposal.json').relative_to(ROOT))}
 rounds.append(row)
 records.append({**row,'question':p['study']['question'],'prespecified_contrasts':p['study']['contrasts']})
 for t in group:
  m=t['metrics'];v=m['validation']
  variant_rows.append({'round':index,'id':t['id'],'role':t['role'],'parent':parent['id'],'auc':v['auc'],'logloss':v['logloss'],
   'delta_parent_auc':v['auc']-parent_v['auc'],'delta_incumbent_auc':v['auc']-incumbent,
   'delta_parent_logloss':v['logloss']-parent_v['logloss'],'parameters':m['parameters'],'dense_parameters':m.get('dense_parameters'),
   'train_seconds':m.get('train_seconds'),'inference_seconds':v.get('inference_seconds'),'source_sha256':t['source_sha256'],
   'classification':category})
 if best:incumbent=max(incumbent,best['metrics']['validation']['auc'])
new=[r for r in rounds if r['refreshes_incumbent']]
summary={'at_unix':time.time(),'scope':'auto only, completed full-protocol validation results; no closed test scores read',
 'state_sha256':sha(ROOT/'auto/state.json'),'committed_training_records':len(trials),'baseline_records':3,
 'complete_research_rounds':42,'research_variant_records':len(variant_rows),'interrupted_rounds':[42],
 'incumbent_improving_rounds':len(new),'candidate_incumbent_winners':sum(r['best_role']=='candidate' for r in new),
 'control_incumbent_winners':sum(r['best_role']=='control' for r in new),'no_incumbent_improvement_rounds':42-len(new),
 'new_best_rounds':[r['round'] for r in new],
 'limitations':['All search models used seed42; no claim of significance or independent model-seed robustness.',
  'Best control is a conservative performance reference, not an automatic substitute for the prespecified matched contrast.',
  'Failure to improve incumbent does not imply an uninformative experiment or a universally invalid mechanism.',
  'AUC-primary failure and Logloss/cost tradeoff remain separate.',
  'Current experiment remains terminated; no model training was launched.'],
 'rounds':records}
write(OUT/'summary.json',summary)
for filename,rows in [('rounds.csv',rounds),('variants.csv',variant_rows)]:
 with (OUT/filename).open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:summary[k] for k in ['committed_training_records','research_variant_records','complete_research_rounds','interrupted_rounds','incumbent_improving_rounds','candidate_incumbent_winners','control_incumbent_winners','no_incumbent_improvement_rounds']},ensure_ascii=False))
