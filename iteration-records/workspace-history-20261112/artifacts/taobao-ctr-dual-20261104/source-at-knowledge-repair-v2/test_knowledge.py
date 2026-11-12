"""Knowledge access, relevance, provenance and maintenance continuity boundaries."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import knowledge
sys.modules['knowledge_engine']=knowledge
import agent_tools
import controller
from common import read,write,sha

PROJECT=controller.PROJECT
CAMPAIGN=PROJECT/'artifacts/taobao-ctr-dual-20261104'


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.previous=agent_tools.ROOT;agent_tools.ROOT=self.root
    def tearDown(self):
        agent_tools.ROOT=self.previous;self.tmp.cleanup()

    def test_actual_source_ids_text_versions_and_missing_reports(self):
        protocol=read(CAMPAIGN/'protocol.json')
        state={'trials':[],'memory':[]}
        resources=controller.knowledge_resources(CAMPAIGN,state,[],protocol)
        registry={'task://t-001':'task.json','run://r-001':'runtime.json','workspace://w-001':'workspace.json'}
        for filename in registry.values():write(self.root/filename,{'contract':filename})
        knowledge.stage_resources(self.root,registry,resources);write(self.root/'registry.json',registry)
        for initial in ['task://t-001','run://r-001','workspace://w-001']:
            self.assertIn('contract',agent_tools.execute('resolve',reference=initial))
        for identifier in ['craft-paper-v1','queryformer-paper-v1','unidot-paper-v1','paper:din','paper:dmr','taac2026-organizer-overview']:
            metadata=agent_tools.execute('resolve',reference=identifier)
            result=agent_tools.execute('read',reference=identifier,lines=5)
            self.assertEqual(metadata['text_sha256'],result['text_sha256'])
            self.assertTrue(result['text']);self.assertLessEqual(len(result['text']),24000)
            self.assertNotIn('text_path',metadata)
        missing=agent_tools.execute('read',reference='unidac-detailed-report')
        self.assertEqual(missing['fallback_ref'],'taac2026-organizer-overview')
        with self.assertRaises(ValueError):agent_tools.execute('resolve',reference='/etc/passwd')
        registry['bad']='../../etc/passwd';write(self.root/'registry.json',registry)
        with self.assertRaises(ValueError):agent_tools.execute('resolve',reference='bad')
        self.assertGreater(len((self.root/'knowledge-access.jsonl').read_text().splitlines()),10)

    def test_length_normalization_separate_recall_and_no_forced_hit(self):
        cards=[{'id':'industry-one','knowledge_type':'industry','source_problem':'query attention bottleneck'},
               {'id':'paper-one','title':'query attention modeling'}]
        memory={'id':'m','task_scope':'d','protocol_hash':'p','analysis':{
            'observations':'query attention '*50000,'limitations':'counterevidence remains provisional'}}
        original=copy.deepcopy(memory)
        cards+=knowledge.scoped_cards([],{'sources':[]},[memory],'d','p')
        result=knowledge.retrieve(cards,'query attention')
        self.assertIn('industry-one',result['selected_ids'])
        self.assertEqual(memory,original)
        self.assertLess(len(json.dumps(cards[-1])),10000)
        self.assertIn('counterevidence',cards[-1]['summary']['limitations']['text'])
        self.assertEqual(len(knowledge.retrieve(cards,'query attention',limit=1)['cards']),1)
        self.assertEqual(knowledge.retrieve(cards,'zzzzunrelated')['cards'],[])
        with self.assertRaises(ValueError):knowledge.retrieve(cards,'query',requested=['invented'])

    def test_scope_and_personal_exclusion(self):
        record={'id':'m','task_scope':'wrong','protocol_hash':'p','analysis':{'observations':'query'}}
        self.assertEqual(knowledge.scoped_cards([],{'sources':[]},[record],'d','p'),[])
        personal={'id':'private','knowledge_type':'personal','title':'query'}
        self.assertEqual(knowledge.rank([personal],'query'),[])

    def test_usage_does_not_turn_consideration_into_adoption(self):
        card={'id':'i','knowledge_type':'industry','source_refs':['s']}
        proposal={'variants':[{'name':'control'}],'knowledge_assessments':[{
            'knowledge_id':'i','decision':'rejected','rationale':'local assumption unsupported','variant_names':[]}]}
        row={'id':'trial','source_sha256':'abc','status':'success'}
        record=knowledge.usage_record(proposal,[card],[row])
        self.assertEqual(record['assessments'][0]['decision'],'rejected')
        self.assertEqual(record['executed_variants'][0]['source_sha256'],'abc')
        self.assertEqual(knowledge.usage_record({},[card])['undocumented_industry_ids'],['i'])

    def fixture_checkpoint(self):
        root=self.root;arm=root/'auto';arm.mkdir()
        protocol={'identity':'p','data_identity':'d','estimated_unit_seconds':1,'estimated_validation_seconds':1}
        state={'mode':'auto','trials':[{'id':'existing','status':'success','wall_seconds':3}],
               'memory':[],'agent_calls':[],'research_closed':False,'incumbent':'existing'}
        write(root/'protocol.json',protocol);write(root/'budget.json',{'deadline_unix':time.time()+100})
        write(arm/'state.json',state);write(arm/'method.json',{'mode':'auto','original':True})
        directory=arm/'rounds/r0005/query';directory.mkdir(parents=True)
        write(arm/'knowledge-handoff.json',{'state_sha256':sha(arm/'state.json'),'protocol_hash':'p','next_round':5,
             'committed_trials':1,'committed_memory':0,'active_children':[]})
        return root,arm,protocol,state

    def test_resume_preserves_state_deadline_and_never_retrains_baselines(self):
        root,arm,protocol,state=self.fixture_checkpoint();before=read(root/'budget.json')
        with (patch.object(controller,'run_jobs',side_effect=AssertionError('No baseline retraining')),
              patch.object(controller,'final_assessment') as final,patch.object(controller,'comparison')):
            controller.worker(root,'auto',[0],resume=True)
        self.assertEqual(read(arm/'state.json'),state)
        self.assertEqual(read(root/'budget.json'),before)
        self.assertTrue(final.called)
        self.assertTrue(read(arm/'method.json')['original'])
        self.assertEqual(read(arm/'method.json')['runtime_revisions'][-1]['start_round'],5)
        self.assertTrue((arm/'maintenance'/knowledge.REVISION/'r0005').exists())
        with self.assertRaises(ValueError):controller.resume_checkpoint(arm,protocol)

    def test_closed_changed_or_started_checkpoint_refused(self):
        root,arm,protocol,state=self.fixture_checkpoint()
        state['research_closed']=True;write(arm/'state.json',state)
        marker=read(arm/'knowledge-handoff.json');marker['state_sha256']=sha(arm/'state.json');write(arm/'knowledge-handoff.json',marker)
        with self.assertRaisesRegex(ValueError,'closed'):controller.resume_checkpoint(arm,protocol)
        state['research_closed']=False;write(arm/'state.json',state)
        with self.assertRaisesRegex(ValueError,'digest'):controller.resume_checkpoint(arm,protocol)
        marker['state_sha256']=sha(arm/'state.json');write(arm/'knowledge-handoff.json',marker)
        (arm/'rounds/r0005/query/codex.jsonl').write_text('{"started":true}\n')
        with self.assertRaisesRegex(ValueError,'started'):controller.resume_checkpoint(arm,protocol)


if __name__=='__main__':unittest.main()
