"""Regression coverage for the v7 review/knowledge handoff failures."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rec_harness.agents import schema_for, validate_schema
from rec_harness.amazon_prefix import check_response
from rec_harness.adapters.amazon_compact import review_code, analyze
from rec_harness.core import ContractError
from rec_harness.knowledge import validate_knowledge_patch


class CompactReviewContractTests(unittest.TestCase):
    def test_positive_checks_do_not_block_but_real_findings_do(self):
        value={'verdict':'approve','checks':['Hashes match; scoring path unchanged.'],
               'blocking_findings':[], 'limitations':['CPU admission is pending.'],
               'evidence_refs':['implementation']}
        check_response({'role':'reviewer'},{'implementation_hash':'implementation'},value)
        value['blocking_findings']=['GELU does not equal identity for positive inputs.']
        with self.assertRaisesRegex(ContractError,'blocking'):
            check_response({'role':'reviewer'},{'implementation_hash':'implementation'},value)
        value['verdict']='unresolved'
        check_response({'role':'reviewer'},{'implementation_hash':'implementation'},value)
        value['evidence_refs']=['wrong-source']
        with self.assertRaises(ContractError):
            check_response({'role':'reviewer'},{'implementation_hash':'implementation'},value)

    def test_historical_review_is_read_only_and_conservative(self):
        from rec_harness.code_review import validate_code_review
        old={'verdict':'approve','findings':[],'evidence_refs':['hash'],'limitations':[]}
        before=json.dumps(old)
        validate_code_review(old,allow_legacy=True,require_approval=True)
        self.assertEqual(json.dumps(old),before)
        with self.assertRaises(ContractError):
            validate_code_review(old)
        old['findings']=['An old observation must not silently become approval.']
        with self.assertRaisesRegex(ContractError,'blocking'):
            validate_code_review(old,allow_legacy=True,require_approval=True)

    def test_review_stage_and_runtime_protocol_are_visible_without_full_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'experiment').mkdir()
            (folder/'experiment/response.json').write_text('{}')
            source=folder/'model.py';source.write_text('x = 1\n')
            seen=[]
            def role(directory,name,context,validator=None):
                seen.append(context)
                value={'verdict':'approve','checks':['Source matches.'],'blocking_findings':[],
                       'limitations':['Host admission pending.'],'evidence_refs':[context['implementation_hash']]}
                validate_schema(value,schema_for(name));validator(value,directory)
                return value
            controller=SimpleNamespace(task='task',protocol_hash='protocol',_role=role)
            context={'bindings':{'candidate_model_version':'candidate'},'protocol':{'eval_users':1000},
                     'baseline_execution_evidence':[{'dataset':'d','seed':42,'protocol':{'eval_users':1000},
                         'receipt_hash':'receipt','training_contract':{'huge':'not needed'}}]}
            review_code(controller,folder,context,{},source,source,{})
            self.assertEqual(len(seen),1)
            self.assertEqual(seen[0]['protocol']['eval_users'],1000)
            self.assertIn('before_host_admission',seen[0]['review_stage']['name'])
            self.assertNotIn('training_contract',seen[0]['baseline_execution_evidence'][0])
            def reject(directory,name,context,validator=None):
                value={'verdict':'unresolved','checks':[],'blocking_findings':['Contradictory GELU check.'],
                       'limitations':[],'evidence_refs':[context['implementation_hash']]}
                validator(value,directory);return value
            controller._role=reject
            with self.assertRaisesRegex(ContractError,'blocked training'):
                review_code(controller,folder,context,{},source,source,{})

    def test_analyst_template_uses_graph_scope_and_missing_scope_is_diagnostic(self):
        graph={'graph_version':3,'task_id':'graph-task','protocol_hash':'protocol','nodes':[],'edges':[]}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'reviewer').mkdir();(folder/'reviewer/response.json').write_text('{}')
            def role(directory,name,context,validator=None):
                attrs=dict(context['patch_contract']['node_attrs_template'])
                self.assertEqual(attrs['task_id'],'graph-task')
                p={'base_version':3,'nodes':[{'id':'claim','type':'Claim','attrs':attrs}],'edges':[]}
                validate_knowledge_patch(p,graph)
                del attrs['task_id']
                with self.assertRaisesRegex(ContractError,'missing or mismatched fields: task_id'):
                    validate_knowledge_patch(p,graph)
                attrs['task_id']='graph-task'
                value={'decision':{},'knowledge':{'patch_json':json.dumps(p)}}
                validator(value,directory);return value
            controller=SimpleNamespace(protocol_hash='protocol',study=folder,_role=role)
            review={'task_id':'campaign-task','bindings':{'proposal_id':'proposal'},'instructions':[]}
            with patch('rec_harness.adapters.amazon_cycle.finalize_cycle',return_value={'done':True}):
                _,cycle=analyze(controller,folder,review,graph,{'summary':{}},lambda v,d:None)
            self.assertEqual(cycle,{'done':True})
