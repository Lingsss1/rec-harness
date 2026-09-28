import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.agents import CodexAgent, search_policy, search_receipt
from rec_harness.jailed_agent import JailedCodexAgent
from rec_harness.core import Artifacts, ContractError, read_json, write_json
from rec_harness.literature import from_archive


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_explicit_modes_and_invalid_configuration(self):
        self.assertEqual(search_policy()['hypothesis'], 'live')
        self.assertEqual(search_policy()['evaluation'], 'disabled')
        self.assertEqual(set(search_policy('disabled').values()), {'disabled'})
        for invalid in ['true', {'hypothsis': 'live'}, {'hypothesis': True}, []]:
            with self.assertRaises(ContractError):
                search_policy(invalid)

    def test_cli_override_reaches_process_and_receipt_is_not_false_search_proof(self):
        response = {'selection':'provisional','conclusion_status':'unknown',
                    'rationale':'fixture','evidence_refs':[],'unresolved_explanations':[]}
        def process(argv, cwd, directory, **kwargs):
            self.assertIn('web_search="disabled"', argv)
            write_json(directory/'response.json', response)
            (directory/'stdout.jsonl').write_text('{"type":"turn.completed"}\n')
            return {'status':'success'}
        with patch('rec_harness.agents.run_process', side_effect=process):
            CodexAgent().run('evaluation', {}, self.root)
        receipt = read_json(self.root/'literature-search.json')
        self.assertFalse(receipt['search_observed'])
        self.assertEqual(receipt['reported_sources'], [])

    def test_search_events_and_reported_citations_keep_distinct_provenance(self):
        event = {'type':'item.completed','item':{'type':'web_search','query':'NextItNet'}}
        (self.root/'stdout.jsonl').write_text('bad-json\n'+json.dumps(event)+'\n')
        search_receipt(self.root, 'live', {'method':{'sources':['https://arxiv.org/abs/1808.05163']}})
        receipt = read_json(self.root/'literature-search.json')
        self.assertEqual(receipt['observed_search_events'], [event])
        self.assertTrue(receipt['search_observed'])
        self.assertEqual(receipt['sources_status'], 'agent_reported_unverified')

    def test_jailed_live_prompt_and_disabled_replay_prompt(self):
        write_json(self.root/'template/jail-manifest.json', {})
        live = JailedCodexAgent(self.root/'template', self.root/'runtime')
        self.assertIn('built-in web search', live.role_prompt('hypothesis'))
        replay = JailedCodexAgent(self.root/'template', self.root/'runtime', web_search='disabled')
        self.assertIn('Do not retrieve external sources', replay.role_prompt('hypothesis'))
        self.assertNotIn('built-in web search', replay.role_prompt('hypothesis'))

    def test_search_trace_requires_exact_log_response_and_receipt_binding(self):
        event={'type':'item.completed','item':{'type':'web_search','query':'fixture'}}
        (self.root/'stdout.jsonl').write_text(json.dumps(event)+'\n')
        response={'method':{'sources':['https://example.org/paper#section-2']}}
        write_json(self.root/'response.json',response)
        search_receipt(self.root,'live',response)
        artifacts=Artifacts(self.root/'objects')
        names=['stdout.jsonl','response.json','literature-search.json']
        archive=artifacts.snapshot(self.root,names)
        evidence=from_archive(archive,artifacts)
        self.assertEqual(evidence['observed_search_events'],[event])
        self.assertEqual(evidence['reported_sources'],response['method']['sources'])
        self.assertEqual(evidence['sources_status'],'agent_reported_unverified')
        # Even internally valid replacement CAS objects cannot change the bound evidence.
        for name,value in [('stdout.jsonl',''),('response.json','{"method":{"sources":[]}}')]:
            (self.root/name).write_text(value)
            altered={**archive,name:artifacts.put(self.root/name)}
            with self.assertRaisesRegex(ContractError,'differs'):from_archive(altered,artifacts)
        missing={k:v for k,v in archive.items() if k!='stdout.jsonl'}
        with self.assertRaisesRegex(ContractError,'lacks archived CLI log'):from_archive(missing,artifacts)

    def test_unaccepted_response_and_legacy_receipts_do_not_invent_citations(self):
        (self.root/'stdout.jsonl').write_text('')
        write_json(self.root/'response.json',{'method':{'sources':['unaccepted']}})
        search_receipt(self.root,'disabled')
        artifacts=Artifacts(self.root/'objects')
        archive=artifacts.snapshot(self.root,['stdout.jsonl','response.json','literature-search.json'])
        value=from_archive(archive,artifacts)
        self.assertFalse(value['search_observed']);self.assertEqual(value['reported_sources'],[])
        self.assertIsNone(value['response_ref'])
        write_json(self.root/'literature-search.json',{'schema':'codex-literature-search-v1'})
        archive['literature-search.json']=artifacts.put(self.root/'literature-search.json')
        self.assertIsNone(from_archive(archive,artifacts))
