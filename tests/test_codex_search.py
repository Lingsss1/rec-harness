import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.agents import CodexAgent, search_policy, search_receipt
from rec_harness.jailed_agent import JailedCodexAgent
from rec_harness.core import ContractError, read_json, write_json


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
