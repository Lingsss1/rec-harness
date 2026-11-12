"""Check the actual staged Agent knowledge, provenance and task boundary."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).absolute().parents[2]
ROOT=(PROJECT/'artifacts/unirank-taobao-latest').resolve()
sys.path.insert(0,str(ROOT/'source'))
os.environ.setdefault('REC_CTR_PROJECT',str(PROJECT))

from common import read,write,sha
import controller
import knowledge
sys.modules['knowledge_engine']=knowledge
import agent_tools


class MethodKnowledge(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'source/task-design/tasks/taobao_display_ctr'
        self.cards=[json.loads(line) for line in (folder/'industry/cards.jsonl').read_text().splitlines()]
        self.papers=read(folder/'paper_sources.json')
        self.protocol={'identity':'knowledge-check','data_identity':'synthetic-check','history_policy':'matured_24h'}

    def test_complete_source_identity_and_generic_boundary(self):
        expected={p.stem for p in (ROOT/'upstream/model_zoo').glob('*.py') if p.stem!='__init__'}
        self.assertEqual(len(self.cards),15)
        self.assertEqual({card['scheme'] for card in self.cards},expected)
        manifest=read(ROOT/'knowledge-assets-v2/manifest.json')
        ids={row['id'] for row in manifest['sources']}
        self.assertEqual(len(ids),len(manifest['sources']))
        self.assertNotIn('taac2026-organizer-overview',ids)
        for card in self.cards:
            self.assertTrue(set(card['source_refs'])<=ids)
            self.assertFalse(card['local_reproduction'])
            self.assertTrue(card['possible_discriminating_checks'])
        for source in manifest['sources']:
            if source.get('text_path'):
                self.assertEqual(sha(ROOT/'knowledge-assets-v2'/source['text_path']),source['text_sha256'])
            if source['id'].startswith('unirank-code:'):
                name=source['id'].split(':')[1]
                self.assertEqual(source['text_sha256'],sha(ROOT/'upstream/model_zoo'/(name+'.py')))
        self.assertEqual(sha(ROOT/'source/task-design/agent/instructions.txt'),
                         sha(PROJECT/'task_designs/autonomous_research_v1/agent/instructions.txt'))
        task=controller.make_task(self.protocol)
        self.assertEqual(len(task['knowledge']['industry']['members']),15)
        self.assertIn('15',task['knowledge']['industry']['delivery'])
        self.assertFalse(task['knowledge']['personal']['enabled'])
        self.assertEqual(read(ROOT/'setup.json')['baseline'],'din')

    def test_agent_can_search_all_methods_and_read_exact_sources(self):
        evidence=knowledge.scoped_cards(self.cards,self.papers,[],'synthetic-check','knowledge-check')
        resources=controller.knowledge_resources(ROOT,{'trials':[],'memory':[]},evidence,self.protocol)
        previous=agent_tools.ROOT
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory);agent_tools.ROOT=workspace
            try:
                registry={};knowledge.stage_resources(workspace,registry,resources)
                write(workspace/'registry.json',registry)
                (workspace/'evidence.jsonl').write_text(''.join(json.dumps(card)+'\n' for card in evidence))
                self.assertNotIn('taac2026-organizer-overview',registry)
                for card in self.cards:
                    result=agent_tools.execute('search',query=card['scheme'],source='industry',top=6)
                    self.assertIn(card['id'],{item['id'] for item in result['matches']})
                    for reference in card['source_refs']:agent_tools.execute('resolve',reference=reference)
                    code=agent_tools.execute('read',reference='unirank-code:'+card['scheme'],find='def forward',lines=8)
                    self.assertIn('def forward',code['text'])
                original=agent_tools.execute('read',reference='paper:rankmixer')
                self.assertEqual(original['fallback_ref'],'unirank-benchmark-v2')
                methods=agent_tools.execute('read',reference='unirank-benchmark-v2',lines=160)
                self.assertIn('UltraHSTU',methods['text'])
                with self.assertRaises(ValueError):agent_tools.execute('resolve',reference='craft-paper-v1')
            finally:agent_tools.ROOT=previous


if __name__=='__main__':unittest.main()
