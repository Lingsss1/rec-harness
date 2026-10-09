import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from rec_harness.adapters.history_diagnostic import summarize,diagnose
from rec_harness.core import ContractError,write_json,file_hash


class HistoryDiagnosticTests(unittest.TestCase):
    def test_development_only_candidate_never_reads_confirmation_and_checks_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'tiny_train.txt';data.write_text('1 2\n1 3\n')
            config={'data':str(root),'data_hashes':{data.name:file_hash(data)},'datasets':['tiny'],
                    'development_seeds':[42],'confirmation_seeds':[17,83],'protocol_hash':'frozen'}
            for model,ranks in [('base',[12]),('candidate',[1])]:
                write_json(root/'jobs'/f'{model}-tiny-42'/'validation_ranks.json',{'user_ids':[1],'ranks':ranks})
            calls=[]
            def audit(root,config,model,dataset,seed):
                self.assertEqual(seed,42);calls.append(model)
                return {'job_id':model,'execution_id':'same'},{'candidates_hash':'same','data_hashes':{}}
            with patch('rec_harness.adapters.history_diagnostic.load_config',return_value=config),patch('rec_harness.adapters.history_diagnostic.audit_job',side_effect=audit):
                report=diagnose(root,root/'report','base','candidate',partitions=['development'])
                self.assertEqual(calls,['base','candidate'])
                self.assertEqual(report['plan']['partitions'],['development'])
                self.assertEqual(len(report['rows']),1)
                self.assertEqual(report['rows'][0]['slices'][0]['delta']['HR@10'],1)
                def mismatched(*args):
                    result,manifest=audit(*args);result['execution_id']=args[2];return result,manifest
                with patch('rec_harness.adapters.history_diagnostic.audit_job',side_effect=mismatched):
                    with self.assertRaisesRegex(ContractError,'execution versions'):
                        diagnose(root,root/'mismatch','base','candidate',partitions=['development'])
            for phases in [[],['development','development'],['test']]:
                with self.assertRaisesRegex(ContractError,'partitions'):
                    diagnose(root,root/'invalid','base','candidate',partitions=phases)
            self.assertFalse((root/'invalid').exists())

    def test_boundary_bins_empty_groups_and_hit_conservation(self):
        left = {'user_ids': [1, 2, 3, 4, 5], 'ranks': [12, 1, 8, 99, 0]}
        right = {'user_ids': [1, 2, 3, 4, 5], 'ranks': [1, 12, 3, 99, 0]}
        rows = summarize(left, right, {1: 5, 2: 6, 3: 128, 4: 129, 5: 1})
        self.assertEqual([r['users'] for r in rows], [2, 1, 0, 1, 1])
        self.assertIsNone(rows[2]['delta']['HR@10'])
        self.assertEqual(rows[0]['hit_wins'], 1)
        self.assertEqual(rows[1]['hit_losses'], 1)
        for row in rows:
            n = row['users']
            self.assertEqual(n, row['hit_wins'] + row['hit_losses'] + row['hit_unchanged'])
            if n:
                self.assertAlmostEqual(row['delta']['HR@10'], (row['hit_wins'] - row['hit_losses']) / n)

    def test_alignment_missing_history_and_bad_rank_rejected(self):
        left = {'user_ids': [1], 'ranks': [10]}
        for right, lengths in [({'user_ids': [2], 'ranks': [1]}, {1: 5}),
                               (left, {}), ({'user_ids': [1], 'ranks': [100]}, {1: 5})]:
            with self.assertRaises(ContractError):
                summarize(left, right, lengths)
