import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collect import Collector,Reader,eligible_trial,incumbent_curve,usage_summary


class MetricBoundaryTests(unittest.TestCase):
    def test_test_calibration_and_other_protocol_data_are_not_curves(self):
        row={'status':'success','metrics':{'data_identity':'d1','validation':{'auc':0.6,'logloss':0.2}}}
        self.assertTrue(eligible_trial(row,'d1'))
        for extra in [{'split':'test'},{'calibration_only':True},{'data_identity':'other'}]:
            self.assertFalse(eligible_trial({**row,'metrics':{**row['metrics'],**extra}},'d1'))

    def test_incumbent_is_monotone_in_completion_order(self):
        rows=[{'id':'c','finished_unix':3,'auc':0.61},{'id':'a','finished_unix':1,'auc':0.60},
              {'id':'b','finished_unix':2,'auc':0.62}]
        self.assertEqual([point['auc'] for point in incumbent_curve(rows)],[0.60,0.62,0.62])

    def test_cached_and_reasoning_tokens_are_not_double_counted(self):
        value=usage_summary([{'usage':[{'input_tokens':1000,'cached_input_tokens':800,'output_tokens':100,'reasoning_output_tokens':30}]}])
        self.assertEqual(value['uncached_input_tokens'],200)
        self.assertEqual(value['output_tokens'],100)

    def test_collector_never_opens_test_scores_or_previous_arm_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'archived-auto-before-restart/assessment').mkdir(parents=True)
            (root/'archived-auto-before-restart/final.json').write_text('{secret test scores')
            (root/'archived-auto-before-restart/state.json').write_text('{old state should not be plotted')
            paths=[]
            original=Reader.json
            def track(reader,path,default=None):
                paths.append(str(path));return original(reader,path,default)
            with patch.object(Reader,'json',track),patch.object(Collector,'resources',return_value={'gpus':[]}):
                result=Collector(root).snapshot()
            self.assertFalse(any('/assessment/' in path or path.endswith('/final.json') or 'archived-auto' in path for path in paths))
            self.assertEqual(result['arms']['auto']['trials'],[])


if __name__=='__main__':unittest.main()
