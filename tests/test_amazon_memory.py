import tempfile
import unittest
import copy
from pathlib import Path
from rec_harness.adapters.amazon_memory import export_development_memory,validate_memory,execution_failure,validate_export_bindings,compact_development_context
from rec_harness.core import ContractError,digest,write_json

class DevelopmentMemoryTests(unittest.TestCase):
    def test_context_round_trip_preserves_evidence_without_mutating_packs(self):
        report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42,'metric':.53}]}
        report['hash']=digest(report)
        entries=[{'round':'old','development':report,'design':{'hypothesis':'scale'},'execution_failure':{'status':'resolved'}},
                 {'round':'new','development_hash':report['hash'],'development_decision':{'selection':'discard'}}]
        before=copy.deepcopy(entries)
        history,compact=compact_development_context([report,report],entries,'p',[42])
        self.assertEqual(entries,before)
        self.assertEqual(history,[report])
        lookup={r['hash']:r for r in history}
        restored={**compact[0],'development':lookup[compact[0]['development_hash']]}
        restored.pop('development_hash')
        self.assertEqual(restored,entries[0]);self.assertEqual(compact[1],entries[1])

    def test_context_retrieval_keeps_referenced_report_outside_history_window(self):
        reports=[]
        for metric in [.4,.5]:
            report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42,'metric':metric}]}
            report['hash']=digest(report);reports.append(report)
        entry={'round':'selected','development_hash':reports[0]['hash']}
        history,entries=compact_development_context(reports[-1:],[entry],'p',[42],reports)
        self.assertEqual({r['hash'] for r in history},{r['hash'] for r in reports})
        self.assertEqual(entries,[entry])
        with self.assertRaisesRegex(ContractError,'missing development'):
            compact_development_context(reports[-1:],[entry],'p',[42])

    def test_context_rejects_changed_or_non_development_evidence(self):
        report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42}]}
        report['hash']=digest(report)
        with self.assertRaisesRegex(ContractError,'reference mismatch'):
            compact_development_context([],[{'development':report,'development_hash':'other'}],'p',[42])
        for bad in [{**report,'protocol_hash':'other'},{**report,'rows':[{'seed':17}]}]:
            with self.assertRaises(ContractError):compact_development_context([bad],[],'p',[42])
            bad['hash']=digest({k:v for k,v in bad.items() if k!='hash'})
            with self.assertRaises(ContractError):compact_development_context([],[{'development':bad}],'p',[42])

    def test_unfinished_recovered_review_cannot_export_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study={'development_seeds':[42]}
            write_json(root/'study/study.json',study)
            write_json(root/'campaign/amazon-auto.json',{'study':str(root/'study'),
                'study_config_hash':digest(study),'review_only':True})
            with self.assertRaisesRegex(ContractError,'must finish before memory export'):
                export_development_memory(root/'campaign',root/'memory.json')
            self.assertFalse((root/'memory.json').exists())

    def test_export_checks_committed_roles_handoff_and_readout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'round-1';events=[]
            values={'evidence':{'gap_id':'gap'},'hypothesis':{'method':{'method_id':'method'},'proposal':{}},
                    'lead':{'selection':'discard'},'knowledge':{'patch_json':'{}'}}
            for role,value in values.items():
                write_json(folder/role/'response.json',value)
                events.append({'kind':'role_completed','payload':{'role':role,'task':'round-1-'+role+'-1','response_hash':digest(value)}})
            cards={'gap':values['evidence'],**values['hypothesis']}
            write_json(folder/'handoff.json',{'cards':cards})
            write_json(folder/'development-readout.json',{'hash':'committed'})
            events.append({'kind':'auto_round_completed','payload':{'state':{'round':1},'result':{'cycle_identity':'cycle','development_hash':'committed'}}})
            validate_export_bindings(folder,events)
            (folder/'lead/response.json').unlink()
            with self.assertRaisesRegex(ContractError,'missing committed response'):validate_export_bindings(folder,events)
            write_json(folder/'lead/response.json',values['lead'])
            for role in values:
                write_json(folder/role/'response.json',{'changed':True})
                with self.assertRaisesRegex(ContractError,'committed response'):validate_export_bindings(folder,events)
                write_json(folder/role/'response.json',values[role])
            write_json(folder/'handoff.json',{'cards':{**cards,'invented':'claim'}})
            with self.assertRaisesRegex(ContractError,'handoff'):validate_export_bindings(folder,events)
            write_json(folder/'handoff.json',{'cards':cards})
            write_json(folder/'development-readout.json',{'hash':'rehashed-edited-metrics'})
            with self.assertRaisesRegex(ContractError,'committed readout'):validate_export_bindings(folder,events)

    def test_recovered_failure_needs_matching_committed_development_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'round-1';write_json(folder/'failure.json',{'reason':'host admission failed'})
            readout={'schema':'amazon-development-selection-v1','rows':[]};readout['hash']=digest(readout)
            write_json(folder/'development-readout.json',readout)
            self.assertEqual(execution_failure(folder,[])['status'],'unresolved')
            event={'kind':'auto_round_completed','payload':{'state':{'round':1},'result':{
                'cycle_identity':'cycle','development_hash':readout['hash'],'confirmation_secret':'DO_NOT_EXPORT'}}}
            value=execution_failure(folder,[event]);self.assertEqual(value['status'],'resolved')
            self.assertNotIn('DO_NOT_EXPORT',str(value));self.assertIn('retained',str(value))
            event['payload']['result']['development_hash']='other'
            self.assertEqual(execution_failure(folder,[event])['status'],'unresolved')
            write_json(folder/'private-confirmation-started.json',{})
            self.assertIsNone(execution_failure(folder,[event]))

    def test_export_excludes_confirmation_results_and_preserves_failed_designs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study={'development_seeds':[42]};write_json(root/'study/study.json',study)
            write_json(root/'campaign/amazon-auto.json',{'study':str(root/'study'),'study_config_hash':digest(study),'protocol_hash':'p','task_id':'task'})
            first=root/'campaign/rounds/round-1';write_json(first/'handoff.json',{'cards':{'method':'model hypothesis'}})
            report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42}]};report['hash']=digest(report)
            write_json(first/'development-readout.json',report)
            write_json(first/'private-confirmation.json',{'metrics':'PRIVATE'})
            write_json(first/'private-confirmation-started.json',{})
            write_json(first/'failure.json',{'reason':'CONFIRMATION FAILURE'})
            second=root/'campaign/rounds/round-2';write_json(second/'handoff.json',{'cards':{'method':'failed implementation'}})
            write_json(second/'failure.json',{'reason':'invalid implementation'})
            pack=export_development_memory(root/'campaign',root/'memory.json')
            self.assertNotIn('PRIVATE',str(pack));self.assertNotIn('CONFIRMATION FAILURE',str(pack))
            self.assertIn('failed implementation',str(pack));self.assertIn('no mechanism refutation',str(pack))
            bad={**pack,'protocol_hash':'other'}
            with self.assertRaises(ContractError):validate_memory(bad,'p')
            report['rows']=[{'seed':17}];report['hash']=digest({k:v for k,v in report.items() if k!='hash'});write_json(first/'development-readout.json',report)
            with self.assertRaisesRegex(ContractError,'non-development seed'):export_development_memory(root/'campaign',root/'invalid.json')
