import tempfile
import unittest
import copy
from unittest.mock import patch
from pathlib import Path
from rec_harness.adapters.amazon_memory import export_development_memory,validate_memory,execution_failure,validate_export_bindings,compact_development_context,gap_applicability
from rec_harness.core import ContractError,digest,write_json


def freeze_config(campaign,config):
    write_json(campaign/'amazon-auto.json',config)
    write_json(campaign/'auto-lock.json',{'config_hash':digest(config)})

class DevelopmentMemoryTests(unittest.TestCase):
    def test_export_rejects_changed_or_missing_controller_lock_before_reading_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);campaign=root/'campaign';output=root/'memory.json'
            # Missing study is intentional: a changed binding must fail before it is opened.
            config={'study':str(root/'unavailable-study'),'study_config_hash':'unused',
                    'task_id':'source','protocol_hash':'p','require_method_sources':True}
            freeze_config(campaign,config)
            for key,value in [('task_id','forged-task'),('require_method_sources',False),('study',str(root/'another-study'))]:
                write_json(campaign/'amazon-auto.json',{**config,key:value})
                with self.assertRaisesRegex(ContractError,'configuration changed'):export_development_memory(campaign,output)
                self.assertFalse(output.exists())
            write_json(campaign/'amazon-auto.json',config);(campaign/'auto-lock.json').unlink()
            with self.assertRaisesRegex(ContractError,'frozen controller lock'):export_development_memory(campaign,output)
            self.assertFalse(output.exists())

    def test_paper_provenance_survives_export_and_context_without_becoming_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);campaign=root/'campaign';study={'development_seeds':[42]}
            write_json(root/'study/study.json',study)
            config={'study':str(root/'study'),'study_config_hash':digest(study),'protocol_hash':'p',
                    'task_id':'source','require_method_sources':True}
            freeze_config(campaign,config)
            folder=campaign/'rounds/round-1'
            method={'origin':'paper_claim','sources':['https://example.org/fictional-paper.pdf']}
            write_json(folder/'handoff.json',{'cards':{'method':method}})
            write_json(folder/'private-confirmation.json',{'private_result':'DO_NOT_EXPORT'})
            first=export_development_memory(campaign,root/'memory.json')
            self.assertEqual(first['schema'],'amazon-development-memory-v2')
            self.assertEqual(first,export_development_memory(campaign,root/'repeat.json'))
            validate_memory(first,'p',[42]);self.assertNotIn('DO_NOT_EXPORT',str(first))
            entry=first['entries'][0];provenance=entry['method_source_provenance']
            self.assertEqual(provenance['method_hash'],digest(method))
            self.assertEqual(provenance['sources'][0]['verification_status'],'agent_reported_unverified')
            self.assertEqual(provenance['sources'][0]['source_version'],'unknown')
            _,context=compact_development_context([],first['entries'],'p',[42])
            self.assertEqual(context[0]['method_source_provenance'],provenance)
            for change in ['verified','different_citation','missing','v1']:
                bad=copy.deepcopy(first)
                if change=='verified':bad['entries'][0]['method_source_provenance']['sources'][0]['verification_status']='verified'
                if change=='different_citation':bad['entries'][0]['design']['method']['sources']=['https://example.org/other.pdf']
                if change=='missing':bad['entries'][0].pop('method_source_provenance')
                if change=='v1':bad['schema']='amazon-development-memory-v1'
                bad['hash']=digest({k:v for k,v in bad.items() if k!='hash'})
                with self.assertRaises(ContractError):validate_memory(bad,'p',[42])
            # Historical tasks retain their original export shape and citation strings.
            config.pop('require_method_sources');legacy_campaign=root/'legacy-campaign';freeze_config(legacy_campaign,config)
            write_json(legacy_campaign/'rounds/round-1/handoff.json',{'cards':{'method':method}})
            legacy=export_development_memory(legacy_campaign,root/'legacy.json')
            self.assertEqual(legacy['schema'],'amazon-development-memory-v1')
            self.assertNotIn('method_source_provenance',legacy['entries'][0]);validate_memory(legacy,'p',[42])

    def test_gap_applicability_rechecks_promoted_models_without_transferring_status(self):
        entries=[{'round':'r1','design':{'gap':{'gap_id':'gap','model_version':'old',
            'protocol_hash':'p','status':'supported','alternative_explanations':['training']}}},
            {'round':'r2','design':{'gap':{'gap_id':'gap','model_version':'new',
            'protocol_hash':'p','status':'mitigated'}}},
            {'round':'r3','design':{'gap':{'gap_id':'gap','model_version':'new',
            'protocol_hash':'other','status':'contradicted'}}},
            {'round':'legacy','design':{'gap':{'gap_id':'unknown'}}},
            {'round':'failure','execution_failure':{'status':'unresolved'}}]
        before=copy.deepcopy(entries)
        report=gap_applicability(entries,'new','p')
        self.assertEqual(entries,before)
        self.assertEqual(report,gap_applicability(entries,'new','p'))
        checks=report['checks']
        self.assertEqual([row['applicability'] for row in checks],
            ['model_changed','same_declared_scope','protocol_changed','scope_unknown'])
        self.assertEqual([row['current_gap_status'] for row in checks],['unknown']*4)
        self.assertEqual(checks[0]['source_card_hash'],digest(entries[0]['design']['gap']))
        self.assertEqual(checks[0]['source_status'],'supported')
        self.assertEqual(checks[0]['required_action'],'remeasure_before_reusing_conclusion')
        self.assertEqual(checks[1]['required_action'],'review_current_evidence')
        self.assertEqual(gap_applicability(entries,'old','p')['checks'][0]['applicability'],'same_declared_scope')

    def test_export_does_not_recover_source_projection_or_include_confirmation(self):
        from rec_harness.store import Store
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);campaign=root/'campaign';study={'development_seeds':[42]}
            write_json(root/'study/study.json',study)
            freeze_config(campaign,{'study':str(root/'study'),
                'study_config_hash':digest(study),'protocol_hash':'p','task_id':'task'})
            folder=campaign/'rounds/round-1'
            report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42}]}
            report['hash']=digest(report);write_json(folder/'development-readout.json',report)
            write_json(folder/'private-confirmation.json',{'secret':'PRIVATE_CONFIRMATION'})
            store=Store(campaign/'memory')
            store.event('auto_round_completed',{'state':{'round':1},'result':{
                'cycle_identity':'cycle','development_hash':report['hash'],'private_result':'PRIVATE_CONFIRMATION'}})
            before=store.verify();store.close()
            projection=campaign/'memory/events.jsonl'
            projection.write_text('interrupted projection\n')
            with patch.object(Store,'export_events',side_effect=AssertionError('export mutated source')):
                first=export_development_memory(campaign,root/'first.json')
                second=export_development_memory(campaign,root/'second.json')
            self.assertEqual(first,second)
            self.assertNotIn('PRIVATE_CONFIRMATION',str(first))
            self.assertEqual(projection.read_text(),'interrupted projection\n')
            store=Store(campaign/'memory',read_only=True)
            try:self.assertEqual(store.verify(),before)
            finally:store.close()

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
            freeze_config(root/'campaign',{'study':str(root/'study'),
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
            freeze_config(root/'campaign',{'study':str(root/'study'),'study_config_hash':digest(study),'protocol_hash':'p','task_id':'task'})
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
