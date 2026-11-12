"""Amazon role wire format and development-only tool exposure; no GPU calls."""
import copy
import unittest
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.store import Store
from rec_harness.tool_exposure import audit_exposure
import test_amazon_project


class AmazonExposureTests(unittest.TestCase):
    setUp=test_amazon_project.AmazonProjectTests.setUp
    config=test_amazon_project.AmazonProjectTests.config
    cpu=test_amazon_project.AmazonProjectTests.cpu
    round=staticmethod(test_amazon_project.AmazonProjectTests.round)
    run_project=test_amazon_project.AmazonProjectTests.run_project

    def test_candidate_review_keeps_incumbent_measurement_explicitly_scoped(self):
        from rec_harness.adapters.amazon_review_evidence import scope_incumbent_measurements
        self.run_project('tool');child=self.campaign/'episodes/model-2/campaign'
        artifacts=Artifacts(child/'artifacts/objects');store=Store(child/'memory')
        try:
            submission=next(e['payload'] for e in store.events() if e['kind']=='agent_submitted')
            original=read_json(artifacts.verify(submission['visible_state_ref']))
            before=copy.deepcopy(original)
            review={**scope_incumbent_measurements(original),'model_version':'new-candidate',
                'bindings':{'base_model_id':original['model_version'],'candidate_model_version':'new-candidate'}}
            self.assertEqual(original,before)
            self.assertNotIn('diagnostic',review)
            self.assertEqual(review['incumbent_measurement_context']['diagnostic'],original['diagnostic'])
            def submit(name,context):
                write_json(child/(name+'.json'),context)
                store.event('agent_submitted',{'task_id':name,'role':'evaluation',
                    'visible_state_ref':artifacts.put(child/(name+'.json')),'input_hash':digest(context)})
            submit('scoped-review',review)
            report=audit_exposure(child)
            self.assertEqual(report['exposures'][-1]['model_version'],original['model_version'])
            self.assertFalse(report['exposures'][-1]['transport_completed'])
            bad=copy.deepcopy(review);bad['incumbent_measurement_context']['model_version']='wrong-base'
            submit('bad-scope',bad)
            with self.assertRaisesRegex(ContractError,'scope differs'):audit_exposure(child)
        finally:store.close()

    def test_unscoped_stale_diagnostic_is_still_rejected(self):
        self.run_project('tool');child=self.campaign/'episodes/model-2/campaign'
        artifacts=Artifacts(child/'artifacts/objects');store=Store(child/'memory')
        try:
            submission=next(e['payload'] for e in store.events() if e['kind']=='agent_submitted')
            context=read_json(artifacts.verify(submission['visible_state_ref']))
            context['model_version']='new-candidate'
            write_json(child/'stale-review.json',context)
            store.event('agent_submitted',{'task_id':'stale-review','role':'evaluation',
                'visible_state_ref':artifacts.put(child/'stale-review.json'),'input_hash':digest(context)})
            with self.assertRaisesRegex(ContractError,'stale model'):audit_exposure(child)
        finally:store.close()

    def test_real_controller_role_format_and_compact_review(self):
        self.run_project('tool')
        child=self.campaign/'episodes/model-2/campaign'
        report=audit_exposure(child)
        self.assertEqual(len(report['exposures']),1)
        self.assertTrue(report['exposures'][0]['transport_completed'])
        self.assertEqual(report,audit_exposure(child))
        artifacts=Artifacts(child/'artifacts/objects');store=Store(child/'memory')
        try:
            submission=next(e['payload'] for e in store.events() if e['kind']=='agent_submitted')
            original=read_json(artifacts.verify(submission['visible_state_ref']))
            context={k:original[k] for k in ['task_id','model_version','protocol_hash']}
            context['baseline']=original['measurement_baseline']['runs']
            context['diagnostic']={k:v for k,v in original['diagnostic'].items() if k in
                ['status','tool_version','implementation_hash','output','tool_registration']}
            write_json(child/'review-fixture.json',context)
            store.event('agent_submitted',{'task_id':'review-fixture','role':'measurement',
                'visible_state_ref':artifacts.put(child/'review-fixture.json'),'input_hash':digest(context)})
            before=store.verify()
        finally:store.close()
        pending=audit_exposure(child)
        self.assertEqual(pending['integrity'],before)
        self.assertEqual(len(pending['exposures']),2)
        self.assertFalse(pending['exposures'][1]['transport_completed'])
        store=Store(child/'memory')
        try:store.event('role_completed',{'task':'review-fixture','role':'measurement','context_hash':digest(context)})
        finally:store.close()
        self.assertTrue(audit_exposure(child)['exposures'][1]['transport_completed'])
        # Even an archived, self-consistent input must not claim confirmation as development.
        changed=copy.deepcopy(context);changed['baseline'][0]['seed']=17
        write_json(child/'confirmation-fixture.json',changed)
        store=Store(child/'memory')
        try:store.event('agent_submitted',{'task_id':'wrong-split','role':'measurement',
            'visible_state_ref':artifacts.put(child/'confirmation-fixture.json'),'input_hash':digest(changed)})
        finally:store.close()
        with self.assertRaisesRegex(ContractError,'outside frozen development'):audit_exposure(child)

    def test_study_changes_rejected_without_role_execution(self):
        self.run_project('tool');child=self.campaign/'episodes/model-2/campaign'
        config=read_json(child/'amazon-auto.json');study=self.root/'study/study.json'
        value=read_json(study);value['development_seeds']=[17];write_json(study,value)
        with self.assertRaisesRegex(ContractError,'bound study changed'):audit_exposure(child)
