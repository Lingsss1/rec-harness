from pathlib import Path
import tempfile
import unittest

from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.adapters.amazon_review_history import retain_predecessor,verify_predecessor
from rec_harness.store import Store


class ReviewHistoryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.study=self.root/'study';write_json(self.study/'study.json',{'frozen':True})
        self.binding={'prior_role_cost':4,'source_round':'original','proposal':'p'}

    def controller(self,name,predecessor=None):
        root=self.root/name
        config={'study':str(self.study),'study_config_hash':digest({'frozen':True}),
            'protocol_hash':'protocol','execution_id':'execution','task_id':name,'review_only':True,
            'role_call_limit':20,'agent_backend':{}}
        if predecessor:config['review_predecessor']=str(predecessor.root)
        write_json(root/'amazon-auto.json',config);write_json(root/'auto-lock.json',{'config_hash':digest(config)})
        write_json(root/'review-recovery.json',self.binding)
        store=Store(root/'memory');store.configure_budget(20)
        store.reserve('recovered-design-prior-calls',4);store.settle('recovered-design-prior-calls',4,'retained');store.close()
        controller=AmazonAutoresearch(root,load_agent=False);self.addCleanup(controller.close)
        return controller

    def charge(self,controller,task,amount):
        controller.store.reserve(task,amount);controller.store.settle(task,amount,'failed')
        write_json(controller.root/'rounds/round-1/evaluation/rejection.json',{'reason':task})

    def test_successor_retains_failures_once_and_reuses_accepted_review(self):
        old=self.controller('old');self.charge(old,'failed',2)
        folder=old.root/'rounds/round-1/lead';response={'selection':'discard'};context={'bound':'readout'}
        for name,value in [('context.json',context),('response.json',response),('resources.json',{'status':'success'})]:
            write_json(folder/name,value)
        accepted={'response_hash':digest(response),'context_hash':digest(context)}
        write_json(folder/'accepted.json',accepted)
        old.store.reserve('round-1-lead-1',1);old.store.settle('round-1-lead-1',1,'success')
        old.store.event('role_completed',{'task':'round-1-lead-1','role':'lead',**accepted,
            'archive':old.artifacts.snapshot(folder,['context.json','response.json','resources.json'])})
        new=self.controller('new',old);target=new.root/'rounds/round-1'
        result=retain_predecessor(new,self.binding,target)
        self.assertEqual(result['additional_cost'],3)
        self.assertEqual(new.store.budget()['charged_or_reserved'],7)
        self.assertEqual(read_json(target/'lead/response.json'),response)
        self.assertFalse((target/'evaluation').exists())
        before=new.store.verify()
        self.assertEqual(retain_predecessor(new,self.binding,target),result)
        self.assertEqual(verify_predecessor(new),result)
        self.assertEqual(new.store.verify(),before)
        self.assertFalse(any(e['kind']=='agent_submitted' for e in new.store.events()))
        # A further successor carries 4 original + 4 review calls, not 4+7+1.
        self.charge(new,'new-failed',1)
        last=self.controller('last',new)
        retain_predecessor(last,self.binding,last.root/'rounds/round-1')
        self.assertEqual(last.store.budget()['charged_or_reserved'],8)
        verify_predecessor(last)
        self.assertEqual(read_json(last.root/'rounds/round-1/lead/reuse.json')['source_role_directory'],str(folder))
        write_json(folder/'response.json',{'changed':True})
        with self.assertRaisesRegex(ContractError,'artifact changed'):verify_predecessor(last)

    def test_changed_ledger_or_binding_and_pending_calls_are_rejected(self):
        old=self.controller('old');self.charge(old,'failed',2)
        new=self.controller('new',old)
        with self.assertRaisesRegex(ContractError,'binding differs'):
            retain_predecessor(new,{**self.binding,'proposal':'other'},new.root/'rounds/round-1')
        old.store.reserve('pending',1)
        with self.assertRaisesRegex(ContractError,'pending calls'):
            retain_predecessor(new,self.binding,new.root/'rounds/round-1')
        old.store.settle('pending',1,'failed')
        retain_predecessor(new,self.binding,new.root/'rounds/round-1')
        self.charge(old,'late-call',1)
        with self.assertRaisesRegex(ContractError,'ledger changed'):verify_predecessor(new)
        with self.assertRaisesRegex(ContractError,'predecessor changed'):
            retain_predecessor(new,self.binding,new.root/'rounds/round-1')

    def test_interruption_after_reservation_settles_without_double_charge(self):
        from unittest.mock import patch
        old=self.controller('old');self.charge(old,'failed',2);new=self.controller('new',old)
        with patch.object(new.store,'settle',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):retain_predecessor(new,self.binding,new.root/'rounds/round-1')
        self.assertEqual(new.store.budget()['charged_or_reserved'],6)
        retain_predecessor(new,self.binding,new.root/'rounds/round-1')
        self.assertEqual(new.store.budget()['charged_or_reserved'],6)
        verify_predecessor(new)
