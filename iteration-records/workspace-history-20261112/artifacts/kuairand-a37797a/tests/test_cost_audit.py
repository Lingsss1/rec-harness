import tempfile
import unittest
from rec_harness.core import ContractError
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost


class CostAuditTests(unittest.TestCase):
    def test_interruption_retains_unknown_cost_and_detects_projection_edits(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);self.addCleanup(store.close);store.configure_budget(10)
            store.reserve('first',4);store.settle('first',2,'success');store.reserve('second',3)
            self.assertEqual(audit_cost(store,10)['pending'],1)
            store.recover();result=audit_cost(store,10)
            self.assertEqual(result['charged_or_reserved'],5);self.assertEqual(result['pending'],0)
            store.db.execute("UPDATE budget SET actual=0 WHERE id='first'")
            with self.assertRaisesRegex(ContractError,'projection'):audit_cost(store,10)

    def test_frozen_limit_and_event_order(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);self.addCleanup(store.close);store.configure_budget(10)
            with self.assertRaisesRegex(ContractError,'frozen limit'):audit_cost(store,11)
            store.event('budget_settled',{'task_id':'unreserved','actual':1,'reserved':1,'status':'success'})
            with self.assertRaisesRegex(ContractError,'without running'):audit_cost(store,10)

    def test_live_validator_prefix_survives_later_settlement_without_double_charge(self):
        from rec_harness.cost_audit import freeze_cost_prefix,verify_cost_prefix
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);self.addCleanup(store.close);store.configure_budget(10)
            store.reserve('design',2);store.settle('design',2,'success');store.reserve('validator',3)
            prefix=freeze_cost_prefix(store,10)
            self.assertEqual(prefix['pending'],1);self.assertEqual(verify_cost_prefix(store,prefix),5)
            store.settle('validator',3,'accepted');store.reserve('later',1)
            self.assertEqual(verify_cost_prefix(store,prefix),5)
            self.assertEqual(audit_cost(store,10)['charged_or_reserved'],6)
            prefix['charged_or_reserved']=2
            with self.assertRaisesRegex(ContractError,'amount changed'):verify_cost_prefix(store,prefix)
