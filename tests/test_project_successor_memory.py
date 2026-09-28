"""Successor restart must reuse the exact parent export, not any valid package."""
import copy
import unittest
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.project import ResearchProject
from rec_harness.store import Store
from rec_harness.memory_policy import validate_package
from rec_harness.adapters.amazon_memory import validate_memory
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
import test_project
import test_amazon_project


class SuccessorMemoryTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp
    cpu=test_project.ProjectTests.cpu

    def exercise(self,path,amazon=False):
        original=ResearchProject.child
        def interrupt(project,directory,config):
            if directory.name=='model-2':raise KeyboardInterrupt()
            return original(project,directory,config)
        with patch.object(ResearchProject,'child',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,path,agent=self.agent)
        package=self.campaign/'episodes/model-2/source-memory.json'
        expected=read_json(package);changed=copy.deepcopy(expected)
        if amazon:
            changed['entries']=[]
        else:
            changed['source_task']='unrelated-parent'
        changed['hash']=digest({k:v for k,v in changed.items() if k!='hash'})
        if amazon:validate_memory(changed,'p')
        else:validate_package(changed)
        write_json(package,changed)
        with patch.object(ResearchProject,'child',side_effect=AssertionError('must reject before child execution')):
            with self.assertRaisesRegex(ContractError,'successor memory differs'):
                autoresearch(self.campaign,agent=self.agent)
        self.assertEqual(read_json(package),changed)
        self.assertFalse((package.parent/'campaign').exists())
        self.assertEqual(list(package.parent.glob('verify-memory-*')),[])
        write_json(package,expected)
        report=autoresearch(self.campaign,agent=self.agent)
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','model'])
        with patch.object(ResearchProject,'child',side_effect=AssertionError('terminal restart')):
            self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)

    def test_generic_valid_replacement_rejected_before_successor(self):
        self.exercise(test_project.ProjectTests.config(self,'harness'))

    def test_amazon_valid_empty_replacement_rejected_before_successor(self):
        with patch.object(AmazonAutoresearch,'_round',test_amazon_project.AmazonProjectTests.round), \
             patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})), \
             patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=AssertionError('cannot train')):
            self.exercise(test_amazon_project.AmazonProjectTests.config(self,'harness'),amazon=True)

    def test_legacy_project_does_not_reexport_historical_package(self):
        project=object.__new__(ResearchProject);project.config={}
        package=self.root/'historical.json';write_json(package,{'historical':True})
        def forbidden(path):raise AssertionError('legacy re-export')
        project.successor_memory(package,forbidden)
        self.assertEqual(read_json(package),{'historical':True})

    def test_valid_parent_event_append_cannot_change_successor_source(self):
        report=autoresearch(self.campaign,test_project.ProjectTests.config(self,'harness'),agent=self.agent)
        parent=self.campaign/report['state']['last_model']
        store=Store(parent/'memory')
        try:
            store.event('late_annotation',{'note':'not part of committed project episode'})
            store.verify()
        finally:store.close()
        project=ResearchProject(self.campaign,agent=self.agent)
        try:
            with self.assertRaisesRegex(ContractError,'parent model history differs'):
                project.model_config(report['state'],self.campaign/'episodes/model-3')
        finally:project.store.close()
