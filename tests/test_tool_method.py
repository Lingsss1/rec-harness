import unittest
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.evolution import Evolution,STAGES
from rec_harness.tool_method import CONTRACT,FILENAME,validate
import test_evolution_auto


class ToolMethodTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_evolution_auto.EvolutionAutomationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.card={'schema':CONTRACT,'route':'original','measurement':'spread','definition':'max(x) - min(x)',
            'inputs':'finite nonempty aggregate values','complexity':'O(n) time, O(1) auxiliary memory',
            'assumptions':['comparable metric definitions'],'failure_modes':['not a causal measurement'],
            'positive_control':'[1, 3] yields 2','null_control':'[4, 4] yields 0',
            'independent_reference':'host scalar range','sources':[],'predecessor_implementation_hash':None}

    def test_new_autonomous_defaults_are_frozen_without_mutating_input(self):
        from rec_harness.evolution_auto import AutonomousEvolution
        config=read_json(self.fixture.config());config['kind']='tool'
        config['context']={}
        root=self.fixture.root/'default-tool'
        AutonomousEvolution.init(root,config)
        frozen=read_json(root/'evolution-auto.json')
        self.assertEqual(frozen['plan']['tool_method_contract'],CONTRACT)
        self.assertNotIn('tool_method_contract',config['plan'])
        # Resuming an already frozen legacy controller never injects defaults.
        legacy=self.fixture.root/'legacy-tool'
        AutonomousEvolution.init(legacy,config,_freeze_tool_method=False)
        research=AutonomousEvolution(legacy,agent=object());self.addCleanup(research.store.close)
        self.assertNotIn('tool_method_contract',research.config['plan'])
        for value in [None,False,'unknown']:
            config['plan']['tool_method_contract']=value
            target=self.fixture.root/('invalid-'+str(value))
            with self.assertRaisesRegex(ContractError,'require diagnostic'):
                AutonomousEvolution.init(target,config)
            self.assertFalse(target.exists())

    def test_project_freezes_default_and_preserves_legacy_child_plan(self):
        from rec_harness.project import ResearchProject
        from unittest.mock import patch
        import test_project
        fixture=test_project.ProjectTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=read_json(fixture.config('tool'))
        ResearchProject.init(fixture.campaign,config)
        project=ResearchProject(fixture.campaign,agent=object());self.addCleanup(project.store.close)
        self.assertEqual(project.config['tool']['plan']['tool_method_contract'],CONTRACT)
        self.assertNotIn('tool_method_contract',config['tool']['plan'])
        child={**config['tool'],'context':{}}
        with patch('rec_harness.autonomous.autoresearch',return_value={'status':'fixture'}) as run:
            directory=fixture.root/'legacy-child'
            project.child(directory,child)
        self.assertIsNone(run.call_args.args[1])
        self.assertNotIn('tool_method_contract',read_json(directory/'campaign/evolution-auto.json')['plan'])

    def test_routes_require_definitions_sources_and_exact_predecessor(self):
        self.assertEqual(validate(self.card),self.card)
        for field in ['definition','complexity','independent_reference']:
            with self.subTest(field=field):
                with self.assertRaises(ContractError):validate({**self.card,field:''})
        paper={**self.card,'route':'paper'}
        with self.assertRaisesRegex(ContractError,'primary source'):validate(paper)
        paper['sources']=[{'locator':'doi:fixture','method_mapping':'Equation 1 to scalar range; fixture only'}]
        validate(paper)
        inherited={**self.card,'route':'existing_tool','predecessor_implementation_hash':'old'}
        with self.assertRaisesRegex(ContractError,'predecessor'):validate(inherited,'different')
        validate(inherited,'old')

    def test_missing_card_is_rejected_before_validation(self):
        plan=read_json(self.fixture.config())['plan'];rule=next(iter(plan['stages'].values()))
        plan['stages']={stage:dict(rule) for stage in STAGES['tool']};plan['tool_method_contract']=CONTRACT
        class Author:
            def run(self,role,context,directory):
                model=Path(directory)/'model';model.mkdir(parents=True)
                (model/'diagnostic.py').write_text('def measure(request): return {}\n')
                return {'changed_files':['diagnostic.py']}
        evolution=Evolution(self.fixture.campaign);self.addCleanup(evolution.store.close)
        result=evolution.run('missing','tool',{},plan,agent=Author())
        self.assertEqual(result['status'],'rejected');self.assertIn('method card',result['reason'])
        self.assertEqual(result['budget']['charged_or_reserved'],1)
        self.assertFalse(any(e['kind']=='validation_stage_started' for e in evolution.store.events()))

    def test_card_archived_separately_bound_to_event_and_used_by_registered_tool(self):
        config=read_json(self.fixture.config());plan=config['plan'];plan['tool_method_contract']=CONTRACT
        plan['tool_contract']='public-aggregate-measurement-v1'
        rule=next(iter(plan['stages'].values()));stages={}
        for stage in STAGES['tool']:
            result={'passed':True,'gain':1,'regression':0,'total_cost':7,'evaluation_ids':[stage+'-a',stage+'-b'],
                'identity':plan['identity'],'comparison_groups':['no_tool','existing_tool','candidate_tool'],'online':True}
            script=self.fixture.root/(stage+'.py')
            import json
            script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('+repr(json.dumps(result))+')\n')
            stages[stage]={**rule,'command':['{python}',str(script),'{output}']}
        plan['stages']=stages
        outer=self
        class Author:
            calls=0
            def run(self,role,context,directory):
                self.calls+=1;outer.assertEqual(context['tool_method_contract']['schema'],CONTRACT)
                model=Path(directory)/'model';model.mkdir(parents=True)
                (model/'diagnostic.py').write_text('def measure(request): return {}\n')
                write_json(model/FILENAME,outer.card)
                return {'changed_files':['diagnostic.py',FILENAME]}
        author=Author();evolution=Evolution(self.fixture.campaign);self.addCleanup(evolution.store.close)
        result=evolution.run('method','tool',{},plan,agent=author)
        self.assertEqual(result['status'],'promoted');self.assertEqual(author.calls,1)
        graph=evolution.store.snapshot()
        method=next(n for n in graph['nodes'] if n['type']=='Method')
        self.assertEqual(method['attrs']['definition'],self.card['definition'])
        self.assertEqual(method['attrs']['origin'],'agent_hypothesis')
        self.assertFalse(method['attrs']['independent_unit'])
        self.assertFalse(any(n['type']=='Claim' for n in graph['nodes']))
        bundle=evolution.store.bundle('method',digest(plan),'tool',query='spread')
        self.assertTrue(any(n['id']==method['id'] for n in bundle['nodes']))
        before=evolution.store.verify()
        self.assertEqual(evolution.run('method','tool',{},plan,agent=author),result)
        self.assertEqual(evolution.store.verify(),before);self.assertEqual(author.calls,1)
        candidate=evolution.verified_candidate('method')
        self.assertNotIn(FILENAME,candidate['implementation'])
        self.assertEqual(read_json(evolution.artifacts.verify(candidate['proposal_ref'])),self.card)
        from rec_harness.diagnostics import freeze_tool
        from rec_harness.core import Artifacts
        tool=freeze_tool({'registry':self.fixture.campaign,'identity':plan['identity'],
            'execution':{'recipe':{},'python':'unused','uid_registry':'unused'},'cost':1,'timeout':5},Artifacts(self.fixture.root/'new-objects'))
        self.assertEqual(tool['registration']['method_card'],self.card)
        self.assertIn('not established',tool['registration']['method_card_status'])
        projection=self.fixture.campaign/'candidates/method/candidate.json'
        changed=read_json(projection);changed['proposal_ref']={}
        write_json(projection,changed)
        with self.assertRaisesRegex(ContractError,'proposal projection'):evolution.verified_candidate('method')
        with self.assertRaisesRegex(ContractError,'proposal projection'):
            evolution.propose('method','tool',{},plan,agent=author)
        self.assertEqual(author.calls,1)

    def test_existing_tool_author_receives_exact_source_hash_and_preserves_helpers(self):
        plan=read_json(self.fixture.config())['plan'];rule=next(iter(plan['stages'].values()))
        plan['stages']={stage:dict(rule) for stage in STAGES['tool']};plan['tool_method_contract']=CONTRACT
        evolution=Evolution(self.fixture.campaign);self.addCleanup(evolution.store.close)
        source_dir=self.fixture.root/'source';source_dir.mkdir()
        (source_dir/'diagnostic.py').write_text('def measure(request): return {}\n')
        (source_dir/'helper.py').write_text('VERSION=1\n')
        source=evolution.artifacts.snapshot(source_dir,['diagnostic.py','helper.py'])
        outer=self
        class Author:
            calls=0
            def run(self,role,context,directory):
                self.calls+=1
                inherited=context['tool_method_contract']['inherited_implementation_hash']
                outer.assertEqual(inherited,digest(source))
                card={**outer.card,'route':'existing_tool','predecessor_implementation_hash':inherited}
                model=Path(directory)/'model';outer.assertTrue((model/'helper.py').exists())
                (model/'diagnostic.py').write_text('from helper import VERSION\ndef measure(request): return {}\n')
                write_json(model/FILENAME,card)
                return {'changed_files':['diagnostic.py',FILENAME]}
        agent=Author();candidate=evolution.propose('inherited','tool',{},plan,agent=agent,source=source)
        self.assertEqual(candidate['implementation']['helper.py'],source['helper.py'])
        card=read_json(evolution.artifacts.verify(candidate['proposal_ref']))
        self.assertEqual(card['predecessor_implementation_hash'],digest(source))
        self.assertNotIn(FILENAME,candidate['implementation'])
        self.assertEqual(evolution.propose('inherited','tool',{},plan,agent=agent,source=source),candidate)
        self.assertEqual(agent.calls,1)

    def test_paper_method_graph_keeps_unverified_sources_and_deduplicates_citations(self):
        from rec_harness.evolution_graph import lifecycle_graph
        card={**self.card,'route':'paper','sources':[
            {'locator':'doi:fixture','method_mapping':'Equation 1'},
            {'locator':'doi:fixture','method_mapping':'Equation 1'},
            {'locator':'doi:fixture','method_mapping':'Section 2'}]}
        events=[{'kind':'candidate_proposed','hash':'a'*64,'created_at':'1','payload':{
            'candidate_id':'paper','plan_hash':'plan','context_hash':'context','kind':'tool','source_hash':None}},
            {'kind':'candidate_implemented','hash':'b'*64,'created_at':'2','payload':{
                'candidate_id':'paper','implementation_hash':'code','proposal_ref':{'sha256':'card'}}}]
        graph=lifecycle_graph(events,proposals={'paper':card})
        source=next(n for n in graph['nodes'] if n['type']=='Source')
        self.assertEqual(sum(n['type']=='Source' for n in graph['nodes']),1)
        self.assertEqual(source['attrs']['verification_status'],'agent_reported_unverified')
        self.assertIsNone(source['attrs']['retrieved_content_ref'])
        method=next(n for n in graph['nodes'] if n['type']=='Method')
        self.assertEqual(method['attrs']['origin'],'paper_claim')
        edge=next(e for e in graph['edges'] if e['relation']=='DERIVED_FROM')
        self.assertEqual(edge['attrs']['method_mappings'],['Equation 1','Section 2'])
        self.assertEqual(edge['attrs']['evidence_refs'],[source['id']])
        self.assertEqual(graph,lifecycle_graph(events,proposals={'paper':card}))
