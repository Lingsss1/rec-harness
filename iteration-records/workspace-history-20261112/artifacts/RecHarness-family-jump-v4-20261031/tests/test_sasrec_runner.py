"""Family boundaries, common budgets and immutable final selection."""
from pathlib import Path
import pytest
import family_runner as runner
from family_launch import ROOT, COMPARISON, blockers


def config(tmp_path, family='gru4rec'):
    return {**runner.read(ROOT/'runs'/family/'config.json'), 'root':str(tmp_path)}


@pytest.mark.parametrize('family',['gru4rec','bert4rec','nextitnet','hstu'])
def test_same_corrected_baseline_and_resources_as_other_methods(tmp_path,family):
    c=runner.FamilyJumpController(config(tmp_path,family))
    peer=runner.read(COMPARISON/'runs'/(family+'-mlestar')/'config.json')
    assert c.baseline()==Path(peer['baseline']['source']).read_text()
    assert c.c['base_model']['sha256']==peer['baseline']['sha256']
    for name in ['data_dir','search_budget_gpu_seconds','trial_timeout_seconds','minimum_trial_seconds',
                 'trial_floor_reference_seconds','final_timeout_seconds','llm_model','llm_effort',
                 'llm_timeout','llm_attempts','single_model_search','global_llm_concurrency']:
        assert c.c[name]==peer[name], name
    assert len(c.c['gpus'])==4 and len(c.c['cpu_cores'])==64
    assert c.environment(c.c['gpus'][0])['GAGC_MODEL_FAMILY']==family
    assert c.c['replication']['fresh_baseline_weights']
    assert not c.c['replication']['cross_run_response_or_memory_reuse']


def test_four_fresh_full_budget_searches():
    plan=runner.read(ROOT/'plan.json')
    assert len(plan['jobs'])==4
    configs=[runner.read(j['config']) for j in plan['jobs']]
    assert {c['search_budget_gpu_seconds'] for c in configs}=={43200}
    assert len({c['root'] for c in configs})==4
    assert plan['total_gpu_seconds']==172800
    assert all(c['prior_protocol_charge']['gpu_seconds']==0 for c in configs)
    assert all(c['alignment']['cold_start'].startswith(c['model_family']) for c in configs)


@pytest.mark.parametrize('family',['gru4rec','bert4rec','nextitnet','hstu'])
def test_transport_retry_preserves_exact_family_request(tmp_path,monkeypatch,family):
    c=runner.FamilyJumpController(config(tmp_path,family));calls=[]
    def request(self,instruction,output_schema,search,key):
        calls.append((instruction,output_schema,search,key,self.scope.get()))
        if len(calls)==1:raise runner.transport.TransportUnavailable('fault injection')
        return {'edits':[]}
    monkeypatch.setattr(runner.transport.ResilientSelfController,'llm',request)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    assert c.llm('a mutation',{},False,'round-1')=={'edits':[]}
    assert calls[0]==calls[1]
    assert 'FAMILY: '+family in calls[0][0]
    assert 'Start from RecHarness sasrec_perdataset' not in runner.transport.TASK
    assert 'appended-mask inference' in calls[0][0]


def test_finish_blocks_changed_checkpoint_before_test(tmp_path):
    c=runner.FamilyJumpController(config(tmp_path));trial=tmp_path/'selected'
    (trial/'working').mkdir(parents=True);(trial/'working/predict.py').write_text('pass\n')
    checkpoint=trial/'working/a.pt';checkpoint.write_text('changed')
    runner.save(trial/'single-model-audit.json',{'eligible':True,
        'predictor_sha256':runner.sha(trial/'working/predict.py'),
        'checkpoint_sha256':{str(checkpoint):'oldhash'}})
    c.best={'output':str(trial)}
    with pytest.raises(RuntimeError,match='checkpoint changed'):c.finish()


def test_prior_job_owns_all_four_gpus_during_llm_phase(tmp_path,monkeypatch):
    import family_launch as launch
    prior=tmp_path/'runs/hstu-mlestar/status.json'
    runner.save(prior,{'status':'running','gpus':[0,1,2,3]})
    monkeypatch.setattr(launch,'COMPARISON',tmp_path)
    assert launch.blockers([0,1,2,3])==['hstu-mlestar']
    assert launch.blockers([4,5,6,7])==[]
    runner.save(prior,{'status':'complete','gpus':[0,1,2,3]})
    assert launch.blockers([0,1,2,3])==[]
