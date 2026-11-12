"""Prevent budget reuse, GPU overlap and accidental protocol changes."""
from collections import Counter
from pathlib import Path
import pytest

from . import launch
from amazon_family_compare.core import read, save
from amazon_family_compare.prepare import sha


def idle_inventory():
    return {g:[2,0] for g in range(8)}


def test_six_new_jobs_three_per_method():
    plan=read(launch.ROOT/'plan.json')
    assert len(plan['jobs'])==6
    assert Counter(j['method'] for j in plan['jobs'])=={'recharness':3,'mlestar':3}
    assert plan['total_new_budget_gpu_seconds']==6*43200
    assert len({j['root'] for j in plan['jobs']})==6
    assert {j['additional_repetition'] for j in plan['jobs']}=={1,2,3}


@pytest.mark.parametrize('method',['recharness','mlestar'])
def test_repeats_preserve_real_protocol_and_start_with_empty_state(method):
    template=(launch.RECHARNESS/'runs/nextitnet/config.json' if method=='recharness'
              else launch.COMPARISON/'runs/nextitnet-mlestar/config.json')
    original=read(template)
    jobs=[j for j in read(launch.ROOT/'plan.json')['jobs'] if j['method']==method]
    for job in jobs:
        c=read(job['config'])
        for key in ['data_dir','worker_python','controller_python','search_budget_gpu_seconds',
                    'trial_floor_reference_seconds','trial_timeout_seconds','minimum_trial_seconds',
                    'final_timeout_seconds','llm_model','llm_effort','llm_timeout','llm_attempts',
                    'retry_backoff_seconds','single_model_search','global_llm_concurrency',
                    'llm_slot_directory']:
            assert c[key]==original[key],key
        for key in ['datasets','preprocessing_seed','training_seed','evaluation_seed',
                    'validation_users_per_dataset','test_users_per_dataset','negative_candidates',
                    'cold_start','training_environment_defaults']:
            assert c['alignment'][key]==original['alignment'][key],key
        assert len(c['gpus'])==4 and len(c['cpu_cores'])==64
        assert c['baseline']['sha256']==original['baseline']['sha256']
        assert sha(c['baseline']['source'])==original['baseline']['sha256']
        assert Path(c['baseline']['source']).read_bytes()==Path(original['baseline']['source']).read_bytes()
        assert not (Path(c['root'])/method).exists()
        assert c['replication']['fresh_baseline_weights']
        assert c['replication']['fresh_baseline_and_llm_calls']
        assert not c['replication']['cross_run_response_or_memory_reuse']
        assert c['replication']['overall_repetition']==job['additional_repetition']+1
        if method=='recharness':
            assert c['prior_protocol_charge']['gpu_seconds']==0
            assert c['arm_policy']==original['arm_policy']
            assert c['memory_policy']==original['memory_policy']
            assert c['jump_fork']==original['jump_fork']


def test_mlestar_waits_for_hstu_even_when_all_gpus_look_idle():
    assert launch.wait_reason('mlestar',[0,1,2,3],idle_inventory(),
                              {'status':'running'},[])=='waiting_for_hstu'


def test_mlestar_waits_if_hstu_status_is_unknown():
    assert launch.wait_reason('mlestar',[0,1,2,3],idle_inventory(),{},[])=='waiting_for_hstu'


def test_mlestar_starts_after_hstu_finishes_and_block_is_free():
    assert launch.wait_reason('mlestar',[0,1,2,3],idle_inventory(),
                              {'status':'complete'},[]) is None


def test_gpu_still_busy_after_hstu_exit_prevents_launch():
    devices=idle_inventory();devices[0]=[1024,25]
    assert launch.wait_reason('mlestar',[0,1,2,3],devices,
                              {'status':'complete'},[])=='waiting_for_idle_gpus'


def test_an_existing_experiment_retains_ownership_during_llm_phase():
    assert launch.wait_reason('mlestar',[0,1,2,3],idle_inventory(),
               {'status':'complete'},['another-owner'])=='waiting_for_existing_experiment'


def test_recharness_uses_other_block_without_waiting_for_hstu():
    assert launch.wait_reason('recharness',[4,5,6,7],idle_inventory(),
                              {'status':'running'},[]) is None


def test_owner_detection_uses_entire_registered_gpu_block(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'WORKSPACE',tmp_path)
    monkeypatch.setattr(launch,'RECHARNESS',tmp_path/'recharness')
    monkeypatch.setattr(launch,'COMPARISON',tmp_path/'comparison')
    save(launch.RECHARNESS/'runs/hstu/status.json',{'status':'running','gpus':[0,1,2,3]})
    assert launch.owners([0,1,2,3])==['recharness/runs/hstu']
    assert launch.owners([4,5,6,7])==[]


def test_started_launcher_is_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    save(tmp_path/'launcher.json',{'pid':123})
    with pytest.raises(RuntimeError,match='Already launched'):
        launch.start()
    assert read(tmp_path/'launcher.json')=={'pid':123}


def test_worker_refuses_existing_budget_without_resetting_it(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch,'verify',lambda _:None)
    folder=tmp_path/'runs/existing';config=folder/'config.json'
    save(config,{'root':str(folder)})
    save(tmp_path/'frozen.json',{'files':{}})
    save(tmp_path/'plan.json',{'jobs':[{'id':'existing','method':'mlestar','root':str(folder),
        'config':str(config),'additional_repetition':1}]})
    budget=folder/'mlestar/budget.json'
    original={'total_gpu_seconds':43200,'spent_gpu_seconds':1234,'reservations':{},'settled':{}}
    save(budget,original)
    with pytest.raises(RuntimeError,match='never reset'):
        launch.worker('existing')
    assert read(budget)==original
